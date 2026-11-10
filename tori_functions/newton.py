"""Invariant tori of a 4D map by Newton's method on their Fourier series, on a GPU (CuPy).

The map comes as two functions of the normalized coordinates (x, px, y, py):
    one_turn(x, px, y, py) -> (x, px, y, py)        jacobian(x, px, y, py) -> A[i][k] = d(output i)/d(input k)
A is the derivative of the map (its jacobian matrix) everywhere in this file.
A torus is sampled on N angles per plane:
    K[t, c, i, j]        coordinate c (x, px, y, py) of torus t at the angles (2 pi i/N, 2 pi j/N)
    Q[t, a], J[t, a]     its tunes and its actions
It is invariant when the map advances every angle by 2 pi Q:   one_turn(K(theta)) = K(theta + 2 pi Q),
and it is labelled by its actions, the enclosed areas / 2 pi:  J_a = < sum_b px_b dx_b/dtheta_a >.
Its spectrum: z_a = x_a - i px_a = sum_m c_m e^{i m.theta}.
A torus with J_y = 0 (an x torus) has one angle, D = 1: y stays at rest. A y torus is solved as an x torus of the
map with the planes x and y exchanged.
"""
import math
import cupy as cp
import numpy as np


def use_gpu(device):
    """Solve on GPU number `device` (numbered as in nvidia-smi)."""
    cp.cuda.Device(device).use()


# ------------------------------------------------------------------------------------------------------------------
# Fourier tools. K has shape (tori, 4, N, ..., N): the last D axes are the angles.
# ------------------------------------------------------------------------------------------------------------------

def angles(N, D):
    """The grid angles theta_a[i, j, ...], shape (D, N, ..., N)."""
    theta = 2*cp.pi*cp.arange(N)/N
    return cp.stack(cp.meshgrid(*[theta]*D, indexing='ij'))


def modes(N, D):
    """The integer Fourier modes m_a[i, j, ...] of the grid, in FFT order, shape (D, N, ..., N)."""
    m = cp.fft.fftfreq(N, 1/N)
    return cp.stack(cp.meshgrid(*[m]*D, indexing='ij'))


def fourier(K):
    return cp.fft.fftn(K, axes=tuple(range(2, K.ndim)))


def inverse_fourier(F):
    return cp.fft.ifftn(F, axes=tuple(range(2, F.ndim))).real


def shift(K, Q):
    """K(theta + 2 pi Q): Fourier mode m gets exp(2 pi i m.Q)."""
    phase = cp.exp(2j*cp.pi*cp.tensordot(Q, modes(K.shape[-1], Q.shape[1]), 1))
    return inverse_fourier(fourier(K)*phase[:, None])


def derivative(K, a):
    """dK/dtheta_a: Fourier mode m gets i m_a."""
    return inverse_fourier(1j*modes(K.shape[-1], K.ndim-2)[a]*fourier(K))


def grid_mean(values):
    """For every torus: the average over the grid."""
    return cp.sum(values.reshape(len(values), -1), axis=1)/values[0, 0].size


def per_torus(numbers, array):
    """One number per torus, shaped to multiply an array (tori, ...)."""
    return numbers.reshape((-1,)+(1,)*(array.ndim-1))


def squared_length(pair):
    """For every torus: the sum of squares of all the entries of a pair of arrays."""
    return sum(cp.sum(part.reshape(len(part), -1)**2, axis=1) for part in pair)


# ------------------------------------------------------------------------------------------------------------------
# The torus: circles, actions, angle origins, the map on it, its equations
# ------------------------------------------------------------------------------------------------------------------

def circle_torus(J, N):
    """The tori of the linear map: x_a = sqrt(2 J_a) cos(theta_a), px_a = -sqrt(2 J_a) sin(theta_a) (y at rest if
    D = 1)."""
    D = J.shape[1]
    theta = angles(N, D)
    amplitude = cp.sqrt(2*J).reshape(J.shape+(1,)*D)
    K = cp.zeros((len(J), 4)+(N,)*D)
    K[:, :2*D] = cp.stack([amplitude*cp.cos(theta), -amplitude*cp.sin(theta)], axis=2).reshape((len(J), 2*D)+(N,)*D)
    return K


def actions(K):
    """J_a = < sum_b px_b dx_b/dtheta_a >, the enclosed areas / 2 pi."""
    x, p = K[:, 0::2], K[:, 1::2]
    return cp.stack([grid_mean(p*derivative(x, a)) for a in range(K.ndim-2)], axis=1)


def on_actions(K, J):
    """K with every plane with an angle scaled by its own factor s_b, so that its actions are exactly J
    (the actions are linear in the s_b^2: a small linear system)."""
    D = J.shape[1]
    x, p = K[:, 0::2], K[:, 1::2]
    M = cp.stack([cp.mean((p*derivative(x, a)).reshape(len(K), 2, -1), axis=2) for a in range(D)], axis=1)
    s = cp.sqrt(cp.linalg.solve(M[:, :, :D], (J-cp.sum(M[:, :, D:], axis=2))[..., None])[..., 0])
    scale = cp.ones((len(K), 4))
    scale[:, :2*D] = cp.repeat(s, 2, axis=1)                                   # x_b and px_b alike
    return K*scale.reshape(scale.shape+(1,)*D)


def phases(K):
    """Where every angle starts: < x_a sin(theta_a) + px_a cos(theta_a) >, zero when the first harmonic of z_a is
    real and positive (else the torus could slide along itself)."""
    D = K.ndim-2
    theta = angles(K.shape[-1], D)
    return cp.stack([grid_mean((K[:, 2*a]*cp.sin(theta[a])+K[:, 2*a+1]*cp.cos(theta[a]))[:, None])
                     for a in range(D)], axis=1)


def apply(one_turn, K):
    """The map at every grid point of every torus."""
    return cp.stack(one_turn(*cp.moveaxis(K, 1, 0)), axis=1)


STEP = 1e-3                     # finite-difference step of the map derivative, in the units of K


def map_derivative(one_turn, K, mapped):
    """A: the derivative of the map (jacobian matrix) at every grid point, shape (tori, 4, 4, N, ..., N), given
    mapped = one_turn(K). Along the torus it is free: A dK/dtheta_a = d(mapped)/dtheta_a. Across, 4 - D directions by
    central differences of the map (2 calls each): the angle directions turned by 90 degrees in each plane, and the
    planes at rest. (Forward differences, 1 call each, are only 1e-3 accurate: Newton at round-off needs 2x the steps.)"""
    D = K.ndim-2
    along = [derivative(K, a) for a in range(D)]
    across = [cp.stack([t[:, 1], -t[:, 0], t[:, 3], -t[:, 2]], axis=1) for t in along]
    for c in range(2*D, 4):                                         # D = 1: y and py, at rest
        e = cp.zeros_like(K)
        e[:, c] = 1
        across.append(e)
    across = [n/cp.sqrt(cp.sum(n**2, axis=1, keepdims=True)) for n in across]
    shifted = apply(one_turn, cp.concatenate([K+s*STEP*n for n in across for s in (1, -1)]))   # one map call for all
    parts = cp.split(shifted, 2*len(across))                        # K + h n, K - h n, for every direction n
    images = [derivative(mapped, a) for a in range(D)]+[(parts[2*i]-parts[2*i+1])/(2*STEP) for i in range(len(across))]
    # A C = B at every grid point, C: the 4 directions as columns, B: their images. Solved as C^T A^T = B^T.
    C, B = [cp.moveaxis(cp.stack(v, axis=-1), 1, -2) for v in (along+across, images)]   # (tori, grid, 4, 4)
    A = cp.linalg.solve(cp.swapaxes(C, -1, -2), cp.swapaxes(B, -1, -2))
    return cp.moveaxis(cp.swapaxes(A, -1, -2), (-2, -1), (1, 2))


def equations(mapped, K, Q, J):
    """What Newton drives to zero, given mapped = one_turn(K): the invariance at every grid point, and the conditions:
    the angle origins and the actions minus their targets."""
    return mapped-shift(K, Q), cp.concatenate([phases(K), actions(K)-J], axis=1)


def condition_rows(K):
    """Weights W[t, j]: N^D grid_mean(W[t, j] dK[t]) is the change of condition j of torus t."""
    D, N = K.ndim-2, K.shape[-1]
    theta = angles(N, D)
    rows = cp.zeros((len(K), 2*D)+K.shape[1:])
    for a in range(D):
        rows[:, a, 2*a] = cp.sin(theta[a])
        rows[:, a, 2*a+1] = cp.cos(theta[a])
        rows[:, D+a, 0::2] = -derivative(K[:, 1::2], a)          # dJ_a = < dpx . dx/dtheta_a - dx . dpx/dtheta_a >
        rows[:, D+a, 1::2] = derivative(K[:, 0::2], a)
    return rows/N**D


# ------------------------------------------------------------------------------------------------------------------
# One Newton step for all the tori: the linearised equations, solved by preconditioned least squares (CGLS)
# ------------------------------------------------------------------------------------------------------------------

CHECK = 10          # CGLS iterations between two convergence checks (a check waits for the GPU)
SPLIT = 10**5       # a batch of tori with fewer numbers is not split when half of it has converged


def newton_step(A, K, Q, invariance, conditions, rest_tunes, tolerance, max_iterations, floor, resonance):
    """The Newton correction (dK, dQ) of every torus: the least-squares solution of the linearised equations
        A(theta) dK(theta) - dK(theta + 2 pi Q) - 2 pi dQ . dK/dtheta(theta + 2 pi Q) = -invariance
        (the linearised conditions)                                                    = -conditions
    with A(theta) the derivative of the map (jacobian matrix) at the torus point K(theta), by conjugate gradients on the normal equations (CGLS), every torus
    with its own step lengths. The matrix is never stored: it is applied (forward) and transposed.
    Preconditioner: the inverse of the same equations for a pure rotation; it multiplies mode m of plane a by
    1 / (exp(-2 pi i Q_a) - exp(2 pi i m.Q)), at most 1/floor (the small divisors of nearby resonances), and by 1 for
    m = -e_a (the mode that slides the torus along itself, fixed by the conditions). The plane at rest of an x torus
    rotates by its linear tune (rest_tunes). Resonance filter: a mode whose divisor is below `resonance` is not solved
    (the round-off of the map, divided by the divisor, would fill it): it stays zero.
    GPU speed (no change to the arithmetic): one CGLS iteration is recorded as a CUDA graph and replayed; when half of
    a large batch has converged, the other tori go on alone."""
    D = Q.shape[1]
    axes = tuple(range(-D, 0))                                            # the angle axes
    phase = cp.exp(2j*cp.pi*cp.tensordot(Q, modes(K.shape[-1], D), 1))    # mode m -> exp(2 pi i m.Q): the turn
    plane_tunes = Q if D == 2 else cp.concatenate([Q, rest_tunes], axis=1)
    rotation = cp.exp(-2j*cp.pi*plane_tunes).reshape((len(Q), 2)+(1,)*D)
    divisor = cp.abs(rotation-phase[:, None])
    weights = 1/cp.maximum(divisor, floor)
    weights[divisor < resonance] = 0                                     # the resonance filter
    for a in range(D):                                                    # the mode m = -e_a of plane a
        index = [slice(None), a]+[0]*D
        index[2+a] = -1
        weights[tuple(index)] = 1

    def turned(v):
        return cp.fft.ifftn(cp.fft.fftn(v, axes=axes)*phase[:, None], axes=axes).real

    tune_columns = cp.stack([2*cp.pi*turned(derivative(K, a)) for a in range(D)], axis=1)
    rows = condition_rows(K)
    # Everything a torus needs (so that a smaller batch is an index); the tune columns and the condition rows get
    # unit length, like the rest of the matrix; the weights include the 1/N^D of the FFTs.
    data = dict(A=A, phase=phase[:, None], back=phase.conj()[:, None], weights=weights/K[0, 0].size,
                tune_columns=tune_columns, rows=rows,
                tune_scale=1/cp.sqrt(cp.sum(tune_columns.reshape(*Q.shape, -1)**2, axis=2)),
                row_scale=1/cp.sqrt(cp.sum(rows.reshape(*rows.shape[:2], -1)**2, axis=2)))

    def packed(*vectors):
        """x + i px of both planes of these vectors, stacked: the input of one FFT."""
        z = cp.empty((len(vectors), len(vectors[0]), 2)+vectors[0].shape[2:], dtype=complex)
        for zi, v in zip(z, vectors):
            zi.real[...], zi.imag[...] = v[:, 0::2], v[:, 1::2]
        return z

    def unpacked(z):
        v = cp.empty((len(z), 4)+z.shape[2:])
        v[:, 0::2], v[:, 1::2] = z.real, z.imag
        return v

    def summed(v):
        return cp.sum(v.reshape(*v.shape[:2], -1), axis=2)

    def forward(d, dK, dQ):
        """The linearised equations applied to (P dK, dQ), P the preconditioner."""
        F = cp.empty((2,)+d['weights'].shape, dtype=complex)
        cp.multiply(cp.fft.fftn(packed(dK), axes=axes)[0], d['weights'], out=F[0])     # P dK
        cp.multiply(F[0], d['phase'], out=F[1])                                          # P dK, turned
        z, z_turned = cp.fft.ifftn(F, axes=axes, norm='forward')
        dK = unpacked(z)
        invariance = (apply_matrix(d['A'], dK)-unpacked(z_turned)
                      - apply_matrix(d['tune_columns'], dQ*d['tune_scale'], transpose=True))
        return invariance, summed(apply_matrix(d['rows'], dK))*d['row_scale']

    def transpose(d, invariance, conditions):
        """The transpose of forward."""
        dK = (apply_matrix(d['A'], invariance, transpose=True)
              + apply_matrix(d['rows'], conditions*d['row_scale'], transpose=True))
        F = cp.fft.fftn(packed(dK, invariance), axes=axes)            # minus the turn back, then P: one inverse FFT
        dK = unpacked(cp.fft.ifftn((F[0]-F[1]*d['back'])*d['weights'], axes=axes, norm='forward'))
        return dK, -summed(apply_matrix(d['tune_columns'], invariance))*d['tune_scale']

    # CGLS for forward(dK~, dQ~) = (-invariance, -conditions), all the tori together. Its state (solution, residual,
    # direction, gamma, active, iteration count) is updated in place, so that a recorded iteration can be replayed.
    r = (-invariance, -conditions*data['row_scale'])
    s = transpose(data, *r)
    gamma = squared_length(s)
    data['target'] = tolerance**2*gamma
    state = [cp.zeros_like(K), cp.zeros_like(Q), *r, *s, gamma, gamma > data['target'], cp.zeros((), dtype=int)]

    def iteration(d, state):
        x, r, p, (gamma, active, count) = state[0:2], state[2:4], state[4:6], state[6:]
        count += cp.any(active)
        q = forward(d, *p)
        alpha = cp.where(active, gamma/cp.maximum(squared_length(q), 1e-300), 0)
        for xi, pi in zip(x, p):
            xi += per_torus(alpha, xi)*pi
        for ri, qi in zip(r, q):
            ri -= per_torus(alpha, ri)*qi
        s = transpose(d, *r)
        new_gamma = squared_length(s)
        active &= new_gamma > d['target']
        beta = cp.where(active, new_gamma/cp.maximum(gamma, 1e-300), 0)  # a converged torus gets p = 0: it stops
        for si, pi in zip(s, p):
            pi *= per_torus(beta, pi)
            pi += si
            pi *= per_torus(active, pi)
        gamma[...] = cp.where(active, new_gamma, 0)

    batch, d, done = cp.arange(len(K)), data, 0             # the tori iterated, their data, the iterations done
    solution = [cp.zeros_like(K), cp.zeros_like(Q)]
    while True:
        iteration(d, state)         # the first iteration of a batch runs directly: it compiles kernels and FFT plans
        done += 1
        replay, active = None, int(cp.sum(state[7]))
        while done < max_iterations and active and not (2*active <= len(batch) and len(batch)*K[0].size > SPLIT):
            replay = replay or replayable(lambda: iteration(d, state))
            for _ in range(min(CHECK, max_iterations-done)):
                replay()
            done += min(CHECK, max_iterations-done)
            active = int(cp.sum(state[7]))
        solution[0][batch], solution[1][batch] = state[0], state[1]
        if done >= max_iterations or not active:
            break
        keep = cp.flatnonzero(state[7])                 # half of a large batch has converged: the others go on
        batch, d = batch[keep], {key: value[keep] for key, value in d.items()}
        state = [value[keep] for value in state[:-1]]+state[-1:]
    precondition = cp.fft.ifftn(cp.fft.fftn(packed(solution[0]), axes=axes)[0]*data['weights'], axes=axes,
                                norm='forward')
    return unpacked(precondition), solution[1]*data['tune_scale']


MATRIX = r'''
const int rows = transpose ? k : n, terms = transpose ? n : k;
const long long p = i % points, row = (i / points) % rows, t = i / (points * rows);
double s = 0.;
for (int j = 0; j < terms; ++j) {
    const int a = transpose ? j : row, b = transpose ? row : j;
    const long long w = spread ? (t * terms + j) * points + p : t * terms + j;
    s += M[((t * n + a) * k + b) * points + p] * v[w];
}
out = s;
'''
_matrix_kernel = []


def apply_matrix(M, v, transpose=False):
    """sum_b M[t, a, b, ...] v[t, b, ...] at every grid point (transpose: sum_a M[t, a, b, ...] v[t, a, ...]), in one
    GPU kernel. M has shape (tori, n, k, N, ..., N); v has the grid axes too, or is one number per torus and index."""
    if not _matrix_kernel:
        _matrix_kernel.append(cp.ElementwiseKernel(
            'raw float64 M, raw float64 v, int32 n, int32 k, int32 points, bool spread, bool transpose',
            'float64 out', MATRIX, 'foliation_apply_matrix'))
    tori, n, k = M.shape[:3]
    out = cp.empty((tori, k if transpose else n)+M.shape[3:])
    _matrix_kernel[0](cp.ascontiguousarray(M), cp.ascontiguousarray(v), n, k, out[0, 0].size, v.ndim > 2,
                      transpose, out)
    return out


def replayable(function):
    """function() recorded once as a CUDA graph: returns a function that replays the recorded GPU work (launching its
    ~100 small kernels from Python takes longer than running them)."""
    pool, stream = cp.cuda.MemoryPool(), cp.cuda.Stream(non_blocking=True)
    with cp.cuda.using_allocator(pool.malloc), stream:
        stream.begin_capture()
        function()
        graph = stream.end_capture()

    def replay():
        graph.launch()
    replay.pool = pool                                  # the temporaries of the graph stay alive with it
    return replay


# ------------------------------------------------------------------------------------------------------------------
# Newton's method for many tori at once
# ------------------------------------------------------------------------------------------------------------------

def solve_tori(one_turn, jacobian, J, points, tolerance, steps=30, inner_tolerance=1e-10, inner_iterations=5000,
               floor=.03, chord=False, resonance=0., circle_tunes=False):
    """The invariant tori with the actions J (tori, D), on `points` angles per plane (odd), all solved together from
    the circles of the linear map. Returns K, Q and the final residual of every torus (NaN: a lost particle).
    Every Newton step: the linearised equations of the tori not converged yet (newton_step), the new torus rescaled
    to the actions J (on_actions), then the step halved until the residual decreases (line search; no halving below
    100 x tolerance, where the residual is at the noise of the map). A torus is
    converged below `tolerance`; one whose residual no step decreases any more (or with a lost particle) stops there.
    chord: the map derivative of a torus is computed again only when a step reduces its residual less than 10 times
    and it is still above 100 x tolerance (not converged yet); below, the old derivative is good enough. A torus that
    no step decreases with an old derivative gets a new one before it stops.
    circle_tunes: start from the tunes of the circles themselves (their mean phase advance under the map) instead of
    the linear tunes, which can be far from the tunes of the tori when the detuning is large."""
    J = cp.asarray(J, dtype=float)
    D = J.shape[1]
    K = circle_torus(J, points)
    Q0 = cp.asarray(np.tile(linear_tunes(jacobian), (len(J), 1)))
    Q, rest_tunes = Q0[:, :D].copy(), Q0[:, D:]
    floor_reached = cp.zeros(len(J), dtype=bool)
    mapped = apply(one_turn, K)
    if circle_tunes:                                    # z = x - i px = r exp(i theta): one map, exp(2 pi i Q)
        z, image = K[:, 0:2*D:2]-1j*K[:, 1:2*D:2], mapped[:, 0:2*D:2]-1j*mapped[:, 1:2*D:2]
        turn = image/(z*cp.exp(2j*cp.pi*Q).reshape(Q.shape+(1,)*D))
        Q = Q+cp.mean(cp.angle(turn).reshape(len(K), D, -1), axis=-1)/(2*cp.pi)
    previous = cp.full(len(J), cp.inf)                  # the residual of every torus at the step before
    for step in range(steps+1):
        invariance, conditions = equations(mapped, K, Q, J)
        largest = cp.maximum(cp.max(cp.abs(invariance).reshape(len(K), -1), axis=1), cp.max(cp.abs(conditions), axis=1))
        todo = cp.flatnonzero((largest >= tolerance) & ~floor_reached)       # NaN (lost) is never >= tolerance
        if len(todo) == 0 or step == steps:
            break
        k, q, j = K[todo], Q[todo], J[todo]
        fresh = cp.full(len(J), step == 0 or not chord)  # the tori whose derivative is computed at this step
        if chord and step == 0:
            A = map_derivative(one_turn, K, mapped)
        elif not chord:
            A = map_derivative(one_turn, k, mapped[todo])
        else:       # chord: the derivative again only for the tori a step did little for, still above 100 x tolerance
            slow = todo[(largest[todo] > .1*previous[todo]) & (largest[todo] > 100*tolerance)]
            if len(slow):
                A[slow] = map_derivative(one_turn, K[slow], mapped[slow])
                fresh[slow] = True
        previous = largest
        dK, dQ = newton_step(A[todo] if chord else A, k, q, invariance[todo], conditions[todo], rest_tunes[todo],
                             inner_tolerance, inner_iterations, floor, resonance)
        dK = on_actions(k+dK, j)-k                      # the step lands exactly on the actions J
        # Line search: the residual (conditions scaled as in newton_step) must decrease, else the step is halved.
        row_scale = 1/cp.sqrt(cp.sum(condition_rows(k).reshape(len(k), 2*D, -1)**2, axis=-1))

        def length(pair, index):
            return squared_length((pair[0], pair[1]*row_scale[index]))
        before = length((invariance[todo], conditions[todo]), slice(None))
        alpha, accepted = cp.ones(len(k)), cp.zeros(len(k), dtype=bool)
        trial_mapped = cp.empty_like(k)
        settled = largest[todo] <= 100*tolerance        # converged already: the full step only, no halving
        for halving in range(16):
            redo = cp.flatnonzero(~accepted & ~(settled & (halving > 0)))   # the tori without a decrease yet
            if len(redo) == 0:
                break
            trial_K = k[redo]+per_torus(alpha[redo], k[redo])*dK[redo]
            trial_Q = q[redo]+per_torus(alpha[redo], q[redo])*dQ[redo]
            trial_mapped[redo] = apply(one_turn, trial_K)
            accepted[redo] = length(equations(trial_mapped[redo], trial_K, trial_Q, j[redo]), redo) < before[redo]
            alpha = cp.where(accepted, alpha, alpha/2)
        alpha = cp.where(accepted, alpha, 0)            # no decrease at all: the torus stays where it is
        # no decrease: the torus stops, unless its derivative is old and it is not converged (it gets a new one)
        floor_reached[todo] = ~accepted & (fresh[todo] | (largest[todo] <= 100*tolerance))
        K[todo], Q[todo] = k+per_torus(alpha, k)*dK, q+per_torus(alpha, q)*dQ
        mapped[todo] = cp.where(per_torus(accepted, k), trial_mapped, mapped[todo])
    return K, Q, cp.asnumpy(largest)


def linear_tunes(jacobian):
    """The tunes at the origin: the rotation angle of each plane of the map derivative there."""
    M = jacobian(0., 0., 0., 0.)
    return np.array([math.atan2(float(M[2*a][2*a+1]), float(M[2*a][2*a])) for a in range(2)])/(2*np.pi) % 1


def swapped(one_turn, jacobian):
    """The map with the planes x and y exchanged."""
    order = [2, 3, 0, 1]

    def new_one_turn(*u):
        z = one_turn(*[u[k] for k in order])
        return tuple(z[k] for k in order)

    def new_jacobian(*u):
        A = jacobian(*[u[k] for k in order])
        return [[A[i][k] for k in order] for i in order]
    return new_one_turn, new_jacobian


def spectra(one_turn, jacobian, J, points, band, **settings):
    """The spectra c (tori, 2, 2 band_x + 1, 2 band_y + 1), the tunes Q (tori, 2) and the final residuals of the tori
    with the actions J (tori, 2). The tori with both actions > 0 are solved together; the x tori (J_y = 0) together,
    y at rest; the y tori as x tori of the map with the planes exchanged. The tune of a plane at rest is NaN."""
    c = np.zeros((len(J), 2, 2*band[0]+1, 2*band[1]+1), complex)
    Q, residual = np.full(J.shape, np.nan), np.full(len(J), np.inf)
    for planes in ([1], [0], [0, 1]):
        tori = np.flatnonzero(np.all((J > 0) == np.isin([0, 1], planes), axis=1))
        if not len(tori):
            continue
        maps = swapped(one_turn, jacobian) if planes == [1] else (one_turn, jacobian)
        K, q, residual[tori] = solve_tori(*maps, J[tori][:, planes], points, **settings)
        part = cp.asnumpy(fourier_coefficients(K, [band[a] for a in planes]))
        if planes == [0, 1]:
            c[tori] = part
        elif planes == [0]:
            c[tori, :, :, band[1]] = part
        else:
            c[tori, :, band[0], :] = part[:, ::-1]      # the planes back in order
        Q[np.ix_(tori, planes)] = cp.asnumpy(q)
    return c, Q, residual


def fourier_coefficients(K, band):
    """c[t, a, m_1 + band_1, ...]: the Fourier coefficients of z_a = x_a - i px_a with |m_b| <= band_b."""
    D, N = K.ndim-2, K.shape[-1]
    F = cp.fft.fftn(K[:, 0::2]-1j*K[:, 1::2], axes=tuple(range(2, K.ndim)))/N**D
    index = np.ix_(*[np.arange(-b, b+1) % N for b in band])
    return F[(slice(None), slice(None))+tuple(cp.asarray(i) for i in index)]
