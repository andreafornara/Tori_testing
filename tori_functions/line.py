"""The one-turn map of an Xsuite line (.json) in normalized coordinates, tracked on a GPU. Nothing here knows tori.

    physical = closed orbit + sqrt(emittance) W u,       u = (x, px, y, py)

W is exactly symplectic and turns the one-turn matrix into two rotations, by Qx and by Qy, with x independent of px
(Courant-Snyder phase). It comes from the one-turn matrix (fourth-order differences of the tracking at 0.1 sigma): the
twiss W_matrix is not symplectic to round-off, and coordinates that are not canonical would spoil the tori. Then
r = sqrt(2J) is the amplitude in beam sigmas. nemitt=None: emittance 1 (the Hénon lines: W = 1, sigma = 1).
The longitudinal plane is frozen (delta = 0): the map is 4D.

    the_map.one_turn(x, px, y, py) -> (x, px, y, py)     CuPy arrays of any shape; NaN for a lost particle
    the_map.jacobian(x, px, y, py) -> A[i][k]            central differences of one_turn, step STEP sigma
    the_map.tunes, the_map.noise (round-off of the map), the_map.device, the_map.info
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

STEP = 1e-3                     # finite-difference step of the jacobian [sigma]
CHUNK = 10**6                   # particles per Xsuite call
BIG = 1000                      # a line with more elements is optimized for tracking
OMEGA = np.kron(np.eye(2), [[0., 1.], [-1., 0.]])


def normal_form(R):
    """A real symplectic W with W^-1 R W = one rotation [[cos, sin], [-sin, cos]] per plane (R a stable one-turn
    matrix): the eigenvectors, then symplectic Gram-Schmidt and the Courant-Snyder phase (W[2a, 2a+1] = 0)."""
    W = np.zeros((4, 4))
    for v in np.linalg.eig(R)[1].T:
        c = np.imag(np.conj(v)@OMEGA@v)
        if c > 0:                                                   # one eigenvector of each mode
            a = int(np.argmax([np.sum(np.abs(v[2*b:2*b+2])**2) for b in range(2)]))   # its plane
            W[:, 2*a], W[:, 2*a+1] = np.sqrt(2)*(v/np.sqrt(c)).real, np.sqrt(2)*(v/np.sqrt(c)).imag
    form = lambda a, b: a@OMEGA@b
    e, f = W[:, 0], W[:, 1]/form(W[:, 0], W[:, 1])
    g, h = [v-form(v, f)*e+form(v, e)*f for v in (W[:, 2], W[:, 3])]   # minus the projection on the first plane
    W = np.stack([e, f, g, h/form(g, h)], axis=1)
    for a in range(2):                                              # the Courant-Snyder phase
        phi = np.arctan2(W[2*a, 2*a+1], W[2*a, 2*a])
        e, f = W[:, 2*a].copy(), W[:, 2*a+1].copy()
        W[:, 2*a], W[:, 2*a+1] = np.cos(phi)*e+np.sin(phi)*f, -np.sin(phi)*e+np.cos(phi)*f
    return W


def gpu_context(device):
    """An Xsuite GPU context whose compiled kernels stay on disk between runs (sorted kernel arguments give identical
    CUDA sources, so CuPy's disk cache is reused)."""
    import xobjects as xo
    import xobjects.context_cupy as context_cupy
    import xtrack as xt
    from xobjects.context import classes_from_kernels
    cache = Path(__file__).resolve().parent.parent/'.cache'/'cupy'/f'xobjects-{xo.__version__}_xtrack-{xt.__version__}'
    os.environ.setdefault('CUPY_CACHE_DIR', str(cache))
    context_cupy.classes_from_kernels = lambda kernels: sorted(classes_from_kernels(kernels), key=lambda c: c.__name__)
    return xo.ContextCupy(device=device)


def load(path, device=2, nemitt=2.5e-6, start=None, log=print):
    """The map of the line in `path` on GPU `device` (numbered as in nvidia-smi), in beam sigmas of the normalized
    emittance nemitt [m] (None: emittance 1), observed at the element `start` (None: the start of the line)."""
    import cupy as cp
    import xtrack as xt

    # The line, its closed orbit (twiss) and the GPU tracker
    clock = time.perf_counter()
    line = xt.Line.from_json(path)
    line.build_tracker()
    twiss = line.twiss4d()                  # the orbit at `start`: the guess of the closed orbit of the cycled line
    i = list(twiss.name).index(start) if start else 0
    guess = {k: float(twiss[k][i]) for k in ('x', 'px', 'y', 'py', 'zeta', 'delta')}
    if start:
        line.cycle(name_first_element=start)
    twiss = line.twiss4d(co_guess=guess,    # small steps: with the default ones the matrix at +150 A has det 1.003
                         steps_R_matrix=dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9))
    co = np.array([twiss.x[0], twiss.px[0], twiss.y[0], twiss.py[0]])
    scale = np.sqrt(nemitt/(line.particle_ref.beta0[0]*line.particle_ref.gamma0[0]) if nemitt else 1.)
    if len(line.element_names) > BIG:
        line.optimize_for_tracking(compile=False, verbose=False)   # markers out, drifts and thin kicks merged
    line.discard_tracker()
    line.freeze_longitudinal(True)
    line.build_tracker(_context=gpu_context(device))
    log(f'{Path(path).name}: {len(line.element_names)} elements, GPU tracker ready in '
        f'{time.perf_counter()-clock:.0f} s')
    templates = {}

    def turn(X):
        """One turn of the physical coordinates X (4, n); NaN for a lost particle. The particles are copies of kept
        templates (building them takes ~0.5 s), of a size rounded up to a power of 2."""
        out = cp.empty_like(X)
        for i in range(0, X.shape[1], CHUNK):
            part = X[:, i:i+CHUNK]
            size = min(CHUNK, max(1024, 1 << (part.shape[1]-1).bit_length()))
            if size not in templates:
                zero = cp.zeros(size)
                templates[size] = line.build_particles(x=zero, px=zero, y=zero, py=zero, zeta=0., delta=0.)
            p = templates[size].copy()
            for name, value in zip(('x', 'px', 'y', 'py'), part):
                getattr(p, name)[:len(value)] = value
            line.track(p, num_turns=1)
            order = cp.argsort(p.particle_id)[:part.shape[1]]           # the particles 0 ... n - 1
            block = cp.stack([p.x[order], p.px[order], p.y[order], p.py[order]])
            block[:, p.state[order] < 1] = cp.nan
            out[:, i:i+CHUNK] = block
        cp.cuda.Device().synchronize()
        return out

    def normalized_turn(co, W):
        """One turn in the normalized coordinates of (co, W): U (4, n) -> (4, n)."""
        forward, backward = cp.asarray(scale*W), cp.asarray(np.linalg.inv(W)/scale)
        orbit = cp.asarray(co)[:, None]
        return lambda U: backward@(turn(orbit+forward@U)-orbit)

    def derivative(mapping, U, h, order):
        """d(mapping)/dU at the points U (4, n) by central differences of order 2 or 4: (4, 4, n)."""
        shifts = [h, -h] if order == 2 else [2*h, h, -h, -2*h]
        eye = cp.eye(4)[:, :, None, None]                                       # (coordinate, direction)
        V = U[:, None, None, :]+eye*cp.asarray(shifts)[None, None, :, None]     # (4, direction, shift, n)
        F = mapping(V.reshape(4, -1)).reshape(V.shape)
        if order == 2:
            return (F[:, :, 0]-F[:, :, 1])/(2*h)
        return (-F[:, :, 0]+8*F[:, :, 1]-8*F[:, :, 2]+F[:, :, 3])/(12*h)

    # The closed orbit, refined by Newton in the twiss coordinates (not symplectic, but fine for a fixed point)
    W0 = twiss.W_matrix[0][:4, :4]
    mapping = normalized_turn(co, W0)
    U = cp.zeros((4, 1))
    for _ in range(4):
        F = mapping(U)-U
        U = U-cp.linalg.solve(derivative(mapping, U, 1e-2, 4)[:, :, 0]-cp.eye(4), F)
    co = co+scale*W0@cp.asnumpy(U[:, 0])
    # The one-turn matrix at the closed orbit, then the symplectic W and the map in its coordinates
    D = cp.asnumpy(derivative(normalized_turn(co, W0), cp.zeros((4, 1)), .1, 4)[:, :, 0])
    W = normal_form(W0@D@np.linalg.inv(W0))
    mapping = normalized_turn(co, W)
    L = cp.asnumpy(derivative(mapping, cp.zeros((4, 1)), .1, 4)[:, :, 0])
    tunes = np.array([np.arctan2(L[2*a, 2*a+1], L[2*a, 2*a])/(2*np.pi) % 1 for a in range(2)])
    # The round-off of the map: rms of M(U + d e) + M(U - d e) - 2 M(U) / sqrt(6), d = 1e-10, at 0.1 sigma
    rng = np.random.default_rng(1)
    U = cp.asarray(rng.normal(size=(4, 256))*.1)
    e = rng.normal(size=(4, 256))
    e = cp.asarray(1e-10*e/np.linalg.norm(e, axis=0))
    noise = float(cp.sqrt(cp.nanmean((mapping(U+e)+mapping(U-e)-2*mapping(U))**2)/6))
    log(f'Q = {tunes}, closed orbit residual {float(cp.abs(mapping(cp.zeros((4, 1)))).max()):.1e}, W symplectic to '
        f'{np.abs(W.T@OMEGA@W-OMEGA).max():.1e}, round-off of the map {noise:.1e} sigma')

    def columns(coordinates):
        """The coordinates (any shapes) as columns (4, n), and their common shape."""
        shape = cp.broadcast(*[cp.asarray(c) for c in coordinates]).shape
        return cp.stack([cp.broadcast_to(cp.asarray(c, dtype=float), shape).ravel() for c in coordinates]), shape

    def one_turn(x, px, y, py):
        U, shape = columns((x, px, y, py))
        return tuple(v.reshape(shape) for v in mapping(U))

    def jacobian(x, px, y, py):
        U, shape = columns((x, px, y, py))
        A = derivative(mapping, U, STEP, 2)
        return [[A[i, k].reshape(shape) for k in range(4)] for i in range(4)]

    info = dict(line=Path(path).name, start=start, nemitt=nemitt, closed_orbit=co.tolist(), W=W.tolist())
    return SimpleNamespace(one_turn=one_turn, jacobian=jacobian, tunes=tunes, noise=noise, device=device, info=info)
