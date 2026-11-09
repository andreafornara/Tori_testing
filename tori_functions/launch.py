"""Single invariant tori of a map, launched from circles, and their spectra. No foliation: every torus on its own.

    result = launch_tori(the_map, n_tori, n_frequencies, max_jx, max_jy=0., layout='diagonal', n_rays=7)

The tori start from the circles of the linear map, at amplitudes r_a = sqrt(2 J_a) equally spaced along rays from
the origin (0 excluded), up to R_a = sqrt(2 max_J_a):
    'diagonal'  one ray, n_tori tori up to (R_x, R_y)
    'fan'       n_rays rays from the x axis (phi = 0) to the y axis (phi = 90 deg), n_tori tori per ray, up to the
                ellipse (R_x cos phi, R_y sin phi)
    'grid'      n_tori x n_tori tori (r_x, r_y); its diagonal is a ray
A torus on an axis (J_y = 0, or J_x = 0) has one angle. Every torus is solved by Newton (newton.py) on n_frequencies
angles per plane (odd); its spectrum is c[a, B_x + m_x, B_y + m_y] of z_a = x_a - i px_a, |m| <= B = n_frequencies // 2.
A torus that does not converge is solved once more (retry: more angles and Newton steps) and keeps its best attempt,
the one with the lowest residual; it stays converged = False (the figures cross it in red).

A torus that does not converge tells which resonance stopped it (record): the map is applied once to the torus, and the
error of every harmonic, divided by its small divisor, is the largest on the harmonics linear harmonic +- n of a line
n.Q = p.
"""
import time
from math import gcd
import numpy as np
from . import newton

AT_ROUNDOFF = dict(tolerance=3e-15)                 # a map exact to double precision (the Hénon lines)
TRACKED = dict(tolerance=5e-12, chord=True, resonance=1e-5, steps=10, circle_tunes=True)    # a tracked lattice
CHUNK = 500                     # tori per Newton call
RETRY = (2, 40)                 # the retry: 2 x the angles, 40 Newton steps


def launch_tori(the_map, n_tori, n_frequencies, max_jx, max_jy=0., layout='diagonal', n_rays=7, log=print):
    """The tori and their spectra, a dict of arrays (np.savez(path, **result)):
        J, r      (tori, 2) actions and amplitudes sqrt(2J) (sigma)
        ray, phi  (tori,) the ray of every torus (-1: none, grid) and its angle atan(r_y / r_x) in degrees
        Q         (tori, 2) tunes (NaN for a plane at rest)
        c         (tori, 2, 2B_x+1, 2B_y+1) spectra of z_x and z_y; band = (B_x, B_y)
        residual  (tori,) the largest Newton residual; converged: below 100 x the tolerance
        angles    (tori,) the angles per plane of the solution kept; seconds: the time taken
    and, for the tori not converged: first_* (the attempt before the retry), failed, error, lost, failed_line, distance (see
    record)."""
    assert n_frequencies % 2 == 1, 'n_frequencies must be odd'
    # The amplitudes of the tori
    R = np.sqrt(2*np.array([max_jx, max_jy], dtype=float))
    s = np.arange(1, n_tori+1)/n_tori
    if layout == 'fan':
        phi = np.linspace(0, 90, n_rays)
        end = np.stack([R[0]*np.cos(np.radians(phi)), R[1]*np.sin(np.radians(phi))], axis=1)
        end[0, 1], end[-1, 0] = 0, 0                                       # the axes exactly
        r = (s[None, :, None]*end[:, None, :]).reshape(-1, 2)
        ray = np.repeat(np.arange(n_rays), n_tori)
    elif layout == 'grid':
        r = np.stack(np.meshgrid(s*R[0], s*R[1], indexing='ij'), axis=-1).reshape(-1, 2)
        ray = np.where(np.arange(n_tori**2) % (n_tori+1) == 0, 0, -1)    # the diagonal i = j
    else:
        r, ray = s[:, None]*R, np.zeros(n_tori, int)
    J = r*r/2
    B = [n_frequencies//2 if R[a] > 0 else 0 for a in range(2)]          # no angle in a plane: only m = 0
    settings = AT_ROUNDOFF if the_map.noise < 1e-14 else TRACKED
    result = dict(J=J, r=r, ray=ray, phi=np.degrees(np.arctan2(r[:, 1], r[:, 0])), Q=np.full((len(J), 2), np.nan),
                  c=np.zeros((len(J), 2, 2*B[0]+1, 2*B[1]+1), complex), residual=np.full(len(J), np.inf),
                  band=np.array(B), angles=np.full(len(J), n_frequencies), layout=layout,
                  tolerance=settings['tolerance'], noise=the_map.noise, tunes0=the_map.tunes, line=the_map.info['line'],
                  unit='sigma' if the_map.info['nemitt'] else '')
    # All the tori, then the retry of the ones not converged
    newton.use_gpu(the_map.device)
    clock = time.perf_counter()
    solve(the_map, result, np.arange(len(J)), n_frequencies, settings)
    result['seconds'] = time.perf_counter()-clock
    retry(the_map, result, log)
    log(f'{len(J)} tori ({result["converged"].sum()} converged) on {n_frequencies} angles per plane in '
        f'{result["seconds"]:.0f} s')
    clock = time.perf_counter()
    record(the_map, result)
    result['record_seconds'] = time.perf_counter()-clock
    return result


def solve(the_map, result, tori, points, settings):
    """Solve the tori `tori` from their circles on `points` angles per plane, CHUNK at a time; a torus keeps the new
    solution only if its residual is lower than the one stored."""
    for part in np.array_split(tori, max(1, int(np.ceil(len(tori)/CHUNK)))):
        c, Q, residual = newton.spectra(the_map.one_turn, the_map.jacobian, result['J'][part], points, result['band'],
                                        **settings)
        stored = result['residual'][part]
        better = (residual < stored) | ~np.isfinite(stored)     # a torus with lost particles is kept too (better than none)
        k = part[better]
        result['c'][k], result['Q'][k], result['residual'][k], result['angles'][k] = \
            c[better], Q[better], residual[better], points


def retry(the_map, result, log=print):
    """The tori of a result that did not converge, solved once more on RETRY[0] x the angles with RETRY[1] Newton
    steps; each keeps its best attempt. The result is updated in place (also a result already saved)."""
    newton.use_gpu(the_map.device)
    settings = AT_ROUNDOFF if the_map.noise < 1e-14 else TRACKED
    n_frequencies = 2*int(np.max(result['band']))+1
    result.setdefault('angles', np.full(len(result['J']), n_frequencies))     # saved before the retry existed
    clock = time.perf_counter()
    left = np.flatnonzero(~(result['residual'] < 100*settings['tolerance']))
    result.update(first_tori=left, first_c=result['c'][left].copy(), first_Q=result['Q'][left].copy(),
                  first_residual=result['residual'][left].copy())
    if len(left):
        points = int(n_frequencies*RETRY[0])//2*2+1                                # odd
        solve(the_map, result, left, points, dict(settings, steps=RETRY[1]))
        log(f'{len(left)} tori not converged, again on {points} angles and {RETRY[1]} Newton steps: '
            f'{np.sum(result["residual"][left] < 100*settings["tolerance"])} converged')
    result['converged'] = result['residual'] < 100*settings['tolerance']
    result['seconds'] = float(result['seconds'])+time.perf_counter()-clock
    return result


def error_spectra(the_map, c, Q, points):
    """The invariance error of the tori (c, Q), harmonic by harmonic: the map is applied once on `points` angles per plane,
    E[t, a, B + m_x, B + m_y] = spectrum of (z_a after the map) - c[t, a, m] exp(2 pi i m.Q).
    A lost particle counts as no error at its point. Returns E and the number of points lost of every torus."""
    B = c.shape[-1]//2
    m = np.arange(-B, B+1)
    index = (slice(None), slice(None))+np.ix_(m % points, m % points)
    F = np.zeros((len(c), 2, points, points), complex)
    F[index] = c
    z = np.fft.ifftn(F, axes=(2, 3))*points**2                               # the torus on the grid of angles
    x, px, y, py = map(newton.cp.asnumpy, the_map.one_turn(z[:, 0].real, -z[:, 0].imag, z[:, 1].real, -z[:, 1].imag))
    image = np.stack([x-1j*px, y-1j*py], axis=1)
    F[index] = c*np.exp(2j*np.pi*(Q[:, 0, None, None]*m[None, :, None]+Q[:, 1, None, None]*m[None, None, :]))[:, None]
    error = image-np.fft.ifftn(F, axes=(2, 3))*points**2                     # after the map - rotated by the tunes
    lost = np.isnan(error)
    error[lost] = 0
    return (np.fft.fftn(error, axes=(2, 3))/points**2)[index], lost.sum(axis=(1, 2, 3))//2


def named_line(Q, E):
    """The line (n_x, n_y, p) that a torus not converged names: of all the lines n.Q = p with the four harmonics
    linear harmonic +- n inside the band, the one with the largest error / small divisor on these four."""
    B = E.shape[-1]//2
    m = np.arange(-B, B+1)
    mx, my = np.meshgrid(m, m, indexing='ij')
    vx, vy = np.stack([mx-1, mx]), np.stack([my, my-1])                     # m - the linear harmonic
    divisor = 2*np.abs(np.sin(np.pi*(vx*Q[0]+vy*Q[1])))
    divisor[(vx == 0) & (vy == 0)] = np.inf
    far = np.abs(mx)+np.abs(my) >= B+8                                       # the far corner of the band: only noise
    size = np.abs(E)/divisor
    size[np.abs(E) < 5*np.median(np.abs(E)[:, far])] = 0                     # an error at the noise level says nothing
    best, line = -1., None
    for nx in range(B):
        for ny in range(-B+1, B):
            p = int(np.round(nx*Q[0]+ny*Q[1]))
            if (nx > 0 or ny > 0) and gcd(gcd(nx, abs(ny)), abs(p)) == 1:
                score = sum(size[a, B+e[0]+s*nx, B+e[1]+s*ny]**2 for a, e in enumerate(((1, 0), (0, 1))) for s in (1, -1))
                if score > best:
                    best, line = score, (nx, ny, p)
    return line


def record(the_map, result):
    """What the tori not converged (with two angles) tell, added to the result:
        failed    (k,) the tori                         error        (k, 2, 2B+1, 2B+1) their invariance error (error_spectra)
        lost      (k,) points with a lost particle      failed_line  (k, 3) the line (n_x, n_y, p) they name (named_line)
        distance  (k,) n.Q - p of the line"""
    usable = np.all(result['J'] > 0, axis=1) & np.all(np.isfinite(result['Q']), axis=1) & \
        np.all(np.isfinite(result['c']), axis=(1, 2, 3))
    failed = np.flatnonzero(~result['converged'] & usable)
    B = result['c'].shape[-1]//2
    result.update(failed=failed, error=np.zeros((0, 2, 2*B+1, 2*B+1), complex), lost=np.zeros(0, int),
                  failed_line=np.zeros((0, 3), int), distance=np.zeros(0))
    if len(failed):
        points = int((2*B+1)*RETRY[0])//2*2+1
        result['error'], result['lost'] = error_spectra(the_map, result['c'][failed], result['Q'][failed], points)
        result['failed_line'] = np.array([named_line(Q, E) for Q, E in zip(result['Q'][failed], result['error'])])
        result['distance'] = np.sum(result['Q'][failed]*result['failed_line'][:, :2], axis=1)-result['failed_line'][:, 2]
