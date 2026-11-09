"""The resonances n.Q = p of the 4D tori, read from their spectra c[torus, a, m_x, m_y] (z_a = sum_m c[a, m] e^{i m.theta}).

method='spectra' (default): the spectra say which resonances there are, with no cutoff in order.
  1. Peaked entry: c[a, m] of a torus rises more than PEAK decades above the straight line through the same entry on
     its two neighbouring tori (along its ray, in log r, or around its arc, in the ray angle), and above FLOOR x the
     noise floor of the spectrum of the torus (floor). The tori not converged are used too.
  2. Type: a peaked entry of z_a points along m - e_a (e_a: the linear harmonic). The torus takes the direction with
     the most peaked entries, times its smallest peaked multiple: n.
  3. Confirmed: p = round(n.Q) and |n.Q - p| is within the change of n.Q to a neighbouring torus. A resonance confirmed
     on MIN_TORI tori is followed along its whole curve: every torus next to the crossing n.Q = p.
method='tunes': every n.Q = p crossed by the tunes of the tori, |n_x| + |n_y| <= ORDER, followed the same way.
Resolved: the resonant power of a torus (its harmonics e_a + k n, 0 < |k| <= RESONANT_K, over the linear power) is above
RESOLVED x the noise power (the noise floor amplified by the same small divisors 2 sin(pi k (n.Q - p))). A resonance
resolved on MIN_TORI tori is flagged. The figures colour the flagged resonances with |n_x| + |n_y| <= PLOT_ORDER and every
resonance named by a torus not converged (launch.record); the others of low order are grey (with 'spectra' they keep only
their confirmed tori).

    found = detect(result, method)
    found['i']                          the converged tori with two angles (indices into result)
    found['every']                      the same with the tori not converged too (the curves n.Q = p of the figures)
    found['resonances']                 all the resonances found (lowest order first)
    found['flagged']                    the resonances flagged, of any order
    found['shown'], ['unresolved']      the ones the figures colour, and the ones they grey
    found['tori'][w], ['resolved'][w]   the tori of w (positions in found['i']) and whether it is resolved on each
    found['peaks']                      the number of peaked entries of every torus
"""
from collections import Counter
from math import gcd
import numpy as np

ORDER = 16              # method 'tunes', two angles: |n_x| + |n_y| <= ORDER
ORDER_1D = 40           # one angle (x or y tori): q Q = p with q <= ORDER_1D
RESONANT_K = 4          # the resonant harmonics e_a + k n, 0 < |k| <= RESONANT_K
PEAK = 1.               # a peaked entry: decades above its two neighbouring tori ...
FLOOR = 1000.           # ... and above FLOOR x the noise floor of the spectrum
MIN_TORI = 2            # a resonance needs MIN_TORI tori
RESOLVED = 10.          # resolved: resonant power above RESOLVED x the noise power
PLOT_ORDER = 13         # the figures colour |n_x| + |n_y| <= PLOT_ORDER
_cache = {}


def grid(c):
    """The bands (B_x, B_y) of the spectra c and the harmonics (m_x, m_y) of every entry."""
    B = [n//2 for n in c.shape[-2:]]
    return B, np.meshgrid(np.arange(-B[0], B[0]+1), np.arange(-B[1], B[1]+1), indexing='ij')


def floor(c):
    """The noise floor of every spectrum c (tori, 2, ...): the median |c| at the far corner of the band."""
    B, (mx, my) = grid(c)
    return np.median(np.abs(c[:, :, np.abs(mx)+np.abs(my) >= max(B)+8]), axis=(1, 2))


def crossed(Q, planes=(0, 1)):
    """Every resonance n.Q = p met by the tunes Q (tori, 2): n.Q - p changes sign among the tori. n and p have no common
    factor, n_x >= 0, |n_x| + |n_y| <= ORDER (ORDER_1D with one plane); lowest order first."""
    planes = list(planes)
    order = ORDER if len(planes) == 2 else ORDER_1D
    found = []
    for nx in range(order+1):
        for ny in range(-order, order+1):
            n = np.array([nx, ny])
            if not 0 < nx+abs(ny) <= order or (nx == 0 and ny < 0) or any(n[b] for b in range(2) if b not in planes):
                continue
            value = Q[:, planes]@n[planes]
            found += [(nx, ny, p) for p in range(int(np.ceil(value.min())), int(np.floor(value.max()))+1)
                      if gcd(gcd(nx, abs(ny)), abs(p)) == 1]
    return sorted(found, key=lambda w: abs(w[0])+abs(w[1]))


def resonant(resonances, B):
    """{(a, m_x, m_y): (n_x, n_y, p)}: the resonant harmonics e_a + k n of every resonance inside the band B; a harmonic
    belongs to the first (lowest-order) resonance that has it."""
    out = {}
    for w in resonances:
        for a in range(2):
            for k in range(-RESONANT_K, RESONANT_K+1):
                m = ((a == 0)+k*w[0], (a == 1)+k*w[1])
                if k and abs(m[0]) <= B[0] and abs(m[1]) <= B[1]:
                    out.setdefault((a, *m), tuple(w[:3]))
    return out


def neighbours(result, i):
    """Pairs (u, v) of positions in i of neighbouring tori: consecutive on a ray, or on adjacent rays at the same
    amplitude."""
    rho = np.round(np.hypot(*result['r'].T), 6)
    position = {t: k for k, t in enumerate(i)}
    by_ray = {}
    for t in i:
        by_ray.setdefault(int(result['ray'][t]), []).append(t)
    keys = sorted(by_ray, key=lambda k: np.median(result['phi'][by_ray[k]]))
    pairs = []
    for k in keys:                                                  # along every ray
        tori = sorted(by_ray[k], key=lambda t: rho[t])
        pairs += [(position[a], position[b]) for a, b in zip(tori, tori[1:])]
    for k0, k1 in zip(keys, keys[1:]):                              # across two adjacent rays
        at = {rho[t]: t for t in by_ray[k1]}
        pairs += [(position[t], position[at[rho[t]]]) for t in by_ray[k0] if rho[t] in at]
    return np.array(pairs, int).reshape(-1, 2)


def triples(result, i):
    """(previous, torus, next, w) of neighbouring tori (positions in i) along every ray and around every arc of tori at
    the same amplitude; w places the torus between the two others (in log r, or in the ray angle)."""
    rho = np.round(np.hypot(*result['r'][i].T), 6)
    phi = result['phi'][i]
    by_ray, by_level = {}, {}
    for k in range(len(i)):
        by_ray.setdefault(int(result['ray'][i[k]]), []).append(k)
        by_level.setdefault(rho[k], []).append(k)
    out = []
    for ks in by_ray.values():
        ks = sorted(ks, key=lambda k: rho[k])
        out += [(a, b, c, np.log(rho[b]/rho[a])/np.log(rho[c]/rho[a])) for a, b, c in zip(ks, ks[1:], ks[2:])]
    for ks in by_level.values():
        ks = sorted(ks, key=lambda k: phi[k])
        out += [(a, b, c, (phi[b]-phi[a])/(phi[c]-phi[a])) for a, b, c in zip(ks, ks[1:], ks[2:]) if phi[c] > phi[a]]
    return out


def linear_power(c):
    """The linear power of every spectrum c (tori, 2, ...): all the harmonics |m_x|, |m_y| <= 1, of both planes."""
    _, (mx, my) = grid(c)
    return np.sum(np.abs(c[:, :, (np.abs(mx) <= 1) & (np.abs(my) <= 1)])**2, axis=(1, 2))


def power(result, i, harmonics):
    """The resonant power and the noise power of the tori i, both over the linear power: harmonics {(a, m_x, m_y): w}."""
    c, Q = result['c'][i], result['Q'][i]
    (Bx, By), _ = grid(c)
    eps = floor(c)                                                          # the error left in every entry
    linear = linear_power(c)
    P, N = np.zeros(len(i)), np.zeros(len(i))
    for (a, x, y), (nx, ny, p) in harmonics.items():
        k = (x-(a == 0))//nx if nx else (y-(a == 1))//ny
        P += np.abs(c[:, a, Bx+x, By+y])**2
        N += (eps/np.maximum(2*np.abs(np.sin(np.pi*k*(Q@[nx, ny]-p))), 1e-300))**2    # amplified by the divisor
    return P/linear, N/linear


def crossing(Q, pairs, w):
    """The tori next to the crossing of n.Q = p: of two neighbouring tori on either side of it, the closer one."""
    d = Q@w[:2]-w[2]
    return [u if abs(d[u]) <= abs(d[v]) else v for u, v in pairs if d[u]*d[v] <= 0]


def detect(result, method='spectra'):
    """See the module docstring. Cached: every figure asks for it."""
    key = (id(result['c']), method)
    if key in _cache and _cache[key][0] is result['c']:
        return _cache[key][1]
    i = np.flatnonzero(np.all(result['J'] > 0, axis=1) & np.all(np.isfinite(result['Q']), axis=1) &
                       np.all(np.isfinite(result['c']), axis=(1, 2, 3)))   # the tori not converged too
    c, Q = result['c'][i], result['Q'][i]
    (Bx, By), (mx, my) = grid(c)
    eps = floor(c)
    pairs = neighbours(result, i)
    # 1. the peaked entries of every torus
    size = np.log10(np.maximum(np.abs(c), 1e-300))
    peaked = np.zeros(size.shape, bool)
    for a, b, c_, w in triples(result, i):
        line = (1-w)*size[a]+w*size[c_]
        peaked[b] |= (size[b]-line > PEAK) & (size[b] > np.log10(FLOOR*eps[b])) & (size[a] > -299) & (size[c_] > -299)
    # 2. and 3. the type of resonance on every torus with peaked entries, confirmed by its tunes
    labels = {}
    for b in np.flatnonzero(peaked.any(axis=(1, 2, 3))) if method == 'spectra' else []:
        directions, multiples = Counter(), {}
        for a, x, y in zip(*np.nonzero(peaked[b])):
            v = (int(mx[x, y])-(a == 0), int(my[x, y])-(a == 1))         # m - e_a
            g = gcd(abs(v[0]), abs(v[1]))
            if g:
                d = (v[0]//g, v[1]//g) if v[0] > 0 or (v[0] == 0 and v[1] > 0) else (-v[0]//g, -v[1]//g)
                directions[d] += 1
                multiples.setdefault(d, set()).add(g)
        if directions:
            d = max(directions, key=lambda q: (directions[q], -abs(q[0])-abs(q[1])))
            n = np.array(d)*min(multiples[d])
            value, p = Q[b]@n, int(np.round(Q[b]@n))
            step = max([abs(value-Q[v]@n) for v in pairs[pairs[:, 0] == b, 1]] +
                       [abs(value-Q[u]@n) for u in pairs[pairs[:, 1] == b, 0]], default=0.)
            if abs(value-p) <= step:
                g = gcd(gcd(abs(int(n[0])), abs(int(n[1]))), abs(p))        # 4Qx-10Qy=-2 is 2Qx-5Qy=-1
                labels[b] = (int(n[0])//g, int(n[1])//g, p//g)
    count = Counter(labels.values())
    named = {tuple(int(v) for v in w) for w in result.get('failed_line', ())}   # named by the tori not converged
    resonances = sorted({w for w, k in count.items() if k >= MIN_TORI} | named, key=lambda w: (abs(w[0])+abs(w[1]), -count[w]))
    if method == 'tunes':
        resonances = crossed(Q)
    # the tori of every resonance, and where it is resolved
    found = dict(i=i, method=method, resonances=resonances, peaks=peaked.sum(axis=(1, 2, 3)), tori={}, resolved={})
    owner = resonant(resonances, (Bx, By))
    for w in resonances:
        confirmed = {b for b, v in labels.items() if v == w}
        for k in (confirmed | set(crossing(Q, pairs, w)), confirmed):      # followed along its curve; if not
            k = np.array(sorted(k), int)                                    # resolved there, only where it peaks
            P, N = power(result, i[k], {h: v for h, v in owner.items() if v == w})
            found['tori'][w], found['resolved'][w] = k, P > RESOLVED*N
            if found['resolved'][w].sum() >= MIN_TORI or method == 'tunes' or w in named:
                break
    low = {w for w in resonances if method == 'tunes' or abs(w[0])+abs(w[1]) <= PLOT_ORDER}
    found['flagged'] = [w for w in resonances if found['resolved'][w].sum() >= MIN_TORI]
    found['shown'] = [w for w in resonances if w in named or (w in low and w in found['flagged'])]
    found['unresolved'] = [w for w in resonances if w in low and w not in found['shown']]
    # the figures use the converged tori only: the others are their red crosses
    keep = result['converged'][i]
    position = np.cumsum(keep)-1
    for w in resonances:
        here = keep[found['tori'][w]]
        found['tori'][w], found['resolved'][w] = position[found['tori'][w][here]], found['resolved'][w][here]
    found.update(every=i, i=i[keep], peaks=found['peaks'][keep])
    _cache.clear()                                       # one result at a time
    _cache[key] = (result['c'], found)
    return found
