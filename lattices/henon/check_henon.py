"""Check that the lines of build_henon.py are the analytic Hénon maps of final_tori_foliation/maps.py: 10^4 random
points tracked 1 and 1000 times by Xsuite (CPU) and by maps.py must agree bit by bit; then the sign of the kick and
the twiss (tunes, beta = 1, alpha = 0). Output in check_henon.log."""
import os
import sys
import numpy as np
import xobjects as xo
import xtrack as xt

sys.path.insert(0, '/afs/cern.ch/work/a/afornara/public/tori_foliation_complete/final_tori_foliation')
import maps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 10_000
CASES = (('henon_2d_q0.2071.json', maps.slide_henon(.2071)[0], 1, .40),       # file, map, planes, largest radius
         ('henon_2d_q0.218.json', maps.slide_henon(.218)[0], 1, .33),
         ('henon_4d_q0.2071_0.28.json', maps.coupled_henon(k2=-2., k3=0., qx=.2071, qy=.28)[0], 2, .15))
rng = np.random.default_rng(2026)


def disk(r_max):
    """N points uniform in the disk of radius r_max of one plane."""
    r, phi = r_max*np.sqrt(rng.uniform(size=N)), rng.uniform(0, 2*np.pi, N)
    return r*np.cos(phi), -r*np.sin(phi)


for name, one_turn, planes, r_max in CASES:
    line = xt.Line.from_json(os.path.join(HERE, name))
    line.build_tracker(_context=xo.ContextCpu())
    line.freeze_longitudinal()
    x, px = disk(r_max)
    y, py = disk(r_max) if planes == 2 else (np.zeros(N), np.zeros(N))
    print(f'\n{name}: {list(line.get_table().element_type[:-1])}, {N} particles, r <= {r_max} per plane')
    for turns in (1, 1000):
        p = line.build_particles(x=x, px=px, y=y, py=py, zeta=0., delta=0.)
        line.track(p, num_turns=turns)
        xs = np.array([p.x, p.px, p.y, p.py])
        ref = np.array(maps.track(one_turn, (x, px, y, py)[:2*planes], turns))
        ok = np.all(np.isfinite(ref), axis=0)                      # orbits that stayed bounded
        diff = np.abs(xs[:2*planes, ok]-ref[:, ok]).max(axis=1)
        same = np.mean(np.all(xs[:2*planes, ok] == ref[:, ok], axis=0))
        rest = '' if planes == 2 else f', y and py stay {np.abs(xs[2:]).max()}'
        print(f'  {turns:4d} turns: max |line - maps.py| per coordinate {diff}, bit-identical {same:.4f}'
              f' ({ok.sum()} bounded orbits){rest}')

    p = line.build_particles(x=[.1, 0., .1], y=[0., .1, .1], px=0., py=0.)     # the kick alone on three points
    line.track(p, ele_start=0, ele_stop=1)
    print(f'  kick alone: px = {p.px} (expect +x^2-y^2 = [.01, -.01, 0]), py = {p.py} (expect -2xy = [0, 0, -.02])')
    tw = line.twiss4d()
    print(f'  twiss4d: qx = {tw.qx:.15f}, qy = {tw.qy:.15f}, betx = {tw.betx[0]:.15f}, bety = {tw.bety[0]:.15f},'
          f' alfx = {tw.alfx[0]:.1e}, alfy = {tw.alfy[0]:.1e}, dx = {tw.dx[0]:.1e}')
