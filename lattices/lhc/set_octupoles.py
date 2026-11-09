"""Other octupole currents from the saved 0 A line (CPU, about 40 s each): set i_oct_b1, match the tunes and the
chromaticity back to (62.31, 60.32) and (5, 6) in 4D, save lhc_top_b1_oct<I>.json.
Run (from lattices/lhc): python set_octupoles.py 50 150
"""
import sys
import time
from pathlib import Path
import xtrack as xt

HERE = Path(__file__).resolve().parent
TARGETS = dict(qx=62.31, qy=60.32, dqx=5., dqy=6.)

for current in map(float, sys.argv[1:]):
    start = time.time()
    line = xt.Line.from_json(HERE/'lhc_top_b1_oct0.json')
    line.vars['i_oct_b1'] = current
    line.match(method='4d', vary=xt.VaryList(['dqx.b1_op', 'dqy.b1_op', 'dqpx.b1_op', 'dqpy.b1_op'], step=1e-5),
               targets=[xt.Target(k, v, tol=1e-6 if k.startswith('q') else 1e-3) for k, v in TARGETS.items()])
    twiss = line.twiss4d()
    name = HERE/f'lhc_top_b1_oct{current:g}.json'
    line.to_json(name)
    print(f'{name.name}: qx {twiss.qx:.6f}, qy {twiss.qy:.6f}, dqx {twiss.dqx:.3f}, dqy {twiss.dqy:.3f}, '
          f'c- {twiss.c_minus:.1e}, betx/bety IP1 {twiss["betx", "ip1"]:.4f}/{twiss["bety", "ip1"]:.4f}, '
          f'{time.time()-start:.0f} s', flush=True)
