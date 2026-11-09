"""Tori of the Hénon lines (lattices/henon): the 2D maps at Q0 = 0.2071 and 0.218 (x tori) and the 4D map (a fan of
rays), and all their figures in results/<line>/. Unit coordinates.
Usage (from the repository root): python examples/launch_henon.py [--angles-2d 255] [--angles-4d 41]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import tori

parser = argparse.ArgumentParser()
parser.add_argument('--angles-2d', type=int, default=255, help='angles of the 2D tori (odd)')
parser.add_argument('--angles-4d', type=int, default=41, help='angles per plane of the 4D tori (odd)')
args = parser.parse_args()

for name, max_jx, max_jy, angles in [('henon_2d_q0.2071', .45**2/2, 0., args.angles_2d),
                                     ('henon_2d_q0.218', .45**2/2, 0., args.angles_2d),
                                     ('henon_4d_q0.2071_0.28', .15**2/2, .15**2/2, args.angles_4d)]:
    the_map = tori.load(f'lattices/henon/{name}.json', device=2, nemitt=None)
    result = tori.launch_tori(the_map, n_tori=100, n_frequencies=angles, max_jx=max_jx, max_jy=max_jy,
                              layout='fan' if max_jy else 'diagonal', n_rays=7)
    out = Path('results')/name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'tori.npz', **result)
    tori.figures(result, out)
