"""Tori of the Hénon lines (lattices/henon) and all their figures in results/test_<line>/: the 2D map (Q0 = 0.2071,
500 x tori, r <= 0.40) and the 4D map (Q = (0.2071, 0.28), a fan of rays, r <= 0.15). Unit coordinates.
Usage (from the repository root): python examples/test_henon_few.py [--angles-2d 511] [--angles-4d 81] [--rays 50]
                                  [--tori-per-ray 20] [--resonances spectra|tunes]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import tori

parser = argparse.ArgumentParser()
parser.add_argument('--angles-2d', type=int, default=511, help='angles of the 2D tori (odd)')
parser.add_argument('--angles-4d', type=int, default=81, help='angles per plane of the 4D tori (odd)')
parser.add_argument('--rays', type=int, default=50)
parser.add_argument('--tori-per-ray', type=int, default=20)
parser.add_argument('--resonances', choices=['spectra', 'tunes'], default='spectra', help='how resonances are found')
args = parser.parse_args()

for name, n_tori, angles, max_jx, max_jy in [('henon_2d_q0.2071', 500, args.angles_2d, .40**2/2, 0.),
                                             ('henon_4d_q0.2071_0.28', args.tori_per_ray, args.angles_4d, .15**2/2,
                                              .15**2/2)]:
    the_map = tori.load(f'lattices/henon/{name}.json', device=2, nemitt=None)
    result = tori.launch_tori(the_map, n_tori, angles, max_jx, max_jy, layout='fan' if max_jy else 'diagonal',
                              n_rays=args.rays)
    out = Path('results')/f'test_{name}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'tori.npz', **result)
    print(f'{name}: {result["converged"].sum()}/{len(result["r"])} converged, {result["seconds"]:.0f} s', flush=True)
    tori.figures(result, out, args.resonances)
