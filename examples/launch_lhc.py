"""Tori of an LHC line, 0 - SIGMA sigma per plane, tracked on a GPU: a fan of rays from the x axis to the y axis,
and all their figures in results/<name>/.
Usage (from the repository root): python examples/launch_lhc.py LINE.json [NAME] [--sigma 2] [--angles 41] [--rays 50]
                                  [--tori-per-ray 20] [--device 2] [--resonances spectra|tunes]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import tori

parser = argparse.ArgumentParser()
parser.add_argument('line', type=Path)
parser.add_argument('name', nargs='?')
parser.add_argument('--sigma', type=float, default=2., help='largest amplitude per plane [sigma]')
parser.add_argument('--angles', type=int, default=41, help='angles per plane (odd)')
parser.add_argument('--rays', type=int, default=50)
parser.add_argument('--tori-per-ray', type=int, default=20)
parser.add_argument('--device', type=int, default=2, help='GPU (1, 2 or 3)')
parser.add_argument('--resonances', choices=['spectra', 'tunes'], default='spectra', help='how resonances are found')
args = parser.parse_args()
out = Path('results')/(args.name or args.line.stem)
out.mkdir(parents=True, exist_ok=True)

the_map = tori.load(args.line, device=args.device, nemitt=2.5e-6, start='bptdv.a4l1.b1')
J = args.sigma**2/2                                     # r = sqrt(2J) = sigma
result = tori.launch_tori(the_map, args.tori_per_ray, args.angles, max_jx=J, max_jy=J, layout='fan', n_rays=args.rays)
np.savez(out/'tori.npz', **result)
print(f'{out.name}: {result["converged"].sum()}/{len(result["r"])} converged, largest residual '
      f'{result["residual"].max():.0e}, {result["seconds"]:.0f} s', flush=True)
tori.figures(result, out, args.resonances)
