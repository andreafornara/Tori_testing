"""Invariant tori of an LHC line: load the line, decide which tori to track, retrieve and save them, then the figures.
Run the # %% cells one by one, or the whole file from this folder. Coordinates in beam sigmas."""

# %% Load the line
import sys
sys.path.insert(0, '..')
from pathlib import Path
import numpy as np
from IPython.display import Image
from tori_functions import load, launch_tori, plots, plots4d

NAME = 'lhc_top_b1_oct0'            # the other currents: lattices/lhc/set_octupoles.py
DEVICE = 2                          # GPU (1, 2 or 3)
the_map = load(f'../lattices/lhc/{NAME}.json', device=DEVICE, nemitt=2.5e-6, start='bptdv.a4l1.b1')

# %% Decide which tori to track: a fan of rays from the x axis to the y axis
RAYS = 50
TORI_PER_RAY = 20
SIGMA = 6                           # largest amplitude per plane [sigma]
ANGLES = 41                         # angles per plane (odd)
OUT = Path('../results')/f'{NAME}_{SIGMA}sigma'
OUT.mkdir(parents=True, exist_ok=True)

# %% Retrieve the invariant tori and save them
J = SIGMA**2/2                      # r = sqrt(2 J) = sigma
result = launch_tori(the_map, TORI_PER_RAY, ANGLES, max_jx=J, max_jy=J, layout='fan', n_rays=RAYS)
np.savez(OUT/'tori.npz', **result)
print(f'{result["converged"].sum()}/{len(result["r"])} converged, {result["seconds"]:.0f} s')

# %% Postprocess: load the saved tori (the cells above can be skipped)
result = dict(np.load(OUT/'tori.npz'))
METHOD = 'spectra'                  # how the resonances are found: 'spectra' or 'tunes'

# %% Resonances: tune plane and amplitude plane
plots.resonance_web(result, OUT/'resonance_web.png', method=METHOD)
Image(OUT/'resonance_web.png')

# %% Resonances, interpolated between the tori
plots.resonance_web(result, OUT/'resonance_web_interpolated.png', method=METHOD, interpolated=True)
Image(OUT/'resonance_web_interpolated.png')

# %% Resonances as lines, coloured by their relative power
plots.resonance_lines(result, OUT/'resonance_web_lines.png', method=METHOD)
Image(OUT/'resonance_web_lines.png')

# %% Resonance maps
plots4d.resonance_maps(result, OUT/'resonance_maps.png', method=METHOD)
Image(OUT/'resonance_maps.png')

# %% Resonance maps, interpolated between the tori
plots4d.resonance_maps(result, OUT/'resonance_maps_interpolated.png', method=METHOD, interpolated=True)
Image(OUT/'resonance_maps_interpolated.png')

# %% 2D spectra, HTML: open it in a browser
plots4d.spectra_2d(result, OUT/'spectra_2d.html', method=METHOD)

# %% Tori, HTML: open it in a browser
plots4d.tori_slider(result, OUT/'tori.html')
