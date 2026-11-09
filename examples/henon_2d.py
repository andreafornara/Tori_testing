"""Invariant tori of the 2D Hénon line: load the line, decide which tori to track, retrieve and save them, then the figures.
Run the # %% cells one by one, or the whole file from this folder. Unit coordinates."""

# %% Load the line
import sys
sys.path.insert(0, '..')
from pathlib import Path
import numpy as np
from IPython.display import Image
from tori_functions import load, launch_tori, plots

NAME = 'henon_2d_q0.2071'           # also: henon_2d_q0.218
DEVICE = 2                          # GPU (1, 2 or 3)
OUT = Path('../results')/NAME
OUT.mkdir(parents=True, exist_ok=True)
the_map = load(f'../lattices/henon/{NAME}.json', device=DEVICE, nemitt=None)

# %% Decide which tori to track: x tori, equally spaced in amplitude
N_TORI = 500
MAX_R = .37                         # largest amplitude r = sqrt(2 J)
ANGLES = 511                        # angles of a torus (odd)

# %% Retrieve the invariant tori and save them
result = launch_tori(the_map, N_TORI, ANGLES, max_jx=MAX_R**2/2)
np.savez(OUT/'tori.npz', **result)
print(f'{result["converged"].sum()}/{len(result["r"])} converged, {result["seconds"]:.0f} s')

# %% Postprocess: load the saved tori (the cells above can be skipped)
result = dict(np.load(OUT/'tori.npz'))

# %% Resonances: the resonant tori in phase space and the harmonics m Q mod 1
plots.resonance_web(result, OUT/'resonance_web.png')
Image(OUT/'resonance_web.png')

# %% Spectra of all the tori against the amplitude
plots.march_spectra(result, OUT/'march_spectra.png')
Image(OUT/'march_spectra.png')

# %% Spectra, HTML: open it in a browser
plots.march_spectra_slider(result, OUT/'march_spectra.html')

# %% Tori in phase space
plots.tori_2d(result, OUT/'tori.png')
Image(OUT/'tori.png')
