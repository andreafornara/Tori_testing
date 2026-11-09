"""Invariant tori of an Xsuite line, one by one, and their spectra.

    import tori
    the_map = tori.load('line.json', device=2)                      # the normalized one-turn map, on a GPU
    result = tori.launch_tori(the_map, 50, 41, max_jx=2., max_jy=2., layout='fan')
    tori.figures(result, 'results/my_line')                         # every figure; method='tunes': the other way
                                                                    # of finding the resonances
The figures one by one: tori.plots.march_spectra, march_spectra_slider, resonance_web and, for 4D tori,
tori.plots4d.spectra_2d, resonance_maps, tori_slider; resonance_web and resonance_maps take interpolated=True.
"""
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')        # GPUs numbered as in nvidia-smi
from pathlib import Path
import numpy as np
from .line import load
from .launch import launch_tori
from . import plots, plots4d


def figures(result, out, method='spectra'):
    """Every figure of a result into the folder `out`; the 4D views when some tori have two angles."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots.march_spectra(result, out/'march_spectra.png', method=method)
    plots.march_spectra_slider(result, out/'march_spectra.html', method=method)
    plots.resonance_web(result, out/'resonance_web.png', method=method)
    if np.any(np.all(result['J'] > 0, axis=1)):
        plots.resonance_web(result, out/'resonance_web_interpolated.png', method=method, interpolated=True)
        plots.resonance_lines(result, out/'resonance_web_lines.png', method=method)
        plots4d.resonance_maps(result, out/'resonance_maps.png', method=method)
        plots4d.resonance_maps(result, out/'resonance_maps_interpolated.png', method=method, interpolated=True)
        plots4d.spectra_2d(result, out/'spectra_2d.html', method=method)
        plots4d.tori_slider(result, out/'tori.html')
