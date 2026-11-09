# tori_studies

Single invariant tori of Xsuite lines and their spectra. From a line saved as `.json`, every torus is solved on its own
by Newton on the 4D one-turn map (longitudinal plane frozen, delta = 0), tracked on a GPU. No foliation.

For every torus: its tunes Q and the spectra c[plane, m_x, m_y] (2D FFT of z_a = x_a - i p_a). From these: the resonances
it feels and a measure of its nonlinear distortion.

## Install

Python 3.12 and an NVIDIA GPU with CUDA.

```bash
bash install_python.sh [PREFIX]    # Miniforge + requirements.txt in PREFIX (default ./python); touches nothing else
```

Or install the pinned versions of `requirements.txt` in your own environment.

## Run

The examples are interactive: run their `# %%` cells one by one (VS Code, Jupyter), or a whole file from `examples/`.
Each one loads the line, decides which tori to track, retrieves and saves them, then makes the figures from the saved tori.

| Script | What |
|---|---|
| `examples/henon_2d.py` | 2D Hénon: resonances, spectra (PNG and HTML), tori in phase space |
| `examples/henon_4d.py` | 4D Hénon: resonances and resonance maps (interpolated and not), 2D spectra and tori (HTML) |
| `examples/lhc.py` | LHC at top energy, 0-6 sigma, 1000 tori (about 7 min): same figures as 4D Hénon |
| `examples/lhc_distortion.py` | bonus: the LHC tori for every octupole current, then the distortion comparison |

The tori and the figures go to `results/<name>/`.

The first LHC run on a GPU compiles the tracking kernel: about 15 min and 18 GB of memory. It is then reused from `.cache/`.

## Contents

| Path | What |
|---|---|
| `tori_functions/` | the package: `line.py` (load a line), `launch.py` and `newton.py` (solve the tori), `detect.py` (resonances from the spectra), `distortion.py`, `plots.py`, `plots4d.py` |
| `examples/` | the scripts above |
| `lattices/henon/` | Hénon 2D and 4D maps as Xsuite lines |
| `lattices/lhc/` | HL-LHC beam 1 at top energy, no errors, no beam-beam, octupole knob |

Only the 0 A LHC line is in the repository (14 MB). The results, logs and compiled kernels are not.

## Definitions

- **Linear power:** sum of |c_m|^2 over |m_x|, |m_y| <= 1, both planes (displaced ellipse and coupling included).
- **Nonlinear harmonics:** |m_x| > 1 or |m_y| > 1.
- **Distortion D:** power of the nonlinear harmonics over the linear power.

## More

- [README_tori.md](README_tori.md): every figure and how the resonances are found.
- [lattices/lhc/README.md](lattices/lhc/README.md): how the LHC lines are built, with their optics and checks.
- [lattices/henon/README.md](lattices/henon/README.md): the Hénon lines.
