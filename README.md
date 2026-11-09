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

From the repository root:

```bash
# Hénon lines, unit coordinates (GPU 2)
python examples/launch_henon.py

# LHC at top energy, 0 A, 0-6 sigma: 1000 tori
python examples/launch_lhc.py lattices/lhc/lhc_top_b1_oct0.json lhc_top_b1_oct0_6sigma --sigma 6 --device 2

# other octupole currents, from the 0 A line (CPU, about 40 s each)
cd lattices/lhc && python set_octupoles.py 50 100 150 200 250 300 && cd ../..

# compare the lattices by their nonlinear distortion
python examples/compare_distortion.py results/lhc_top_b1_oct0_6sigma results/lhc_top_b1_oct50_6sigma
```

Each run writes `tori.npz` and its figures in `results/<name>/`.

The first LHC run on a GPU compiles the tracking kernel: about 15 min and 18 GB of memory. It is then reused from `.cache/`.

From Python:

```python
import tori
the_map = tori.load('lattices/lhc/lhc_top_b1_oct0.json', device=2)     # normalized one-turn map, in beam sigmas
result = tori.launch_tori(the_map, n_tori=50, n_frequencies=41, max_jx=2., max_jy=2., layout='fan', n_rays=7)
tori.figures(result, 'results/my_line')
```

## Contents

| Path | What |
|---|---|
| `tori/` | the package: `line.py` (load a line), `launch.py` and `newton.py` (solve the tori), `detect.py` (resonances from the spectra), `distortion.py`, `plots.py`, `plots4d.py` |
| `examples/` | the scripts above; `test_henon_few.py` (Hénon, many tori); `compare_swarm.py` (extra tori around the resonance lines) |
| `lattices/henon/` | Hénon 2D and 4D maps as Xsuite lines |
| `lattices/lhc/` | HL-LHC beam 1 at top energy, no errors, no beam-beam, octupole knob |
| `run_octupole_scan.sh` | the 0-300 A scan on three GPUs (edit the paths at the top) |

Only the 0 A LHC line is in the repository (14 MB). The results, logs and compiled kernels are not.

## Definitions

- **Linear power:** sum of |c_m|^2 over |m_x|, |m_y| <= 1, both planes (displaced ellipse and coupling included).
- **Nonlinear harmonics:** |m_x| > 1 or |m_y| > 1.
- **Distortion D:** power of the nonlinear harmonics over the linear power.

## More

- [README_tori.md](README_tori.md): every figure and how the resonances are found.
- [lattices/lhc/README.md](lattices/lhc/README.md): how the LHC lines are built, with their optics and checks.
- [lattices/henon/README.md](lattices/henon/README.md): the Hénon lines.
