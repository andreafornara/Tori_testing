# Hénon maps as Xsuite lines

These are the reference Hénon maps of `final_tori_foliation/maps.py`, written as Xsuite `xt.Line` files.
Tracking a line gives the analytic map **bit for bit**.

| file | analytic map in maps.py | (Qx, Qy) |
|---|---|---|
| `henon_2d_q0.2071.json` | `slide_henon(0.2071)` with y = py = 0 | (0.2071, 0.28) |
| `henon_2d_q0.218.json` | `slide_henon(0.218)` with y = py = 0 (strong 3/14 resonance at r = 0.3385) | (0.218, 0.28) |
| `henon_4d_q0.2071_0.28.json` | `coupled_henon(k2=-2, k3=0, qx=0.2071, qy=0.28)` | (0.2071, 0.28) |

## Map

One turn applies the kick first and then the rotation, in the same order as maps.py:

- kick: px → px + x² − y², py → py − 2xy
- rotation by 2πQ in each plane: x' = c x + s px, px' = −s x + c px, with c = cos 2πQ and s = sin 2πQ

With y = py = 0 the kick is px → px + x², which is the 2D map z' = e^{2πiQ0}[z − i(Re z)²] with z = x − i px.
So a single line family covers both cases: a 2D line is the 4D line with Qx = Q0, started at y = py = 0. In that
case y and py stay exactly 0. The vertical tune 0.28 of the 2D lines has no effect there; it only keeps twiss4d
well defined.

## Elements

- `kick` = `xt.Multipole(knl=[0, 0, -2])`. Xsuite's thin multipole gives dpx − i dpy = −Σ knl_n (x+iy)^n / n!,
  so knl₂ = −2 produces the kick above. Checked numerically on the kick element alone:
  (x, y) = (0.1, 0), (0, 0.1), (0.1, 0.1) give px = +0.01, −0.01, 0 and py = 0, 0, −0.02.
- `rot` = `xt.FirstOrderTaylorMap(m1=R)`, where R is the exact 6×6 rotation matrix and its longitudinal block is
  the identity. No drift, no length, no chromaticity, no detuning.
  I used this instead of `LineSegmentMap(betx=bety=1)`. Both track bit-identically, but in xtrack 0.104.3 a
  LineSegmentMap does not survive `from_dict`/`from_json`: its frozen longitudinal mode is read back as
  'nonlinear' and an assertion fails.
- Reference particle: protons with p0c = 450 GeV. The choice does not matter. delta = zeta = 0 and
  `line.freeze_longitudinal()` is applied after loading, so tracking is 4D.

**Units:** β = 1 and α = 0 in both planes, so the Xsuite coordinates (x, px, y, py) are directly the normalized
coordinates of maps.py, with no normalization step. One "sigma" of this line is 1 map unit. This is not the
detuning-calibrated sigma of the fortori henon4d README.

## Check (`check_henon.py`, output in `check_henon.log`)

The check tracks 10⁴ random initial conditions on the CPU (`xo.ContextCpu`). They are uniform in the disk
r ≤ r_max of each active plane: r_max = 0.40 (2D, Q0 = 0.2071), 0.33 (2D, Q0 = 0.218), and 0.15 per plane (4D).
Each line is compared with maps.py (NumPy).

| line | max abs(line − maps.py), 1 turn | 1000 turns | twiss4d |
|---|---|---|---|
| 2D Q0 = 0.2071 | 0 (all 10⁴ bit-identical) | 0 (bit-identical) | qx = 0.2071, qy = 0.28, β = 1, α = 0 |
| 2D Q0 = 0.218 | 0 | 0 | qx = 0.218, qy = 0.28, β = 1, α = 0 |
| 4D | 0 | 0 | qx = 0.2071, qy = 0.28, β = 1, α = 0 |

All orbits stayed bounded for 1000 turns. The agreement is exact, not just at round-off level, because both
codes evaluate the same floating-point expressions: c x + s (px + x²).

To rebuild and check:

```
python build_henon.py
python check_henon.py
```

Use the python at `/home/afornara/ion_DA_study/miniforge/envs/d2-da/bin/python`. The check takes about 80 s, most
of it compiling the kernels.
