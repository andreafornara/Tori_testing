## The `tori_functions` package

Single invariant tori of an Xsuite line (`.json`) and their spectra. No foliation: every torus on its own.

```python
from tori_functions import load, launch_tori, plots
the_map = load('lattices/lhc/lhc_top_b1_oct0.json', device=2)                # normalized one-turn map, beam sigmas
result = launch_tori(the_map, n_tori=50, n_frequencies=41, max_jx=2., max_jy=2., layout='fan', n_rays=7)
plots.resonance_web(result, 'resonance_web.png')
```
Figures: `plots.march_spectra`, `march_spectra_slider`, `resonance_web`, `resonance_lines`, `tori_2d` (x tori) and, for 4D tori,
`plots4d.spectra_2d`, `resonance_maps`, `tori_slider`. Every resonance figure takes `method='spectra'` (default) or
`'tunes'` (how the resonances are found, below); `resonance_maps` and `resonance_web` also take `interpolated=True`.
Tori that did not converge are red crosses (their best approximation is kept in the data).

Figures (methodology; the docstrings of `tori_functions/plots.py` and `tori_functions/plots4d.py` have the details):
- Harmonic m = (m_x, m_y) of z_a = x_a - i p_a: the term c_m e^{i m.theta} of the torus; its frequency is m.Q mod 1.
- Resonance n.Q = p crossed: the tunes Q(J) of the tori meet it; there the harmonics m and m + n have the same
  frequency. Its resonant harmonics are m = e_a + k n: their divisor |e^{2 pi i k (n.Q - p)} - 1| (the denominator of
  c_m in the torus equation) vanishes there, so they are amplified. Every resonance keeps one colour in all figures.
- Every torus is drawn as one marker or one row: no interpolation between tori. A resonance is located by rings on
  the tori next to it (of two neighbouring tori on either side of n.Q = p, the closer one).
- Interpolated version (`interpolated=True`, files `*_interpolated.png`): the same fields filled linearly between the
  tori over (r_x, r_y), nothing filled across an axis torus or one that did not converge (the white gaps); every
  resonance is the curve n.Q(J) = p of the interpolated tunes instead of rings, hatched where not resolved.
- `march_spectra`: log10 |c_m| of the tori of one ray (rows: amplitude; columns: harmonics by signed order
  +-(|m_x| + |m_y|), 0 in the middle); a crossing is a horizontal line on the torus next to it, the resonant
  harmonics are marked in its colour. The HTML page has two sliders, angle and amplitude: the march in amplitude
  along the ray and the march in angle at the amplitude, through the selected torus.
- `resonance_web` (4D): the tunes of the tori and the resonance lines they cross; the amplitude plane; both coloured
  by the tune shift |Q(J) - Q(0)|, the tori next to every resonance ringed in its colour.
- `tori_2d` (x tori): every torus in phase space, rebuilt from its spectrum, coloured by its amplitude.
- `resonance_lines` (4D, `resonance_web_lines.png`): the resonances as lines with their name written on them, in the tune
  plane and the amplitude plane, coloured on one scale by the relative power of the line: its harmonics e_a + k n outside
  |m_x|, |m_y| <= 1, over the linear power. The tori that did not converge count with their best approximation.
- `spectra_2d`: for one torus, the relative power |c_m|^2 / linear power (|m_x|, |m_y| <= 1, both planes) of every harmonic
  as an image over (m_x, m_y), resonant harmonics circled; the torus marked in the tune plane.
- `resonance_maps`: over (r_x, r_y), all resonances together, then per resonance the distance log10 |n.Q - p| and the
  power of its resonant harmonics relative to the linear ones; faded and crossed where that power is below 10x the
  noise power (the noise floor of the spectrum amplified by the same divisor): not resolved there.
- `tori_slider` (`tori.html`): two sliders, angle and amplitude, select one torus; the torus itself, rebuilt in the
  browser from its spectrum (x_a = Re z_a, p_a = -Im z_a on 64 x 64 angles, coloured by theta_y), in (x, p_x),
  (y, p_y), (x, y) and (p_x, p_y), with the linear torus of the same J dashed; fixed axes, so the deformation grows
  with the slider. Only the coefficients above 1e-7 of the linear one are written (a few MB for 1000 tori).

### How the resonances are retrieved (`tori_functions/detect.py`)

The resonances are flagged from the spectra at any order; the figures show the low orders. Each torus has one FFT matrix per plane,
c[a, m_x, m_y] for z_a; its tunes Q are unknowns of the Newton solve of M(K(theta)) = K(theta + 2 pi Q).

1. **Peaked entries.** An entry (a, m) of a torus is peaked when it rises more than 1 decade above the line through
   the same entry on its two neighbouring tori: on its ray (in log r) or around its arc at the same amplitude (in the
   ray angle). It must also lie above 1000x the noise floor of the spectrum of the torus (the median |c| at the far
   corner of the band). The tori not converged are used too.
2. **Type.** A peaked entry of z_a points along m - e_a (e_a: the linear harmonic). The torus takes the direction with
   the most peaked entries. A resonance also raises sidebands, in other directions with fewer entries. The multiple
   is the smallest k with e_a + k d peaked: n = k d.
3. **Confirmed.** p = round(n.Q) at that torus, and the crossing is within one sampling step: |n.Q - p| is no larger
   than the change of n.Q to a neighbouring torus. Otherwise the label was a sideband and is dropped. The tunes only
   check the label and name p.
   **Located.** A peak marks a resonance only where one torus is much closer to it than its neighbours. Where the
   curve n.Q = p runs between two rays, the tori on either side are equally amplified and neither peaks. So a
   resonance found is followed along its whole curve: every torus next to the crossing (of two neighbouring tori on
   either side of n.Q = p, the closer one) is added, and step 4 decides on each whether the spectrum resolves it.
   A resonance not resolved on 2 of these tori keeps only the tori where its entries peaked.
4. **Resolved.** For every torus where it is confirmed:
   - resonant power P = sum over a and 0 < |k| <= 4 of |c[a, e_a + k n]|^2, divided by the linear power
     (the sum of |c|^2 over |m_x|, |m_y| <= 1, both planes);
   - noise power N = sum over the same entries of (eps / |2 sin(pi k (n.Q - p))|)^2, eps = the noise floor of the
     spectrum: the error left in an entry, amplified by its own divisor.
   It is resolved there when P > 10 N.
5. **What is flagged and shown.** A resonance found (confirmed on at least 2 tori) and resolved on at least 2 tori is
   flagged (`found['flagged']`), at any order. The figures colour the flagged ones with |n_x| + |n_y| <= 13 and every
   resonance named by a torus not converged (`launch.record`, by its invariance error over its small divisors): a
   legend entry ("8 tori, 8 resolved"), rings on its tori and a row in `resonance_maps`. The others found of low order
   are grey: rings and one legend entry.

**The other method, `method='tunes'`** (as before the spectral detection): the resonances are guessed from the
tunes. Every n.Q = p with |n_x| + |n_y| <= 16 (n and p without a common factor) that the tunes of the tori cross is
taken; it is located on the tori next to its crossing and gets the same resolved test (step 4) and the same rule for
what is shown (step 5). Which to choose: `'tunes'` lists every low-order line the footprint crosses, also where the
spectrum shows nothing, and never sees one above order 16; `'spectra'` lists what the tori actually feel, at any order.
In the examples: `METHOD`.

The axis tori (one angle) keep the tune crossings q Q_a = p, q <= 40, in `march_spectra`.

- `load(path, device=2, nemitt=2.5e-6, start=None)`: loads the line, freezes the longitudinal plane (4D,
  delta = 0), simplifies big lines for tracking, tracks on the GPU `device` (nvidia-smi numbering), finds the closed
  orbit and an exactly symplectic normalizing matrix W from the one-turn matrix. Coordinates in beam sigmas of the
  normalized emittance `nemitt`; `nemitt=None`: unit coordinates (the Hénon lines). `start`: the observation point.
- `launch_tori(the_map, n_tori, n_frequencies, max_jx, max_jy=0, layout='diagonal'|'fan'|'grid', n_rays=7)`:
  initial conditions equally spaced in r = sqrt(2J) along rays from the origin; `fan`: n_rays rays from the x axis
  (phi = 0) to the y axis (phi = 90 deg), ending on the ellipse (R_x cos phi, R_y sin phi). Tori on an axis are
  lower-dimensional (x tori, J_y = 0, or y tori). Each torus: Newton from the circle (map, FFT, least squares) on
  `n_frequencies` angles per plane (odd). Returns a dict (`np.savez`-able): J, r, ray, phi, Q, spectra
  c[torus, plane, m_x, m_y], residual, converged, angles.
- Newton settings follow the round-off of the map: at double-precision round-off (Hénon) tolerance 3e-15; a tracked
  lattice 5e-12 sigma, chord jacobian, resonance filter 1e-5. A torus not converged is solved once more on twice the
  angles with 40 Newton steps (`launch.RETRY`; `launch.retry(the_map, result)` on a saved result) and keeps its best
  attempt.
- `tori_functions/newton.py`: the solver of `final_tori_foliation/foliation.py` (GPU only, without the fit); a torus with a lost
  particle stops alone.

Examples: the `# %%` scripts of `examples/`, see `README.md`; outputs in `results/` (not in git).
Only the 0 A LHC line is in git: the other currents come from it with `lattices/lhc/set_octupoles.py`.

### Nonlinear distortion ranking (`tori_functions/distortion.py`)

From the spectra of a torus, with |c_m|^2 = |c[x, m]|^2 + |c[y, m]|^2:
- action of plane k: I_k = 1/2 sum_m m_k |c_m|^2 (the enclosed area / 2 pi; equal to the J of the torus to 1e-13);
- linear power: the sum of |c_m|^2 over |m_x|, |m_y| <= 1 (displaced ellipse and coupling included);
- distortion: D = sum of |c_m|^2 over the nonlinear harmonics (|m_x| > 1 or |m_y| > 1), over the linear power. A linear torus has D = 0.

`examples/lhc_distortion.py` solves the tori of every octupole current and writes
`results/distortion_comparison.png`: D versus the action (mean over the ray angles and min - max band), the ranking at
1, 2, 4, 6 sigma, D along the x axis, the diagonal and the y axis, and a map of D per lattice (red crosses: tori that did
not converge, left out of the means).
LHC top energy, octupoles 0-300 A: D grows with the current at every amplitude (0 A lowest). Order 2 dominates up to
6 sigma: the octupoles act as sextupoles through the closed orbit at the octupoles (up to 3 mm, feed-down), so sqrt(D)
of order 2 grows linearly with the current.
