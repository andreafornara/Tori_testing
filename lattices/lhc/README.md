# HL-LHC beam 1 at top energy: no errors, no beam-beam, octupole knob

`build_lhc_top.py --octupoles I` writes `lhc_top_b1_oct<I>.json`, an `xt.Line` (beam 1 only, starts at IP3), which
you load with `xt.Line.from_json`.

```bash
export PYTHONPATH=/eos/user/a/afornara/25_11_18_sorting/xmask
/home/afornara/ion_DA_study/miniforge/envs/d2-da/bin/python build_lhc_top.py --octupoles 0
```

Build time on lxplus: about 70 s per line on the CPU (67 s alone; 75-80 s with three builds in parallel at 2 threads
each). Loading the collider takes about 10 s of that.

`set_octupoles.py 50 100 150 200 250 300` writes `lhc_top_b1_oct<I>.json` from the 0 A line, without the collider
(no EOS, about 40 s per line on the CPU): it sets `i_oct_b1` (positive polarity here), then re-matches Q and Q' in 4D
with `dqx/dqy.b1_op` and `dqpx/dqpy.b1_op`. Result for every current: Q = (62.31, 60.32), Q' = (5.00, 6.00),
c- = 4.5-5.5e-5, so no coupling correction is needed.

## Recipe

The steps follow `sorting_final/` (config.yaml, functions.py) and the xmask example `hllhc19/` (`000_build.py`,
`002_tuning.py`). The error and correction steps are left out.

1. Collider: `xmask/examples/hllhc19/lhc_thin_test_00_prepared.json` (the `paths.collider_in` of
   sorting_final/config.yaml). This is the HL-LHC v1.9 lattice, sliced thin, with round optics `opt_150_thin`
   (beta* = 15 cm at IP1/IP5) at 7 TeV.
2. No field-quality errors are installed (arc/WISE, triplet, D1, D2), so no nonlinear correction is run. The line has
   no `on_error_*` knobs.
3. No beam-beam: the 224 inactive lenses that `000_build.py` installs (BeamBeamBiGaussian2D/3D) are removed from b1.
4. All `knob_settings` of `hllhc19/config.yaml` are applied, without luminosity levelling. Then `i_oct_b1 = I`.
5. `xm.machine_tuning`, twice, with the line's default twiss set to 4D during the match: closed-orbit correction
   against `lhc_co_ref_thin_test.json` (`corr_co_b1.json`), then c- -> 0 (`cmrs/cmis.b1_op`, tolerance 1e-4), then
   tune and chromaticity (`dqx/dqy.b1_op`, `dqpx/dqpy.b1_op`).

Knobs that are on:

| knob | value |
|---|---|
| crossing IP1 (H), IP5 (V) | `on_x1hs = on_x5vs = 250` urad, `on_disp = 1` (spurious-dispersion bumps on) |
| IP2 / IP8 | `on_x2v = -170`, `on_sep2h = 0.138` mm, `on_x8v = -200`, `on_sep8h = -0.043` mm |
| IP1 / IP5 separation | `on_sep1v = on_sep5h = 0` |
| crab cavities | `on_crab1 = on_crab5 = -190` urad (they give no kick at zeta = 0, so none in 4D tracking) |
| spectrometers | `on_alice_normalized = on_lhcb_normalized = 1`; solenoids off |
| RF | `vrf400 = 16` MV, `lagrf400.b1 = 0.5` |
| octupoles | `i_oct_b1 = I` (0, -100, -200, -300 A); the knob stays live in the saved line |
| targets (hllhc19 tuning) | Qx = 62.31, Qy = 60.32, Q'x = 5, Q'y = 6, c- -> 0 |

## Twiss summary (reloaded line, `twiss4d`)

| i_oct_b1 [A] | qx | qy | dqx | dqy | c_minus | betx/bety IP1 [m] | betx/bety IP5 [m] |
|---|---|---|---|---|---|---|---|
| 0    | 62.31000 | 60.32000 | 5.000 | 6.000 | 4.4e-5 | 0.1502 / 0.1500 | 0.1496 / 0.1501 |
| -100 | 62.31000 | 60.32000 | 5.000 | 6.000 | 5.7e-5 | 0.1505 / 0.1509 | 0.1511 / 0.1507 |
| -200 | 62.31000 | 60.32000 | 5.001 | 6.000 | 8.4e-5 | 0.1507 / 0.1519 | 0.1525 / 0.1514 |
| -300 | 62.31000 | 60.32000 | 5.000 | 6.000 | 7.4e-5 | 0.1510 / 0.1528 | 0.1540 / 0.1521 |

- The residual c- (4-8e-5) is already inside the 1e-4 tolerance. The coupling knobs therefore stay at 0, except at
  -300 A, where `cmrs.b1_op = -1.2e-4`.
- The closed orbit is found for every line: px(IP1) and py(IP5) are 248-250 urad.
- The spurious-dispersion bumps carry orbit through the arc octupoles. Their quadrupole feed-down shifts the tune
  (`dqx.b1_op` goes up by 1.3e-3 for every -100 A) and beats beta* by up to 2.6% at -300 A.

## Checks

- Every saved line has no beam-beam elements and no `on_error_*` knobs.
- No `Multipole` element carries a sextupole or higher component: there are no errors, and the spool pieces and IR
  correctors are at zero.
- Every knl/ksl/k1-k3 of the saved line equals that of the error-free collider with the same knob values, to 1e-20.
- Amplitude detuning from tracking: `line.get_amplitude_detuning_coefficients`, 4D with frozen longitudinal motion,
  1000 turns, amplitudes 0.01/1/2 sigma, nemitt 2.5e-6 m. dQ is the tune shift at 2 sigma.

| i_oct_b1 [A] | dQx/dJx [1/m] | dQy/dJy [1/m] | dQx/dJy [1/m] | dQx(2 sigma x) | dQy(2 sigma y) |
|---|---|---|---|---|---|
| 0    | +4.9e4  | +4.2e4  | -8.6e4 | +3.3e-5 | +2.8e-5 |
| -100 | -1.20e5 | -1.16e5 | -1.3e4 | -8.0e-5 | -7.8e-5 |
| -200 | -2.84e5 | -2.61e5 | +5.9e4 | -1.9e-4 | -1.7e-4 |
| -300 | -4.44e5 | -3.93e5 | +1.31e5 | -3.0e-4 | -2.6e-4 |

The direct terms change linearly with the octupole current, by about -1.6e5 1/m for every -100 A. The signs are the
usual ones for negative polarity: dQx/dJx < 0 and dQx/dJy > 0. At 0 A the remaining detuning comes from the lattice
sextupoles.
