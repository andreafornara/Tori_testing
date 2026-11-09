"""Build the HL-LHC v1.9 beam-1 line at top energy (7 TeV, beta* = 15 cm), without errors and without beam-beam
(CPU, about 70 s), from the collider prepared by xmask/examples/hllhc19/000_build.py.
  - knobs: those of hllhc19/config.yaml (crossing 250 urad at IP1/IP5, IP2/IP8, crab cavities, RF 16 MV)
  - octupoles: i_oct_b1 = --octupoles [A]
  - tuning: orbit correction, c- -> 0, tunes (62.31, 60.32) and chromaticity (5, 6) matched in 4D, twice
The line starts at IP3. Other currents from this line: set_octupoles.py.

Run (PYTHONPATH=/eos/user/a/afornara/25_11_18_sorting/xmask): python build_lhc_top.py --octupoles 0
    -> lhc_top_b1_oct0.json
"""
import argparse
import time
from pathlib import Path
import yaml
import xmask as xm
import xtrack as xt

HLLHC19 = Path('/eos/user/a/afornara/25_11_18_sorting/xmask/examples/hllhc19')
HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--octupoles', type=float, default=0., help='octupole current i_oct_b1 [A]')
args = parser.parse_args()
config = yaml.safe_load(open(HLLHC19/'config.yaml'))
tuning = config['tuning']
start = time.time()
collider = xt.load(HLLHC19/'lhc_thin_test_00_prepared.json')
reference = xt.load(HLLHC19/'lhc_co_ref_thin_test.json')      # the same machine, for the orbit correction
line = collider['b1']

# no beam-beam: remove the (inactive) lenses
line.remove([name for name, kind in zip(*line.get_table().cols['name', 'element_type'].values())
             if kind.startswith('BeamBeam')])

# the knobs of hllhc19 (no errors are installed, so the error knobs are skipped), then the octupoles
for name, value in config['knob_settings'].items():
    if not name.startswith('on_error'):
        collider[name] = value
collider['i_oct_b1'] = args.octupoles

# orbit, coupling, tune and chromaticity, matched in 4D (twice: the knobs move the orbit)
line.twiss_default['method'] = '4d'
for _ in range(2):
    xm.machine_tuning(line=line, enable_closed_orbit_correction=True, enable_linear_coupling_correction=True,
                      enable_tune_correction=True, enable_chromaticity_correction=True,
                      knob_names=tuning['knob_names']['b1'],
                      targets=dict(qx=tuning['qx']['b1'], qy=tuning['qy']['b1'], dqx=tuning['dqx']['b1'],
                                   dqy=tuning['dqy']['b1']),
                      step_q_knob=1e-4, step_dq_knob=1e-3, step_c_minus_knob=1e-5,
                      tol_tune=1e-4, tol_chromaticity=.05, tol_c_minus=1e-4,
                      line_co_ref=reference['b1'], co_corr_config=HLLHC19/tuning['closed_orbit_correction']['b1'])
line.twiss_default.pop('method')

twiss = line.twiss4d()
name = HERE/f'lhc_top_b1_oct{args.octupoles:g}.json'
line.to_json(name)
print(f'{name.name} in {time.time()-start:.0f} s: Q = ({twiss.qx:.4f}, {twiss.qy:.4f}), '
      f"Q' = ({twiss.dqx:.2f}, {twiss.dqy:.2f}), c- = {twiss.c_minus:.1e}")
