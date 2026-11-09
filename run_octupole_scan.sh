#!/bin/bash
# Octupole scan at top energy, 0-6 sigma: build the missing lines (CPU), then the tori on GPUs 1, 2 and 3.
cd /home/afornara/test_tori/tori_studies
PY=~/ion_DA_study/miniforge/envs/d2-da/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs
missing=$(for I in 100 200 250 300; do [ -f lattices/lhc/lhc_top_b1_oct$I.json ] || echo $I; done)
[ -n "$missing" ] && (cd lattices/lhc && $PY set_octupoles.py $missing) >> logs/set_octupoles.log 2>&1
run() { for I in "${@:2}"; do
          $PY examples/launch_lhc.py lattices/lhc/lhc_top_b1_oct$I.json lhc_top_b1_oct${I}_6sigma --sigma 6 --device $1 \
              > logs/oct${I}_6sigma.log 2>&1; done; }
run 1 0 300 &
run 2 100 250 &
run 3 200 &
wait
echo done > logs/octupole_scan.done
