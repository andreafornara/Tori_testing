#!/bin/bash
# Self-contained Python for tori_studies: Miniforge (python 3.12 + CUDA 11.8 libraries) and requirements.txt.
#
#   bash install_python.sh [PREFIX]        default PREFIX: ./python in the repository (gitignored)
#   PREBUILD=1 bash install_python.sh ...  also precompile the Xsuite CPU kernels (xsuite-prebuild, ~5 min)
#   GPU=3 bash install_python.sh ...       GPU of the smoke test (nvidia-smi numbering, default 2)
#
# Touches nothing outside PREFIX (no ~/.bashrc, no conda init, no other environment).
# Use afterwards:  PREFIX/bin/python examples/launch_henon.py   (or: export PATH=PREFIX/bin:$PATH)
set -euo pipefail

MINIFORGE_VERSION=26.9.4-1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFIX=$(realpath -m "${1:-$REPO/python}")
GPU=${GPU:-2}
export PYTHONNOUSERSITE=1          # ignore packages in ~/.local
unset PYTHONPATH CONDA_PREFIX
start=$SECONDS

# 1. Miniforge, batch mode (-b: no conda init, nothing written to ~/.bashrc)
if [ ! -x "$PREFIX/bin/conda" ]; then
    installer=$(mktemp -d)/miniforge.sh
    curl -fsSL -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/Miniforge3-$MINIFORGE_VERSION-Linux-x86_64.sh"
    bash "$installer" -b -p "$PREFIX"
    rm -rf "$(dirname "$installer")"
fi
echo "== Miniforge: $((SECONDS-start)) s"

# 2. Base env: python 3.12 and the CUDA 11.8 runtime libraries (nvrtc, cublas, ...) that cupy-cuda11x loads
"$PREFIX/bin/conda" install -y -n base -c conda-forge --override-channels python=3.12 cudatoolkit=11.8
echo "== conda packages: $((SECONDS-start)) s"

# 3. Pinned Python packages
"$PREFIX/bin/python" -m pip install --upgrade-strategy only-if-needed -r "$REPO/requirements.txt"
echo "== pip packages: $((SECONDS-start)) s"

# 4. Optional: precompiled Xsuite CPU kernels (faster first twiss/tracking of standard lines)
if [ "${PREBUILD:-0}" = 1 ]; then
    "$PREFIX/bin/python" -m pip install xsuite==0.51.4
    "$PREFIX/bin/xsuite-prebuild" regenerate
    echo "== xsuite-prebuild: $((SECONDS-start)) s"
fi

# 5. Smoke test: imports, GPU, the 4D Henon line tracked on CPU and on GPU
CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR="$PREFIX/cupy_cache" \
"$PREFIX/bin/python" - "$REPO/lattices/henon/henon_4d_q0.2071_0.28.json" <<'EOF'
import sys
import numpy as np, scipy, matplotlib, cupy as cp
import xobjects as xo, xtrack as xt, xpart, xdeps, xmask
print('python', sys.version.split()[0], '| numpy', np.__version__, '| scipy', scipy.__version__,
      '| matplotlib', matplotlib.__version__, '| cupy', cp.__version__, '| xtrack', xt.__version__,
      '| xmask', xmask.__version__)
print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
      '| CUDA runtime', cp.cuda.runtime.runtimeGetVersion(), '| driver', cp.cuda.runtime.driverGetVersion())

coords = dict(x=np.linspace(.01, .1, 10), px=np.linspace(0, .05, 10), y=np.linspace(.05, 0, 10), py=np.zeros(10))
result = {}
for name, context in (('CPU', xo.ContextCpu()), ('GPU', xo.ContextCupy(device=0))):
    line = xt.Line.from_json(sys.argv[1])
    line.freeze_longitudinal(True)
    line.build_tracker(_context=context)
    particles = line.build_particles(**coords, zeta=0., delta=0.)
    line.track(particles, num_turns=100)
    particles.move(_context=xo.ContextCpu())
    order = np.argsort(particles.particle_id)
    result[name] = np.array([getattr(particles, k)[order] for k in ('x', 'px', 'y', 'py')])
    assert np.all(particles.state > 0), f'{name}: particles lost'
difference = np.max(np.abs(result['CPU']-result['GPU']))
print(f'Henon 4D, 10 particles x 100 turns: max |CPU - GPU| = {difference:.1e}')
assert difference < 1e-12, 'CPU and GPU tracking disagree'
print('smoke test OK')
EOF
echo "== done in $((SECONDS-start)) s: $PREFIX/bin/python"
