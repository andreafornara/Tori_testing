"""What the swarm of tori adds around the resonance lines named by the tori that did not converge, against the same
scan without it. For every line: residual and relative power of the line against the distance |n.Q - p|.
Usage (from the repository root): python examples/compare_swarm.py results/lhc_top_b1_oct50_6sigma_swarm results/...
Writes swarm_comparison.png in each folder and prints what was found."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tori import detect, plots


def name(w):
    return f'{w[0]}$Q_x${w[1]:+d}$Q_y$ = {w[2]}'


def text(w):
    return f'{w[0]}Qx{w[1]:+d}Qy={w[2]}'


def satellites(c, w):
    """The relative power of the line w on the tori c."""
    return plots.line_power(dict(c=c), slice(None), w)


for folder in map(Path, sys.argv[1:]):
    r = dict(np.load(folder/'tori.npz'))
    before = dict(np.load(Path(str(folder).replace('_swarm', ''))/'tori.npz'))
    lines = np.unique(r['swarm_w'], axis=0) if len(r['swarm_w']) else np.zeros((0, 3), int)
    print(f"\n{folder.name}: tori {r['converged'].sum()}/{len(r['J'])} converged, before "
          f"{before['converged'].sum()}/{len(before['J'])}")
    print(f"  time [s]: tori {float(r['seconds']):.0f} (before {float(before['seconds']):.0f}), record "
          f"{float(r['record_seconds']):.1f}, swarm {float(r['swarm_seconds']):.0f}")
    print(f"  named by the tori not converged: {[text(w) for w in np.unique(r['failed_line'], axis=0)]}")
    print(f"  named by the swarm tori not converged: {[text(w) for w in np.unique(r['swarm_failed_line'], axis=0)]}")
    found = detect.detect(r)
    print(f"  detect.py on the tori: colour {[text(w) for w in found['shown']]}")
    fig, axes = plt.subplots(2, max(len(lines), 1), figsize=(5.2*max(len(lines), 1), 7), constrained_layout=True,
                             squeeze=False, sharex=True, sharey='row')
    for j, w in enumerate(lines):
        here = np.all(r['swarm_w'] == w, axis=1)
        d = np.abs(r['swarm_Q'][here]@w[:2]-w[2])
        ok = r['swarm_converged'][here]
        P = satellites(r['swarm_c'][here], w)
        res = r['swarm_residual'][here]
        # the scan in the same region as the swarm: its converged tori within 1 sigma of a swarm torus
        reach = np.min(np.linalg.norm(r['r'][:, None]-r['swarm_r'][here][None], axis=2), axis=1)
        fan = np.all(r['J'] > 0, axis=1) & r['converged'] & (reach < 1.)
        fan_d = np.abs(r['Q'][fan]@w[:2]-w[2])
        fan_P = satellites(r['c'][fan], w)
        # the fit of the power against the distance, on the converged swarm tori well above the noise
        select = ok & (P > 1e-26) & (d > 2e-5)
        slope = np.polyfit(np.log10(d[select]), np.log10(P[select]), 1)[0] if select.sum() > 3 else np.nan
        failed = np.sort(d[~ok])
        print(f"  {text(w)}: swarm {here.sum()} tori, {ok.sum()} converged; closest converged |n.Q-p|: swarm {d[ok].min():.1e}, "
              f"scan {fan_d.min():.1e}; failed at {[f'{v:.1e}' for v in failed[:8]]}; slope of the power {slope:+.2f}")
        for ax, y, fan_y in ((axes[0, j], res, r['residual'][fan]), (axes[1, j], P, fan_P)):
            ax.scatter(fan_d, fan_y, s=9, c='#b9b9c9', label='the scan (converged)')
            ax.scatter(d[ok], y[ok], s=26, c='#2a78d6', label='swarm: converged')
            ax.scatter(d[~ok], y[~ok], s=46, marker='x', c='#d62a2a', label='swarm: not converged')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=.3)
        axes[0, j].set_title(name(w), fontsize=11)
        axes[1, j].set_xlabel('distance from the line  $|n\\cdot Q-p|$')
    axes[0, 0].set_ylabel('Newton residual')
    axes[1, 0].set_ylabel('relative power of the line')
    axes[0, 0].legend(fontsize=8)
    fig.savefig(folder/'swarm_comparison.png', dpi=130)
