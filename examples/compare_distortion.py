"""Compare lattices by the nonlinear distortion D of their tori (see tori/distortion.py), as a function of the action.
Usage (from the repository root): python examples/compare_distortion.py results/lhc_top_b1_oct0_6sigma ...
Writes results/distortion_comparison.png and prints the ranking."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tori.distortion import distortion, spectral_actions

AMPLITUDES = [1, 2, 4, 6]      # sigma: where the lattices are ranked
OUT = Path('results/distortion_comparison.png')

# one entry per lattice, ordered by octupole current
lattices = []
for folder in sys.argv[1:]:
    result = dict(np.load(Path(folder)/'tori.npz'))
    current = float(re.search(r'oct(-?[\d.]+)', str(result['line'])).group(1))
    D = distortion(result)
    rho = np.round(np.hypot(*result['r'].T), 6)             # the amplitude levels of the fan, shared by all rays
    levels = np.unique(rho)
    lattices.append(dict(current=current, result=result, D=D, I=spectral_actions(result).sum(axis=1),
                         levels=levels, action=levels**2/2,
                         mean=np.array([np.nanmean(D[rho == v]) for v in levels]),
                         low=np.array([np.nanmin(D[rho == v]) for v in levels]),
                         high=np.array([np.nanmax(D[rho == v]) for v in levels])))
    print(f'{current:g} A: {result["converged"].sum()}/{len(D)} converged')
lattices.sort(key=lambda l: l['current'])
colors = plt.cm.viridis(np.linspace(0, .85, len(lattices)))
action_label = r'action $I_x + I_y$ [$\sigma^2$]'

COLUMNS = 4                    # maps per row
map_rows = (len(lattices)+COLUMNS-1)//COLUMNS
fig = plt.figure(figsize=(17, 9+4.7*map_rows))
rows = fig.add_gridspec(3, 1, height_ratios=[.8, 1, 1.1*map_rows], hspace=.28)
top = rows[0].subgridspec(1, 4, wspace=.3)
middle = rows[1].subgridspec(1, 3, wspace=.22)
bottom = rows[2].subgridspec(map_rows, COLUMNS+1, width_ratios=[1]*COLUMNS+[.06], wspace=.22, hspace=.3)

# D versus the action: mean over the ray angles, band from the smallest to the largest
ax = fig.add_subplot(top[0, :3])
for l, color in zip(lattices, colors):
    ax.fill_between(l['action'], l['low'], l['high'], color=color, alpha=.2, lw=0)
    ax.plot(l['action'], l['mean'], 'o-', color=color, ms=4, label=f'{l["current"]:g} A')
ax.set(xscale='log', yscale='log', xlabel=action_label, ylabel='distortion $D$')
ax.legend(title='octupoles: mean over the ray angles, band min - max', frameon=False, ncols=4)
amplitude = ax.secondary_xaxis('top', functions=(lambda I: np.sqrt(2*np.abs(I)), lambda r: r**2/2))
amplitude.set_xlabel(r'amplitude $\sqrt{2(I_x + I_y)}$ [$\sigma$]')
amplitude.set_xticks([.3, .5, 1, 2, 4, 6], ['0.3', '0.5', '1', '2', '4', '6'])

# ranking: at each amplitude the lattices from the lowest D (top) to the highest
ax = fig.add_subplot(top[0, 3])
for column, a in enumerate(AMPLITUDES):
    at = [l['mean'][np.argmin(abs(l['levels']-a))] for l in lattices]
    ranked = np.argsort(at)
    print(f'{a} sigma, lowest D first: ' + ', '.join(f'{lattices[q]["current"]:g} A ({at[q]:.1e})' for q in ranked))
    for place, q in enumerate(ranked):
        ax.add_patch(plt.Rectangle((column, place), .94, .92, color=colors[q]))
        ax.text(column+.47, place+.46, f'{lattices[q]["current"]:g}', ha='center', va='center',
                color='k' if q > len(lattices)/2 else 'w')
ax.set(xlim=(0, len(AMPLITUDES)), ylim=(len(lattices), 0), xticks=np.arange(len(AMPLITUDES))+.47,
       xticklabels=[f'{a} $\\sigma$' for a in AMPLITUDES], yticks=np.arange(len(lattices))+.46,
       yticklabels=range(1, len(lattices)+1), title='ranking [A], lowest D first')
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)

# D along three rays: the x axis, the diagonal, the y axis
for column, (angle, title) in enumerate([(0, r'$x$ axis ($J_y = 0$)'), (45, 'diagonal'), (90, r'$y$ axis ($J_x = 0$)')]):
    ax = fig.add_subplot(middle[0, column])
    for l, color in zip(lattices, colors):
        phi = l['result']['phi']
        ray = np.isclose(phi, phi[np.argmin(abs(phi-angle))])
        ax.plot(l['I'][ray], l['D'][ray], 'o-', color=color, ms=3, lw=1)
    ax.set(xscale='log', yscale='log', xlabel=action_label, ylabel='distortion $D$' if column == 0 else None,
           title=title)

# maps: log10 D of every torus over the amplitude plane; red crosses: tori that did not converge
values = np.log10(np.concatenate([l['D'] for l in lattices]))
low, high = np.nanpercentile(values, [1, 99.9])
for column, (l, color) in enumerate(zip(lattices, colors)):
    ax = fig.add_subplot(bottom[column//COLUMNS, column % COLUMNS])
    r, converged = l['result']['r'], l['result']['converged']
    dots = ax.scatter(*r[converged].T, c=np.log10(l['D'][converged]), s=16, cmap='magma', vmin=low, vmax=high, lw=0)
    ax.scatter(*r[~converged].T, s=14, marker='x', color='red', lw=1.2, zorder=3, label='not converged')
    ax.set(aspect='equal', xlabel=r'$r_x$ [$\sigma$]', xticks=[0, 3, 6], yticks=[0, 3, 6],
           ylabel=r'$r_y$ [$\sigma$]' if column % COLUMNS == 0 else None)
    ax.set_title(f'{l["current"]:g} A', color=color)
fig.colorbar(dots, cax=fig.add_subplot(bottom[:, -1]), label=r'$\log_{10} D$')
ax.legend(loc='upper right', frameon=False)
fig.suptitle(r'nonlinear distortion of the tori: $D = \sum_{|m_x| > 1\ \mathrm{or}\ |m_y| > 1} |c_m|^2\ /\ \sum_{|m_x|, |m_y| \leq 1} |c_m|^2$,'
             r'  $I_k = \frac{1}{2}\sum_m m_k |c_m|^2$', fontsize=13, y=.95)
fig.savefig(OUT, dpi=110, bbox_inches='tight')
print(OUT)
