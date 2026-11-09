"""Figures of the tori of launch_tori.

    march_spectra(result, path)          the spectra of the tori of one ray against the amplitude (PNG)
    march_spectra_slider(result, path)   two sliders, ray angle and amplitude: the march in amplitude along the ray
                                         and the march in angle at that amplitude (HTML)
    resonance_web(result, path, interpolated=False)
                                         x tori: the tori in phase space and the harmonics m Q mod 1; 4D tori: the tunes
                                         and the resonance lines, and where they are in the amplitude plane (PNG)
    resonance_lines(result, path)        4D tori: the resonances as lines with their name on them, coloured by their
                                         strength on one scale (PNG)
    tori_2d(result, path)                x tori: the tori in phase space, rebuilt from their spectra (PNG)
Every figure takes method='spectra' or 'tunes': how the resonances of the 4D tori are found (detect.py).
A resonance n.Q = p keeps one colour in all the figures of a result. Its resonant harmonics in plane a are m = e_a + k n:
their frequency differs from the linear one by k (n.Q - p), which vanishes on the resonance. Every torus is one marker
or one row (no interpolation), except with interpolated=True. Tori that did not converge are red crosses.
"""
import base64
import json
from math import gcd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, to_hex
from .detect import ORDER, ORDER_1D, crossed, detect, grid, linear_power, resonant

ACTIVE = 1e-12          # a harmonic is shown when it rises above this on some torus (units of r)
COLORS = [to_hex(plt.get_cmap('tab10')(k)) for k in (0, 1, 2, 3, 4, 5, 6, 8, 9)] + \
         [to_hex(v) for v in plt.get_cmap('Dark2').colors]
STROKE = [patheffects.withStroke(linewidth=4, foreground='w')]    # a white edge: the line reads on any colour map
DARK = '#14141f'      # the box behind the names of the resonance lines
MAGMA = json.dumps([[k/10, to_hex(plt.get_cmap('magma')(k/10))] for k in range(11)])   # Plotly has no magma
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


# ---------- small helpers ----------

def job(result):
    """The line and the time taken, for the titles."""
    return f'{result["line"]}, {len(result["r"])} tori in {float(result["seconds"]):.0f} s'


def unit(result, latex=True):
    """' [sigma]' for a line in beam sigmas, '' for the Hénon lines (their own coordinates)."""
    if str(result.get('unit', '')) != 'sigma':
        return ''
    return r' [$\sigma$]' if latex else ' [sigma]'


def rays(result):
    """{ray: its converged tori, by amplitude}."""
    out = {}
    for k in np.unique(result['ray'][result['ray'] >= 0]):
        i = np.flatnonzero((result['ray'] == k) & result['converged'])
        out[int(k)] = i[np.argsort(np.hypot(*result['r'][i].T))]
    return out


def planes_of(result, i):
    """The planes with an angle on the tori i: [0], [1] or [0, 1]."""
    return [a for a in range(2) if np.all(result['J'][i, a] > 0)]


def palette(keys):
    """A colour for every resonance (n_x, n_y, p), in the order given."""
    return {k: COLORS[j % len(COLORS)] for j, k in enumerate(dict.fromkeys(tuple(k[:3]) for k in keys))}


def label(nx, ny, p, planes=(0, 1), latex=True):
    """'5Qx + 7Qy = 3' (two planes) or 'Qx = 1/5' (one plane)."""
    if len(planes) == 1:
        q, xy = (nx, ny)[planes[0]], 'xy'[planes[0]]
        return f'$Q_{xy}$ = {p}/{q}' if latex else f'Q{xy} = {p}/{q}'
    terms = ''
    for k, xy in ((nx, 'x'), (ny, 'y')):
        if k:
            terms += (' - ' if k < 0 else ' + ' if terms else '-' if k < 0 else '')+(str(abs(k)) if abs(k) != 1 else '')
            terms += f'$Q_{xy}$' if latex else f'Q{xy}'
    return f'{terms} = {p}'


def tune_shift(result, i):
    """|Q(J) - Q(0)| of the tori i."""
    return np.hypot(*(result['Q'][i]-np.asarray(result['tunes0'])).T)


def tune_window(Q, margin=.08):
    """The (low, high) corners of the tunes Q, widened by `margin` of their span."""
    low, high = Q.min(axis=0), Q.max(axis=0)
    pad = np.maximum(high-low, 1e-6)*margin
    return low-pad, high+pad


def ring(ax, x, y, color, size=60):
    """Open rings around the tori of a resonance."""
    ax.scatter(x, y, s=size, facecolors='none', edgecolors=color, linewidths=1.5, zorder=4)


def unconverged(result, ax=None, tunes=False, size=36):
    """Red crosses on the tori that did not converge (in the amplitude plane, or at their tunes) and their legend entry
    (None if every torus converged). Their best approximation is kept in the data."""
    k = np.flatnonzero(~result['converged'])
    if tunes:
        k = k[np.all(np.isfinite(result['Q'][k]), axis=1)]
    if not len(k):
        return None
    xy = (result['Q'] if tunes else result['r'])[k]
    if ax is not None:
        ax.scatter(xy[:, 0], xy[:, 1], marker='x', color='red', s=size, linewidths=1.4, zorder=6)
    return plt.Line2D([], [], ls='', marker='x', ms=7, mew=1.4, color='red', label=unconverged_label(result, k))


def unconverged_label(result, k):
    """'5 not converged: best approximation kept, residual 1e-05 - 5e-05 (2 lost particles)'."""
    worst = result['residual'][k]
    finite = worst[np.isfinite(worst)]
    text = f'{len(k)} not converged'
    if len(finite):
        text += f': best approximation kept, residual {finite.min():.0e} - {finite.max():.0e}'
    if len(finite) < len(k):
        text += f' ({len(k)-len(finite)} lost particles)'
    return text


def grey(found, ax=None, r=None):
    """Grey rings on the tori of the resonances found but not resolved (with ax), and their one legend entry."""
    if ax is not None:
        k = np.unique(np.concatenate([found['tori'][w] for w in found['unresolved']] or [np.zeros(0, int)]))
        ring(ax, r[k, 0], r[k, 1], '.6', 40)
    return f'{len(found["unresolved"])} more {"found" if found["method"] == "spectra" else "crossed"}, ' \
           'not resolved (amplified noise)'


def legend_title(found):
    """How the resonances of the legend were found."""
    by = '' if found['method'] == 'spectra' else f' crossed by the tunes\n($|n_x| + |n_y| \\leq {ORDER}$),'
    return f'{"resonances" if found["shown"] else "no resonance"}{by} resolved in the spectra'


def legend_label(found, w, text):
    """'4Qx - 7Qy = -1  (24 tori, 3 resolved)'."""
    return f'{text}  ({len(found["tori"][w])} tori, {int(found["resolved"][w].sum())} resolved)'


def amplitude_grid(result, i, n=400):
    """Linear interpolation over the amplitude plane between the tori i: the grid X, Y and field(values of the tori i).
    A triangle with a corner outside i (an axis torus, one that did not converge) is left empty."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    r = result['r']
    triangles = Triangulation(r[:, 0], r[:, 1])
    inside = np.zeros(len(r), bool)
    inside[i] = True
    triangles.set_mask(~inside[triangles.triangles].all(axis=1))
    X, Y = np.meshgrid(np.linspace(0, r[:, 0].max(), n), np.linspace(0, r[:, 1].max(), n))

    def field(values):
        z = np.zeros(len(r))
        z[i] = values
        return LinearTriInterpolator(triangles, z)(X, Y)
    return X, Y, field


def curve(ax, X, Y, F, color, lw=2.2):
    """The curve F = 0 of an interpolated field (a resonance n.Q - p = 0); a thin one (grey) under, without edge."""
    if np.nanmin(F) < 0 < np.nanmax(F):
        lines = ax.contour(X, Y, F, levels=[0], colors=[color], linewidths=lw, zorder=4 if lw >= 2 else 3)
        if lw >= 2:
            lines.set(path_effects=STROKE)


# ---------- the spectra along a ray: march_spectra ----------

def signed_order(x, y):
    """+-(|m_x| + |m_y|), with the sign of the first non-zero component: m and -m on either side of 0."""
    return (abs(x)+abs(y))*(1 if x > 0 or (x == 0 and y > 0) else -1 if x or y else 0)


def columns(c, a):
    """The harmonics of z_a shown in the marches (above ACTIVE on some torus of c), by signed order, 0 in the middle and
    -m at the mirror place of m: their flat indices in c[t, a], (m_x, m_y), signed orders and x positions (the
    harmonics of one order spread evenly over one unit)."""
    _, (mx, my) = grid(c)
    flat = np.flatnonzero(np.abs(c[:, a]).max(axis=0).ravel() > ACTIVE)
    m = [(int(mx.flat[k]), int(my.flat[k])) for k in flat]

    def key(j):
        s = signed_order(*m[j])
        return (s, *m[j]) if s > 0 else (s, -m[j][0], -m[j][1]) if s < 0 else (0, 0, 0)
    ordering = sorted(range(len(m)), key=key)
    m = [m[j] for j in ordering]
    order = np.array([signed_order(*h) for h in m])
    x = np.zeros(len(order))
    for s in np.unique(order):
        k = np.flatnonzero(order == s)
        x[k] = s-.5+(np.arange(len(k))+.5)/len(k)
    return flat[ordering], m, order.tolist(), x


def crossings_on(result, i, method='spectra'):
    """(n_x, n_y, p, rho) of the resonances on the tori i of one ray. Two angles: the resonances of detect, each on its
    torus of the ray with the most peaked entries. One angle: the tunes crossing q Q = p, on the torus closest to it."""
    rho = np.hypot(*result['r'][i].T)
    planes = planes_of(result, i)
    out = []
    if len(planes) == 1:
        Q = result['Q'][i][:, planes]
        for w in crossed(result['Q'][i], planes):
            d = Q@np.array(w[:2])[planes]-w[2]
            k = np.flatnonzero(np.diff(np.sign(d)))
            if len(k):
                out.append((*w, float(rho[k[0] if abs(d[k[0]]) <= abs(d[k[0]+1]) else k[0]+1])))
        return sorted(out, key=lambda w: (abs(w[0])+abs(w[1]), w[3]))
    found = detect(result, method)
    where = {t: k for k, t in enumerate(i)}
    for w in found['shown']:
        mine = [b for b in found['tori'][w] if found['i'][b] in where]
        if mine:
            b = max(mine, key=lambda b: found['peaks'][b])
            out.append((*w, float(rho[where[found['i'][b]]])))
    return out


def march_spectra(result, path, ray=None, method='spectra'):
    """The spectra along one ray (default: the one crossing the most resonances, then the closest to 45 degrees): one
    panel per plane, the tori by amplitude (rows), the harmonics by signed order (columns); a resonance crossed is a
    horizontal line and its resonant harmonics are marked in its colour."""
    all_rays = rays(result)
    if ray is None:
        ray = max(all_rays, key=lambda k: (len(crossings_on(result, all_rays[k], method)),
                                           -abs(np.median(result['phi'][all_rays[k]])-45)))
    i = all_rays[ray]
    bad = np.flatnonzero((result['ray'] == ray) & ~result['converged'] & np.isfinite(result['residual']))
    i = np.concatenate([i, bad])                       # the tori that did not converge: their best approximation
    i = i[np.argsort(np.hypot(*result['r'][i].T))]
    rho = np.hypot(*result['r'][i].T)
    planes = planes_of(result, i)
    webs = crossings_on(result, i, method)
    owner = resonant(webs, grid(result['c'])[0])
    colors = palette(webs)
    c = result['c'][i]
    panels = [a for a in range(2) if len(columns(c, a)[0])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 5.5*len(panels)), layout='constrained', sharey=True,
                             squeeze=False)
    for ax, a in zip(axes[:, 0], panels):
        flat, m, order, x = columns(c, a)
        image = ax.pcolormesh(x, rho, np.log10(np.maximum(np.abs(c[:, a].reshape(len(c), -1)[:, flat]), 1e-17)),
                              cmap='magma', vmin=-15, vmax=.5, shading='nearest', rasterized=True)
        for w in webs:
            ax.axhline(w[3], color=colors[w[:3]], lw=1.8, path_effects=STROKE, label=label(*w[:3], planes))
        mine = [owner.get((a, *h)) for h in m]
        if len(planes) == 2:                            # one unit of x per signed order, resonant harmonics marked
            top = max(abs(min(order)), max(order))
            step = 1 if top <= 25 else 2
            ax.set_xticks(np.arange(-top//step*step, top+1, step))
            ax.set_xlim(-top-.5, top+.5)
            ax.tick_params(axis='x', labelsize=8)
            for k in range(-top, top):
                ax.axvline(k+.5, color='w', lw=.3, alpha=.35)
            for k, w in enumerate(mine):
                if w:
                    ax.plot(x[k], 1.0, 'v', color=colors[w], ms=7, transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_xlabel(f'harmonics $m = (m_x, m_y)$ of $z_{"xy"[a]}$ by signed order '
                          r'$\pm(|m_x| + |m_y|)$, $-m$ mirrors $m$ (markers: resonant harmonics)')
        else:                                           # one plane: every harmonic m labelled
            many = len(m) > 60
            ax.set_xticks(x, [f'{h[planes[0]]}' for h in m], rotation=90 if many else 0, fontsize=4 if many else 7)
            for tick, w in zip(ax.get_xticklabels(), mine):
                if w:
                    tick.set_color(colors[w])
                    tick.set_fontweight('bold')
            ax.set_xlabel(f'harmonic $m$ of $z_{"xy"[a]}$')
        ax.set_ylabel(r'amplitude $\sqrt{r_x^2+r_y^2}$'+unit(result))
        if len(bad):
            ax.plot(np.zeros(len(bad)), np.hypot(*result['r'][bad].T), 'x', color='red', ms=9, mew=2,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
    fig.colorbar(image, ax=axes[:, 0], label=r'$\log_{10}|c_m|$'+unit(result), shrink=.6)
    if len(bad):
        axes[0, 0].plot([], [], 'x', color='red', ms=7, mew=1.4, label=unconverged_label(result, bad))
    if webs or len(bad):
        axes[0, 0].legend(loc='upper left', fontsize=8, framealpha=.9, title='resonances crossed' if webs else None)
    phi = float(np.median(result['phi'][i]))
    fig.suptitle(f'{job(result)}: spectra of the {len(i)} tori on the ray $\\phi$ = {phi:.0f}$^\\circ$')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def slots(result, allowed):
    """For the two sliders of the HTML pages, among the tori `allowed`: the rays sorted by angle, their angles phi, the
    amplitudes rho and slot[ray, amplitude] = the torus there (-1: none)."""
    ray, phi = result['ray'][allowed], result['phi'][allowed]
    keys = sorted({int(k) for k in ray if k >= 0}, key=lambda k: np.median(phi[ray == k]))
    rho = np.hypot(*result['r'][allowed].T)
    levels = np.unique(np.round(rho, 6))
    slot = -np.ones((len(keys), len(levels)), int)
    for t, k, v in zip(allowed, ray, rho):
        if k >= 0:
            slot[keys.index(k), np.argmin(np.abs(levels-v))] = t
    return keys, [float(np.median(phi[ray == k])) for k in keys], levels, slot


def march_spectra_slider(result, path, method='spectra'):
    """One HTML page, two sliders (ray angle phi, amplitude r) select one torus. Per plane: the march in amplitude
    along the ray and the march in angle at the amplitude, with the resonances crossed as horizontal lines and the
    selected torus dashed. The spectra are packed as bytes: log10 |c| in tenths of a decade from -17."""
    ok = np.flatnonzero(result['converged'])
    keys, phis, levels, slot = slots(result, ok)
    by_ray = rays(result)
    planes = sorted({a for a in range(2) if np.any(result['J'][ok, a] > 0)})
    # the resonances along every ray, and around every arc (4D: each on its torus with the most peaked entries)
    webs_ray = [crossings_on(result, by_ray[k], method) for k in keys]
    webs_arc = [[] for _ in levels]
    shown = []
    if len(planes) == 2:
        found = detect(result, method)
        shown = found['shown']
        place = {int(t): (j, v) for (j, v), t in np.ndenumerate(slot) if t >= 0}
        for w in shown:
            best = {}
            for b in found['tori'][w]:
                j, v = place[int(found['i'][b])]
                if v not in best or found['peaks'][b] > found['peaks'][best[v][1]]:
                    best[v] = (j, b)
            for v, (j, b) in best.items():
                webs_arc[v].append((*w, phis[j]))
    others = [w[:3] for ws in webs_ray for w in ws if w[:3] not in shown]
    every = list(dict.fromkeys(shown+sorted(others, key=lambda w: (abs(w[0])+abs(w[1]), w))))
    colors, index = palette(every), {w: j for j, w in enumerate(every)}
    owner = resonant(every, grid(result['c'])[0])
    # the spectra of every plane, packed
    c, used = result['c'], slot[slot >= 0]
    panels = []
    for a in planes:
        flat, m, order, x = columns(c[used], a)
        size = np.log10(np.maximum(np.abs(c[:, a].reshape(len(c), -1)[:, flat]), 1e-17))
        packed = np.zeros((len(c), len(flat)), np.uint8)
        packed[used] = np.clip(np.round((size[used]+17)*10), 0, 254).astype(np.uint8)
        panels.append(dict(plane='xy'[a], labels=[f'{h[0]},{h[1]}' if len(planes) == 2 else f'{h[a]}' for h in m],
                           order=order if len(planes) == 2 else None, x=np.round(x, 4).tolist(),
                           owner=[index.get(owner.get((a, *h))) for h in m],
                           bytes=base64.b64encode(packed.tobytes()).decode()))
    data = dict(resonances=[dict(label=label(*w, planes if len(planes) == 1 else [0, 1], latex=False), color=colors[w])
                            for w in every],
                phi=np.round(phis, 1).tolist(), rho=np.round(levels, 4).tolist(), slot=slot.tolist(), tori=len(c),
                panels=panels, webs_ray=[[[index[w[:3]], round(w[3], 4)] for w in ws] for ws in webs_ray],
                webs_arc=[[[index[w[:3]], round(w[3], 2)] for w in ws] for ws in webs_arc])
    page = PAGE.replace('__TITLE__', f'{job(result)}: spectra of the tori, in amplitude and in angle') \
               .replace('__DATA__', json.dumps(data, separators=(',', ':'))) \
               .replace('__UNIT__', unit(result, latex=False)).replace('__MAGMA__', MAGMA)
    Path(path).write_text(page, encoding='utf-8')


PAGE = '''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tori spectra</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{font-family:sans-serif;margin:16px;background:#fff;color:#222} .bar{display:flex;gap:12px;
align-items:center;margin:4px 0} input{width:360px} .plot{width:100%;height:40vh}</style></head><body>
<h3>__TITLE__</h3>
<div class="bar">ray angle &phi; = atan(r<sub>y</sub>/r<sub>x</sub>) <input id="ray" type="range" min="0" step="1">
<b id="phi"></b></div>
<div class="bar">amplitude r = (r<sub>x</sub><sup>2</sup> + r<sub>y</sub><sup>2</sup>)<sup>1/2</sup>
<input id="level" type="range" min="0" step="1"><b id="rho"></b></div>
<div id="note"></div>
<div id="x_r" class="plot"></div><div id="x_phi" class="plot"></div>
<div id="y_r" class="plot"></div><div id="y_phi" class="plot"></div>
<script>
const DATA = __DATA__, RES = DATA.resonances, NONE = -1;
const ray = document.getElementById('ray'), level = document.getElementById('level');
ray.max = DATA.phi.length-1; ray.value = Math.floor(DATA.phi.length/2);
level.max = DATA.rho.length-1; level.value = DATA.rho.length-1;
for (const p of DATA.panels) {           // the packed spectra: one row of bytes per torus
  const raw = atob(p.bytes), n = p.labels.length;
  p.rows = new Uint8Array(raw.length);
  for (let k = 0; k < raw.length; k++) p.rows[k] = raw.charCodeAt(k);
  p.row = t => t === NONE ? Array(n).fill(null) : Array.from(p.rows.subarray(t*n, (t+1)*n), b => b/10 - 17);
  const index = p.labels.map((l, i) => i);
  p.mine = index.filter(i => p.owner[i] !== null);
  if (p.order) {             // one unit of x per signed order: 0 in the middle, +-1, +-2, ... evenly spaced
    const top = Math.max(...p.order.map(Math.abs));
    p.axis = {tickmode: 'linear', tick0: 0, dtick: top <= 25 ? 1 : 2, tickfont: {size: 10}, tickangle: 0,
              range: [-top - 0.5, top + 0.5], zeroline: false,
              title: 'harmonics m of z_' + p.plane + ' by signed order \\u00b1(|m_x|+|m_y|), -m mirrors m (triangles: resonant)'};
  } else {
    p.axis = {tickvals: p.x, tickangle: -90, tickfont: {size: 7}, title: 'harmonic m of z_' + p.plane,
              ticktext: p.labels.map((l, i) => p.owner[i] === null ? l :
                                     '<span style="color:' + RES[p.owner[i]].color + '"><b>' + l + '</b></span>')};
  }
}
function heatmap(p, rows, y, ytitle, webs, here, top, title) {
  const lines = webs.map(([j, at]) => ({type: 'line', xref: 'x domain', x0: 0, x1: 1, y0: at, y1: at,
                                        line: {width: 2, color: RES[j].color}}))
    .concat([{type: 'line', xref: 'x domain', x0: 0, x1: 1, y0: here, y1: here, line: {width: 1.5, color: '#fff', dash: 'dash'}}]);
  const keys = webs.map(([j]) => ({type: 'scatter', mode: 'lines', x: [null], y: [null], name: RES[j].label,
                                   line: {width: 3, color: RES[j].color}}));
  const marks = {type: 'scatter', mode: 'markers', x: p.mine.map(i => p.x[i]), y: p.mine.map(() => top), showlegend: false,
                 marker: {symbol: 'triangle-down', size: 9, color: p.mine.map(i => RES[p.owner[i]].color)},
                 text: p.mine.map(i => 'm = ' + p.labels[i] + ': ' + RES[p.owner[i]].label), hoverinfo: 'text',
                 cliponaxis: false};
  return [[{type: 'heatmap', z: rows, x: p.x, y: y, colorscale: __MAGMA__, zmin: -15, zmax: 0.5, showlegend: false,
            colorbar: {title: 'log|c|', len: 0.9}, text: rows.map(() => p.labels),
            hovertemplate: 'm=%{text}<br>' + ytitle + '=%{y}<br>log|c|=%{z}<extra></extra>'}].concat(keys, [marks]),
          {margin: {l: 60, r: 20, t: 30, b: 60}, title: {text: title, font: {size: 13}}, xaxis: p.axis,
           yaxis: {title: ytitle}, shapes: lines,
           legend: {x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)', title: {text: 'resonances crossed'}}}];
}
function draw() {
  const j = +ray.value, v = +level.value, t = DATA.slot[j][v];
  document.getElementById('phi').textContent = DATA.phi[j] + '\\u00b0';
  document.getElementById('rho').textContent = DATA.rho[v] + '__UNIT__';
  document.getElementById('note').textContent = t === NONE ? '✖ no converged torus at this angle and amplitude' : '';
  document.getElementById('note').style.color = 'red';
  for (const plane of ['x', 'y']) {
    const p = DATA.panels.find(q => q.plane === plane);
    for (const kind of ['r', 'phi']) {
      const div = document.getElementById(plane + '_' + kind);
      if (!p || (kind === 'phi' && DATA.phi.length < 2)) { Plotly.purge(div); div.style.display = 'none'; continue; }
      div.style.display = '';     // the plot div holds only Plotly's elements: never write text into it
      const [traces, layout] = kind === 'r' ?
        heatmap(p, DATA.slot[j].map(s => p.row(s)), DATA.rho, 'r__UNIT__', DATA.webs_ray[j], DATA.rho[v],
                DATA.rho[DATA.rho.length-1], 'z_' + plane + ': march in amplitude at \\u03c6 = ' + DATA.phi[j] + '\\u00b0') :
        heatmap(p, DATA.slot.map(s => p.row(s[v])), DATA.phi, '\\u03c6 [deg]', DATA.webs_arc[v], DATA.phi[j],
                DATA.phi[DATA.phi.length-1], 'z_' + plane + ': march in angle at r = ' + DATA.rho[v] + '__UNIT__');
      Plotly.react(div, traces, layout, {responsive: true, displayModeBar: false});
    }
  }
}
let pending = false;            // one draw per frame, with the last slider positions: dragging never queues draws
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; draw(); }); } };
ray.addEventListener('input', schedule);
level.addEventListener('input', schedule);
draw();
</script></body></html>
'''


# ---------- the resonances met by the tori: resonance_web ----------

def resonance_web(result, path, method='spectra', interpolated=False):
    """x tori (one angle): resonant tori in phase space and the harmonics m Q mod 1. 4D tori: the tunes of the tori
    and the lines n.Q = p of the resonances found (left), where they are in the amplitude plane (right), both
    coloured by the tune shift; interpolated: the amplitude plane filled between the tori, the resonances as curves."""
    ok = result['converged']
    if sum(np.any(result['J'][ok, a] > 0) for a in range(2)) == 1:
        return _web_1d(result, path)
    found = detect(result, method)
    i, met = found['i'], found['shown']
    Q, r = result['Q'][i], result['r'][i]
    shift = tune_shift(result, i)
    colors = palette(met)
    low, high = tune_window(Q)
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')
    fig.suptitle(job(result)+(', interpolated between the tori in the amplitude plane' if interpolated else ''))
    norm = plt.Normalize(0, shift.max())
    left.scatter(Q[:, 0], Q[:, 1], c=shift, cmap='viridis', norm=norm, s=7, zorder=3)
    if interpolated:
        X, Y, field = amplitude_grid(result, i)
        image = right.pcolormesh(X, Y, field(shift), cmap='viridis', norm=norm, shading='auto', rasterized=True)
    else:
        image = right.scatter(r[:, 0], r[:, 1], c=shift, cmap='viridis', norm=norm, s=22, linewidths=0)
    if interpolated:                                    # the curves n.Q = p: from the tori not converged too
        X, Y, over_all = amplitude_grid(result, found['every'])
    Q_all = result['Q'][found['every']]
    qx = np.array([low[0], high[0]])
    for w in met:                                       # every resonance: its line, its tori ringed (or its curve)
        nx, ny, p = w
        style = dict(color=colors[w], lw=2.2, path_effects=STROKE, label=legend_label(found, w, label(*w)))
        if ny:
            left.plot(qx, (p-nx*qx)/ny, **style)
        else:
            left.axvline(p/nx, **style)
        k = found['tori'][w]
        ring(left, Q[k, 0], Q[k, 1], colors[w], 40)
        if interpolated:
            curve(right, X, Y, over_all(Q_all@[nx, ny]-p), colors[w])
        else:
            ring(right, r[k, 0], r[k, 1], colors[w])
    left.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='$Q_x$', ylabel='$Q_y$',
             title='tunes of the tori and the resonances found')
    left.ticklabel_format(useOffset=False)
    right.set(aspect='equal', xlim=(0, None), ylim=(0, None), xlabel=r'$r_x$'+unit(result), ylabel=r'$r_y$'+unit(result),
              title='where in amplitude: $n.Q(J) = p$')
    fig.colorbar(image, ax=[left, right], shrink=.6, location='bottom', format='%.1e',
                 label='tune shift $|Q(J) - Q(0)|$')
    unconverged(result, left, True, 20)
    cross = unconverged(result, right)
    handles, texts = left.get_legend_handles_labels()
    if cross:
        handles.append(cross)
        texts.append(cross.get_label())
    if found['unresolved']:                             # the others: grey
        if interpolated:
            for w in found['unresolved']:
                curve(right, X, Y, over_all(Q_all@w[:2]-w[2]), '.6', .8)
            handles.append(plt.Line2D([], [], color='.6', lw=.8))
            texts.append(grey(found))
        else:
            handles.append(plt.Line2D([], [], ls='', marker='o', ms=8, mfc='none', mew=1.5, color='.6'))
            texts.append(grey(found, right, r))
    fig.legend(handles, texts, loc='outside right upper', fontsize=9, title=legend_title(found))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def line_power(result, i, w):
    """The strength of the resonance w on the tori i: the power of its nonlinear harmonics (|m_x| > 1 or |m_y| > 1) over
    the linear power (all the harmonics |m_x|, |m_y| <= 1 of both planes)."""
    c = result['c'][i]
    (Bx, By), _ = grid(c)
    return sum(np.abs(c[:, a, Bx+x, By+y])**2 for a, x, y in resonant([w], (Bx, By)) if max(abs(x), abs(y)) > 1)/linear_power(c)


def resonance_lines(result, path, method='spectra'):
    """The resonances of the 4D tori as lines with their name written on them, in the tune plane and in the amplitude
    plane. Colour: the relative power of the harmonics +- n on one scale for all; on the amplitude plane it follows the line, from its
    nearest torus. The tori that did not converge count with their best approximation."""
    from contourpy import contour_generator
    found = detect(result, method)
    i, every, met = found['i'], found['every'], found['shown']
    Q, r, shift = result['Q'][i], result['r'][i], tune_shift(result, i)
    X, Y, field = amplitude_grid(result, every)
    failed, failed_line = result.get('failed', np.zeros(0, int)), result.get('failed_line', np.zeros((0, 3), int))
    # the tori of a line: the resolved ones and the not converged ones that name it
    tori = {w: np.concatenate([i[found['tori'][w][found['resolved'][w]]], failed[np.all(failed_line == w, axis=1)]]) for w in met}
    values = {w: line_power(result, k, w) for w, k in tori.items()}
    known = np.concatenate([v for v in values.values()])
    norm = LogNorm(10.**np.floor(np.log10(known.min())), 10.**np.ceil(np.log10(known.max())))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('lines', plt.get_cmap('magma')(np.linspace(.15, .92, 256)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.4), layout='constrained')
    fig.suptitle(job(result))
    low, high = tune_window(Q)
    dots = left.scatter(Q[:, 0], Q[:, 1], c=shift, cmap='viridis', s=7, zorder=2)
    right.scatter(r[:, 0], r[:, 1], c=shift, cmap='viridis', s=16, linewidths=0, zorder=2)
    for w in met:
        S, at = values[w], result['r'][tori[w]]
        # the tune plane: one colour, the strongest value of the line
        x = np.linspace(low[0], high[0], 400)
        y = (w[2]-w[0]*x)/w[1]
        x, y = x[(y > low[1]) & (y < high[1])], y[(y > low[1]) & (y < high[1])]
        left.plot(x, y, color=cmap(norm(S.max())), lw=2.8, path_effects=STROKE, zorder=3)
        j = int(.7*len(x))
        name(left, x[j], y[j], np.degrees(np.arctan2(y[j+1]-y[j], x[j+1]-x[j])), label(*w))
        # the amplitude plane: the colour along the line, from the nearest torus
        pieces = contour_generator(x=X[0], y=Y[:, 0], z=field(result['Q'][every]@w[:2]-w[2])).lines(0)
        pieces = [q for q in pieces if len(q) > 1]
        for q in pieces:
            nearest = np.linalg.norm(((q[1:]+q[:-1])/2)[:, None]-at[None], axis=2).argmin(axis=1)
            right.plot(*q.T, color='w', lw=5.5, zorder=2.9, solid_capstyle='round')     # a white edge
            right.add_collection(LineCollection(np.stack([q[:-1], q[1:]], axis=1), colors=cmap(norm(S[nearest])), linewidths=3.2,
                                                capstyle='round', zorder=3))
        right.scatter(*at.T, s=46, facecolors='none', edgecolors=cmap(norm(S)), linewidths=1.3, zorder=4)
        q = max(pieces, key=len)
        j = int(.6*(len(q)-1))
        name(right, *q[j], np.degrees(np.arctan2(*(q[min(j+6, len(q)-1)]-q[j])[::-1])), label(*w))
    unconverged(result, left, True, 20)
    unconverged(result, right)
    left.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='$Q_x$', ylabel='$Q_y$')
    left.ticklabel_format(useOffset=False)
    top = result['r'].max()*1.03
    right.set(aspect='equal', xlim=(0, top), ylim=(0, top), xlabel=r'$r_x$'+unit(result), ylabel=r'$r_y$'+unit(result))
    if not result['converged'].all():
        right.legend(handles=[plt.Line2D([], [], ls='', marker='x', ms=7, mew=1.4, color='red', label='not converged')],
                     loc='upper right', fontsize=9, frameon=False)
    bar = fig.colorbar(dots, ax=left, location='bottom', shrink=.7, format='%.1e', label='tune shift $|Q(J) - Q(0)|$')
    bar.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=right, location='bottom', shrink=.7,
                 label=r'$\dfrac{\sum\ |c^{\,x}_{m_x,m_y}|^2 + |c^{\,y}_{m_x,m_y}|^2\quad (|m_x| > 1\ \mathrm{or}\ |m_y| > 1)}{\sum\ |c^{\,x}_{m_x,m_y}|^2 + |c^{\,y}_{m_x,m_y}|^2\quad (|m_x|, |m_y| \leq 1)}$'
                       '\n' r'top: the harmonics of the line, $(m_x - 1, m_y)$ for $c^{\,x}$ and $(m_x, m_y - 1)$ for $c^{\,y}$ equal to $k\,(n_x, n_y)$')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def tori_2d(result, path):
    """x tori (one angle): every torus in phase space, x = Re z and p = -Im z with z = sum_m c[m] e^{i m theta}, coloured
    by its amplitude; the tori that did not converge in red, with their best approximation."""
    keep = np.isfinite(result['residual'])
    (Bx, By), _ = grid(result['c'])
    z = result['c'][keep, 0, :, By]@np.exp(1j*np.outer(np.arange(-Bx, Bx+1), 2*np.pi*np.arange(513)/512))
    r = result['r'][keep, 0]
    norm = plt.Normalize(0, r.max())
    colors = plt.get_cmap('viridis')(norm(r))
    colors[~result['converged'][keep]] = (1, 0, 0, 1)
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    ax.add_collection(LineCollection(np.stack([z.real, -z.imag], axis=-1), colors=colors, linewidths=.5))
    ax.autoscale()
    ax.set(aspect='equal', xlabel=f'$x${unit(result)}', ylabel=f'$p_x${unit(result)}', title=job(result))
    fig.colorbar(plt.cm.ScalarMappable(norm, 'viridis'), ax=ax, label=f'amplitude $r${unit(result)}')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def name(ax, x, y, angle, text):
    """The name of a resonance written on its line, above everything else."""
    ax.text(x, y, text, fontsize=9, color='w', rotation=angle if abs(angle) <= 90 else angle-180*np.sign(angle), rotation_mode='anchor',
            transform_rotates_text=True, ha='center', va='center', zorder=7,
            bbox=dict(boxstyle='round,pad=.15', fc=DARK, ec='none', alpha=.85))


def strength(c, n, roundoff=1e-14):
    """|c_m| at |m| = n of a spectrum c (harmonics -B ... B) beyond what it holds: a straight line through log10 |c_m|
    against |m| where it decays (before round-off, and before a harmonic raised by a nearby resonance, |m| > 10)."""
    B = len(c)//2
    order = np.arange(1, B+1)
    size = np.log10([max(abs(c[B+m]), abs(c[B-m]), 1e-300) for m in order])
    lowest = np.minimum.accumulate(np.where(order < 10, np.inf, size))
    last = max(3, int(np.argmax((size < np.log10(roundoff)) | (size > lowest+.5))) or B)
    a, b = np.polyfit(order[last//3:last], size[last//3:last], 1)
    return 10**(a*n+b)


def _web_1d(result, path):
    """x tori (or y tori): the resonant tori Q = p/q in phase space (left) and the harmonics m Q mod 1 against the
    action with their crossings (right), coloured by the strength of the resonance."""
    ray = max(rays(result).values(), key=len)
    a = planes_of(result, ray)[0]
    (Bx, By), _ = grid(result['c'])
    B = (Bx, By)[a]
    I, Q = result['J'][ray, a], result['Q'][ray, a]
    c = result['c'][ray, a][:, :, By] if a == 0 else result['c'][ray, a][:, Bx, :]    # (tori, 2B + 1)
    m = np.arange(-B, B+1)
    E = np.exp(1j*np.outer(m, 2*np.pi*np.arange(513)/512))

    def coefficients(action):                           # the spectrum at any action, between two tori
        k = np.clip(np.searchsorted(I, action), 1, len(I)-1)
        w = (action-I[k-1])/(I[k]-I[k-1])
        return (1-w)*c[k-1]+w*c[k]

    found = []                                          # (p, q, action, strength) of every Q = p/q crossed
    for q in range(1, min(2*B, ORDER_1D)+1):
        for p in range(int(np.ceil(q*Q.min())), int(np.floor(q*Q.max()))+1):
            if gcd(p, q) == 1:
                at = float(np.interp(p/q, *(Q, I) if Q[-1] > Q[0] else (Q[::-1], I[::-1])))
                found.append((p, q, at, strength(coefficients(at), q, max(10*result['noise'], 1e-15))))
    cmap, shade = plt.get_cmap('plasma_r'), LogNorm(1e-60, 1e-8)
    x, p_ = ('x', 'p_x') if a == 0 else ('y', 'p_y')
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.8), layout='constrained', gridspec_kw={'width_ratios': [1, 1.1]})
    fig.suptitle(job(result))
    for k in np.linspace(0, len(I)-1, 20).astype(int):
        z = c[k]@E
        left.plot(z.real, -z.imag, color='.8', lw=.6)
    for p, q, at, s in sorted(found, key=lambda f: -f[1]):
        z = coefficients(at)@E
        left.plot(z.real, -z.imag, color=cmap(shade(s)), lw=55/q)
    for n, (p, q, at, s) in enumerate(sorted([f for f in found if f[1] < 50], key=lambda f: f[2])):
        z = coefficients(at)@E
        j = np.argmin(np.abs(np.angle(np.exp(1j*np.radians(125-22*n))*np.conj(z.real-1j*z.imag))))
        left.text(z[j].real, -z[j].imag, f'{p}/{q}', ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85))
    left.set(aspect='equal', xlabel=f'${x}$', ylabel=f'${p_}$', title='resonant tori $Q = p/q$')
    segments, shades = [], []
    for k in m:
        nu = (k*Q) % 1
        cuts = np.flatnonzero(np.abs(np.diff(nu)) > .5)+1
        for u, v in zip(np.split(nu, cuts), np.split(I, cuts)):
            segments.append(np.c_[u, v])
            shades.append(abs(k))
    right.add_collection(LineCollection(segments, array=np.array(shades), cmap='Greys_r',
                                        norm=plt.Normalize(-10, 1.25*B), lw=.6))
    for p, q, at, s in sorted(found, key=lambda f: -f[1]):
        residues, counts = np.unique((m*p) % q, return_counts=True)
        k = residues[counts > 1]
        nu = np.r_[k, q*np.ones(np.sum(k == 0), int)]/q
        right.scatter(nu, np.full(len(nu), at), s=900/q, color=cmap(shade(s)), edgecolor='none', zorder=3)
    for p, q, at, s in found:
        if q < 50:
            right.annotate(f'{p}/{q}', (1, at), xytext=(4, 0), textcoords='offset points', va='center', fontsize=9,
                           annotation_clip=False)
    right.set(xlim=(0, 1), ylim=(0, I.max()), xlabel=f'frequency in the base band,  $mQ_{x}$ mod 1',
              ylabel=f'action $J_{x}$', title=f'harmonics $|m| \\leq {B}$ and their crossings')
    colorbar = fig.colorbar(plt.cm.ScalarMappable(shade, cmap), ax=right, pad=.06, shrink=.6, extend='min')
    colorbar.set_label('resonance strength $|c_m|$, $|m| = q$')
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
