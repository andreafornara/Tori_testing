"""Two views of the 4D tori (two angles) and of the resonances they cross.

    spectra_2d(result, path)        HTML, sliders over the rays and the tori: the tune plane (all the tori, the selected
                                    one, the resonance lines) and the spectra of z_x and z_y of the selected torus as
                                    images over (m_x, m_y), relative to the linear power; resonant harmonics circled
    resonance_maps(result, path, interpolated=False)
                                    PNG, maps over the amplitude plane (r_x, r_y): all the resonances together, then per
                                    resonance its distance n.Q(J) - p and the power of its resonant harmonics
    tori_slider(result, path)       HTML, sliders over the ray angle and the amplitude: the selected torus itself,
                                    rebuilt from its spectrum, in (x, px), (y, py), (x, y) and (px, py)

Relative power of a harmonic: |c_m|^2 / the linear power (all the harmonics |m_x|, |m_y| <= 1, both planes). Resonant power of n.Q = p:
the sum of |c_m|^2 over m = e_a + k n (both planes, 0 < |k| <= RESONANT_K) over the linear power. Where it is below
RESOLVED x the noise power (detect.power) the torus is faded and crossed: it can be amplified noise there.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .detect import RESOLVED, RESONANT_K, detect, grid, linear_power, power, resonant
from matplotlib.colors import to_hex
from .plots import (ACTIVE, MAGMA, amplitude_grid, curve, grey, job, label, legend_label, legend_title, palette,
                    planes_of, rays, ring, slots, tune_shift, tune_window, unconverged, unit)

MAPS = 8                # resonances with a row of their own in resonance_maps (resolved ones, lowest order first)
MARKER = 22             # marker area of one torus in the amplitude-plane maps [pt^2]
SCRIPT = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
STYLE = ('<style>body{font-family:sans-serif;margin:16px;background:#fff;color:#222} .bar{display:flex;gap:12px;'
         'align-items:center;margin:4px 0} input{width:320px} .row{display:flex;flex-wrap:wrap} '
         '.plot{width:min(32vw,600px);height:min(32vw,560px);min-width:340px;min-height:340px}</style>')


def spectra_2d(result, path, method='spectra'):
    """The tune plane and the 2D spectra of the selected torus (see the module docstring)."""
    c, Q = result['c'], result['Q']
    (Bx, By), (mx, my) = grid(c)
    found = detect(result, method)
    i, met = found['i'], found['shown']
    colors = palette(met)
    low, high = tune_window(Q[i])
    shift = tune_shift(result, i)
    data = dict(B=[Bx, By], range=[low.tolist(), high.tolist()], smax=float(shift.max()),
                foot=[[round(float(q[0]), 8), round(float(q[1]), 8), float(f'{s:.3g}')] for q, s in zip(Q[i], shift)],
                resonances=[dict(label=legend_label(found, w, label(*w, latex=False)), color=colors[w], n=list(w[:2]),
                                 p=w[2]) for w in met],
                K=RESONANT_K, rays=[])
    bad = np.flatnonzero(~result['converged'] & np.isfinite(result['residual']) & np.all(np.isfinite(Q), axis=1))
    data['bad'] = [[round(float(q[0]), 8), round(float(q[1]), 8)] for q in Q[bad]]
    linear = linear_power(c)
    for k, j in rays(result).items():
        if len(planes_of(result, j)) != 2:
            continue
        j = np.concatenate([j, bad[result['ray'][bad] == k]])          # with their best approximation, flagged
        j = j[np.argsort(np.hypot(*result['r'][j].T))]
        tori = []
        for t in j:
            planes = []
            for a in range(2):
                keep = np.abs(c[t, a]) > ACTIVE
                power = np.log10(np.abs(c[t, a][keep])**2/linear[t])
                planes.append([[int(x), int(y), round(float(s), 1)] for x, y, s in zip(mx[keep], my[keep], power)])
            tori.append(dict(r=[round(float(v), 4) for v in result['r'][t]], Q=[round(float(q), 8) for q in Q[t]],
                             z=planes, bad=0 if result['converged'][t] else float(f"{result['residual'][t]:.1e}")))
        data['rays'].append(dict(phi=round(float(np.median(result['phi'][j])), 1), tori=tori))
    page = SPECTRA.replace('__TITLE__', f'{job(result)}: spectra of the 4D tori') \
                  .replace('__DATA__', json.dumps(data, separators=(',', ':'))).replace('__UNIT__', unit(result, latex=False)).replace('__MAGMA__', MAGMA)
    Path(path).write_text(page, encoding='utf-8')


SPECTRA = '''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>4D spectra</title>''' + SCRIPT + STYLE + '''</head><body>
<h3>__TITLE__</h3>
<div class="bar">ray angle &phi; = atan(r<sub>y</sub>/r<sub>x</sub>) <input id="ray" type="range" min="0" step="1">
<b id="phi"></b></div>
<div class="bar">torus on the ray <input id="torus" type="range" min="0" step="1"><b id="where"></b></div>
<div class="row"><div id="tunes" class="plot"></div><div id="x" class="plot"></div><div id="y" class="plot"></div></div>
<script>
const DATA = __DATA__, RES = DATA.resonances, B = DATA.B;
const ray = document.getElementById('ray'), torus = document.getElementById('torus');
ray.max = DATA.rays.length-1; ray.value = Math.floor(DATA.rays.length/2);
const [low, high] = DATA.range;
const footprint = {type: 'scatter', mode: 'markers', x: DATA.foot.map(f => f[0]), y: DATA.foot.map(f => f[1]),
  marker: {size: 4, color: DATA.foot.map(f => f[2]), colorscale: 'Viridis', cmin: 0, cmax: DATA.smax,
           colorbar: {title: 'tune shift', len: 0.6, y: 0.3, exponentformat: 'e'}},
  hoverinfo: 'skip', showlegend: false};
const crosses = {type: 'scatter', mode: 'markers', x: DATA.bad.map(q => q[0]), y: DATA.bad.map(q => q[1]),
  name: 'not converged: best approximation', marker: {symbol: 'x', size: 8, color: 'red'}, hoverinfo: 'name'};
const lines = RES.map(R => {
  const [nx, ny] = R.n;
  const x = ny ? [low[0], high[0]] : [R.p/nx, R.p/nx], y = ny ? x.map(q => (R.p - nx*q)/ny) : [low[1], high[1]];
  return {type: 'scatter', mode: 'lines', x: x, y: y, name: R.label, legendgroup: R.label,
          line: {color: R.color, width: 2.5}, hoverinfo: 'name'};
});
function circles(a) {          // the resonant harmonics e_a + k n of every resonance, in its colour
  return RES.map(R => {
    const x = [], y = [];
    for (let k = -DATA.K; k <= DATA.K; k++) {
      const m = [(a === 0) + k*R.n[0], (a === 1) + k*R.n[1]];
      if (k && Math.abs(m[0]) <= B[0] && Math.abs(m[1]) <= B[1]) { x.push(m[0]); y.push(m[1]); }
    }
    return {type: 'scatter', mode: 'markers', x: x, y: y, name: R.label, legendgroup: R.label, showlegend: false,
            marker: {symbol: 'circle-open', size: 11, line: {width: 2.5, color: R.color}}, hoverinfo: 'name'};
  });
}
function draw() {
  const Rr = DATA.rays[+ray.value];
  torus.max = Rr.tori.length-1;
  if (+torus.value > Rr.tori.length-1) torus.value = Rr.tori.length-1;
  const T = Rr.tori[+torus.value];
  document.getElementById('phi').textContent = Rr.phi + '\\u00b0';
  document.getElementById('where').textContent = 'r = (' + T.r[0] + ', ' + T.r[1] + ')__UNIT__,  Q = (' +
      T.Q[0].toFixed(6) + ', ' + T.Q[1].toFixed(6) + ')' +
      (T.bad ? '   NOT CONVERGED: best approximation, residual ' + T.bad.toExponential(0) : '');
  document.getElementById('where').style.color = T.bad ? 'red' : '';
  const here = {type: 'scatter', mode: 'markers', x: [T.Q[0]], y: [T.Q[1]], name: 'this torus',
                marker: {symbol: 'star', size: 18, color: '#000', line: {width: 1.5, color: '#fff'}}};
  Plotly.react('tunes', [footprint].concat(DATA.bad.length ? [crosses] : [], lines, [here]),
    {title: {text: 'tune plane', font: {size: 13}}, margin: {l: 70, r: 20, t: 40, b: 50},
     xaxis: {title: 'Q_x', range: [low[0], high[0]], exponentformat: 'none'},
     yaxis: {title: 'Q_y', range: [low[1], high[1]], exponentformat: 'none'},
     legend: {orientation: 'h', y: -0.2, font: {size: 10}}}, {responsive: true, displayModeBar: false});
  ['x', 'y'].forEach((plane, a) => {
    const z = Array.from({length: 2*B[1]+1}, () => Array(2*B[0]+1).fill(null));
    T.z[a].forEach(([mx, my, s]) => { z[my + B[1]][mx + B[0]] = s; });
    const axis = n => Array.from({length: 2*n+1}, (v, k) => k - n);
    Plotly.react(plane, [{type: 'heatmap', x: axis(B[0]), y: axis(B[1]), z: z, colorscale: __MAGMA__, zmin: -24, zmax: 0,
        colorbar: {title: 'log P/P_lin', len: 0.8}, hoverongaps: false,
        hovertemplate: 'm = (%{x}, %{y})<br>log10 |c_m|^2/|c_lin|^2 = %{z}<extra></extra>'}].concat(circles(a)),
      {title: {text: 'z_' + plane + ': relative power of every harmonic m', font: {size: 13}},
       margin: {l: 50, r: 20, t: 40, b: 50}, showlegend: false,
       xaxis: {title: 'm_x', zeroline: false}, yaxis: {title: 'm_y', zeroline: false, scaleanchor: 'x'}},
      {responsive: true, displayModeBar: false});
  });
}
let pending = false;            // one draw per frame, with the last slider positions
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; draw(); }); } };
ray.addEventListener('input', schedule);
torus.addEventListener('input', schedule);
torus.value = 1e9;              // start from the outermost torus of the ray
draw();
</script></body></html>
'''


def resonance_maps(result, path, method='spectra', interpolated=False):
    """Maps over the amplitude plane: one marker per torus (interpolated: filled between the tori)."""
    found = detect(result, method)
    i, met = found['i'], found['shown']
    Q, c, r = result['Q'][i], result['c'][i], result['r'][i]
    (Bx, By), _ = grid(c)
    colors = palette(met)
    owner = resonant(met, (Bx, By))
    if interpolated:
        X, Y, field = amplitude_grid(result, i)

    def logs(harmonics):
        P, N = power(result, i, harmonics)
        return np.log10(np.maximum(P, 1e-40)), np.log10(np.maximum(P, 1e-300)/np.maximum(N, 1e-300))

    rows = met[:MAPS]
    mosaic = [['all', 'legend']]+[[f'd{j}', f'p{j}'] for j in range(len(rows))]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 5.2*len(mosaic)), layout='constrained')
    fig.suptitle(f'{job(result)}: the resonances found, in the amplitude plane'
                 +(', interpolated between the tori' if interpolated else ''))

    def show(ax, values, cmap, title, bar, curves, resolved=None, filled=None):
        """One panel: every torus a marker coloured by `values`, faded and crossed where not resolved, the tori of the
        resonances `curves` ringed; interpolated: the values filled, the unresolved part hatched, the curves n.Q = p."""
        top = float(np.ceil(4*values.max())/4)
        norm = plt.Normalize(max(values.min(), top-20), top)
        good = np.ones(len(values), bool) if resolved is None else resolved >= np.log10(RESOLVED)
        if interpolated:
            image = ax.pcolormesh(X, Y, field(values) if filled is None else filled, cmap=cmap, norm=norm,
                                  shading='auto', rasterized=True)
            if not good.all():
                below = dict(levels=[-1e300, np.log10(RESOLVED)], zorder=2)
                ax.contourf(X, Y, field(resolved), colors='w', alpha=.55, **below)
                ax.contourf(X, Y, field(resolved), colors='none', hatches=['xxxx'], **below).set(edgecolor='.45',
                                                                                                linewidth=0)
            for w in curves:
                curve(ax, X, Y, field(Q@w[:2]-w[2]), colors[w])
        else:
            image = ax.scatter(r[good, 0], r[good, 1], c=values[good], cmap=cmap, norm=norm, s=MARKER, linewidths=0)
            if not good.all():
                ax.scatter(r[~good, 0], r[~good, 1], c=values[~good], cmap=cmap, norm=norm, s=MARKER, linewidths=0,
                           alpha=.25)
                ax.scatter(r[~good, 0], r[~good, 1], marker='x', s=MARKER/3, color='.45', linewidths=.6)
            for w in curves:
                k = found['tori'][w]
                ring(ax, r[k, 0], r[k, 1], colors[w])
        unconverged(result, ax, size=MARKER*1.5)
        fig.colorbar(image, ax=ax, shrink=.85, label=bar, ticks=MaxNLocator(8, integer=True), extend='min')
        ax.set(aspect='equal', xlim=(0, None), ylim=(0, None), xlabel=r'$r_x$'+unit(result),
               ylabel=r'$r_y$'+unit(result))
        ax.set_title(title)

    total, resolved = logs(owner)
    show(axes['all'], total, 'magma', 'all the resonant harmonics together',
         'log$_{10}$ resonant power / linear power', met, resolved)
    if interpolated:
        handles = [plt.Line2D([], [], color=colors[w], lw=2.2, label=legend_label(found, w, label(*w))) for w in met]
        if found['unresolved']:
            for w in found['unresolved']:
                curve(axes['all'], X, Y, field(Q@w[:2]-w[2]), '.6', .8)
            handles.append(plt.Line2D([], [], color='.6', lw=.8, label=grey(found)))
        handles.append(plt.Rectangle((0, 0), 1, 1, fc='w', ec='.45', hatch='xxxx', lw=0,
                                     label=f'below {RESOLVED:g}x amplified noise'))
    else:
        handles = [plt.Line2D([], [], ls='', marker='o', ms=9, mfc='none', mew=1.5, color=colors[w],
                              label=legend_label(found, w, label(*w))) for w in met]
        if found['unresolved']:
            handles.append(plt.Line2D([], [], ls='', marker='o', ms=8, mfc='none', mew=1.5, color='.6',
                                      label=grey(found, axes['all'], r)))
        handles.append(plt.Line2D([], [], ls='', marker='o', ms=6, color='.3', label='one computed torus'))
        handles.append(plt.Line2D([], [], ls='', marker='x', ms=6, color='.45',
                                  label=f'below {RESOLVED:g}x amplified noise'))
    cross = unconverged(result)
    if cross:
        handles.append(cross)
    axes['legend'].axis('off')
    axes['legend'].legend(handles=handles, loc='center left', fontsize=9, frameon=False, title=legend_title(found))
    for j, w in enumerate(rows):
        d = Q@w[:2]-w[2]
        show(axes[f'd{j}'], np.log10(np.maximum(np.abs(d), 1e-16)), 'Blues_r', label(*w),
             r'log$_{10}|n.Q - p|$: distance to the resonance', [w],
             filled=np.log10(np.maximum(np.abs(field(d)), 1e-16)) if interpolated else None)
        axes[f'd{j}'].title.set_color(colors[w])
        mine, ok = logs({h: v for h, v in owner.items() if v == w})
        show(axes[f'p{j}'], mine, 'magma', f'its resonant harmonics $e_a + kn$, $|k| \\leq {RESONANT_K}$',
             'log$_{10}$ resonant power / linear power', [w], ok)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tori_slider(result, path, keep=1e-7):
    """HTML: two sliders (ray angle, amplitude) select one torus, rebuilt in the browser from its spectrum on 64 x 64
    angles: x_a = Re z_a, p_a = -Im z_a with z_a = sum_m c[a, m] e^{i m.theta}, coloured by theta_y; the linear torus
    of the same J dashed. Only the coefficients above `keep` x the linear one are written, to keep the page small."""
    allowed = np.flatnonzero(np.isfinite(result['residual']))          # with the best approximation of the others
    keys, phis, levels, slot = slots(result, allowed)
    c = result['c']
    (Bx, By), (mx, my) = grid(c)
    position = {int(t): k for k, t in enumerate(slot[slot >= 0])}
    tori = []
    for t in position:
        linear = max(abs(c[t, 0, Bx+1, By]), abs(c[t, 1, Bx, By+1]))
        terms = [[[int(mx.flat[k]), int(my.flat[k]), float(f'{v.real:.5g}'), float(f'{v.imag:.5g}')]
                  for k, v in enumerate(c[t, a].ravel()) if abs(v) > keep*linear] for a in range(2)]
        tori.append(dict(r=np.round(result['r'][t], 4).tolist(), J=[float(f'{j:.4g}') for j in result['J'][t]],
                         Q=[round(float(q), 6) if np.isfinite(q) else None for q in result['Q'][t]],
                         bad=0 if result['converged'][t] else float(f"{result['residual'][t]:.1e}"), c=terms))
    reach = [1.1*float(np.abs(c[list(position), a]).sum(axis=(1, 2)).max()) for a in range(2)]   # fixed axis ranges
    data = dict(phi=np.round(phis, 1).tolist(), rho=np.round(levels, 4).tolist(), reach=reach, tori=tori,
                slot=[[position.get(int(t), -1) for t in row] for row in slot])
    twilight = json.dumps([[k/10, to_hex(plt.get_cmap('twilight')(k/10))] for k in range(11)])
    page = TORI.replace('__TITLE__', f'{job(result)}: the tori, rebuilt from their spectra') \
               .replace('__DATA__', json.dumps(data, separators=(',', ':'))) \
               .replace('__UNIT__', unit(result, latex=False)).replace('__TWILIGHT__', twilight)
    Path(path).write_text(page, encoding='utf-8')


TORI = '''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tori</title>''' + SCRIPT + STYLE + '''</head><body>
<h3>__TITLE__</h3>
<div class="bar">ray angle &phi; = atan(r<sub>y</sub>/r<sub>x</sub>) <input id="ray" type="range" min="0" step="1">
<b id="phi"></b></div>
<div class="bar">amplitude r = (r<sub>x</sub><sup>2</sup> + r<sub>y</sub><sup>2</sup>)<sup>1/2</sup>
<input id="level" type="range" min="0" step="1"><b id="rho"></b></div>
<div id="info"></div>
<div class="row"><div id="xpx" class="plot"></div><div id="ypy" class="plot"></div>
<div id="xy" class="plot"></div><div id="pxpy" class="plot"></div></div>
<script>
const DATA = __DATA__, N = 64, NONE = -1, [RX, RY] = DATA.reach;
const ray = document.getElementById('ray'), level = document.getElementById('level');
ray.max = DATA.phi.length-1; ray.value = Math.floor(DATA.phi.length/2);
level.max = DATA.rho.length-1; level.value = DATA.rho.length-1;
const cos = Array.from({length: N}, (v, k) => Math.cos(2*Math.PI*k/N)), sin = cos.map((v, k) => Math.sin(2*Math.PI*k/N));
const thetaY = Array.from({length: N*N}, (v, n) => 360*(n % N)/N);       // the colour of every point: theta_y
function rebuild(terms) {        // x = Re z, p = -Im z of z = sum_m c_m e^{i m.theta}, on N x N angles (theta_x, theta_y)
  const x = new Float64Array(N*N), p = new Float64Array(N*N);
  for (const [mx, my, re, im] of terms)
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const k = ((mx*i + my*j) % N + N) % N, n = i*N + j;
      x[n] += re*cos[k] - im*sin[k];
      p[n] -= re*sin[k] + im*cos[k];
    }
  return [Array.from(x), Array.from(p)];
}
const circle = Array.from({length: 101}, (w, k) => 2*Math.PI*k/100);
function panel(div, u, v, ref, U, V, ru, rv, title) {
  Plotly.react(div, [
    {type: 'scatter', mode: 'markers', x: u, y: v, hoverinfo: 'skip', showlegend: false,
     marker: {size: 3, color: thetaY, colorscale: __TWILIGHT__, cmin: 0, cmax: 360, showscale: div === 'xpx',
              colorbar: {title: 'θ_y [deg]', len: 0.8}}},
    {type: 'scatter', mode: 'lines', x: ref[0], y: ref[1], name: 'linear torus, same J', showlegend: div === 'xpx',
     line: {dash: 'dash', color: '#444', width: 1.5}, hoverinfo: 'skip'}],
    {title: {text: title, font: {size: 13}}, margin: {l: 55, r: 10, t: 35, b: 45}, legend: {x: 0, y: -0.15, orientation: 'h'},
     xaxis: {title: U + '__UNIT__', range: [-ru, ru]}, yaxis: {title: V + '__UNIT__', range: [-rv, rv], scaleanchor: 'x'}},
    {responsive: true, displayModeBar: false});
}
function draw() {
  const j = +ray.value, v = +level.value, s = DATA.slot[j][v], info = document.getElementById('info');
  document.getElementById('phi').textContent = DATA.phi[j] + '°';
  document.getElementById('rho').textContent = DATA.rho[v] + '__UNIT__';
  if (s === NONE) { info.innerHTML = '<span style="color:red">✖ no torus at this angle and amplitude</span>'; return; }
  const T = DATA.tori[s], q = T.Q.map(x => x === null ? '-' : x.toFixed(6));
  info.innerHTML = 'r = (' + T.r.join(', ') + ')__UNIT__, J = (' + T.J.join(', ') + '), Q = (' + q.join(', ') + ')' +
    (T.bad ? '  <span style="color:red"><b>not converged: best approximation, residual ' + T.bad.toExponential(0) + '</b></span>' : '');
  const [x, px] = rebuild(T.c[0]), [y, py] = rebuild(T.c[1]), [rx, ry] = T.r;
  const box = [[-rx, rx, rx, -rx, -rx], [-ry, -ry, ry, ry, -ry]];         // the linear torus: circles of radius r,
  panel('xpx', x, px, [circle.map(a => rx*Math.cos(a)), circle.map(a => rx*Math.sin(a))], 'x', 'p_x', RX, RX, '(x, p_x)');
  panel('ypy', y, py, [circle.map(a => ry*Math.cos(a)), circle.map(a => ry*Math.sin(a))], 'y', 'p_y', RY, RY, '(y, p_y)');
  panel('xy', x, y, box, 'x', 'y', RX, RY, '(x, y)');                       // a rectangle in (x, y) and (px, py)
  panel('pxpy', px, py, box, 'p_x', 'p_y', RX, RY, '(p_x, p_y)');
}
let pending = false;            // one draw per frame, with the last slider positions
const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; draw(); }); } };
ray.addEventListener('input', schedule);
level.addEventListener('input', schedule);
draw();
</script></body></html>
'''
