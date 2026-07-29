/* PROXY GAP -- a small dependency-free SVG chart layer.
 *
 * Forms: line (with crosshair + tooltip), scatter, bar, and stat tiles.
 * Every chart ships a legend when it has >=2 series, direct labels on <=4
 * series, and a table view, because three of the light-mode series colors sit
 * below 3:1 against the surface and colour must never be the only channel.
 *
 * Colours come from CSS custom properties defined once in styles.css, so the
 * light/dark swap happens in one place and nothing here hardcodes a hex.
 */

const NS = 'http://www.w3.org/2000/svg';

const SERIES_VARS = [
  '--series-1', '--series-2', '--series-3', '--series-4',
  '--series-5', '--series-6', '--series-7', '--series-8',
];

export const seriesColor = (i) => `var(${SERIES_VARS[i % SERIES_VARS.length]})`;

const el = (name, attrs = {}, parent = null) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
};

const fmt = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '--';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  if (Number.isInteger(v) && a < 1e5) return String(v);
  return v.toFixed(digits);
};

/* Nice axis ticks: 1/2/5 x 10^k covering the domain. */
function ticks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return out;
}

function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  return [lo, hi];
}

function pad([lo, hi], frac = 0.08) {
  const d = (hi - lo) * frac;
  return [lo - d, hi + d];
}

/* ---------------------------------------------------------------- scales -- */

function linear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * span;
  f.domain = domain;
  return f;
}

function log(domain, range) {
  const d0 = Math.log10(Math.max(domain[0], 1e-9));
  const d1 = Math.log10(Math.max(domain[1], 1e-9));
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const f = (v) => r0 + ((Math.log10(Math.max(v, 1e-9)) - d0) / span) * (r1 - r0);
  f.invert = (p) => 10 ** (d0 + ((p - r0) / (r1 - r0 || 1)) * span);
  f.domain = domain;
  f.isLog = true;
  return f;
}

/* ------------------------------------------------------------- chrome ----- */

function frame(host, opts) {
  const { width = 720, height = 340, margin } = opts;
  const m = Object.assign({ top: 24, right: 76, bottom: 46, left: 62 }, margin);

  host.innerHTML = '';
  const wrap = document.createElement('figure');
  wrap.className = 'chart';

  if (opts.title) {
    const cap = document.createElement('figcaption');
    cap.className = 'chart-title';
    cap.textContent = opts.title;
    if (opts.subtitle) {
      const sub = document.createElement('span');
      sub.className = 'chart-sub';
      sub.textContent = opts.subtitle;
      cap.appendChild(sub);
    }
    wrap.appendChild(cap);
  }

  const scroller = document.createElement('div');
  scroller.className = 'chart-scroll';
  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: '100%',
    role: 'img',
    'aria-label': opts.title || 'chart',
    preserveAspectRatio: 'xMidYMid meet',
  });
  svg.style.minWidth = `${Math.min(width, 560)}px`;
  scroller.appendChild(svg);
  wrap.appendChild(scroller);
  host.appendChild(wrap);

  const plot = {
    x0: m.left, x1: width - m.right, y0: height - m.bottom, y1: m.top,
    w: width - m.left - m.right, h: height - m.bottom - m.top,
  };
  return { wrap, svg, plot, width, height };
}

function axes(svg, plot, xs, ys, opts) {
  const g = el('g', {}, svg);
  const xt = xs.isLog
    ? logTicks(xs.domain)
    : ticks(xs.domain[0], xs.domain[1], opts.xTicks || 6);
  const yt = ticks(ys.domain[0], ys.domain[1], opts.yTicks || 5);

  for (const t of yt) {
    const y = ys(t);
    if (y < plot.y1 - 1 || y > plot.y0 + 1) continue;
    el('line', { x1: plot.x0, x2: plot.x1, y1: y, y2: y, class: 'grid' }, g);
    el('text', { x: plot.x0 - 10, y: y + 4, class: 'tick tick-y' }, g).textContent =
      opts.yFormat ? opts.yFormat(t) : fmt(t, 2);
  }
  for (const t of xt) {
    const x = xs(t);
    if (x < plot.x0 - 1 || x > plot.x1 + 1) continue;
    el('text', { x, y: plot.y0 + 22, class: 'tick tick-x' }, g).textContent =
      opts.xFormat ? opts.xFormat(t) : fmt(t, 2);
  }
  el('line', { x1: plot.x0, x2: plot.x1, y1: plot.y0, y2: plot.y0, class: 'axis' }, g);

  if (opts.xLabel) {
    el('text', {
      x: (plot.x0 + plot.x1) / 2, y: plot.y0 + 42, class: 'axis-label',
    }, g).textContent = opts.xLabel;
  }
  if (opts.yLabel) {
    const t = el('text', {
      x: 0, y: 0, class: 'axis-label',
      transform: `translate(14 ${(plot.y0 + plot.y1) / 2}) rotate(-90)`,
    }, g);
    t.textContent = opts.yLabel;
  }
  return g;
}

function logTicks([lo, hi]) {
  const out = [];
  for (let e = Math.floor(Math.log10(Math.max(lo, 1e-9))); e <= Math.ceil(Math.log10(hi)); e++) {
    const v = 10 ** e;
    if (v >= lo * 0.99 && v <= hi * 1.01) out.push(v);
  }
  return out.length >= 2 ? out : [lo, hi];
}

function legend(wrap, series) {
  if (series.length < 2) return;
  const box = document.createElement('div');
  box.className = 'legend';
  series.forEach((s, i) => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = s.color || seriesColor(i);
    if (s.dashed) sw.classList.add('swatch-dashed');
    item.appendChild(sw);
    item.appendChild(document.createTextNode(s.label));
    box.appendChild(item);
  });
  wrap.appendChild(box);
}

/* A table view is mandatory: it is the non-colour channel, and it is what the
 * light-mode contrast warning obliges us to ship. */
function tableView(wrap, columns, rows) {
  if (!rows || !rows.length) return;
  const det = document.createElement('details');
  det.className = 'table-view';
  const sum = document.createElement('summary');
  sum.textContent = `Table view (${rows.length} rows)`;
  det.appendChild(sum);

  const scroll = document.createElement('div');
  scroll.className = 'table-scroll';
  const t = document.createElement('table');
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  for (const c of columns) {
    const th = document.createElement('th');
    th.textContent = c;
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  t.appendChild(thead);
  const tb = document.createElement('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    for (const cell of r) {
      const td = document.createElement('td');
      td.textContent = typeof cell === 'number' ? fmt(cell) : String(cell);
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  scroll.appendChild(t);
  det.appendChild(scroll);
  wrap.appendChild(det);
}

/* ------------------------------------------------------------------ line -- */

export function lineChart(host, opts) {
  const { series } = opts;
  const { wrap, svg, plot } = frame(host, opts);

  const allX = series.flatMap((s) => s.points.map((p) => p[0]));
  const allY = series.flatMap((s) => s.points.flatMap((p) =>
    p[2] === undefined ? [p[1]] : [p[1], p[1] - p[2], p[1] + p[2]]));

  const xs = (opts.xScale === 'log' ? log : linear)(
    opts.xDomain || (opts.xScale === 'log' ? extent(allX) : pad(extent(allX), 0.02)),
    [plot.x0, plot.x1]
  );
  const ys = linear(opts.yDomain || pad(extent(allY)), [plot.y0, plot.y1]);

  axes(svg, plot, xs, ys, opts);

  // Optional shaded reference band (used for "predicted KL*" intervals).
  for (const band of opts.bands || []) {
    const a = xs(band.from);
    const b = xs(band.to);
    el('rect', {
      x: Math.min(a, b), y: plot.y1, width: Math.abs(b - a), height: plot.h,
      class: 'band',
    }, svg);
    if (band.label) {
      el('text', { x: (a + b) / 2, y: plot.y1 + 14, class: 'band-label' }, svg)
        .textContent = band.label;
    }
  }
  for (const rule of opts.vrules || []) {
    el('line', {
      x1: xs(rule.at), x2: xs(rule.at), y1: plot.y0, y2: plot.y1, class: 'vrule',
    }, svg);
    if (rule.label) {
      const tx = el('text', {
        x: xs(rule.at), y: plot.y1 - 6, class: 'vrule-label',
      }, svg);
      tx.textContent = rule.label;
    }
  }

  series.forEach((s, i) => {
    const color = s.color || seriesColor(i);
    const pts = s.points.filter((p) => Number.isFinite(p[1]));
    if (!pts.length) return;

    if (pts.some((p) => p[2] !== undefined)) {
      const up = pts.map((p) => `${xs(p[0])},${ys(p[1] + (p[2] || 0))}`);
      const dn = pts.slice().reverse().map((p) => `${xs(p[0])},${ys(p[1] - (p[2] || 0))}`);
      el('polygon', {
        points: up.concat(dn).join(' '), fill: color, class: 'ci-band',
      }, svg);
    }

    el('polyline', {
      points: pts.map((p) => `${xs(p[0])},${ys(p[1])}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': 2,
      'stroke-dasharray': s.dashed ? '5 4' : null,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }, svg);

    if (opts.markers !== false) {
      for (const p of pts) {
        el('circle', {
          cx: xs(p[0]), cy: ys(p[1]), r: 3.5, fill: color, class: 'marker',
        }, svg);
      }
    }

    // Direct label at the last point -- identity without relying on colour.
    if (series.length <= 4 && opts.directLabels !== false) {
      const last = pts[pts.length - 1];
      const t = el('text', {
        x: Math.min(xs(last[0]) + 8, plot.x1 + 68),
        y: ys(last[1]) + 4, fill: color, class: 'direct-label',
      }, svg);
      t.textContent = s.label;
    }
  });

  legend(wrap, series);
  crosshair(svg, plot, xs, ys, series, opts);

  const cols = [opts.xLabel || 'x', ...series.map((s) => s.label)];
  const keys = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  tableView(wrap, cols, keys.map((k) => [
    k, ...series.map((s) => {
      const hit = s.points.find((p) => p[0] === k);
      return hit ? hit[1] : '--';
    }),
  ]));
  return wrap;
}

function crosshair(svg, plot, xs, ys, series, opts) {
  const g = el('g', { class: 'hover', style: 'opacity:0' }, svg);
  const rule = el('line', { y1: plot.y1, y2: plot.y0, class: 'crosshair' }, g);
  const dots = series.map((s, i) => el('circle', {
    r: 5, fill: s.color || seriesColor(i), stroke: 'var(--surface-1)',
    'stroke-width': 2,
  }, g));

  const tip = document.createElement('div');
  tip.className = 'tooltip';
  tip.style.opacity = '0';
  svg.parentElement.parentElement.appendChild(tip);

  const xsAll = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))]
    .sort((a, b) => a - b);

  const hit = el('rect', {
    x: plot.x0, y: plot.y1, width: plot.w, height: plot.h,
    fill: 'transparent', style: 'cursor:crosshair',
  }, svg);

  const move = (evt) => {
    const box = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const px = ((evt.clientX - box.left) / box.width) * vb.width;
    const target = xs.invert(px);
    let best = xsAll[0];
    for (const v of xsAll) {
      if (Math.abs(v - target) < Math.abs(best - target)) best = v;
    }
    rule.setAttribute('x1', xs(best));
    rule.setAttribute('x2', xs(best));

    const rows = [];
    series.forEach((s, i) => {
      const p = s.points.find((q) => q[0] === best);
      if (p && Number.isFinite(p[1])) {
        dots[i].setAttribute('cx', xs(p[0]));
        dots[i].setAttribute('cy', ys(p[1]));
        dots[i].style.opacity = '1';
        rows.push(`<tr><td><i style="background:${s.color || seriesColor(i)}"></i>${s.label}</td><td>${fmt(p[1])}${p[2] !== undefined ? ` <span class="pm">&plusmn;${fmt(p[2], 3)}</span>` : ''}</td></tr>`);
      } else {
        dots[i].style.opacity = '0';
      }
    });

    g.style.opacity = '1';
    tip.innerHTML =
      `<div class="tip-head">${opts.xLabel || 'x'} = ${opts.xFormat ? opts.xFormat(best) : fmt(best)}</div>` +
      `<table>${rows.join('')}</table>`;
    tip.style.opacity = '1';
    const frac = (xs(best) - plot.x0) / plot.w;
    const host = svg.parentElement;
    tip.style.left = `${host.offsetLeft + (xs(best) / vb.width) * host.clientWidth + (frac > 0.6 ? -12 : 12)}px`;
    tip.style.transform = frac > 0.6 ? 'translateX(-100%)' : 'none';
    tip.style.top = `${svg.parentElement.offsetTop + 18}px`;
  };

  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerdown', move);
  hit.addEventListener('pointerleave', () => {
    g.style.opacity = '0';
    tip.style.opacity = '0';
  });
}

/* --------------------------------------------------------------- scatter -- */

export function scatterChart(host, opts) {
  const { wrap, svg, plot } = frame(host, opts);
  const groups = opts.groups;
  const allX = groups.flatMap((g) => g.points.map((p) => p[0]));
  const allY = groups.flatMap((g) => g.points.map((p) => p[1]));

  const xs = linear(opts.xDomain || pad(extent(allX)), [plot.x0, plot.x1]);
  const ys = linear(opts.yDomain || pad(extent(allY)), [plot.y0, plot.y1]);
  axes(svg, plot, xs, ys, opts);

  if (opts.identity) {
    const lo = Math.max(xs.domain[0], ys.domain[0]);
    const hi = Math.min(xs.domain[1], ys.domain[1]);
    el('line', {
      x1: xs(lo), y1: ys(lo), x2: xs(hi), y2: ys(hi), class: 'identity',
    }, svg);
    el('text', { x: xs(hi) - 6, y: ys(hi) - 8, class: 'identity-label' }, svg)
      .textContent = 'y = x';
  }

  groups.forEach((grp, i) => {
    const color = grp.color || seriesColor(i);
    for (const p of grp.points) {
      const c = el('circle', {
        cx: xs(p[0]), cy: ys(p[1]), r: grp.radius || 4,
        fill: color, class: 'dot',
      }, svg);
      const label = p[2] !== undefined ? `${p[2]}: ` : '';
      el('title', {}, c).textContent =
        `${label}${opts.xLabel || 'x'} ${fmt(p[0])}, ${opts.yLabel || 'y'} ${fmt(p[1])}`;
    }
  });

  legend(wrap, groups);
  tableView(
    wrap,
    [opts.xLabel || 'x', opts.yLabel || 'y', 'series'],
    groups.flatMap((g) => g.points.map((p) => [p[0], p[1], g.label]))
  );
  return wrap;
}

/* ------------------------------------------------------------------- bar -- */

export function barChart(host, opts) {
  const { wrap, svg, plot } = frame(host, Object.assign({ height: 300 }, opts));
  const rows = opts.rows;
  const vals = rows.map((r) => r.value);
  const lo = Math.min(0, ...vals);
  const hi = Math.max(0, ...vals);
  const xs = linear(opts.xDomain || pad([lo, hi], 0.12), [plot.x0, plot.x1]);
  const step = plot.h / Math.max(rows.length, 1);
  const bh = Math.min(26, step - 8);

  for (const t of ticks(xs.domain[0], xs.domain[1], 5)) {
    el('line', { x1: xs(t), x2: xs(t), y1: plot.y1, y2: plot.y0, class: 'grid' }, svg);
    el('text', { x: xs(t), y: plot.y0 + 20, class: 'tick tick-x' }, svg).textContent = fmt(t, 2);
  }

  rows.forEach((r, i) => {
    const y = plot.y1 + i * step + (step - bh) / 2;
    const zero = xs(0);
    const x = Math.min(zero, xs(r.value));
    const w = Math.abs(xs(r.value) - zero);
    el('rect', {
      x, y, width: Math.max(w, 1), height: bh, rx: 4,
      fill: r.color || seriesColor(r.slot ?? 0), class: 'bar',
    }, svg);
    el('text', { x: plot.x0 - 10, y: y + bh / 2 + 4, class: 'bar-label' }, svg)
      .textContent = r.label;
    const vx = xs(r.value) + (r.value >= 0 ? 8 : -8);
    const t = el('text', {
      x: vx, y: y + bh / 2 + 4, class: 'bar-value',
      'text-anchor': r.value >= 0 ? 'start' : 'end',
    }, svg);
    t.textContent = r.format ? r.format(r.value) : fmt(r.value);
  });

  el('line', { x1: xs(0), x2: xs(0), y1: plot.y1, y2: plot.y0, class: 'axis' }, svg);
  tableView(wrap, [opts.yLabel || 'item', opts.xLabel || 'value'],
    rows.map((r) => [r.label, r.value]));
  return wrap;
}

/* ------------------------------------------------------------- stat tile -- */

export function statTiles(host, tiles) {
  host.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'stat-grid';
  for (const t of tiles) {
    const card = document.createElement('div');
    card.className = 'stat';
    if (t.tone) card.classList.add(`tone-${t.tone}`);
    card.innerHTML =
      `<div class="stat-label">${t.label}</div>` +
      `<div class="stat-value">${t.value}</div>` +
      (t.note ? `<div class="stat-note">${t.note}</div>` : '');
    grid.appendChild(card);
  }
  host.appendChild(grid);
}

export { fmt };

/* ------------------------------------------------------------------ data -- */

/* Prefer the inlined bundle: fetch() is blocked on file:// URLs, so the bundle
 * is what makes `open site/index.html` work without a web server. Fall back to
 * the JSON files when the bundle is absent. */
export async function loadData(name) {
  const bundle = typeof window !== 'undefined' ? window.__PROXYGAP__ : null;
  if (bundle && bundle[name]) return bundle[name];
  const res = await fetch(`data/${name}.json`);
  if (!res.ok) throw new Error(`could not load data/${name}.json (${res.status})`);
  return res.json();
}

export function failNotice(host, err) {
  host.innerHTML =
    `<div class="notice notice-bad"><strong>This figure did not render.</strong> ` +
    `${String(err.message || err)}<br>Regenerate with <code>make all</code>.</div>`;
}
