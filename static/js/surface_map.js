/* Surface Map / Route Explorer
 *
 * Renders each world's road network coloured by surface type, overlays a
 * selected route, and shows a surface-tagged elevation profile. The backend
 * returns geometry in a single 2D "plot space" (y increasing downward): map
 * PNG pixels for GPS-calibrated worlds, or flipped local metres otherwise.
 */

'use strict';

const state = {
    world: null,        // { mapID, name, bounds, segments, routes, colors, background }
    route: null,        // route detail from API
    routeFilter: '',
    sportFilter: 'all', // 'all' | 'cycling' | 'running'
    activeHash: null,
    // Affine transform: screenX = X*scale + offsetX ; screenY = Y*scale + offsetY
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    baseScale: 1,       // scale when the whole world is fitted (for line widths)
    dpr: 1,
    showNetwork: true,
    bgImage: null,      // HTMLImageElement for the map PNG, or null
    hoverPoint: null,   // {x, y} in plot space, from elevation hover
    pinnedPoint: null,  // {x, y} in plot space, frozen on elevation click
    combined: [],       // [{dist_m, x, y}] leadin+route for hover linkage
};

let canvas, ctx, baseCanvas, baseCtx;

document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('map-canvas');
    ctx = canvas.getContext('2d');
    baseCanvas = document.createElement('canvas');
    baseCtx = baseCanvas.getContext('2d');

    bindEvents();
    resizeCanvas();
    loadWorlds();
});

function setStatus(msg, isError) {
    const el = document.getElementById('status');
    el.textContent = msg || '';
    el.classList.toggle('error', !!isError);
}

/* ---------------------------------------------------------------- data --- */

async function loadWorlds() {
    setStatus('Loading worlds…');
    try {
        const res = await fetch('/api/surface_map/worlds');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load worlds');
        const sel = document.getElementById('world-select');
        sel.innerHTML = '';
        data.worlds.forEach(w => {
            const opt = document.createElement('option');
            opt.value = w.mapID;
            opt.textContent = `${w.name} (${w.route_count} routes)`;
            sel.appendChild(opt);
        });
        setStatus('');
        if (data.worlds.length) {
            sel.value = data.worlds[0].mapID;
            await loadWorld(data.worlds[0].mapID);
        }
    } catch (e) {
        setStatus(e.message, true);
    }
}

async function loadWorld(mapId) {
    setStatus('Loading world…');
    state.route = null;
    state.activeHash = null;
    state.hoverPoint = null;
    state.bgImage = null;
    document.getElementById('route-detail').style.display = 'none';
    try {
        const res = await fetch(`/api/surface_map/world/${mapId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load world');
        state.world = data;
        buildLegend(data.colors);
        renderRouteList();
        // Record the full-world fit scale for line-width scaling.
        const b = data.bounds;
        state.baseScale = Math.min(canvas.width / Math.max(b.max_x - b.min_x, 1),
                                   canvas.height / Math.max(b.max_y - b.min_y, 1)) * 0.94;
        fitToBounds(data.bounds);
        loadBackground(data.background);
        document.getElementById('map-hint').style.display = 'none';
        setStatus('');
    } catch (e) {
        setStatus(e.message, true);
    }
}

function loadBackground(bg) {
    if (!bg) { state.bgImage = null; renderBase(); return; }
    const img = new Image();
    img.onload = () => {
        if (state.world && state.world.background &&
            state.world.background.image === bg.image) {
            state.bgImage = img;
            renderBase();
        }
    };
    img.src = bg.image;
}

async function loadRoute(hash) {
    setStatus('Loading route…');
    try {
        const res = await fetch(`/api/surface_map/route/${state.world.mapID}/${hash}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load route');
        state.route = data;
        state.activeHash = hash;
        state.hoverPoint = null;
        state.pinnedPoint = null;
        buildCombinedPath(data);
        renderRouteDetail(data);
        renderRouteList();
        if (data.bounds) fitToBounds(data.bounds, 0.82);
        else renderBase();
        setStatus('');
    } catch (e) {
        setStatus(e.message, true);
    }
}

/* ------------------------------------------------------------- sidebar --- */

// Zwift route sportType: 2 = running. Every other value (-1, 0, 1, 3) is a
// cycling route entry; runnable routes get their own separate value-2 entry.
function sportIsCycling(v) { return v !== 2; }
function sportIsRunning(v) { return v === 2; }
function sportLabel(v) { return v === 2 ? 'Running' : 'Cycling'; }

function renderRouteList() {
    const ul = document.getElementById('route-list');
    ul.innerHTML = '';
    if (!state.world) return;
    const filter = state.routeFilter.toLowerCase();
    const sf = state.sportFilter;
    const routes = state.world.routes.filter(r => {
        if (filter && !r.name.toLowerCase().includes(filter)) return false;
        if (sf === 'cycling') return sportIsCycling(r.sport_type);
        if (sf === 'running') return sportIsRunning(r.sport_type);
        return true;
    });

    document.getElementById('route-count').textContent =
        `${routes.length}/${state.world.routes.length}`;

    routes.forEach(r => {
        const li = document.createElement('li');
        if (r.nameHash === state.activeHash) li.classList.add('active');
        const dist = ((r.distance_m + r.leadin_distance_m) / 1000).toFixed(1);
        const asc = Math.round(r.ascent_m + r.leadin_ascent_m);
        const badge = r.event_only ? '<span class="r-badge">Event</span>' : '';
        const sport = sportLabel(r.sport_type);
        const sportBadge =
            `<span class="sport-tag sport-${sport.toLowerCase()}">${sport}</span>`;
        li.innerHTML = `${escapeHtml(r.name)}${badge}` +
            `<span class="r-meta">${dist} km · ${asc} m ↑${sportBadge}</span>`;
        li.addEventListener('click', () => {
            if (r.nameHash === state.activeHash) clearRoute();
            else loadRoute(r.nameHash);
        });
        ul.appendChild(li);
    });
}

function clearRoute() {
    state.route = null;
    state.activeHash = null;
    state.hoverPoint = null;
    state.pinnedPoint = null;
    state.combined = [];
    document.getElementById('route-detail').style.display = 'none';
    renderRouteList();
    if (state.world) fitToBounds(state.world.bounds);
}

function renderRouteDetail(d) {
    const panel = document.getElementById('route-detail');
    panel.style.display = '';
    document.getElementById('rd-name').textContent = d.name;

    const totalKm = ((d.distance_m + d.leadin_distance_m) / 1000).toFixed(1);
    const ascent = Math.round(d.ascent_m);
    const leadinKm = (d.leadin_distance_m / 1000).toFixed(1);
    document.getElementById('rd-stats').innerHTML = `
        <div class="stat"><span class="v">${totalKm} km</span><span class="k">Distance</span></div>
        <div class="stat"><span class="v">${ascent} m</span><span class="k">Ascent</span></div>
        <div class="stat"><span class="v">${leadinKm} km</span><span class="k">Lead-in</span></div>`;

    renderBreakdown(d.breakdown);
    renderElevation(d);
}

function renderBreakdown(breakdown) {
    const el = document.getElementById('rd-breakdown');
    if (!breakdown || !breakdown.length) { el.innerHTML = ''; return; }
    const total = breakdown.reduce((s, b) => s + b.distance_m, 0) || 1;
    const bar = breakdown.map(b =>
        `<span style="width:${(b.distance_m / total * 100).toFixed(2)}%;background:${b.color}" title="${b.surface}"></span>`
    ).join('');
    const key = breakdown.map(b => {
        const pct = (b.distance_m / total * 100).toFixed(0);
        const km = (b.distance_m / 1000).toFixed(1);
        return `<span class="item"><span class="swatch" style="background:${b.color}"></span>${b.surface} · ${km} km (${pct}%)</span>`;
    }).join('');
    el.innerHTML = `<div class="bar">${bar}</div><div class="bkey">${key}</div>`;
}

function buildLegend(colors) {
    const el = document.getElementById('legend');
    el.innerHTML = Object.entries(colors).map(([surf, col]) =>
        `<span class="item"><span class="swatch" style="background:${col}"></span>${surf}</span>`
    ).join('');
}

/* ------------------------------------------------------------ elevation --- */

function elevationShapes() {
    const shapes = [];
    if (state.elevationLeadinKm != null) {
        shapes.push({
            type: 'line', x0: state.elevationLeadinKm, x1: state.elevationLeadinKm,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: 'rgba(255,255,255,0.35)', width: 1, dash: 'dash' },
        });
    }
    if (state.pinnedPoint) {
        shapes.push({
            type: 'line', x0: state.pinnedPoint.km, x1: state.pinnedPoint.km,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: '#2ec4ff', width: 1.5, dash: 'dash' },
        });
    }
    return shapes;
}

function buildCombinedPath(d) {
    const combined = [];
    const push = (leg, offset) => {
        if (!leg) return;
        for (let i = 0; i < leg.d.length; i++) {
            combined.push({ dist_m: leg.d[i] + offset, x: leg.x[i], y: leg.y[i] });
        }
    };
    push(d.leadin, 0);
    push(d.route, d.leadin_distance_m || 0);
    state.combined = combined;
}

function renderElevation(d) {
    const colors = (state.world && state.world.colors) || {};
    const traces = [];
    let combinedIdx = 0;

    // Emit one line trace per contiguous run of the same surface so each run is
    // coloured. legendgroup keeps one legend entry per surface.
    const seenSurface = new Set();
    const addLeg = (leg, offset, dash) => {
        if (!leg || !leg.d.length) return combinedIdx;
        let runStart = 0;
        for (let i = 1; i <= leg.d.length; i++) {
            const boundary = (i === leg.d.length) || (leg.surface[i] !== leg.surface[runStart]);
            if (!boundary) continue;
            const surf = leg.surface[runStart];
            const xs = [], ys = [], cd = [];
            for (let j = runStart; j <= Math.min(i, leg.d.length - 1); j++) {
                xs.push((leg.d[j] + offset) / 1000);
                ys.push(leg.alt[j]);
                cd.push(combinedIdx + j);
            }
            const showLegend = !seenSurface.has(surf) && dash !== 'dot';
            seenSurface.add(surf);
            traces.push({
                x: xs, y: ys, customdata: cd,
                mode: 'lines', type: 'scatter',
                line: { color: colors[surf] || '#888', width: dash === 'dot' ? 2 : 3, dash: dash || 'solid' },
                name: surf, legendgroup: surf, showlegend: showLegend,
                hovertemplate: `%{x:.2f} km · %{y:.0f} m<br>${surf}<extra></extra>`,
            });
            runStart = i;
        }
        return combinedIdx + leg.d.length;
    };

    combinedIdx = addLeg(d.leadin, 0, 'dot');
    combinedIdx = addLeg(d.route, d.leadin_distance_m || 0, 'solid');

    // Lead-in boundary; the pinned-point line is added on click via relayout.
    state.elevationLeadinKm = d.leadin_distance_m > 0 ? d.leadin_distance_m / 1000 : null;

    const layout = {
        margin: { l: 48, r: 12, t: 10, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c8ccd2', size: 11 },
        xaxis: { title: 'Distance (km)', gridcolor: 'rgba(255,255,255,0.07)', zeroline: false },
        yaxis: { title: 'Elevation (m)', gridcolor: 'rgba(255,255,255,0.07)', zeroline: false },
        legend: { orientation: 'h', y: 1.12, font: { size: 10 } },
        shapes: elevationShapes(),
        hovermode: 'closest',
    };

    const chart = document.getElementById('elevation-chart');
    Plotly.react(chart, traces, layout, { displayModeBar: false, responsive: true });

    chart.removeAllListeners?.('plotly_hover');
    chart.removeAllListeners?.('plotly_unhover');
    chart.removeAllListeners?.('plotly_click');
    chart.on('plotly_hover', ev => {
        const pt = ev.points[0];
        const idx = pt.customdata;
        const c = state.combined[idx];
        if (c) { state.hoverPoint = { x: c.x, y: c.y }; paint(); }
    });
    chart.on('plotly_unhover', () => { state.hoverPoint = null; paint(); });
    chart.on('plotly_click', ev => {
        const c = state.combined[ev.points[0].customdata];
        if (!c) return;
        // Toggle the frozen marker when clicking the same point again.
        const same = state.pinnedPoint
            && state.pinnedPoint.x === c.x && state.pinnedPoint.y === c.y;
        state.pinnedPoint = same ? null : { x: c.x, y: c.y, km: c.dist_m / 1000 };
        Plotly.relayout(chart, { shapes: elevationShapes() });
        paint();
    });
}

/* --------------------------------------------------------------- canvas --- */

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    state.dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * state.dpr);
    const h = Math.round(rect.height * state.dpr);
    canvas.width = w; canvas.height = h;
    baseCanvas.width = w; baseCanvas.height = h;
    if (state.world) {
        if (state.route && state.route.bounds) fitToBounds(state.route.bounds, 0.82);
        else fitToBounds(state.world.bounds);
    }
}

function fitToBounds(b, pad = 0.94) {
    const w = canvas.width, h = canvas.height;
    const bw = Math.max(b.max_x - b.min_x, 1);
    const bh = Math.max(b.max_y - b.min_y, 1);
    state.scale = Math.min(w / bw, h / bh) * pad;
    const cx = (b.min_x + b.max_x) / 2;
    const cy = (b.min_y + b.max_y) / 2;
    state.offsetX = w / 2 - cx * state.scale;
    state.offsetY = h / 2 - cy * state.scale;
    renderBase();
}

function toScreen(x, y) {
    return { sx: x * state.scale + state.offsetX, sy: y * state.scale + state.offsetY };
}
function toWorld(sx, sy) {
    return { x: (sx - state.offsetX) / state.scale, y: (sy - state.offsetY) / state.scale };
}

function renderBase() {
    if (!state.world) return;
    const g = baseCtx;
    g.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    g.lineJoin = 'round';
    g.lineCap = 'round';

    // Map PNG background (GPS-calibrated worlds), drawn in plot space.
    const bg = state.world.background;
    if (state.bgImage && bg) {
        const p0 = toScreen(0, 0);
        const p1 = toScreen(bg.width, bg.height);
        g.drawImage(state.bgImage, p0.sx, p0.sy, p1.sx - p0.sx, p1.sy - p0.sy);
    }

    // Line widths scale with zoom relative to the full-world fit.
    const zoomK = state.baseScale ? state.scale / state.baseScale : 1;

    if (state.showNetwork) {
        const roadW = Math.max(1.1, 3.0 * zoomK);
        const colors = state.world.colors;
        state.world.segments.forEach(seg => {
            g.strokeStyle = colors[seg.surface] || '#555a63';
            g.lineWidth = roadW;
            g.beginPath();
            const xs = seg.x, ys = seg.y;
            for (let i = 0; i < xs.length; i++) {
                const p = toScreen(xs[i], ys[i]);
                if (i === 0) g.moveTo(p.sx, p.sy); else g.lineTo(p.sx, p.sy);
            }
            g.stroke();
        });
    }

    if (state.route) {
        drawLeg(g, state.route.leadin, 'rgba(255,255,255,0.5)', 2.5, true, zoomK);
        drawLeg(g, state.route.route, '#f7931e', 3.5, false, zoomK);
    }
}

function drawLeg(g, leg, color, baseWidth, dashed, zoomK) {
    if (!leg || !leg.x.length) return;
    const w = Math.max(baseWidth, baseWidth * (zoomK || 1));
    // Dark casing for contrast against same-coloured roads.
    g.lineWidth = w + 3;
    g.strokeStyle = 'rgba(0,0,0,0.55)';
    g.setLineDash([]);
    strokePath(g, leg);
    g.lineWidth = w;
    g.strokeStyle = color;
    g.setLineDash(dashed ? [8, 6] : []);
    strokePath(g, leg);
    g.setLineDash([]);
}

function strokePath(g, leg) {
    g.beginPath();
    for (let i = 0; i < leg.x.length; i++) {
        const p = toScreen(leg.x[i], leg.y[i]);
        if (i === 0) g.moveTo(p.sx, p.sy); else g.lineTo(p.sx, p.sy);
    }
    g.stroke();
}

function paint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(baseCanvas, 0, 0);
    if (state.pinnedPoint) drawMarker(state.pinnedPoint, '#2ec4ff', true);
    if (state.hoverPoint) drawMarker(state.hoverPoint, '#f7931e', false);
}

function drawMarker(pt, color, filled) {
    const p = toScreen(pt.x, pt.y);
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, 7 * state.dpr, 0, Math.PI * 2);
    ctx.fillStyle = filled ? color : '#fff';
    ctx.strokeStyle = filled ? '#fff' : color;
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
}

// Repaint after every base render on the next frame.
const _origRenderBase = renderBase;
renderBase = function () { _origRenderBase(); requestAnimationFrame(paint); };

/* --------------------------------------------------------------- events --- */

function bindEvents() {
    document.getElementById('world-select').addEventListener('change', e => {
        if (e.target.value) loadWorld(parseInt(e.target.value, 10));
    });
    document.getElementById('route-search').addEventListener('input', e => {
        state.routeFilter = e.target.value;
        renderRouteList();
    });
    document.querySelectorAll('#sport-filter .sport-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            state.sportFilter = btn.dataset.sport;
            document.querySelectorAll('#sport-filter .sport-btn').forEach(
                b => b.classList.toggle('active', b === btn));
            renderRouteList();
        });
    });
    document.getElementById('btn-fit').addEventListener('click', () => {
        if (state.route && state.route.bounds) fitToBounds(state.route.bounds, 0.82);
        else if (state.world) fitToBounds(state.world.bounds);
    });
    document.getElementById('toggle-network').addEventListener('change', e => {
        state.showNetwork = e.target.checked;
        renderBase();
    });

    // Pan
    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('mousedown', e => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        const dx = (e.clientX - lastX) * state.dpr;
        const dy = (e.clientY - lastY) * state.dpr;
        lastX = e.clientX; lastY = e.clientY;
        state.offsetX += dx; state.offsetY += dy;
        renderBase();
    });

    // Zoom around cursor
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        if (!state.world) return;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * state.dpr;
        const my = (e.clientY - rect.top) * state.dpr;
        const world = toWorld(mx, my);
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        state.scale *= factor;
        state.offsetX = mx - world.x * state.scale;
        state.offsetY = my - world.y * state.scale;
        renderBase();
    }, { passive: false });

    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resizeCanvas, 150);
    });
}

/* ----------------------------------------------------------------- util --- */

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
