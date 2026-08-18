/* Surface Map (Dev) — road / XML inspector.
 *
 * Lists every road in a world's road.xml, highlights a selected road on the
 * map, shows its raw XML, and lets you click a ROADNODE (segment) or ROADMARKER
 * line in the XML to trace that piece on the map. Geometry arrives in the same
 * plot space as the main surface map (y increases downward).
 */

'use strict';

const state = {
    world: null,        // { mapID, name, bounds, roads, colors, background }
    road: null,         // road detail from API
    roadFilter: '',
    viewportOnly: false, // list only roads visible in the current map view
    activeRoadId: null,
    highlight: null,    // { x:[], y:[], color } segment/marker overlay, or null
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    baseScale: 1,
    dpr: 1,
    showNetwork: true,
    bgImage: null,
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
        const res = await fetch('/api/surface_map_dev/worlds');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load worlds');
        const sel = document.getElementById('world-select');
        sel.innerHTML = '';
        data.worlds.forEach(w => {
            const opt = document.createElement('option');
            opt.value = w.mapID;
            opt.textContent = w.name;
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
    setStatus('Reading road.xml…');
    state.road = null;
    state.activeRoadId = null;
    state.highlight = null;
    state.bgImage = null;
    document.getElementById('road-detail').style.display = 'none';
    try {
        const res = await fetch(`/api/surface_map_dev/world/${mapId}/roads`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load world');
        state.world = data;
        buildLegend(data.colors);
        renderRoadList();
        const b = data.bounds;
        state.baseScale = Math.min(canvas.width / Math.max(b.max_x - b.min_x, 1),
                                   canvas.height / Math.max(b.max_y - b.min_y, 1)) * 0.94;
        fitToBounds(data.bounds);
        loadBackground(data.background);
        document.getElementById('map-hint').style.display = 'none';
        setStatus(`${data.roads.length} roads`);
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

async function loadRoad(roadId) {
    setStatus('Loading road…');
    try {
        const res = await fetch(`/api/surface_map_dev/world/${state.world.mapID}/road/${roadId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load road');
        state.road = data;
        state.activeRoadId = roadId;
        state.highlight = null;
        renderRoadDetail(data);
        renderRoadList();
        if (data.bounds) fitToBounds(data.bounds, 0.82);
        else renderBase();
        setStatus('');
    } catch (e) {
        setStatus(e.message, true);
    }
}

function clearRoad() {
    state.road = null;
    state.activeRoadId = null;
    state.highlight = null;
    document.getElementById('road-detail').style.display = 'none';
    renderRoadList();
    if (state.world) fitToBounds(state.world.bounds);
}

/* ------------------------------------------------------------- sidebar --- */

function renderRoadList() {
    const ul = document.getElementById('road-list');
    ul.innerHTML = '';
    if (!state.world) return;
    const filter = state.roadFilter.toLowerCase();
    let roads = state.world.roads.filter(r =>
        !filter ||
        String(r.id).includes(filter) ||
        (r.name && r.name.toLowerCase().includes(filter)) ||
        r.surface.toLowerCase().includes(filter));

    if (state.viewportOnly) {
        const vb = viewportWorldBounds();
        roads = roads.filter(r => roadInView(r, vb));
    }

    document.getElementById('road-count').textContent =
        `${roads.length}/${state.world.roads.length}`;

    const colors = state.world.colors || {};
    roads.forEach(r => {
        const li = document.createElement('li');
        if (r.id === state.activeRoadId) li.classList.add('active');
        const km = (r.length_m / 1000).toFixed(2);
        const swatch = `<span class="swatch" style="background:${colors[r.surface] || '#555a63'}"></span>`;
        const nm = r.name ? escapeHtml(r.name) : `<span class="muted">road ${r.id}</span>`;
        const loop = r.looped ? '<span class="r-badge">loop</span>' : '';
        li.innerHTML =
            `<span class="road-id">#${r.id}</span> ${swatch}${nm}${loop}` +
            `<span class="r-meta">${r.surface} · ${km} km · ${r.node_count} nodes</span>`;
        li.addEventListener('click', () => {
            if (r.id === state.activeRoadId) clearRoad();
            else loadRoad(r.id);
        });
        ul.appendChild(li);
    });
}

function renderRoadDetail(d) {
    const panel = document.getElementById('road-detail');
    panel.style.display = '';
    document.getElementById('rd-name').textContent =
        d.name ? `${d.name} (#${d.id})` : `Road #${d.id}`;

    const km = (d.length_m / 1000).toFixed(2);
    document.getElementById('rd-stats').innerHTML = `
        <div class="stat"><span class="v">${km} km</span><span class="k">Length</span></div>
        <div class="stat"><span class="v">${d.node_count}</span><span class="k">Nodes</span></div>
        <div class="stat"><span class="v">${d.default_surface}</span><span class="k">${escapeHtml(d.default_style)}</span></div>
        <div class="stat"><span class="v">${d.looped ? 'Yes' : 'No'}</span><span class="k">Looped</span></div>`;

    renderXml(d);
}

/* ----------------------------------------------------------------- xml --- */

function renderXml(d) {
    // Index segments by their starting node and markers by id for click lookup.
    const segByNode = new Map();
    const nodePoint = new Map();  // node index -> {x, y}, for nodes without a segment
    (d.segments || []).forEach(s => {
        segByNode.set(s.node, s);
        if (s.x.length) {
            if (!nodePoint.has(s.node)) nodePoint.set(s.node, { x: s.x[0], y: s.y[0] });
            nodePoint.set(s.to, { x: s.x[s.x.length - 1], y: s.y[s.y.length - 1] });
        }
    });
    const markers = d.markers || [];

    const pre = document.getElementById('road-xml');
    pre.innerHTML = '';
    let nodeIdx = 0;
    let markerIdx = 0;
    d.xml.split('\n').forEach(line => {
        const span = document.createElement('span');
        span.className = 'xml-line';
        span.textContent = line + '\n';

        if (line.includes('ENTITY_TYPE_ROADNODE')) {
            const idx = nodeIdx++;
            const seg = segByNode.get(idx);
            span.classList.add('clickable', 'node-line');
            if (seg) {
                span.title = `Segment node ${seg.node} → ${seg.to}`;
                span.addEventListener('click', () => selectXml(span, {
                    x: seg.x, y: seg.y, color: '#2ec4ff',
                }));
            } else {
                // Terminal node of an open road: no outgoing segment, mark the point.
                const p = nodePoint.get(idx);
                span.title = `Node ${idx}`;
                span.addEventListener('click', () => selectXml(span, {
                    x: p ? [p.x] : [], y: p ? [p.y] : [], color: '#2ec4ff',
                }));
            }
        } else if (line.includes('ENTITY_TYPE_ROADMARKER')) {
            // Markers are mapped by document order (backend keeps them all, in order).
            const m = markers[markerIdx++];
            if (m) {
                span.classList.add('clickable', 'marker-line');
                const label = m.surface ? `${m.style} → ${m.surface}` : 'no surface override';
                span.title = `${label}  (t ${m.t0}–${m.t1})`;
                const col = (m.surface && state.world.colors && state.world.colors[m.surface]) || '#e6e6e6';
                span.addEventListener('click', () => selectXml(span, {
                    x: m.x, y: m.y, color: col,
                }));
            }
        }
        pre.appendChild(span);
    });
}

function selectXml(span, hl) {
    document.querySelectorAll('#road-xml .xml-line.selected')
        .forEach(el => el.classList.remove('selected'));
    span.classList.add('selected');
    state.highlight = hl;
    renderBase();
}

function buildLegend(colors) {
    const el = document.getElementById('legend');
    el.innerHTML = Object.entries(colors).map(([surf, col]) =>
        `<span class="item"><span class="swatch" style="background:${col}"></span>${surf}</span>`
    ).join('');
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
        if (state.road && state.road.bounds) fitToBounds(state.road.bounds, 0.82);
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

// World-space bounds of the visible canvas area.
function viewportWorldBounds() {
    const a = toWorld(0, 0);
    const b = toWorld(canvas.width, canvas.height);
    return {
        min_x: Math.min(a.x, b.x), max_x: Math.max(a.x, b.x),
        min_y: Math.min(a.y, b.y), max_y: Math.max(a.y, b.y),
    };
}

// True if any of the road's plotted points falls inside the viewport box.
function roadInView(r, vb) {
    for (let i = 0; i < r.x.length; i++) {
        if (r.x[i] >= vb.min_x && r.x[i] <= vb.max_x &&
            r.y[i] >= vb.min_y && r.y[i] <= vb.max_y) return true;
    }
    return false;
}

// Refresh the sidebar when the view changes, but only if it's view-filtered.
function refreshViewFilter() {
    if (state.viewportOnly) renderRoadList();
}

function polyline(g, xs, ys) {
    g.beginPath();
    for (let i = 0; i < xs.length; i++) {
        const p = toScreen(xs[i], ys[i]);
        if (i === 0) g.moveTo(p.sx, p.sy); else g.lineTo(p.sx, p.sy);
    }
    g.stroke();
}

function renderBase() {
    if (!state.world) return;
    const g = baseCtx;
    g.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    g.lineJoin = 'round';
    g.lineCap = 'round';

    const bg = state.world.background;
    if (state.bgImage && bg) {
        const p0 = toScreen(0, 0);
        const p1 = toScreen(bg.width, bg.height);
        g.drawImage(state.bgImage, p0.sx, p0.sy, p1.sx - p0.sx, p1.sy - p0.sy);
    }

    const zoomK = state.baseScale ? state.scale / state.baseScale : 1;

    // Faint full road network for context.
    if (state.showNetwork) {
        g.strokeStyle = 'rgba(150,160,175,0.35)';
        g.lineWidth = Math.max(0.8, 1.6 * zoomK);
        state.world.roads.forEach(r => {
            if (r.id !== state.activeRoadId && r.x.length > 1) polyline(g, r.x, r.y);
        });
    }

    // Selected road in orange with a dark casing.
    if (state.road && state.road.full && state.road.full.x.length > 1) {
        const w = Math.max(2.5, 3.5 * zoomK);
        g.lineWidth = w + 3;
        g.strokeStyle = 'rgba(0,0,0,0.55)';
        polyline(g, state.road.full.x, state.road.full.y);
        g.lineWidth = w;
        g.strokeStyle = '#f7931e';
        polyline(g, state.road.full.x, state.road.full.y);
    }

    // Selected segment / marker highlight.
    if (state.highlight && state.highlight.x.length > 1) {
        const w = Math.max(4, 6 * zoomK);
        g.lineWidth = w + 4;
        g.strokeStyle = 'rgba(0,0,0,0.65)';
        polyline(g, state.highlight.x, state.highlight.y);
        g.lineWidth = w;
        g.strokeStyle = state.highlight.color;
        polyline(g, state.highlight.x, state.highlight.y);
    }

    requestAnimationFrame(paint);
}

function paint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(baseCanvas, 0, 0);
    // Endpoint dots for the highlighted piece.
    if (state.highlight && state.highlight.x.length) {
        const xs = state.highlight.x, ys = state.highlight.y;
        drawDot(xs[0], ys[0], state.highlight.color);
        drawDot(xs[xs.length - 1], ys[ys.length - 1], state.highlight.color);
    }
}

function drawDot(x, y, color) {
    const p = toScreen(x, y);
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, 5 * state.dpr, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
}

/* --------------------------------------------------------------- events --- */

function bindEvents() {
    document.getElementById('world-select').addEventListener('change', e => {
        if (e.target.value) loadWorld(parseInt(e.target.value, 10));
    });
    document.getElementById('road-search').addEventListener('input', e => {
        state.roadFilter = e.target.value;
        renderRoadList();
    });
    document.getElementById('btn-view-filter').addEventListener('click', e => {
        state.viewportOnly = !state.viewportOnly;
        e.target.classList.toggle('active', state.viewportOnly);
        renderRoadList();
    });
    document.getElementById('btn-fit').addEventListener('click', () => {
        if (state.road && state.road.bounds) fitToBounds(state.road.bounds, 0.82);
        else if (state.world) fitToBounds(state.world.bounds);
    });
    document.getElementById('toggle-network').addEventListener('change', e => {
        state.showNetwork = e.target.checked;
        renderBase();
    });

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('mousedown', e => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => {
        if (dragging) { dragging = false; refreshViewFilter(); }
    });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        const dx = (e.clientX - lastX) * state.dpr;
        const dy = (e.clientY - lastY) * state.dpr;
        lastX = e.clientX; lastY = e.clientY;
        state.offsetX += dx; state.offsetY += dy;
        renderBase();
    });

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
        refreshViewFilter();
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
