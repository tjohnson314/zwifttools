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
    arch: null,
    spawn: null,
    browserMode: 'roads',
    worldRequest: 0,
    roadRequest: 0,
    roadFilter: '',
    viewportOnly: false, // list only roads visible in the current map view
    activeRoadId: null,
    loadingRoadId: null,
    highlight: null,    // { x:[], y:[], color } segment/marker overlay, or null
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    baseScale: 1,
    dpr: 1,
    showNetwork: true,
    showArches: true,
    showSpawns: true,
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
    new ResizeObserver(resizeCanvas).observe(canvas);
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
            const params = new URLSearchParams(window.location.search);
            const initial = data.worlds.find(world => String(world.mapID) === params.get('world')) || data.worlds[0];
            sel.value = initial.mapID;
            await loadWorld(initial.mapID);
            const arch = state.world?.arches.find(item => item.hash === params.get('arch'));
            if (arch) { setBrowserMode('arches'); selectArch(arch); }
            const spawn = state.world?.spawn_markers.find(item => item.name === params.get('spawn'));
            if (spawn) { setBrowserMode('spawns'); selectSpawn(spawn); }
        }
    } catch (e) {
        setStatus(e.message, true);
    }
}

async function loadWorld(mapId) {
    const request = ++state.worldRequest;
    ++state.roadRequest;
    setRoadLoading(null);
    setStatus('Loading world...');
    state.world = null;
    state.road = null;
    state.arch = null;
    state.spawn = null;
    state.activeRoadId = null;
    state.highlight = null;
    state.bgImage = null;
    document.getElementById('road-detail').style.display = 'none';
    document.getElementById('arch-detail').hidden = true;
    document.getElementById('spawn-detail').hidden = true;
    document.getElementById('arch-tooltip').hidden = true;
    renderRoadList();
    try {
        const responses = await Promise.all([
            fetch(`/api/surface_map_dev/world/${mapId}/roads`),
            fetch(`/api/surface_map_dev/world/${mapId}/timing-arches`),
        ]);
        const [data, timing] = await Promise.all(responses.map(response => response.json()));
        if (request !== state.worldRequest) return;
        if (!responses[0].ok) throw new Error(data.error || 'Failed to load world');
        if (!responses[1].ok) throw new Error(timing.error || 'Failed to load timing arches');
        state.world = { ...data, arches: timing.arches || [], spawn_markers: timing.spawn_markers || [] };
        buildLegend(data.colors);
        renderRoadList();
        fitToBounds(data.bounds);
        loadBackground(data.background);
        document.getElementById('map-hint').style.display = 'none';
        setStatus(`${data.roads.length} roads, ${timing.arches.length} directional timing entries, ${state.world.spawn_markers.length} spawn markers`);
    } catch (e) {
        if (request === state.worldRequest) setStatus(e.message, true);
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
    const world = state.world;
    if (!world) return;
    const request = ++state.roadRequest;
    setStatus('');
    setRoadLoading(roadId);
    try {
        const res = await fetch(`/api/surface_map_dev/world/${world.mapID}/road/${roadId}`);
        const data = await res.json();
        if (state.world !== world || request !== state.roadRequest) return;
        if (!res.ok) throw new Error(data.error || 'Failed to load road');
        state.arch = null;
        state.spawn = null;
        document.getElementById('arch-detail').hidden = true;
        document.getElementById('spawn-detail').hidden = true;
        state.road = data;
        state.activeRoadId = roadId;
        state.highlight = null;
        renderRoadDetail(data);
        renderRoadList();
        if (data.bounds) fitToBounds(data.bounds, 0.82);
        else renderBase();
        setStatus('');
    } catch (e) {
        if (state.world === world && request === state.roadRequest) setStatus(e.message, true);
    } finally {
        if (state.world === world && request === state.roadRequest) setRoadLoading(null);
    }
}

function clearRoad() {
    ++state.roadRequest;
    setRoadLoading(null);
    state.road = null;
    state.arch = null;
    state.spawn = null;
    state.activeRoadId = null;
    state.highlight = null;
    document.getElementById('road-detail').style.display = 'none';
    document.getElementById('arch-detail').hidden = true;
    document.getElementById('spawn-detail').hidden = true;
    renderRoadList();
    if (state.world) fitToBounds(state.world.bounds);
}

/* ------------------------------------------------------------- sidebar --- */

function setRoadLoading(roadId) {
    state.loadingRoadId = roadId;
    for (const item of document.getElementById('road-list').children) {
        item.setAttribute('aria-busy', String(roadId != null && item.dataset.roadId === String(roadId)));
    }
}

function renderRoadList() {
    const ul = document.getElementById('road-list');
    ul.innerHTML = '';
    document.getElementById('arch-list').innerHTML = '';
    document.getElementById('spawn-list').innerHTML = '';
    if (state.browserMode === 'arches') { renderArchList(); return; }
    if (state.browserMode === 'spawns') { renderSpawnList(); return; }
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
        li.dataset.roadId = String(r.id);
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
    setRoadLoading(state.loadingRoadId);
}

function setBrowserMode(mode) {
    state.browserMode = mode;
    for (const [value, tab, list] of [['roads', 'tab-roads', 'road-list'], ['arches', 'tab-arches', 'arch-list'], ['spawns', 'tab-spawns', 'spawn-list']]) {
        document.getElementById(tab).setAttribute('aria-selected', String(mode === value));
        document.getElementById(list).hidden = mode !== value;
    }
    document.getElementById('browser-heading').textContent = { roads: 'Roads', arches: 'Timing Arches', spawns: 'Spawn Markers' }[mode];
    const search = document.getElementById('road-search');
    search.placeholder = mode === 'spawns' ? 'Search name or road...' : mode === 'arches' ? 'Search name, hash or road...' : 'Search by id, name or surface...';
    search.setAttribute('aria-label', `Search ${document.getElementById('browser-heading').textContent.toLowerCase()}`);
    renderRoadList();
}

function archBounds(arch) {
    const points = [arch.start, arch.finish].filter(Boolean);
    if (!points.length) return null;
    return { min_x: Math.min(...points.map(point => point.x)) - 30,
        max_x: Math.max(...points.map(point => point.x)) + 30,
        min_y: Math.min(...points.map(point => point.y)) - 30,
        max_y: Math.max(...points.map(point => point.y)) + 30 };
}

function renderArchList() {
    if (!state.world) return;
    const filter = state.roadFilter.toLowerCase();
    const bounds = viewportWorldBounds();
    const arches = state.world.arches.filter(arch => {
        const matches = [arch.name, arch.entity_name, arch.hash, arch.road_id, arch.direction]
            .some(value => String(value ?? '').toLowerCase().includes(filter));
        const visible = [arch.start, arch.finish].some(point => point &&
            point.x >= bounds.min_x && point.x <= bounds.max_x && point.y >= bounds.min_y && point.y <= bounds.max_y);
        return matches && (!state.viewportOnly || visible);
    });
    document.getElementById('road-count').textContent = `${arches.length}/${state.world.arches.length}`;
    const list = document.getElementById('arch-list');
    arches.forEach(arch => {
        const item = document.createElement('li');
        item.classList.toggle('active', state.arch?.id === arch.id);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'arch-row';
        const missing = !arch.start || !arch.finish;
        button.innerHTML = `<span class="road-id">${arch.direction}</span> ${escapeHtml(arch.name)}` +
            `<span class="r-meta">Road ${arch.road_id ?? '?'} &middot; ${escapeHtml(arch.hash ?? 'No hash')}</span>` +
            (missing ? '<span class="r-meta arch-unlocated">Unlocated boundary</span>' : '');
        button.addEventListener('click', () => selectArch(arch));
        item.appendChild(button);
        list.appendChild(item);
    });
}

function selectArch(arch) {
    ++state.roadRequest;
    setRoadLoading(null);
    state.arch = arch;
    state.spawn = null;
    state.road = null;
    state.activeRoadId = null;
    state.highlight = null;
    document.getElementById('road-detail').style.display = 'none';
    document.getElementById('spawn-detail').hidden = true;
    document.getElementById('arch-detail').hidden = false;
    document.getElementById('arch-name').textContent = `${arch.name} (${arch.direction})`;
    const status = document.getElementById('arch-location-status');
    const missing = ['start', 'finish'].filter(boundary => !arch[boundary]);
    status.textContent = missing.length ? `Unlocated: ${missing.join(', ')}` : 'Start and finish located';
    status.classList.toggle('incomplete', !!missing.length);
    const coordinates = point => point ? `${point.local_x}, ${point.local_z}` : 'Unlocated';
    const fields = [
        ['Entity', arch.entity_name], ['Hash', arch.hash], ['Road', arch.road_id],
        ['Direction', arch.direction], ['Start road time', arch.start_time], ['Finish road time', arch.end_time],
        ['Declared length (m)', arch.declared_distance_m], ['Position source', arch.position_source],
        ['Start source route', arch.start?.source_route], ['Finish source route', arch.finish?.source_route],
        ['Start local x, z (m)', coordinates(arch.start)], ['Finish local x, z (m)', coordinates(arch.finish)],
    ];
    document.getElementById('arch-fields').innerHTML = fields.map(([key, value]) =>
        `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value ?? 'Not specified'))}</dd>`).join('');
    document.getElementById('arch-xml').textContent = arch.xml;
    renderRoadList();
    const bounds = archBounds(arch);
    if (bounds) fitToBounds(bounds, 0.78);
    else renderBase();
    setStatus('');
}

function spawnBounds(spawn) {
    if (!spawn?.position) return null;
    const road = state.world?.roads.find(item => item.id === spawn.road_id);
    const xs = [...(road?.x || []), spawn.position.x];
    const ys = [...(road?.y || []), spawn.position.y];
    return { min_x: Math.min(...xs) - 30, max_x: Math.max(...xs) + 30,
        min_y: Math.min(...ys) - 30, max_y: Math.max(...ys) + 30 };
}

function renderSpawnList() {
    if (!state.world) return;
    const filter = state.roadFilter.toLowerCase();
    const bounds = viewportWorldBounds();
    const markers = state.world.spawn_markers.filter(marker => {
        const matches = [marker.name, marker.label, marker.road_id]
            .some(value => String(value ?? '').toLowerCase().includes(filter));
        const point = marker.position;
        const visible = point && point.x >= bounds.min_x && point.x <= bounds.max_x &&
            point.y >= bounds.min_y && point.y <= bounds.max_y;
        return matches && (!state.viewportOnly || visible);
    });
    document.getElementById('road-count').textContent = `${markers.length}/${state.world.spawn_markers.length}`;
    const list = document.getElementById('spawn-list');
    for (const marker of markers) {
        const item = document.createElement('li');
        item.classList.toggle('active', state.spawn?.id === marker.id);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'arch-row';
        const location = !marker.position ? 'Unlocated' : marker.position_source === 'road_geometry_midpoint' ? 'Road anchor' : 'Checkpoint-derived position';
        button.innerHTML = `${escapeHtml(marker.name)}<span class="r-meta">Road ${marker.road_id ?? '?'} &middot; ${escapeHtml(location)}</span>`;
        button.addEventListener('click', () => selectSpawn(marker));
        item.appendChild(button);
        list.appendChild(item);
    }
}

function selectSpawn(spawn) {
    ++state.roadRequest;
    setRoadLoading(null);
    state.spawn = spawn;
    state.arch = null;
    state.road = null;
    state.activeRoadId = null;
    const road = state.world.roads?.find(item => item.id === spawn.road_id);
    state.highlight = road ? { x: road.x, y: road.y, color: '#55c9ff' } : null;
    document.getElementById('road-detail').style.display = 'none';
    document.getElementById('arch-detail').hidden = true;
    document.getElementById('spawn-detail').hidden = false;
    document.getElementById('spawn-name').textContent = spawn.name;
    const locationStatus = document.getElementById('spawn-location-status');
    locationStatus.textContent = !spawn.position ? 'Unlocated' :
        spawn.position_source === 'road_geometry_midpoint' ? 'Road midpoint anchor; exact spawn position unlocated' : 'Checkpoint-derived position';
    locationStatus.classList.toggle('incomplete', spawn.position_source !== 'native_route_checkpoint_interpolation');
    const fields = [['Label', spawn.label], ['Road', spawn.road_id], ['Road time', spawn.road_time],
        ['Position source', spawn.position_source], ['Source route', spawn.position?.source_route],
        ['Local x, z (m)', spawn.position ? `${spawn.position.local_x}, ${spawn.position.local_z}` : 'Unlocated']];
    document.getElementById('spawn-fields').innerHTML = fields.map(([key, value]) =>
        `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value ?? 'Not specified'))}</dd>`).join('');
    document.getElementById('spawn-xml').textContent = spawn.xml;
    document.getElementById('spawn-road').disabled = spawn.road_id == null;
    renderRoadList();
    const bounds = spawnBounds(spawn);
    if (bounds) fitToBounds(bounds, 0.78);
    else renderBase();
    setStatus('');
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
        if (state.spawn && spawnBounds(state.spawn)) fitToBounds(spawnBounds(state.spawn), 0.78);
        else if (state.arch && archBounds(state.arch)) fitToBounds(archBounds(state.arch), 0.78);
        else if (state.road && state.road.bounds) fitToBounds(state.road.bounds, 0.82);
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

    const b = state.world.bounds;
    state.baseScale = Math.min(canvas.width / Math.max(b.max_x - b.min_x, 1),
                               canvas.height / Math.max(b.max_y - b.min_y, 1)) * 0.94;
    const zoomK = state.baseScale ? state.scale / state.baseScale : 1;

    // Faint full road network for context.
    if (state.showNetwork) {
        g.strokeStyle = 'rgba(150,160,175,0.35)';
        g.lineWidth = mapStrokeWidth(1.5, zoomK, state.dpr);
        state.world.roads.forEach(r => {
            if (r.id !== state.activeRoadId && r.x.length > 1) polyline(g, r.x, r.y);
        });
    }

    // Selected road in orange with a dark casing.
    if (state.road && state.road.full && state.road.full.x.length > 1) {
        const w = mapStrokeWidth(3.5, zoomK, state.dpr);
        g.lineWidth = w + 2 * state.dpr;
        g.strokeStyle = 'rgba(0,0,0,0.55)';
        polyline(g, state.road.full.x, state.road.full.y);
        g.lineWidth = w;
        g.strokeStyle = '#f7931e';
        polyline(g, state.road.full.x, state.road.full.y);
    }

    // Selected segment / marker highlight.
    if (state.highlight && state.highlight.x.length > 1) {
        const w = mapStrokeWidth(6, zoomK, state.dpr);
        g.lineWidth = w + 2 * state.dpr;
        g.strokeStyle = 'rgba(0,0,0,0.65)';
        polyline(g, state.highlight.x, state.highlight.y);
        g.lineWidth = w;
        g.strokeStyle = state.highlight.color;
        polyline(g, state.highlight.x, state.highlight.y);
    }

    if (state.showArches) drawArchMarkers(g);
    if (state.showSpawns) drawSpawnMarkers(g);
    requestAnimationFrame(paint);
}

function drawSpawnMarkers(g) {
    const markers = (state.world.spawn_markers || []).filter(marker => marker.position);
    const labels = [];
    for (const marker of markers) {
        const screen = toScreen(marker.position.x, marker.position.y);
        if (screen.sx < 0 || screen.sy < 0 || screen.sx > canvas.width || screen.sy > canvas.height) continue;
        g.beginPath();
        g.arc(screen.sx, screen.sy, (marker.id === state.spawn?.id ? 7 : 4) * state.dpr, 0, Math.PI * 2);
        g.fillStyle = marker.position_source === 'road_geometry_midpoint' ? '#111922' : '#55c9ff';
        g.strokeStyle = '#55c9ff';
        g.lineWidth = 2 * state.dpr;
        g.fill();
        g.stroke();
    }
    for (const marker of [...markers].sort((first, last) => Number(last.id === state.spawn?.id) - Number(first.id === state.spawn?.id))) {
        const screen = toScreen(marker.position.x, marker.position.y);
        if (screen.sx < 0 || screen.sy < 0 || screen.sx > canvas.width || screen.sy > canvas.height) continue;
        g.font = `${12 * state.dpr}px sans-serif`;
        let width = g.measureText(marker.name).width;
        if (width > canvas.width - 12 * state.dpr) {
            g.font = `${12 * state.dpr * (canvas.width - 12 * state.dpr) / width}px sans-serif`;
            width = g.measureText(marker.name).width;
        }
        const left = Math.max(6 * state.dpr, Math.min(screen.sx + 12 * state.dpr, canvas.width - width - 6 * state.dpr));
        const top = Math.max(16 * state.dpr, Math.min(screen.sy - 12 * state.dpr, canvas.height - 6 * state.dpr));
        const box = { left, right: left + width, top: top - 14 * state.dpr, bottom: top + 4 * state.dpr };
        if (labels.some(other => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top)) continue;
        labels.push(box);
        g.strokeStyle = '#111922';
        g.lineWidth = 4 * state.dpr;
        g.strokeText(marker.name, left, top);
        g.fillStyle = '#55c9ff';
        g.fillText(marker.name, left, top);
    }
}

function pickRoad(clientX, clientY) {
    if (!state.world) return null;
    const rect = canvas.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.left + rect.width ||
        clientY < rect.top || clientY > rect.top + rect.height) return null;
    const point = toWorld((clientX - rect.left) * state.dpr, (clientY - rect.top) * state.dpr);
    let closest = (8 * state.dpr / state.scale) ** 2;
    let hit = null;
    for (const road of state.world.roads) {
        const path = road.id === state.activeRoadId ? state.road?.full :
            state.showNetwork ? road : state.spawn?.road_id === road.id ? state.highlight : null;
        if (!path) continue;
        for (let index = 1; index < path.x.length; index++) {
            const startX = path.x[index - 1], startY = path.y[index - 1];
            const dx = path.x[index] - startX, dy = path.y[index] - startY;
            const length2 = dx * dx + dy * dy;
            const fraction = length2 ? Math.max(0, Math.min(1,
                ((point.x - startX) * dx + (point.y - startY) * dy) / length2)) : 0;
            const distance = (point.x - startX - fraction * dx) ** 2 +
                (point.y - startY - fraction * dy) ** 2;
            if (distance < closest || (distance === closest && road.id === state.activeRoadId)) {
                closest = distance;
                hit = road;
            }
        }
    }
    return hit;
}

function pickSpawn(clientX, clientY) {
    if (!state.world || !state.showSpawns) return null;
    const rect = canvas.getBoundingClientRect();
    const screenX = (clientX - rect.left) * state.dpr;
    const screenY = (clientY - rect.top) * state.dpr;
    let closest = 11 * state.dpr;
    let hit = null;
    for (const marker of state.world.spawn_markers || []) {
        if (!marker.position) continue;
        const screen = toScreen(marker.position.x, marker.position.y);
        const distance = Math.hypot(screen.sx - screenX, screen.sy - screenY);
        if (distance < closest) { closest = distance; hit = marker; }
    }
    return hit;
}

function sharedTimingFinish(arch) {
    if (!arch.finish) return null;
    return (state.world.arches || []).find(other => other.id !== arch.id &&
        other.id.split(':')[0] === arch.id.split(':')[0] && other.direction !== arch.direction &&
        other.road_id === arch.road_id && other.end_time === arch.end_time && other.finish &&
        other.finish.x === arch.finish.x && other.finish.y === arch.finish.y) || null;
}

function drawArchMarkers(g) {
    const arches = [...(state.world.arches || []).filter(arch => arch.id !== state.arch?.id),
        ...(state.arch ? [state.arch] : [])];
    for (const arch of arches) {
        const selected = arch.id === state.arch?.id;
        for (const boundary of ['start', 'finish']) {
            const point = arch[boundary];
            if (!point) continue;
            const peer = boundary === 'finish' ? sharedTimingFinish(arch) : null;
            if (peer && (peer.id === state.arch?.id || (!selected && arch.direction === 'R'))) continue;
            const screen = toScreen(point.x, point.y);
            if (screen.sx < -20 || screen.sy < -20 || screen.sx > canvas.width + 20 || screen.sy > canvas.height + 20) continue;
            const color = boundary === 'start' ? '#64e3bb' : '#ffffff';
            const radius = (selected ? (boundary === 'start' ? 8 : 5) : (boundary === 'start' ? 4 : 3)) * state.dpr;
            g.beginPath();
            g.arc(screen.sx, screen.sy, radius, 0, Math.PI * 2);
            g.fillStyle = color;
            g.strokeStyle = '#111922';
            g.lineWidth = 2 * state.dpr;
            g.fill();
            g.stroke();
            if (selected) {
                const name = peer && arch.direction === 'R' ? peer.name : arch.name;
                const text = peer ? `${name} finish` : `${name} (${arch.direction}) ${boundary}`;
                g.font = `${12 * state.dpr}px sans-serif`;
                const width = g.measureText(text).width;
                const left = Math.max(6 * state.dpr, Math.min(screen.sx + 12 * state.dpr, canvas.width - width - 6 * state.dpr));
                const top = Math.max(16 * state.dpr, Math.min(screen.sy + (boundary === 'start' ? -14 : 24) * state.dpr, canvas.height - 6 * state.dpr));
                g.strokeStyle = '#111922';
                g.lineWidth = 4 * state.dpr;
                g.strokeText(text, left, top);
                g.fillStyle = color;
                g.fillText(text, left, top);
            }
        }
    }
}

function pickArch(clientX, clientY) {
    if (!state.world || !state.showArches) return null;
    const rect = canvas.getBoundingClientRect();
    const screenX = (clientX - rect.left) * state.dpr;
    const screenY = (clientY - rect.top) * state.dpr;
    let closest = 11 * state.dpr;
    let hit = null;
    const arches = state.arch ? [state.arch, ...state.world.arches.filter(arch => arch.id !== state.arch.id)] : state.world.arches;
    for (const arch of arches) {
        for (const boundary of ['start', 'finish']) {
            const point = arch[boundary];
            if (!point) continue;
            const screen = toScreen(point.x, point.y);
            const distance = Math.hypot(screen.sx - screenX, screen.sy - screenY);
            if (distance < closest) { closest = distance; hit = { arch, boundary }; }
        }
    }
    return hit;
}

function showArchTooltip(event) {
    const tooltip = document.getElementById('arch-tooltip');
    const spawn = pickSpawn(event.clientX, event.clientY);
    const hit = pickArch(event.clientX, event.clientY);
    tooltip.hidden = !spawn && !hit;
    if (spawn) {
        tooltip.textContent = `${spawn.name}\nRoad ${spawn.road_id}${spawn.position_source === 'road_geometry_midpoint' ? ' - road midpoint anchor' : ''}`;
    } else if (hit) {
        const peer = hit.boundary === 'finish' ? sharedTimingFinish(hit.arch) : null;
        const name = peer && hit.arch.direction === 'R' ? peer.name : hit.arch.name;
        tooltip.textContent = `${name}${peer ? '' : ` (${hit.arch.direction})`}\n${hit.boundary} - road ${hit.arch.road_id}`;
    } else return;
    const rect = canvas.getBoundingClientRect();
    tooltip.style.left = `${Math.max(8, Math.min(event.clientX - rect.left + 12, rect.width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(event.clientY - rect.top + 12, rect.height - tooltip.offsetHeight - 8))}px`;
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
    document.getElementById('tab-roads').addEventListener('click', () => setBrowserMode('roads'));
    document.getElementById('tab-arches').addEventListener('click', () => setBrowserMode('arches'));
    document.getElementById('tab-spawns').addEventListener('click', () => setBrowserMode('spawns'));
    document.getElementById('spawn-road').addEventListener('click', () => {
        if (state.spawn?.road_id != null) { const roadId = state.spawn.road_id; setBrowserMode('roads'); loadRoad(roadId); }
    });
    document.getElementById('toggle-spawns').addEventListener('change', event => {
        state.showSpawns = event.target.checked;
        document.getElementById('arch-tooltip').hidden = true;
        renderBase();
    });
    document.getElementById('toggle-arches').addEventListener('change', event => {
        state.showArches = event.target.checked;
        document.getElementById('arch-tooltip').hidden = true;
        renderBase();
    });
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
        if (state.spawn && spawnBounds(state.spawn)) fitToBounds(spawnBounds(state.spawn), 0.78);
        else if (state.arch && archBounds(state.arch)) fitToBounds(archBounds(state.arch), 0.78);
        else if (state.road && state.road.bounds) fitToBounds(state.road.bounds, 0.82);
        else if (state.world) fitToBounds(state.world.bounds);
    });
    document.getElementById('toggle-network').addEventListener('change', e => {
        state.showNetwork = e.target.checked;
        renderBase();
    });

    let dragging = false, moved = false, lastX = 0, lastY = 0, startX = 0, startY = 0;
    canvas.addEventListener('pointerdown', e => {
        if (e.button !== 0) return;
        moved = false; startX = e.clientX; startY = e.clientY;
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
        document.getElementById('arch-tooltip').hidden = true;
    });
    window.addEventListener('pointerup', e => {
        if (!dragging) return;
        dragging = false;
        if (moved) refreshViewFilter();
        else {
            const spawn = pickSpawn(e.clientX, e.clientY);
            if (spawn) { setBrowserMode('spawns'); selectSpawn(spawn); }
            else {
                const hit = pickArch(e.clientX, e.clientY);
                if (hit) { setBrowserMode('arches'); selectArch(hit.arch); }
                else {
                    const road = pickRoad(e.clientX, e.clientY);
                    if (road) { setBrowserMode('roads'); loadRoad(road.id); }
                }
            }
        }
    });
    canvas.addEventListener('pointerleave', () => { document.getElementById('arch-tooltip').hidden = true; });
    window.addEventListener('pointercancel', () => { dragging = false; });
    window.addEventListener('pointermove', e => {
        if (!dragging) { if (e.target === canvas) showArchTooltip(e); return; }
        moved = moved || Math.abs(e.clientX - startX) + Math.abs(e.clientY - startY) > 4;
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
