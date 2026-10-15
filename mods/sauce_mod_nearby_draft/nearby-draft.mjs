import * as Common from '/pages/src/common.mjs';
import {MODEL} from './bike-model.mjs';
import * as Surface from './surface.mjs';
import {DEFAULTS, DraftTracker, PositionInterpolator, positionOffset,
    selectDisplayedRiders, riderInitials, donutSegments} from './draft-model.mjs';

Common.enableSentry();

let settings;
let self;
let nearby = [];
let mapEl, canvas, emptyEl, countEl, detailsEl, statusEl, latencyEl;
let tracker = new DraftTracker();
const positions = new PositionInterpolator();
const markers = new Map();
const roadCache = new Map();
let currentSelf;
let selected = [];
let actualRiderCount = 0;
let displayedWorldTime = null;
let selectedId;
let zoom = 1;
let drawPending = false;
let lastDraw = -Infinity;
const FRAME_INTERVAL = 1000 / 30;

function loadSettings() {
    settings = {...DEFAULTS};
    for (const key of Object.keys(DEFAULTS)) {
        const value = Common.settingsStore.get(key);
        if (value !== undefined) settings[key] = value;
    }
    if (!Number.isFinite(settings.radiusM) || settings.radiusM <= 0) settings.radiusM = DEFAULTS.radiusM;
    if (!Number.isInteger(settings.maxRiders) || settings.maxRiders < 0) settings.maxRiders = DEFAULTS.maxRiders;
}

export function main() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();
    mapEl = document.querySelector('.draft-map');
    canvas = document.querySelector('.road-map');
    emptyEl = document.querySelector('.empty');
    countEl = document.querySelector('.rider-count');
    detailsEl = document.querySelector('.rider-details');
    statusEl = document.querySelector('.map-status');
    latencyEl = document.querySelector('.position-latency');
    const latencyTimer = setInterval(() => updateLatency(), 1000);
    globalThis.addEventListener('pagehide', () => clearInterval(latencyTimer), {once: true});
    new ResizeObserver(requestDraw).observe(mapEl);
    for (const button of document.querySelectorAll('[data-zoom]')) {
        button.addEventListener('click', () => {
            zoom = button.dataset.zoom === 'reset' ? 1 :
                Math.max(1, Math.min(8, zoom * Number(button.dataset.zoom)));
            requestDraw();
        });
    }
    mapEl.addEventListener('wheel', event => {
        event.preventDefault();
        zoom = Math.max(1, Math.min(8, zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
        requestDraw();
    }, {passive: false});
    Common.settingsStore.addEventListener('changed', () => {
        loadSettings();
        tracker = new DraftTracker();
        refresh();
    });
    Common.subscribe('athlete/self', data => {
        if (self && String(self.athleteId) !== String(data?.athleteId)) tracker = new DraftTracker();
        self = data;
        refresh();
    });
    Common.subscribe('nearby', data => {
        nearby = Array.isArray(data) ? data : [];
        refresh();
    });
}

function refresh() {
    const nearbySelf = nearby.find(entry => self?.athleteId != null && String(entry.athleteId) === String(self.athleteId));
    const latest = nearbySelf?.state?.worldTime > (self?.state?.worldTime ?? -Infinity) ? nearbySelf : self;
    currentSelf = latest ? {...latest, athlete: {...self?.athlete, ...nearbySelf?.athlete}} : null;
    const selection = selectDisplayedRiders(nearby, currentSelf, settings);
    selected = selection.riders;
    actualRiderCount = selection.actual;
    const ids = new Set(selected.map(entry => String(entry.athleteId)));
    tracker.retain([...ids]);
    positions.retain([...ids]);
    const now = performance.now();
    for (const [id, marker] of markers) {
        if (!ids.has(id)) {
            marker.element.remove();
            markers.delete(id);
        }
    }
    for (const entry of selected) {
        const id = String(entry.athleteId);
        let marker = markers.get(id);
        if (!marker) {
            marker = createMarker(id);
            markers.set(id, marker);
            mapEl.appendChild(marker.element);
        }
        positions.update(id, entry.state, now);
        Surface.preload(entry.state.courseId);
        const {style} = Surface.lookupStyle(entry.state.courseId, entry.state.roadId, entry.state.roadTime);
        marker.entry = entry;
        marker.summary = tracker.update(entry, settings, style);
        renderMarker(marker);
    }
    if (!ids.has(selectedId)) selectedId = currentSelf?.athleteId == null ? null : String(currentSelf.athleteId);
    document.querySelector('.radius-label').textContent = `${settings.radiusM} m radius`;
    mapEl.setAttribute('aria-label', `Rider positions within ${settings.radiusM} metres`);
    document.querySelector('[data-zoom="reset"]').title = `Show full ${settings.radiusM} m radius`;
    loadRoads(currentSelf?.state?.courseId);
    renderDetails();
    requestDraw();
}

function createMarker(id) {
    const element = document.createElement('button');
    element.type = 'button';
    element.className = 'rider-marker';
    element.hidden = true;
    element.innerHTML = `
        <canvas class="donut" aria-hidden="true"></canvas>
        <span class="initials"></span><span class="signed-flag" aria-hidden="true">!</span>`;
    const select = () => { selectedId = id; renderDetails(); };
    element.addEventListener('pointerenter', select);
    element.addEventListener('focus', select);
    element.addEventListener('click', select);
    return {element, canvas: element.querySelector('canvas'), initials: element.querySelector('.initials')};
}

function renderMarkerLabel(marker) {
    const {entry, summary, element} = marker;
    const segments = donutSegments(summary);
    const name = entry.athlete?.sanitizedFullname || `Rider ${entry.athleteId}`;
    const distance = marker.displayDistance ?? entry.distance;
    const label = `${name}${entry.isSelf ? ' (you)' : ''}: ${distance.toFixed(1)} m; ` +
        `effective ${summary.effective.toFixed(0)} W; wasted ${summary.residual.toFixed(0)} W; ` +
        `remaining ${segments.values[2].toFixed(0)} W; max draft ${summary.maxDraft.toFixed(0)} W` +
        (segments.signed ? '. Signed values; dashed segments are negative.' : '');
    element.title = label;
    element.setAttribute('aria-label', label);
}

function renderMarker(marker) {
    const {entry, summary, element} = marker;
    const segments = donutSegments(summary);
    renderMarkerLabel(marker);
    element.classList.toggle('self', entry.isSelf);
    element.classList.toggle('signed', segments.signed);
    marker.initials.textContent = riderInitials(entry.athlete, entry.athleteId);
    const ratio = devicePixelRatio || 1;
    marker.canvas.width = marker.canvas.height = Math.round(36 * ratio);
    const context = marker.canvas.getContext('2d');
    context.scale(ratio, ratio);
    context.lineWidth = 5;
    const colors = ['#45a86e', '#4c8fd6', '#737b80'];
    let angle = -Math.PI / 2;
    segments.fractions.forEach((fraction, index) => {
        if (fraction <= 0) return;
        const end = angle + fraction * Math.PI * 2;
        context.beginPath();
        context.strokeStyle = colors[index];
        context.setLineDash(segments.values[index] < 0 ? [2, 2] : []);
        context.arc(18, 18, 14.5, angle, end);
        context.stroke();
        angle = end;
    });
}

function renderDetails() {
    const marker = markers.get(selectedId);
    for (const [id, value] of markers) value.element.classList.toggle('selected', id === selectedId);
    detailsEl.hidden = !marker;
    if (!marker) return;
    const {entry, summary} = marker;
    detailsEl.querySelector('.detail-name').textContent =
        `${entry.athlete?.sanitizedFullname || `Rider ${entry.athleteId}`}${entry.isSelf ? ' (you)' : ''}`;
    detailsEl.querySelector('.detail-distance').textContent = `${(marker.displayDistance ?? entry.distance).toFixed(1)} m`;
    detailsEl.querySelector('.effective-value').textContent = `${summary.effective.toFixed(0)} W`;
    detailsEl.querySelector('.wasted-value').textContent = `${summary.residual.toFixed(0)} W`;
    detailsEl.querySelector('.remaining-value').textContent = `${donutSegments(summary).values[2].toFixed(0)} W`;
}

async function loadRoads(courseId) {
    if (courseId == null || roadCache.has(courseId)) return;
    roadCache.set(courseId, {loading: true, paths: []});
    requestDraw();
    try {
        const roads = await Common.getRoads(courseId);
        const paths = (roads || []).map(road => road.path || []).filter(points => points.length > 1).map(points => {
            const path = new Path2D();
            points.forEach((point, index) => {
                if (index === 0) path.moveTo(point[0], point[1]);
                else path.lineTo(point[0], point[1]);
            });
            return path;
        });
        roadCache.set(courseId, {paths});
    } catch (error) {
        roadCache.set(courseId, {paths: [], error: true});
    }
    requestDraw();
}

function requestDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(now => {
        drawPending = false;
        if (now - lastDraw < FRAME_INTERVAL - 0.5) {
            requestDraw();
            return;
        }
        lastDraw = now;
        drawMap(now);
        if (positions.isAnimating(now, currentSelf?.athleteId)) requestDraw();
    });
}

function updateLatency(now = performance.now()) {
    const latency = positions.latencyMs(now, displayedWorldTime);
    latencyEl.textContent = latency == null ? 'Latency: --' : `Latency: ${(latency / 1000).toFixed(1)} s`;
}

function drawMap(now) {
    const {width, height} = mapEl.getBoundingClientRect();
    const ratio = devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const scale = Math.max(1, Math.min(width, height) - 52) / (2 * settings.radiusM) * zoom;
    const centerX = width / 2;
    const centerY = height / 2;
    context.strokeStyle = '#ffffff0b';
    context.lineWidth = 1;
    const rulerM = settings.radiusM / 5;
    const spacing = rulerM * scale;
    context.beginPath();
    for (let column = centerX % spacing; column < width; column += spacing) {context.moveTo(column, 0); context.lineTo(column, height);}
    for (let row = centerY % spacing; row < height; row += spacing) {context.moveTo(0, row); context.lineTo(width, row);}
    context.stroke();
    const timeline = positions.timeline(now, currentSelf?.athleteId);
    const origin = positions.sample(currentSelf?.athleteId, now, timeline);
    const roads = roadCache.get(origin?.courseId);
    const native = Number.isFinite(origin?.x) && Number.isFinite(origin?.y);
    statusEl.textContent = roads?.loading ? 'Loading roads...' : roads?.error ? 'Road map unavailable' : '';
    if (native) {
        context.save();
        context.translate(centerX, centerY);
        context.scale(scale / 100, scale / 100);
        context.translate(-origin.x, -origin.y);
        context.strokeStyle = '#9aabb340';
        context.lineWidth = Math.max(3, 4 * scale) * 100 / scale;
        context.lineCap = 'round';
        context.lineJoin = 'round';
        for (const path of roads?.paths || []) context.stroke(path);
        context.restore();
    }
    context.beginPath();
    context.strokeStyle = '#aab9bf66';
    context.lineWidth = 1;
    context.setLineDash([4, 6]);
    context.arc(centerX, centerY, settings.radiusM * scale, 0, Math.PI * 2);
    context.stroke();
    context.setLineDash([]);
    const ruler = document.querySelector('.map-scale');
    ruler.style.width = `${rulerM * scale}px`;
    ruler.textContent = `${Number(rulerM.toFixed(1))} m`;
    let displayed = 0;
    for (const marker of markers.values()) {
        const state = positions.sample(marker.entry.athleteId, now, timeline);
        const position = positionOffset(origin, state);
        marker.element.hidden = !position;
        if (!position) continue;
        displayed++;
        marker.displayDistance = Math.hypot(position.x, position.y);
        renderMarkerLabel(marker);
        if (String(marker.entry.athleteId) === selectedId) {
            detailsEl.querySelector('.detail-distance').textContent = `${marker.displayDistance.toFixed(1)} m`;
        }
        marker.element.style.left = `${centerX + position.x * scale}px`;
        marker.element.style.top = `${centerY + position.y * scale}px`;
    }
    countEl.textContent = `${displayed}/${actualRiderCount} riders`;
    displayedWorldTime = displayed > 0 ? timeline.time : null;
    updateLatency(now);
    emptyEl.hidden = displayed > 0;
    emptyEl.textContent = selected.length ? 'Synchronizing rider positions...' :
        self?.state ? 'Waiting for rider positions...' : 'Waiting for rider data...';
}

export function settingsMain() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();
    const framePicker = document.querySelector('[data-set="frameId"]');
    const wheelPicker = document.querySelector('[data-set="wheelId"]');
    for (const frame of MODEL.frames) framePicker.add(new Option(frame.name, frame.id));
    wheelPicker.add(new Option('Frame built-in wheels', ''));
    for (const wheel of MODEL.wheels) wheelPicker.add(new Option(`${wheel.make} ${wheel.model}`.trim(), wheel.id));
    document.querySelector('form').addEventListener('submit', event => event.preventDefault());
    for (const element of document.querySelectorAll('[data-set]')) {
        const key = element.dataset.set;
        if (element.type === 'checkbox') element.checked = settings[key];
        else element.value = settings[key];
        element.addEventListener('change', () => {
            if (!element.checkValidity()) {
                element.reportValidity();
                return;
            }
            const value = element.type === 'checkbox' ? element.checked :
                element.type === 'number' || key === 'upgradeLevel' ? Number(element.value) : element.value;
            settings[key] = value;
            Common.settingsStore.set(key, value);
        });
    }
}