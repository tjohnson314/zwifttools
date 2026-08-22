/**
 * TT Pacing Planner — frontend logic.
 *
 * Loads routes, then POSTs rider/bike/route inputs to /api/tt_pacing_plan and
 * renders the optimal power-vs-distance plan (Chart.js) plus a section table.
 */

'use strict';

let allRoutes = [];       // full list from /api/tt_pacing/routes
let bikeDatabase = null;  // { frames, wheels, bikes } from /api/bike_database
let selectedRoute = null; // currently selected route object
let planChart = null;     // Chart.js instance
let lastPlanData = null;  // most recent plan response, for re-rendering on unit change
let serverDividerIdx = []; // profile-point indices of the current buckets' dividers
let bucketSliderTimer = null;
let laps = 1;             // number of laps for looped routes
let routeSlugMap = {};    // slug → route object, for URL deep-linking

let routeGeometry = null; // { key, leadin, route, combined, background, bounds, stitchGap }
let mapView = null;       // { canvas, ctx, dpr, project() } cached map transform
let mapSelectionKm = null;// currently highlighted distance (km) on the map
let mapDividerKm = [];    // smart-bucket divider distances (km) to mark on the map

// ── Units ────────────────────────────────────────────────────────────────────
// The API always works in metric; imperial is a display/input-only conversion.
const KG_TO_LB = 2.2046226;
const CM_TO_IN = 0.3937008;
const KM_TO_MI = 0.6213712;
const M_TO_FT  = 3.2808399;

let currentUnit = 'metric'; // 'metric' | 'imperial'

function isImperial() { return currentUnit === 'imperial'; }

function fmtDistance(km, dp = 1) {
    return isImperial()
        ? (km * KM_TO_MI).toFixed(dp) + ' mi'
        : km.toFixed(dp) + ' km';
}

function fmtElevation(m) {
    return isImperial()
        ? Math.round(m * M_TO_FT) + ' ft'
        : Math.round(m) + ' m';
}

function fmtSpeed(kph, dp = 1) {
    return isImperial()
        ? (kph * KM_TO_MI).toFixed(dp) + ' mph'
        : kph.toFixed(dp) + ' km/h';
}

function fmtWeight(kg, dp = 2) {
    return isImperial()
        ? (kg * KG_TO_LB).toFixed(dp) + ' lb'
        : kg.toFixed(dp) + ' kg';
}

// Rider inputs are shown in the active unit but the API needs metric.
function getWeightKg() {
    const v = parseFloat(document.getElementById('riderWeight').value);
    return isImperial() ? v / KG_TO_LB : v;
}

function getHeightCm() {
    const v = parseFloat(document.getElementById('riderHeight').value);
    return isImperial() ? v / CM_TO_IN : v;
}

function setUnits(unit) {
    if (unit === currentUnit) return;
    convertInputs(unit);          // convert field values before switching
    currentUnit = unit;
    try { localStorage.setItem('ttUnit', unit); } catch (e) { /* ignore */ }
    applyUnitToUI();
}

// Convert the rider weight/height fields from the current unit to `newUnit`.
function convertInputs(newUnit) {
    const wEl = document.getElementById('riderWeight');
    const hEl = document.getElementById('riderHeight');
    const w = parseFloat(wEl.value);
    const h = parseFloat(hEl.value);
    if (newUnit === 'imperial') {
        if (!isNaN(w)) wEl.value = Math.round(w * KG_TO_LB);
        if (!isNaN(h)) hEl.value = (h * CM_TO_IN).toFixed(1);
    } else {
        if (!isNaN(w)) wEl.value = (w / KG_TO_LB).toFixed(1);
        if (!isNaN(h)) hEl.value = Math.round(h / CM_TO_IN);
    }
}

function updateInputConstraints() {
    const wEl = document.getElementById('riderWeight');
    const hEl = document.getElementById('riderHeight');
    if (isImperial()) {
        document.getElementById('weightLabel').textContent = 'Weight (lb)';
        wEl.min = 66; wEl.max = 440; wEl.step = 1;
        document.getElementById('heightLabel').textContent = 'Height (in)';
        hEl.min = 55; hEl.max = 87; hEl.step = 0.5;
    } else {
        document.getElementById('weightLabel').textContent = 'Weight (kg)';
        wEl.min = 30; wEl.max = 200; wEl.step = 0.5;
        document.getElementById('heightLabel').textContent = 'Height (cm)';
        hEl.min = 140; hEl.max = 220; hEl.step = 1;
    }
}

function updateUnitToggleUI() {
    document.querySelectorAll('#unitToggle .unit-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.unit === currentUnit);
    });
}

// Refresh every unit-dependent part of the page.
function applyUnitToUI() {
    updateInputConstraints();
    updateUnitToggleUI();
    filterRoutes();                                   // rebuilds route option labels
    if (selectedRoute) showRouteStats(selectedRoute);
    updateBikeStats();
    updateWperKg();
    if (lastPlanData) displayResults(lastPlanData);
}

// ── Persisted settings ────────────────────────────────────────────────────────
// Rider/bike/route choices are saved to localStorage so they survive reloads.
// Rider weight/height are stored in canonical metric regardless of display unit.
const TT_SETTINGS_KEY = 'ttSettings';

function loadSettings() {
    try { return JSON.parse(localStorage.getItem(TT_SETTINGS_KEY)) || {}; }
    catch (e) { return {}; }
}

function saveSettings() {
    const w = getWeightKg();
    const h = getHeightCm();
    const np = parseFloat(document.getElementById('avgPower').value);
    const data = {
        weightKg: isNaN(w) ? null : +w.toFixed(2),
        heightCm: isNaN(h) ? null : +h.toFixed(1),
        npTarget: isNaN(np) ? null : np,
        world: document.getElementById('worldFilter').value,
        routeId: document.getElementById('routeSelect').value,
        frameId: document.getElementById('frameSelect').value,
        wheelId: document.getElementById('wheelSelect').value,
        upgradeLevel: document.getElementById('upgradeLevel').value,
        includeLeadin: document.getElementById('includeLeadin').checked,
        laps: laps,
    };
    try { localStorage.setItem(TT_SETTINGS_KEY, JSON.stringify(data)); } catch (e) { /* ignore */ }
}

// Restore the saved dropdown choices once routes + bike database are populated.
function restoreSelections(settings) {
    const hasOption = (el, val) => [...el.options].some(o => o.value === val);

    // A ?route= URL param takes precedence over the saved route.
    const fromUrl = selectRouteFromUrl();

    if (settings.world && !fromUrl) {
        const wf = document.getElementById('worldFilter');
        if (hasOption(wf, settings.world)) { wf.value = settings.world; filterRoutes(); }
    }
    if (settings.routeId && !fromUrl) {
        const rs = document.getElementById('routeSelect');
        if (hasOption(rs, settings.routeId)) { rs.value = settings.routeId; onRouteChange(); }
    }
    // Restore the saved lap count for a looped route (onRouteChange reset it).
    if (settings.laps && selectedRoute && selectedRoute.is_loop) {
        laps = Math.max(1, parseInt(settings.laps, 10) || 1);
        updateLapUI();
        showRouteStats(selectedRoute);
    }
    if (settings.frameId) {
        const fs = document.getElementById('frameSelect');
        if (hasOption(fs, settings.frameId)) { fs.value = settings.frameId; onFrameChange(); }
    }
    if (settings.wheelId) {
        const ws = document.getElementById('wheelSelect');
        if (hasOption(ws, settings.wheelId)) ws.value = settings.wheelId;
    }
    if (settings.upgradeLevel != null) {
        const ul = document.getElementById('upgradeLevel');
        if (hasOption(ul, String(settings.upgradeLevel))) ul.value = settings.upgradeLevel;
    }
    updateBikeStats();
    updatePlanButton();
}

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const settings = loadSettings();

    // Restore rider inputs (canonical metric) before any unit conversion.
    if (settings.weightKg != null) document.getElementById('riderWeight').value = settings.weightKg;
    if (settings.heightCm != null) document.getElementById('riderHeight').value = settings.heightCm;
    if (settings.npTarget != null) document.getElementById('avgPower').value = settings.npTarget;
    if (settings.includeLeadin != null) document.getElementById('includeLeadin').checked = settings.includeLeadin;

    let savedUnit = null;
    try { savedUnit = localStorage.getItem('ttUnit'); } catch (e) { /* ignore */ }
    if (savedUnit === 'imperial') {
        convertInputs('imperial');
        currentUnit = 'imperial';
    }
    updateInputConstraints();
    updateUnitToggleUI();

    Promise.all([loadRoutes(), loadBikeDatabase()])
        .then(() => { restoreSelections(settings); updatePlanButton(); })
        .catch(err => showError('Failed to load data: ' + err.message));
    updateWperKg();
    document.getElementById('includeLeadin')
        .addEventListener('change', () => { if (selectedRoute) showRouteStats(selectedRoute); });

    // Persist choices whenever the user changes them.
    ['riderWeight', 'riderHeight', 'avgPower', 'worldFilter', 'routeSelect',
     'frameSelect', 'wheelSelect', 'upgradeLevel', 'includeLeadin'].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('change', saveSettings);
        if (el.type === 'number') el.addEventListener('input', saveSettings);
    });
});

// ── Data loading ─────────────────────────────────────────────────────────────
async function loadRoutes() {
    const resp = await fetch('/api/tt_pacing/routes');
    if (!resp.ok) throw new Error('Could not load routes');
    const data = await resp.json();
    allRoutes = data.routes || [];
    buildRouteSlugMap();
    populateWorldFilter();
    filterRoutes();
}

// Normalise a route name into a URL-safe slug (must match the shared convention).
function convertToSlug(routeName) {
    return routeName
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '');
}

// Map each route's slug back to the route object so a ?route= param resolves to
// the original route. First match wins when two worlds share a slug.
function buildRouteSlugMap() {
    routeSlugMap = {};
    allRoutes.forEach(r => {
        const slug = convertToSlug(r.name);
        if (slug && !(slug in routeSlugMap)) routeSlugMap[slug] = r;
    });
}

async function loadBikeDatabase() {
    const resp = await fetch('/api/bike_database');
    if (!resp.ok) throw new Error('Could not load bike database');
    bikeDatabase = await resp.json();
    populateFrames();
}

// ── World filter ─────────────────────────────────────────────────────────────
function populateWorldFilter() {
    const worlds = [...new Set(allRoutes.map(r => r.world))].sort();
    const sel = document.getElementById('worldFilter');
    worlds.forEach(w => {
        const opt = document.createElement('option');
        opt.value = w;
        opt.textContent = formatWorldName(w);
        sel.appendChild(opt);
    });
}

function formatWorldName(world) {
    const MAP = {
        WATOPIA: 'Watopia', LONDON: 'London', RICHMOND: 'Richmond',
        NEWYORK: 'New York', INNSBRUCK: 'Innsbruck', FRANCE: 'France',
        MAKURIISLANDS: 'Makuri Islands', YORKSHIRE: 'Yorkshire',
        SCOTLAND: 'Scotland', PARIS: 'Paris', CRITCITY: 'Crit City',
        BOLOGNATT: 'Bologna', 'GRAVEL MOUNTAIN': 'Gravel Mountain',
    };
    return MAP[world] || world;
}

function filterRoutes() {
    const world = document.getElementById('worldFilter').value;
    const filtered = world ? allRoutes.filter(r => r.world === world) : allRoutes;

    const sel = document.getElementById('routeSelect');
    sel.innerHTML = '<option value="">Select a route…</option>';
    filtered.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.id;
        opt.textContent = `${r.name}  (${fmtDistance(r.distance_km)}, ${fmtElevation(r.ascent_m)}↑)`;
        sel.appendChild(opt);
    });

    if (selectedRoute && filtered.find(r => r.id === selectedRoute.id)) {
        sel.value = selectedRoute.id;
    } else {
        selectedRoute = null;
        hideRouteStats();
        updateLapUI();
    }
    updatePlanButton();
}

function onRouteChange() {
    const id = document.getElementById('routeSelect').value;
    selectedRoute = allRoutes.find(r => r.id === id) || null;
    laps = 1;
    updateLapUI();
    if (selectedRoute) {
        showRouteStats(selectedRoute);
        setRouteUrlParam(selectedRoute);
    } else {
        hideRouteStats();
        clearRouteUrlParam();
    }
    updatePlanButton();
}

// ── Laps (looped routes only) ─────────────────────────────────────────────────
function updateLapUI() {
    const group = document.getElementById('lapGroup');
    const isLoop = !!(selectedRoute && selectedRoute.is_loop);
    group.style.display = isLoop ? 'block' : 'none';
    if (!isLoop) laps = 1;
    document.getElementById('lapCount').textContent = laps;
    document.getElementById('lapMinus').disabled = laps <= 1;
}

function changeLaps(delta) {
    laps = Math.max(1, laps + delta);
    updateLapUI();
    if (selectedRoute) showRouteStats(selectedRoute);
    saveSettings();
}

// ── URL deep-linking ──────────────────────────────────────────────────────────
function setRouteUrlParam(route) {
    try {
        const url = new URL(window.location.href);
        url.searchParams.set('route', convertToSlug(route.name));
        window.history.replaceState(null, '', url);
    } catch (e) { /* ignore */ }
}

function clearRouteUrlParam() {
    try {
        const url = new URL(window.location.href);
        url.searchParams.delete('route');
        window.history.replaceState(null, '', url);
    } catch (e) { /* ignore */ }
}

// Resolve a ?route= slug to a route and select it. Returns true on success.
function selectRouteFromUrl() {
    let slug = null;
    try { slug = new URL(window.location.href).searchParams.get('route'); }
    catch (e) { return false; }
    if (!slug) return false;
    const route = routeSlugMap[slug];
    if (!route) return false;

    const wf = document.getElementById('worldFilter');
    if ([...wf.options].some(o => o.value === route.world)) {
        wf.value = route.world;
        filterRoutes();
    }
    const rs = document.getElementById('routeSelect');
    if ([...rs.options].some(o => o.value === route.id)) {
        rs.value = route.id;
        onRouteChange();
        return true;
    }
    return false;
}

function showRouteStats(route) {
    document.getElementById('routeStats').style.display = 'flex';
    const includeLeadin = document.getElementById('includeLeadin').checked;
    const dist = (includeLeadin ? (route.leadin_distance_km || 0) : 0)
        + route.distance_km * laps;
    const ascent = (includeLeadin ? (route.leadin_ascent_m || 0) : 0)
        + route.ascent_m * laps;
    document.getElementById('statDist').textContent = fmtDistance(dist);
    document.getElementById('statAscent').textContent = fmtElevation(ascent);
}

function hideRouteStats() {
    document.getElementById('routeStats').style.display = 'none';
}

// ── Bike database ─────────────────────────────────────────────────────────────
function populateFrames() {
    if (!bikeDatabase) return;
    const sel = document.getElementById('frameSelect');
    sel.innerHTML = '<option value="">Select a frame…</option>';
    bikeDatabase.frames.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.frameid;
        opt.textContent = `${f.framemake} ${f.framemodel}`;
        sel.appendChild(opt);
    });
    onFrameChange();
}

function onFrameChange() {
    if (!bikeDatabase) return;
    const frameId = document.getElementById('frameSelect').value;
    const wheelSel = document.getElementById('wheelSelect');
    wheelSel.innerHTML = '';

    if (!frameId) {
        wheelSel.innerHTML = '<option value="">Select a frame first…</option>';
        updateBikeStats();
        return;
    }

    const combos = bikeDatabase.bikes.filter(b => b.frameid === frameId);
    const wheelIds = combos.map(b => b.wheelid).filter(Boolean);

    if (wheelIds.length === 0) {
        wheelSel.innerHTML = '<option value="">(Built-in wheels)</option>';
    } else {
        wheelSel.innerHTML = '<option value="">Select wheels…</option>';
        const seenIds = new Set();
        wheelIds.forEach(wid => {
            if (seenIds.has(wid)) return;
            seenIds.add(wid);
            const wh = bikeDatabase.wheels.find(w => w.wheelid === wid);
            if (!wh) return;
            const opt = document.createElement('option');
            opt.value = wid;
            opt.textContent = `${wh.wheelmake} ${wh.wheelmodel}`;
            wheelSel.appendChild(opt);
        });
    }
    updateBikeStats();
}

function updateBikeStats() {
    if (!bikeDatabase) return;
    const frameId = document.getElementById('frameSelect').value;
    const wheelId = document.getElementById('wheelSelect').value;
    const level = parseInt(document.getElementById('upgradeLevel').value, 10);

    if (!frameId) { clearBikeStats(); return; }

    const combo = bikeDatabase.bikes.find(
        b => b.frameid === frameId && (b.wheelid || '') === (wheelId || '')
    );
    if (!combo) { clearBikeStats(); return; }

    const frame = bikeDatabase.frames.find(f => f.frameid === frameId);
    document.getElementById('bikeCd').textContent = combo.cd[level].toFixed(4);
    document.getElementById('bikeWeightStat').textContent = fmtWeight(combo.weight[level]);
    document.getElementById('bikeType').textContent = frame ? (frame.frametype || 'Standard') : 'Standard';
    updatePlanButton();
}

function clearBikeStats() {
    ['bikeCd', 'bikeWeightStat', 'bikeType'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    updatePlanButton();
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function updateWperKg() {
    const power = parseFloat(document.getElementById('avgPower').value) || 0;
    const weight = getWeightKg() || 1;
    document.getElementById('wpkgDisplay').textContent =
        (power / weight).toFixed(2) + ' W/kg';
}

function updatePlanButton() {
    const ready = !!selectedRoute
        && !!document.getElementById('frameSelect').value
        && document.getElementById('bikeCd').textContent !== '—';
    const btn = document.getElementById('planBtn');
    btn.disabled = !ready;
    document.getElementById('planHint').textContent =
        ready ? 'Ready — click to build the plan' : 'Select a route and bike to continue';
}

function showError(msg) {
    const toast = document.getElementById('errorToast');
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 6000);
}

// ── Plan request ──────────────────────────────────────────────────────────────
async function runPlan() {
    serverDividerIdx = [];        // a fresh build starts from the full optimal plan
    await requestPlan({ numBuckets: null, isBuild: true });
}

// Slider moved: request that many "smart" buckets (max = the full optimal plan).
function onBucketSliderInput() {
    const slider = document.getElementById('bucketSlider');
    const k = parseInt(slider.value, 10);
    const max = parseInt(slider.max, 10);
    document.getElementById('bucketCount').textContent = k >= max ? `${k} · max detail` : k;
    clearTimeout(bucketSliderTimer);
    bucketSliderTimer = setTimeout(() => {
        requestPlan({ numBuckets: k >= max ? null : k, isBuild: false });
    }, 220);
}

async function requestPlan({ numBuckets = null, isBuild = false } = {}) {
    if (!selectedRoute) return;

    const weightKg  = getWeightKg();
    const heightCm  = getHeightCm();
    const avgPowerW = parseFloat(document.getElementById('avgPower').value);
    const frameId   = document.getElementById('frameSelect').value;
    const wheelId   = document.getElementById('wheelSelect').value || null;
    const level     = parseInt(document.getElementById('upgradeLevel').value, 10);

    if (!frameId || isNaN(weightKg) || isNaN(heightCm) || isNaN(avgPowerW) || avgPowerW <= 0) {
        showError('Please fill in all fields correctly.');
        return;
    }

    const btn = document.getElementById('planBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    if (isBuild) btn.textContent = '⏳ Optimising…';
    document.getElementById('bucketSlider').disabled = true;
    document.getElementById('planHint').textContent = '';

    try {
        const resp = await fetch('/api/tt_pacing_plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                route_id:        selectedRoute.id,
                route_name:      selectedRoute.name,
                world:           selectedRoute.world,
                include_leadin:  document.getElementById('includeLeadin').checked,
                rider_weight_kg: weightKg,
                rider_height_cm: heightCm,
                avg_power_watts: avgPowerW,
                frame_id:        frameId,
                wheel_id:        wheelId,
                upgrade_level:   level,
                laps:            (selectedRoute.is_loop ? laps : 1),
                num_buckets:     numBuckets != null ? numBuckets : undefined,
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            showError(data.error || 'Planning failed');
            return;
        }
        displayResults(data, isBuild);
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.textContent = '▶ Build Pacing Plan';
        document.getElementById('bucketSlider').disabled = false;
        updatePlanButton();
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────
function displayResults(data, isBuild = false) {
    lastPlanData = data;
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsRouteName').textContent = data.route_name;

    document.getElementById('totalTime').textContent = data.total_time_formatted;
    document.getElementById('totalDist').textContent = fmtDistance(data.total_distance_km);
    document.getElementById('totalAscent').textContent = fmtElevation(data.total_ascent_m);
    document.getElementById('avgSpeed').textContent = fmtSpeed(data.avg_speed_kph);
    document.getElementById('powerRange').textContent =
        `${Math.round(data.min_power_w)}–${Math.round(data.max_power_w)} W`;
    document.getElementById('normPower').textContent = Math.round(data.normalized_power_w) + ' W';
    document.getElementById('avgPowerStat').textContent = Math.round(data.avg_power_w) + ' W';

    // Divider distances (km) → nearest profile index for drawing the lines.
    serverDividerIdx = (data.dividers_km || [])
        .map(km => nearestProfileIndex(data.profile, km));
    mapDividerKm = data.bucketed ? (data.dividers_km || []) : [];

    renderChart(data.profile);
    renderTable(data.sections);
    if (isBuild) configureBucketSlider(data.max_buckets);
    updateBucketHint(data);

    loadRouteMap();

    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Smart buckets ─────────────────────────────────────────────────────────────
// The server places the dividers (best step-fit of the optimal power curve) and
// returns their distances; we draw them and drive everything from the slider.

function nearestProfileIndex(profile, km) {
    const arr = profile.distance_km;
    let best = 0, bd = Infinity;
    for (let i = 0; i < arr.length; i++) {
        const d = Math.abs(arr[i] - km);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

function configureBucketSlider(maxBuckets) {
    const slider = document.getElementById('bucketSlider');
    const max = Math.max(1, maxBuckets || 1);
    slider.max = max;
    slider.value = max;
    slider.disabled = false;
    document.getElementById('bucketCount').textContent = `${max} · max detail`;
}

function updateBucketHint(data) {
    const hint = document.getElementById('bucketHint');
    if (!hint) return;
    if (data.bucketed) {
        const nb = (data.dividers_km ? data.dividers_km.length : 0) + 1;
        hint.textContent =
            `Holding ${nb} constant-power bucket${nb > 1 ? 's' : ''} — dividers placed automatically. ` +
            `Drag the slider for more or fewer.`;
        hint.classList.add('active');
    } else {
        hint.textContent =
            'Full optimal plan. Drag the slider left to simplify it into a few constant-power buckets.';
        hint.classList.remove('active');
    }
}

function dividerPixelForIndex(chart, idx) {
    const area = chart.chartArea;
    const n = chart.data.labels.length;
    if (n <= 1) return area.left;
    return area.left + (area.right - area.left) * (idx / (n - 1));
}

// Custom plugin: draw a dashed vertical line at each bucket divider.
const dividerPlugin = {
    id: 'ttDividers',
    afterDatasetsDraw(chart) {
        if (!serverDividerIdx.length) return;
        const { ctx, chartArea: area } = chart;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        serverDividerIdx.forEach(idx => {
            const x = dividerPixelForIndex(chart, idx);
            ctx.beginPath();
            ctx.moveTo(x, area.top);
            ctx.lineTo(x, area.bottom);
            ctx.stroke();
        });
        ctx.restore();
    },
};

function renderChart(p) {
    const ctx = document.getElementById('planChart').getContext('2d');
    if (planChart) planChart.destroy();

    const imp = isImperial();
    const distUnit = imp ? 'mi' : 'km';
    const elevUnit = imp ? 'ft' : 'm';
    const spdUnit  = imp ? 'mph' : 'km/h';
    const distLabels = imp ? p.distance_km.map(d => +(d * KM_TO_MI).toFixed(2)) : p.distance_km;
    const elevData   = imp ? p.altitude_m.map(a => Math.round(a * M_TO_FT)) : p.altitude_m;
    const speedData  = imp ? p.speed_kph.map(s => +(s * KM_TO_MI).toFixed(1)) : p.speed_kph;

    planChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: distLabels,
            datasets: [
                {
                    label: 'Power (W)',
                    data: p.power_w,
                    borderColor: '#f7931e',
                    backgroundColor: 'rgba(247,147,30,0.15)',
                    yAxisID: 'yPower',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.15,
                    fill: true,
                },
                {
                    label: `Elevation (${elevUnit})`,
                    data: elevData,
                    borderColor: 'rgba(120,180,255,0.9)',
                    backgroundColor: 'rgba(120,180,255,0.10)',
                    yAxisID: 'yElev',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.15,
                    fill: true,
                },
            ],
        },
        plugins: [dividerPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            onHover: (evt, els) => {
                if (els && els.length) highlightMapAtKm(p.distance_km[els[0].index]);
            },
            onClick: (evt, els) => {
                if (els && els.length) highlightMapAtKm(p.distance_km[els[0].index], true);
            },
            plugins: {
                legend: { labels: { color: '#ccc' } },
                tooltip: {
                    callbacks: {
                        title: items => `${items[0].label} ${distUnit}`,
                        afterBody: items => {
                            const i = items[0].dataIndex;
                            const spd = speedData[i];
                            const grd = p.gradient_pct[i];
                            return `Speed: ${spd} ${spdUnit}   Grade: ${grd}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: `Distance (${distUnit})`, color: '#888' },
                    ticks: { color: '#888', maxTicksLimit: 12 },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                yPower: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Power (W)', color: '#f7931e' },
                    ticks: { color: '#f7931e' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    beginAtZero: true,
                },
                yElev: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: `Elevation (${elevUnit})`, color: '#78b4ff' },
                    ticks: { color: '#78b4ff' },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });
}

function renderTable(sections) {
    const tbody = document.querySelector('#pacingTable tbody');
    tbody.innerHTML = '';
    const imp = isImperial();
    const distUnit = imp ? 'mi' : 'km';
    let cumTime = 0;
    sections.forEach((s, i) => {
        const tr = document.createElement('tr');
        const gradeCls = s.avg_gradient_pct > 0.2 ? 'grade-up'
            : s.avg_gradient_pct < -0.2 ? 'grade-down' : '';
        const startD = imp ? (s.start_km * KM_TO_MI).toFixed(1) : s.start_km;
        const endD   = imp ? (s.end_km * KM_TO_MI).toFixed(1) : s.end_km;
        cumTime += s.time_seconds;
        tr.innerHTML =
            `<td>${i + 1} &nbsp;<small style="color:#777">${startD}–${endD} ${distUnit}</small></td>` +
            `<td>${fmtDistance(s.distance_m / 1000, 2)}</td>` +
            `<td class="${gradeCls}">${s.avg_gradient_pct.toFixed(1)}%</td>` +
            `<td class="power">${s.power_w} W</td>` +
            `<td>${fmtSpeed(s.avg_speed_kph)}</td>` +
            `<td>${formatTime(s.time_seconds)}</td>` +
            `<td>${formatTime(cumTime)}</td>`;
        tbody.appendChild(tr);
    });
}

function formatTime(sec) {
    const s = Math.round(sec);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}:${String(r).padStart(2, '0')}` : `${r}s`;
}

// ── Route map (lead-in + route) ───────────────────────────────────────────────

// Fetch the lead-in + route geometry for the current route and draw it.
async function loadRouteMap() {
    if (!selectedRoute) return;
    const includeLeadin = document.getElementById('includeLeadin').checked;
    const key = `${selectedRoute.id}|${selectedRoute.name}|${selectedRoute.world}|${includeLeadin}`;

    if (!routeGeometry || routeGeometry.key !== key) {
        const params = new URLSearchParams({
            route_id: selectedRoute.id || '',
            route_name: selectedRoute.name || '',
            world: selectedRoute.world || '',
            include_leadin: includeLeadin ? 'true' : 'false',
        });
        try {
            const res = await fetch('/api/tt_pacing/route_geometry?' + params.toString());
            const data = await res.json();
            if (!res.ok) { hideRouteMap(data.error); return; }
            routeGeometry = buildGeometry(key, data);
        } catch (e) {
            hideRouteMap('Map unavailable: ' + e.message);
            return;
        }
    }
    document.getElementById('mapSection').style.display = 'block';
    drawRouteMap();
}

function hideRouteMap() {
    routeGeometry = null;
    document.getElementById('mapSection').style.display = 'none';
}

// Combine the two legs into one ordered point list.
function buildGeometry(key, data) {
    const legs = [];
    const push = (leg, name) => {
        if (!leg || !leg.x || !leg.x.length) return;
        const pts = leg.x.map((x, i) => ({ x, y: leg.y[i], d: leg.d[i], leg: name }));
        legs.push({ name, pts });
    };
    push(data.leadin, 'leadin');
    push(data.route, 'route');

    const combined = legs.flatMap(l => l.pts);

    // Bounds across all drawn points.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    combined.forEach(p => {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    });

    return {
        key, legs, combined,
        background: data.background,
        bounds: { minX, maxX, minY, maxY },
        bgImg: null,
    };
}

function drawRouteMap() {
    const geo = routeGeometry;
    if (!geo) return;
    const canvas = document.getElementById('routeMap');
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    // Fit transform (plot space → CSS pixels); plot y already increases downward.
    const b = geo.bounds;
    const bw = Math.max(b.maxX - b.minX, 1);
    const bh = Math.max(b.maxY - b.minY, 1);
    const pad = 0.07;
    const scale = Math.min(cw / bw, ch / bh) * (1 - pad * 2);
    const offX = (cw - bw * scale) / 2 - b.minX * scale;
    const offY = (ch - bh * scale) / 2 - b.minY * scale;
    const project = (x, y) => [x * scale + offX, y * scale + offY];
    mapView = { scale, offX, offY, project };

    // Background world image (drawn in the same plot space), if available.
    if (geo.background && geo.background.image) {
        if (!geo.bgImg) {
            const img = new Image();
            img.onload = () => { if (routeGeometry === geo) drawRouteMap(); };
            img.src = geo.background.image;
            geo.bgImg = img;
        }
        if (geo.bgImg.complete && geo.bgImg.naturalWidth) {
            const [ix, iy] = project(0, 0);
            ctx.globalAlpha = 0.5;
            ctx.drawImage(geo.bgImg, ix, iy,
                geo.background.width * scale, geo.background.height * scale);
            ctx.globalAlpha = 1;
        }
    }

    const drawLeg = (pts, color, width) => {
        if (!pts || !pts.length) return;
        ctx.beginPath();
        pts.forEach((p, i) => {
            const [sx, sy] = project(p.x, p.y);
            if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineJoin = 'round';
        ctx.stroke();
    };

    const leadin = geo.legs.find(l => l.name === 'leadin');
    const route  = geo.legs.find(l => l.name === 'route');
    drawLeg(route && route.pts, '#35d0ff', 3);
    drawLeg(leadin && leadin.pts, '#f7931e', 3);

    // Smart-bucket dividers: gray dots where each constant-power bucket begins.
    for (const km of mapDividerKm) {
        const p = nearestGeometryPoint(km * 1000);
        if (!p) continue;
        const [sx, sy] = project(p.x, p.y);
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#9aa3b2';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.stroke();
    }

    // Selection marker linked to the chart.
    if (mapSelectionKm != null) {
        const p = nearestGeometryPoint(mapSelectionKm * 1000);
        if (p) {
            const [sx, sy] = project(p.x, p.y);
            ctx.beginPath();
            ctx.arc(sx, sy, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#111';
            ctx.stroke();
        }
    }
}

// Nearest stitched-geometry point to a cumulative distance (metres).
function nearestGeometryPoint(targetM) {
    if (!routeGeometry) return null;
    const arr = routeGeometry.combined;
    let best = null, bd = Infinity;
    for (const p of arr) {
        const d = Math.abs(p.d - targetM);
        if (d < bd) { bd = d; best = p; }
    }
    return best;
}

// Called from the chart's hover/click to mark a distance on the map.
function highlightMapAtKm(km) {
    if (km == null || !routeGeometry) return;
    mapSelectionKm = km;
    drawRouteMap();
}

window.addEventListener('resize', () => { if (routeGeometry) drawRouteMap(); });
