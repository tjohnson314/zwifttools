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

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    let saved = null;
    try { saved = localStorage.getItem('ttUnit'); } catch (e) { /* ignore */ }
    if (saved === 'imperial') {
        convertInputs('imperial');
        currentUnit = 'imperial';
    }
    updateInputConstraints();
    updateUnitToggleUI();

    Promise.all([loadRoutes(), loadBikeDatabase()])
        .then(updatePlanButton)
        .catch(err => showError('Failed to load data: ' + err.message));
    updateWperKg();
    document.getElementById('includeLeadin')
        .addEventListener('change', () => { if (selectedRoute) showRouteStats(selectedRoute); });
});

// ── Data loading ─────────────────────────────────────────────────────────────
async function loadRoutes() {
    const resp = await fetch('/api/tt_pacing/routes');
    if (!resp.ok) throw new Error('Could not load routes');
    const data = await resp.json();
    allRoutes = data.routes || [];
    populateWorldFilter();
    filterRoutes();
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
    }
    updatePlanButton();
}

function onRouteChange() {
    const id = document.getElementById('routeSelect').value;
    selectedRoute = allRoutes.find(r => r.id === id) || null;
    if (selectedRoute) showRouteStats(selectedRoute);
    else hideRouteStats();
    updatePlanButton();
}

function showRouteStats(route) {
    document.getElementById('routeStats').style.display = 'flex';
    const includeLeadin = document.getElementById('includeLeadin').checked;
    const dist = route.distance_km + (includeLeadin ? (route.leadin_distance_km || 0) : 0);
    const ascent = route.ascent_m + (includeLeadin ? (route.leadin_ascent_m || 0) : 0);
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
    btn.textContent = '⏳ Optimising…';
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
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            showError(data.error || 'Planning failed');
            return;
        }
        displayResults(data);
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.textContent = '▶ Build Pacing Plan';
        updatePlanButton();
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────
function displayResults(data) {
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

    renderChart(data.profile);
    renderTable(data.sections);

    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

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
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
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
    sections.forEach((s, i) => {
        const tr = document.createElement('tr');
        const gradeCls = s.avg_gradient_pct > 0.2 ? 'grade-up'
            : s.avg_gradient_pct < -0.2 ? 'grade-down' : '';
        const startD = imp ? (s.start_km * KM_TO_MI).toFixed(1) : s.start_km;
        const endD   = imp ? (s.end_km * KM_TO_MI).toFixed(1) : s.end_km;
        tr.innerHTML =
            `<td>${i + 1} &nbsp;<small style="color:#777">${startD}–${endD} ${distUnit}</small></td>` +
            `<td>${fmtDistance(s.distance_m / 1000, 2)}</td>` +
            `<td class="${gradeCls}">${s.avg_gradient_pct.toFixed(1)}%</td>` +
            `<td class="power">${s.power_w} W</td>` +
            `<td>${fmtSpeed(s.avg_speed_kph)}</td>` +
            `<td>${formatTime(s.time_seconds)}</td>`;
        tbody.appendChild(tr);
    });
}

function formatTime(sec) {
    const s = Math.round(sec);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}:${String(r).padStart(2, '0')}` : `${r}s`;
}
