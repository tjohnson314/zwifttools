/**
 * Ride Simulator — frontend logic.
 *
 * Loads routes and bike data, then POSTs to /api/simulate_ride and renders
 * the result summary + Chart.js elevation/speed overlay.
 */

'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let allRoutes = [];          // full list from /api/ride_simulator/routes
let bikeDatabase = null;     // { frames, wheels } from /api/bike_database
let selectedRoute = null;    // currently selected route object
let profileChart = null;     // Chart.js instance

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    Promise.all([loadRoutes(), loadBikeDatabase()])
        .then(() => updateSimulateButton())
        .catch(err => showError('Failed to load data: ' + err.message));
    updateWperKg();
});

// ── Data loading ─────────────────────────────────────────────────────────────
async function loadRoutes() {
    const resp = await fetch('/api/ride_simulator/routes');
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
        WATOPIA: 'Watopia',
        LONDON: 'London',
        RICHMOND: 'Richmond',
        NEWYORK: 'New York',
        INNSBRUCK: 'Innsbruck',
        FRANCE: 'France',
        MAKURIISLANDS: 'Makuri Islands',
        YORKSHIRE: 'Yorkshire',
        SCOTLAND: 'Scotland',
        PARIS: 'Paris',
        CRITCITY: 'Crit City',
        BOLOGNATT: 'Bologna',
        'GRAVEL MOUNTAIN': 'Gravel Mountain',
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
        opt.textContent = `${r.name}  (${r.distance_km} km, ${r.ascent_m} m↑)`;
        sel.appendChild(opt);
    });

    // Restore selection if still in filtered list
    if (selectedRoute && filtered.find(r => r.id === selectedRoute.id)) {
        sel.value = selectedRoute.id;
    } else {
        selectedRoute = null;
        hideRouteStats();
    }
    updateSimulateButton();
}

function onRouteChange() {
    const id = document.getElementById('routeSelect').value;
    selectedRoute = allRoutes.find(r => r.id === id) || null;
    if (selectedRoute) showRouteStats(selectedRoute);
    else hideRouteStats();
    updateSimulateButton();
}

function showRouteStats(route) {
    document.getElementById('routeStats').style.display = 'flex';
    document.getElementById('statDist').textContent = route.distance_km + ' km';
    document.getElementById('statAscent').textContent = route.ascent_m + ' m';
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

    // Find valid wheel combos for this frame
    const combos = bikeDatabase.bikes.filter(b => b.frameid === frameId);
    const wheelIds = combos.map(b => b.wheelid).filter(Boolean);

    if (wheelIds.length === 0) {
        // Built-in wheels (e.g., Tron)
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

    if (!frameId) {
        clearBikeStats(); return;
    }

    // Find the combo
    const combo = bikeDatabase.bikes.find(
        b => b.frameid === frameId && (b.wheelid || '') === (wheelId || '')
    );

    if (!combo) {
        clearBikeStats(); return;
    }

    const cd = combo.cd[level];
    const wt = combo.weight[level];
    const frame = bikeDatabase.frames.find(f => f.frameid === frameId);
    const frameType = frame ? (frame.frametype || 'Standard') : 'Standard';

    document.getElementById('bikeCd').textContent = cd.toFixed(4);
    document.getElementById('bikeWeight').textContent = wt.toFixed(2) + ' kg';
    document.getElementById('bikeType').textContent = frameType;

    updateSimulateButton();
}

function clearBikeStats() {
    ['bikeCd', 'bikeWeight', 'bikeType'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    updateSimulateButton();
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function updateWperKg() {
    const power = parseFloat(document.getElementById('powerWatts').value) || 0;
    const weight = parseFloat(document.getElementById('riderWeight').value) || 1;
    document.getElementById('wpkgDisplay').textContent =
        (power / weight).toFixed(2) + ' W/kg';
}

function updateSimulateButton() {
    const hasRoute  = !!selectedRoute;
    const hasFrame  = !!document.getElementById('frameSelect').value;
    const hasCd     = document.getElementById('bikeCd').textContent !== '—';
    const ready = hasRoute && hasFrame && hasCd;
    const btn = document.getElementById('simulateBtn');
    btn.disabled = !ready;
    document.getElementById('simulateHint').textContent =
        ready ? 'Ready — click to simulate' : 'Select a route and bike to continue';
}

function showError(msg) {
    const toast = document.getElementById('errorToast');
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 6000);
}

// ── Simulation ────────────────────────────────────────────────────────────────
async function runSimulation() {
    if (!selectedRoute) return;

    const frameId   = document.getElementById('frameSelect').value;
    const wheelId   = document.getElementById('wheelSelect').value || null;
    const level     = parseInt(document.getElementById('upgradeLevel').value, 10);
    const weightKg  = parseFloat(document.getElementById('riderWeight').value);
    const heightCm  = parseFloat(document.getElementById('riderHeight').value);
    const powerW    = parseFloat(document.getElementById('powerWatts').value);

    if (!frameId || isNaN(weightKg) || isNaN(heightCm) || isNaN(powerW) || powerW <= 0) {
        showError('Please fill in all fields correctly.');
        return;
    }

    const btn = document.getElementById('simulateBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = '⏳ Simulating…';
    document.getElementById('simulateHint').textContent = '';

    try {
        const resp = await fetch('/api/simulate_ride', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                route_id:        selectedRoute.id,
                route_name:      selectedRoute.name,
                world:           selectedRoute.world,
                rider_weight_kg: weightKg,
                rider_height_cm: heightCm,
                power_watts:     powerW,
                frame_id:        frameId,
                wheel_id:        wheelId,
                upgrade_level:   level,
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            showError(data.error || 'Simulation failed');
            return;
        }
        displayResults(data);
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.textContent = '▶ Simulate Ride';
        updateSimulateButton();
    }
}

// ── Results rendering ─────────────────────────────────────────────────────────
function displayResults(data) {
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';

    document.getElementById('resultsRouteName').textContent = data.route_name;

    document.getElementById('totalTime').textContent   = data.total_time_formatted;
    document.getElementById('totalDist').textContent   = data.total_distance_km.toFixed(1) + ' km';
    document.getElementById('totalAscent').textContent = data.total_ascent_m + ' m';
    document.getElementById('avgSpeed').textContent    = data.avg_speed_kph.toFixed(1) + ' km/h';

    renderChart(data.profile);

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChart(profile) {
    const ctx = document.getElementById('profileChart').getContext('2d');

    if (profileChart) {
        profileChart.destroy();
        profileChart = null;
    }

    // Build {x, y} points so the x-axis is a true linear distance scale
    // (a category axis would label ticks by point index, not distance).
    const elevationPoints = profile.distance_km.map((d, i) => ({ x: d, y: profile.altitude_m[i] }));
    const speedPoints     = profile.distance_km.map((d, i) => ({ x: d, y: profile.speed_kph[i] }));

    profileChart = new Chart(ctx, {
        data: {
            datasets: [
                {
                    type: 'line',
                    label: 'Elevation (m)',
                    data: elevationPoints,
                    borderColor: 'rgba(247,147,30,0.9)',
                    backgroundColor: 'rgba(247,147,30,0.12)',
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0,
                    yAxisID: 'y',
                },
                {
                    type: 'line',
                    label: 'Speed (km/h)',
                    data: speedPoints,
                    borderColor: 'rgba(79,195,247,0.9)',
                    backgroundColor: 'transparent',
                    tension: 0.35,
                    pointRadius: 0,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#ccc', font: { size: 12 } },
                },
                tooltip: {
                    callbacks: {
                        title: ctx => ctx[0].parsed.x.toFixed(2) + ' km',
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    bounds: 'data',
                    ticks: {
                        color: '#888',
                        maxTicksLimit: 10,
                        callback: v => v.toFixed(1) + ' km',
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Distance (km)', color: '#888' },
                },
                y: {
                    position: 'left',
                    ticks: { color: '#f7931e', callback: v => v + ' m' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Elevation (m)', color: '#f7931e' },
                },
                y1: {
                    position: 'right',
                    ticks: { color: '#4fc3f7', callback: v => v + ' km/h' },
                    grid: { display: false },
                    title: { display: true, text: 'Speed (km/h)', color: '#4fc3f7' },
                },
            },
        },
    });
}
