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

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
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
        opt.textContent = `${r.name}  (${r.distance_km} km, ${r.ascent_m} m↑)`;
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
    document.getElementById('statDist').textContent = dist.toFixed(1) + ' km';
    document.getElementById('statAscent').textContent = Math.round(ascent) + ' m';
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
    document.getElementById('bikeWeightStat').textContent = combo.weight[level].toFixed(2) + ' kg';
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
    const weight = parseFloat(document.getElementById('riderWeight').value) || 1;
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

    const weightKg  = parseFloat(document.getElementById('riderWeight').value);
    const heightCm  = parseFloat(document.getElementById('riderHeight').value);
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
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsRouteName').textContent = data.route_name;

    document.getElementById('totalTime').textContent = data.total_time_formatted;
    document.getElementById('totalDist').textContent = data.total_distance_km.toFixed(1) + ' km';
    document.getElementById('totalAscent').textContent = Math.round(data.total_ascent_m) + ' m';
    document.getElementById('avgSpeed').textContent = data.avg_speed_kph.toFixed(1) + ' km/h';
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

    planChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: p.distance_km,
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
                    label: 'Elevation (m)',
                    data: p.altitude_m,
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
                        title: items => `${items[0].label} km`,
                        afterBody: items => {
                            const i = items[0].dataIndex;
                            const spd = p.speed_kph[i];
                            const grd = p.gradient_pct[i];
                            return `Speed: ${spd} km/h   Grade: ${grd}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Distance (km)', color: '#888' },
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
                    title: { display: true, text: 'Elevation (m)', color: '#78b4ff' },
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
    sections.forEach((s, i) => {
        const tr = document.createElement('tr');
        const gradeCls = s.avg_gradient_pct > 0.2 ? 'grade-up'
            : s.avg_gradient_pct < -0.2 ? 'grade-down' : '';
        tr.innerHTML =
            `<td>${i + 1} &nbsp;<small style="color:#777">${s.start_km}–${s.end_km} km</small></td>` +
            `<td>${(s.distance_m / 1000).toFixed(2)} km</td>` +
            `<td class="${gradeCls}">${s.avg_gradient_pct.toFixed(1)}%</td>` +
            `<td class="power">${s.power_w} W</td>` +
            `<td>${s.avg_speed_kph.toFixed(1)} km/h</td>` +
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
