'use strict';

// TTT Pacing Planner front-end: route + bike selection (shared with the TT
// planner), an editable rider table, and plan results.

const TTT_SETTINGS_KEY = 'tttSettings';
const MIN_TEAM = 4;
const MAX_TEAM = 8;
const DEFAULT_W_PRIME_KJ = 20;
const RIDER_COLORS = ['#f7931e', '#4fc3f7', '#81c784', '#e57373', '#ba68c8', '#fff176', '#4db6ac', '#ff8a65'];

let allRoutes = [];
let selectedRoute = null;
let bikeSelector = null;
let laps = 1;
let riders = [];   // [{ name, height_cm, weight_kg, cp_w, w_prime_kj, zwift_id?, error? }]
const charts = {};
let isAuthenticated = false;
let isPlanning = false;

// ── Login ────────────────────────────────────────────────────────────────────
async function checkAuth() {
    const text = document.getElementById('authStatusText');
    const button = document.getElementById('authBtn');
    try {
        const resp = await fetch('/auth/status');
        const auth = await resp.json();
        isAuthenticated = !!auth.authenticated;
    } catch (e) {
        isAuthenticated = false;
    }
    text.textContent = isAuthenticated ? '✓ Logged in via Zwift' : 'Not logged in';
    text.className = isAuthenticated ? 'auth-status logged-in' : 'auth-status';
    button.textContent = isAuthenticated ? 'Log out' : 'Log in with Zwift';
    button.className = isAuthenticated ? 'btn-auth logout' : 'btn-auth';
}

function handleAuth() {
    window.location.href = isAuthenticated ? '/auth/logout' : '/auth/login?next=/ttt-pacing';
}

// ── Persistence ──────────────────────────────────────────────────────────────
function loadSettings() {
    try { return JSON.parse(localStorage.getItem(TTT_SETTINGS_KEY)) || {}; }
    catch (e) { return {}; }
}

function saveSettings() {
    const bike = bikeSelector ? bikeSelector.getConfig() : {};
    const data = {
        world: document.getElementById('worldFilter').value,
        routeId: document.getElementById('routeSelect').value,
        includeLeadin: document.getElementById('includeLeadin').checked,
        customDistanceKm: getCustomDistanceKm(),
        laps,
        zwiftIds: document.getElementById('zwiftIds').value,
        reservePct: document.getElementById('reservePct').value,
        maxPowerPct: document.getElementById('maxPowerPct').value,
        draftSecondPct: document.getElementById('draftSecondPct').value,
        draftRestPct: document.getElementById('draftRestPct').value,
        allowDrops: document.getElementById('allowDrops').checked,
        frameId: bike.frameId,
        wheelId: bike.wheelId,
        upgradeLevel: bike.level,
        riders: riders.map(({ error, ...r }) => r),
    };
    try { localStorage.setItem(TTT_SETTINGS_KEY, JSON.stringify(data)); } catch (e) { /* ignore */ }
}

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    const settings = loadSettings();
    if (settings.zwiftIds) document.getElementById('zwiftIds').value = settings.zwiftIds;
    if (settings.reservePct != null) document.getElementById('reservePct').value = settings.reservePct;
    if (settings.maxPowerPct != null) document.getElementById('maxPowerPct').value = settings.maxPowerPct;
    if (settings.draftSecondPct != null) document.getElementById('draftSecondPct').value = settings.draftSecondPct;
    if (settings.draftRestPct != null) document.getElementById('draftRestPct').value = settings.draftRestPct;
    if (settings.allowDrops != null) document.getElementById('allowDrops').checked = settings.allowDrops;
    if (settings.includeLeadin != null) document.getElementById('includeLeadin').checked = settings.includeLeadin;
    if (settings.customDistanceKm != null) {
        document.getElementById('customDistance').value = settings.customDistanceKm.toFixed(1);
    }
    if (Array.isArray(settings.riders)) riders = settings.riders.slice(0, MAX_TEAM);
    renderRiderTable();

    bikeSelector = new BikeSelector({
        frameSelect: 'frameSelect',
        wheelSelect: 'wheelSelect',
        levelSelect: 'upgradeLevel',
        weightOut: 'bikeWeightStat',
        cdaOut: 'bikeCda',
        framePlaceholder: 'Select a frame…',
        onChange: () => { updatePlanButton(); saveSettings(); },
    });

    Promise.all([loadRoutes(), bikeSelector.load()])
        .then(() => restoreSelections(settings))
        .catch(err => showError('Failed to load data: ' + err.message));

    document.getElementById('includeLeadin')
        .addEventListener('change', () => { if (selectedRoute) showRouteStats(selectedRoute); });
    ['worldFilter', 'routeSelect', 'includeLeadin', 'customDistance', 'zwiftIds',
    'reservePct', 'maxPowerPct', 'draftSecondPct', 'draftRestPct', 'allowDrops'].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('change', saveSettings);
    });
    ['draftSecondPct', 'draftRestPct'].forEach(id => {
        document.getElementById(id).addEventListener('input', updatePlanButton);
    });
});

function restoreSelections(settings) {
    const hasOption = (el, val) => [...el.options].some(o => o.value === val);
    if (settings.world) {
        const wf = document.getElementById('worldFilter');
        if (hasOption(wf, settings.world)) { wf.value = settings.world; filterRoutes(); }
    }
    if (settings.routeId) {
        const rs = document.getElementById('routeSelect');
        if (hasOption(rs, settings.routeId)) { rs.value = settings.routeId; onRouteChange(); }
    }
    if (settings.laps && selectedRoute && selectedRoute.is_loop) {
        laps = Math.max(1, parseInt(settings.laps, 10) || 1);
        updateLapUI();
        showRouteStats(selectedRoute);
    }
    bikeSelector.setConfig({
        frameId: settings.frameId,
        wheelId: settings.wheelId,
        level: settings.upgradeLevel,
    });
    updatePlanButton();
}

// ── Route selection ──────────────────────────────────────────────────────────
async function loadRoutes() {
    const resp = await fetch('/api/tt_pacing/routes');
    if (!resp.ok) throw new Error('Could not load routes');
    const data = await resp.json();
    allRoutes = data.routes || [];
    const worldNames = new Map(allRoutes.map(r => [r.world, r.world_name || r.world]));
    const sel = document.getElementById('worldFilter');
    [...worldNames.keys()].sort().forEach(w => {
        const opt = document.createElement('option');
        opt.value = w;
        opt.textContent = worldNames.get(w);
        sel.appendChild(opt);
    });
    filterRoutes();
}

function filterRoutes() {
    const world = document.getElementById('worldFilter').value;
    const filtered = world ? allRoutes.filter(r => r.world === world) : allRoutes;
    const sel = document.getElementById('routeSelect');
    sel.innerHTML = '<option value="">Select a route…</option>';
    filtered.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.id;
        opt.textContent = `${r.name}  (${r.distance_km.toFixed(1)} km, ${Math.round(r.ascent_m)} m↑)`;
        sel.appendChild(opt);
    });
    if (selectedRoute && filtered.find(r => r.id === selectedRoute.id)) {
        sel.value = selectedRoute.id;
    } else {
        selectedRoute = null;
        document.getElementById('routeStats').style.display = 'none';
        updateLapUI();
    }
    updatePlanButton();
}

function onRouteChange() {
    const id = document.getElementById('routeSelect').value;
    selectedRoute = allRoutes.find(r => r.id === id) || null;
    laps = 1;
    updateLapUI();
    if (selectedRoute) showRouteStats(selectedRoute);
    else document.getElementById('routeStats').style.display = 'none';
    updatePlanButton();
}

function updateLapUI() {
    const isLoop = !!(selectedRoute && selectedRoute.is_loop);
    document.getElementById('lapGroup').style.display = isLoop ? 'block' : 'none';
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

function showRouteStats(route) {
    const includeLeadin = document.getElementById('includeLeadin').checked;
    const dist = (includeLeadin ? (route.leadin_distance_km || 0) : 0) + route.distance_km * laps;
    const ascent = (includeLeadin ? (route.leadin_ascent_m || 0) : 0) + route.ascent_m * laps;
    document.getElementById('routeStats').style.display = 'flex';
    document.getElementById('statDist').textContent = `${dist.toFixed(1)} km`;
    document.getElementById('statAscent').textContent = `${Math.round(ascent)} m`;
}

function getCustomDistanceKm() {
    const v = parseFloat(document.getElementById('customDistance').value);
    return isNaN(v) || v <= 0 ? null : v;
}

// ── Riders ───────────────────────────────────────────────────────────────────
function parseZwiftIds() {
    return document.getElementById('zwiftIds').value
        .split(/[\s,;]+/)
        .map(s => s.trim())
        .filter(Boolean);
}

async function fetchRiders() {
    const ids = parseZwiftIds();
    if (!ids.length) { showError('Enter at least one Zwift ID.'); return; }
    const bad = ids.filter(id => !/^\d+$/.test(id));
    if (bad.length) { showError(`Invalid Zwift ID: ${bad[0]}`); return; }
    if (ids.length > MAX_TEAM) { showError(`A WTRL team has at most ${MAX_TEAM} riders.`); return; }

    const btn = document.getElementById('fetchRidersBtn');
    btn.disabled = true;
    btn.textContent = 'Fetching…';
    try {
        const resp = await fetch('/api/ttt_pacing/riders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zwift_ids: ids }),
        });
        const data = await resp.json();
        if (resp.status === 401) {
            isAuthenticated = false;
            checkAuth();
            showError('Log in with Zwift (top right) to look up riders.');
            return;
        }
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        // Keep manual edits for riders that were already in the table.
        const previous = new Map(riders.filter(r => r.zwift_id).map(r => [r.zwift_id, r]));
        riders = data.riders.map(r => {
            const old = previous.get(r.zwift_id);
            return {
                zwift_id: r.zwift_id,
                name: r.name,
                height_cm: r.height_cm ?? old?.height_cm ?? null,
                weight_kg: r.weight_kg ?? old?.weight_kg ?? null,
                cp_w: old?.cp_w ?? r.ftp_w ?? null,
                w_prime_kj: old?.w_prime_kj ?? r.w_prime_kj ?? DEFAULT_W_PRIME_KJ,
                error: r.error,
            };
        });
        renderRiderTable();
        saveSettings();
    } catch (err) {
        showError('Rider lookup failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Fetch riders';
    }
}

function addRider() {
    if (riders.length >= MAX_TEAM) { showError(`A WTRL team has at most ${MAX_TEAM} riders.`); return; }
    riders.push({
        name: `Rider ${riders.length + 1}`, height_cm: 180, weight_kg: 75,
        cp_w: 280, w_prime_kj: DEFAULT_W_PRIME_KJ,
    });
    renderRiderTable();
    saveSettings();
}

function moveRider(i, delta) {
    const j = i + delta;
    if (j < 0 || j >= riders.length) return;
    [riders[i], riders[j]] = [riders[j], riders[i]];
    renderRiderTable();
    saveSettings();
}

function removeRider(i) {
    riders.splice(i, 1);
    renderRiderTable();
    saveSettings();
}

const RIDER_FIELDS = [
    { key: 'name', type: 'text', cls: 'name-input' },
    { key: 'height_cm', type: 'number', min: 120, max: 230, step: 1 },
    { key: 'weight_kg', type: 'number', min: 30, max: 200, step: 0.1 },
    { key: 'cp_w', type: 'number', min: 50, max: 700, step: 1 },
    { key: 'w_prime_kj', type: 'number', min: 1, max: 80, step: 0.5 },
];

function isValidValue(field, value) {
    if (field.type === 'text') return true;
    return typeof value === 'number' && isFinite(value) && value >= field.min && value <= field.max;
}

function renderRiderTable() {
    const tbody = document.querySelector('#riderTable tbody');
    tbody.innerHTML = '';
    riders.forEach((r, i) => {
        const tr = document.createElement('tr');
        const idx = document.createElement('td');
        idx.textContent = i + 1;
        tr.appendChild(idx);

        RIDER_FIELDS.forEach(field => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = field.type;
            if (field.cls) input.className = field.cls;
            if (field.type === 'number') {
                input.min = field.min;
                input.max = field.max;
                input.step = field.step;
            }
            input.value = r[field.key] ?? '';
            input.classList.toggle('invalid', !isValidValue(field, r[field.key]));
            input.addEventListener('input', () => {
                r[field.key] = field.type === 'number'
                    ? (input.value === '' ? null : parseFloat(input.value))
                    : input.value;
                input.classList.toggle('invalid', !isValidValue(field, r[field.key]));
                updateWkg(tr, r);
                updatePlanButton();
                saveSettings();
            });
            td.appendChild(input);
            if (field.key === 'name' && r.error) {
                const err = document.createElement('div');
                err.className = 'lookup-error';
                err.textContent = r.error;
                td.appendChild(err);
            }
            tr.appendChild(td);
        });

        const wkg = document.createElement('td');
        wkg.className = 'wkg-cell';
        tr.appendChild(wkg);
        updateWkg(tr, r);

        const actions = document.createElement('td');
        actions.className = 'row-actions';
        [['↑', () => moveRider(i, -1), i === 0, 'Move up'],
         ['↓', () => moveRider(i, 1), i === riders.length - 1, 'Move down'],
         ['✕', () => removeRider(i), false, 'Remove rider']].forEach(([label, fn, disabled, title]) => {
            const b = document.createElement('button');
            b.type = 'button';
            b.textContent = label;
            b.title = title;
            b.disabled = disabled;
            b.addEventListener('click', fn);
            actions.appendChild(b);
        });
        tr.appendChild(actions);
        tbody.appendChild(tr);
    });
    document.getElementById('riderCount').textContent = riders.length ? `(${riders.length})` : '';
    updatePlanButton();
}

function updateWkg(tr, r) {
    const cell = tr.querySelector('.wkg-cell');
    cell.textContent = r.cp_w > 0 && r.weight_kg > 0 ? (r.cp_w / r.weight_kg).toFixed(2) : '—';
}

function ridersValid() {
    return riders.length >= MIN_TEAM && riders.length <= MAX_TEAM &&
        riders.every(r => RIDER_FIELDS.every(f => isValidValue(f, r[f.key])));
}

// ── Plan ─────────────────────────────────────────────────────────────────────
function draftSettingsValid() {
    return ['draftSecondPct', 'draftRestPct'].every(id => {
        const input = document.getElementById(id);
        return input.value !== '' && input.validity.valid;
    });
}

function updatePlanButton() {
    if (isPlanning) {
        document.getElementById('planBtn').disabled = true;
        return;
    }
    const routeOk = !!selectedRoute;
    const bikeOk = !!bikeSelector && bikeSelector.isReady();
    const ridersOk = ridersValid();
    const draftsOk = draftSettingsValid();
    document.getElementById('planBtn').disabled = !(routeOk && bikeOk && ridersOk && draftsOk);
    let hint = 'Ready - planning can take several minutes';
    if (!routeOk) hint = 'Select a route';
    else if (!bikeOk) hint = 'Select a bike';
    else if (riders.length < MIN_TEAM || riders.length > MAX_TEAM) hint = `Add ${MIN_TEAM}–${MAX_TEAM} riders`;
    else if (!ridersOk) hint = 'Fix the highlighted rider values';
    else if (!draftsOk) hint = 'Draft reductions must be between 0% and 99.9%';
    document.getElementById('planHint').textContent = hint;
}

async function runPlan() {
    if (isPlanning) return;
    if (!draftSettingsValid()) {
        updatePlanButton();
        return;
    }
    const btn = document.getElementById('planBtn');
    const bike = bikeSelector.getConfig();
    const body = {
        route_id: selectedRoute.id,
        route_name: selectedRoute.name,
        world: selectedRoute.world,
        include_leadin: document.getElementById('includeLeadin').checked,
        laps,
        frame_id: bike.frameId,
        wheel_id: bike.wheelId,
        upgrade_level: bike.level,
        reserve_pct: parseFloat(document.getElementById('reservePct').value) || 0,
        max_power_pct: parseFloat(document.getElementById('maxPowerPct').value) || 150,
        draft_second_pct: Number(document.getElementById('draftSecondPct').value),
        draft_rest_pct: Number(document.getElementById('draftRestPct').value),
        allow_drops: document.getElementById('allowDrops').checked,
        stream: true,
        riders: riders.map(r => ({
            name: r.name, height_cm: r.height_cm, weight_kg: r.weight_kg,
            cp_w: r.cp_w, w_prime_kj: r.w_prime_kj,
        })),
    };
    const customKm = getCustomDistanceKm();
    if (customKm != null) body.custom_distance_km = customKm;

    isPlanning = true;
    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = '⏳ Optimising…';
    document.getElementById('resultsSection').style.display = 'none';
    const started = Date.now();
    let progressText = 'Preparing course';
    let hasPlan = false;
    let finalStatus = '';
    const timer = setInterval(() => {
        document.getElementById('planHint').textContent =
            `${progressText} - ${Math.round((Date.now() - started) / 1000)} s`;
    }, 1000);
    try {
        const resp = await fetch('/api/ttt_pacing_plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        if ((resp.headers.get('Content-Type') || '').includes('text/event-stream')) {
            await readPlanStream(resp, update => {
                if (update.type === 'progress') {
                    progressText = `Iteration ${update.iteration}: ${update.stage} - ` +
                        `${update.step} ${update.candidate_iteration}/${update.candidate_iterations}`;
                    document.getElementById('planHint').textContent =
                        `${progressText} - ${Math.round((Date.now() - started) / 1000)} s`;
                    if (update.plan && (update.improved || !hasPlan)) {
                        displayResults(update.plan, { scroll: !hasPlan, provisional: true });
                        hasPlan = true;
                    }
                } else if (update.type === 'complete') {
                    displayResults(update.plan, { scroll: !hasPlan });
                    hasPlan = true;
                }
            });
        } else {
            displayResults(await resp.json());
            hasPlan = true;
        }
        finalStatus = `Finished in ${Math.round((Date.now() - started) / 1000)} s`;
    } catch (err) {
        showError('Plan failed: ' + err.message);
        finalStatus = hasPlan ? 'Stopped - showing best feasible plan so far' : 'Planning failed';
    } finally {
        clearInterval(timer);
        isPlanning = false;
        btn.classList.remove('loading');
        btn.textContent = '▶ Build TTT Plan';
        updatePlanButton();
        document.getElementById('planHint').textContent = finalStatus;
    }
}

async function readPlanStream(response, onUpdate) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let complete = false;
    let ended = false;

    function handleEvent(block) {
        const data = block.split(/\r?\n/)
            .filter(line => line.startsWith('data:'))
            .map(line => line.slice(5).trimStart()).join('\n');
        if (!data) return;
        const update = JSON.parse(data);
        if (update.type === 'error') throw new Error(update.error || 'Optimization failed');
        onUpdate(update);
        if (update.type === 'complete') complete = true;
    }

    try {
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value, { stream: !done });
            let boundary;
            while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
                if (!complete) handleEvent(buffer.slice(0, boundary.index));
                buffer = buffer.slice(boundary.index + boundary[0].length);
            }
            if (done) {
                ended = true;
                if (!complete && buffer.trim()) handleEvent(buffer);
                if (!complete) throw new Error('Connection ended before optimization finished');
                break;
            }
        }
    } finally {
        if (!ended) await reader.cancel().catch(() => {});
        reader.releaseLock();
    }
}

// ── Results ──────────────────────────────────────────────────────────────────
function formatTime(sec) {
    const s = Math.round(sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
        : `${m}:${String(r).padStart(2, '0')}`;
}

function cell(text, cls) {
    const td = document.createElement('td');
    td.textContent = text;
    if (cls) td.className = cls;
    return td;
}

function displayResults(data, { scroll = true, provisional = false } = {}) {
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsRouteName').textContent =
        data.route_name + (provisional ? ' (provisional)' : '');
    document.getElementById('infeasibleWarning').style.display = data.feasible ? 'none' : 'block';
    document.getElementById('totalTime').textContent = data.total_time_formatted;
    document.getElementById('totalDist').textContent = `${data.total_distance_km.toFixed(1)} km`;
    document.getElementById('totalAscent').textContent = `${data.total_ascent_m} m`;
    document.getElementById('avgSpeed').textContent = `${data.avg_speed_kph} km/h`;
    document.getElementById('scoringRider').textContent =
        `${data.scoring_rider_count}${data.scoring_rider_count === 3 ? 'rd' : 'th'} of ${data.team_size}`;
    document.getElementById('flatSpeed').textContent = `${data.flat_rotation.speed_kph} km/h`;

    const riderBody = document.querySelector('#riderResultTable tbody');
    riderBody.innerHTML = '';
    data.riders.forEach(r => {
        const tr = document.createElement('tr');
        tr.appendChild(cell(r.name));
        tr.appendChild(cell(`${Math.round(r.cp_w)} W`));
        tr.appendChild(r.finishes
            ? cell('Finishes', 'outcome-finish')
            : cell(`Dropped at ${r.drop_km} km (${formatTime(r.drop_time_s)})`, 'outcome-drop'));
        tr.appendChild(cell(formatTime(r.time_on_front_s)));
        tr.appendChild(cell(r.avg_lead_power_w != null ? `${Math.round(r.avg_lead_power_w)} W` : '—'));
        tr.appendChild(cell(`${Math.round(r.avg_power_w)} W`));
        tr.appendChild(cell(`${(r.min_wbal_j / 1000).toFixed(1)} kJ (${r.min_wbal_pct}%)`));
        riderBody.appendChild(tr);
    });

    const phaseBody = document.querySelector('#phaseTable tbody');
    phaseBody.innerHTML = '';
    data.phases.forEach(p => {
        const tr = document.createElement('tr');
        tr.appendChild(cell(`${p.start_km}–${p.end_km} km`));
        tr.appendChild(cell(p.riders.join(', ')));
        const pulls = Object.entries(p.pulls_s)
            .map(([name, s]) => `${name}: ${s > 0 ? Math.round(s) + ' s' : 'no pulls'}`)
            .join(' · ');
        tr.appendChild(cell(pulls));
        phaseBody.appendChild(tr);
    });

    renderCharts(data.profile);
    if (scroll) document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function baseChartOptions(yTitle) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: '#ccc' } } },
        scales: {
            x: {
                title: { display: true, text: 'Distance (km)', color: '#888' },
                ticks: { color: '#888', maxTicksLimit: 12 },
                grid: { color: 'rgba(255,255,255,0.05)' },
            },
            y: {
                title: { display: true, text: yTitle, color: '#ccc' },
                ticks: { color: '#ccc' },
                grid: { color: 'rgba(255,255,255,0.05)' },
            },
        },
    };
}

function drawChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id).getContext('2d'), config);
}

function riderDatasets(series, scale = 1) {
    return Object.entries(series).map(([name, values], i) => ({
        label: name,
        data: values.map(v => (v == null ? null : v * scale)),
        borderColor: RIDER_COLORS[i % RIDER_COLORS.length],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
        spanGaps: false,
    }));
}

function renderCharts(p) {
    const speedOpts = baseChartOptions('Speed (km/h)');
    speedOpts.scales.yElev = {
        position: 'right',
        title: { display: true, text: 'Elevation (m)', color: '#78b4ff' },
        ticks: { color: '#78b4ff' },
        grid: { drawOnChartArea: false },
    };
    speedOpts.plugins.tooltip = {
        callbacks: {
            afterBody: items => {
                const i = items[0].dataIndex;
                const lines = [`Leader: ${p.leader[i]}   Grade: ${p.gradient_pct[i]}%`];
                if (p.entry_speed_kph?.[i] != null && p.exit_speed_kph?.[i] != null) {
                    lines.push(`Entry: ${p.entry_speed_kph[i]} km/h   Exit: ${p.exit_speed_kph[i]} km/h`);
                }
                return lines;
            },
        },
    };
    drawChart('speedChart', {
        type: 'line',
        data: {
            labels: p.distance_km,
            datasets: [
                { label: 'Group speed (km/h)', data: p.speed_kph, borderColor: '#f7931e',
                  borderWidth: 2, pointRadius: 0, tension: 0.1 },
                { label: 'Elevation (m)', data: p.altitude_m, yAxisID: 'yElev',
                  borderColor: 'rgba(120,180,255,0.9)', backgroundColor: 'rgba(120,180,255,0.10)',
                  borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.1 },
            ],
        },
        options: speedOpts,
    });

    drawChart('wbalChart', {
        type: 'line',
        data: { labels: p.distance_km, datasets: riderDatasets(p.wbal_j, 0.001) },
        options: baseChartOptions('W′ balance (kJ)'),
    });

    const powerOpts = baseChartOptions('Power (W)');
    powerOpts.plugins.tooltip = {
        callbacks: {
            afterLabel: context => {
                const name = context.dataset.label;
                const i = context.dataIndex;
                const lines = [];
                const peak = p.peak_power_w?.[name]?.[i];
                const braking = p.braking_w?.[name]?.[i];
                if (peak != null) lines.push(`Peak: ${peak} W`);
                if (braking > 0) lines.push(`Speed-matching dissipation: ${braking} W`);
                return lines;
            },
        },
    };
    drawChart('powerChart', {
        type: 'line',
        data: { labels: p.distance_km, datasets: riderDatasets(p.power_w) },
        options: powerOpts,
    });
}

function showError(msg) {
    const toast = document.getElementById('errorToast');
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 6000);
}
