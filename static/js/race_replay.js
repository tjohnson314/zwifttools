/**
 * Race Replay — client-side playback engine.
 *
 * Loads full race data as JSON from the server, then handles:
 *   - Playback controls (play / pause / scrub / speed)
 *   - Rider table with live positions, gaps, power, HR, speed, 1-min power
 *   - Elevation profile with rider dots (Plotly.js)
 *   - Peloton elevation chart (colored group bars)
 *   - Peloton details panel
 *   - Zoom & follow-rider controls
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let raceData = null;          // Full race JSON from server
let currentTime = 0;          // Current playback time (seconds)
let isPlaying = false;
let isAuthenticated = false;
let playbackSpeed = 5;        // Multiplier
let lastFrameTs = null;       // For requestAnimationFrame delta
let animFrameId = null;
let selectedRank = null;      // Clicked rider rank
let checkedRanks = new Set();  // Checked riders for highlighting
let sortField = 'position';
let sortAsc = true;

// Rider lookup for O(1) access: rank -> rider object with typed arrays
let riderLookup = {};

// Pre-built elevation trace (shared across frames)
let elevationTrace = null;

// Zoom state
let viewXMin = 0;
let viewXMax = 25;            // Will be set to finish_line_km on load
let viewWidth = 25;           // Current zoom level in km
let followRider = true;       // Auto-follow selected rider
let showWkg = localStorage.getItem('powerUnit') === 'wkg';  // Toggle: false=watts, true=W/kg
let chartMode = 'riders';  // 'riders' or 'peloton'

// YouTube stream links data
let streamLinks = [];         // [{streamer_name, youtube_url, offset_seconds, stream_title}]

// Plotly chart div references
const DARK_BG = 'rgba(0,0,0,0)';
const GRID_COLOR = 'rgba(255,255,255,0.08)';
const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: DARK_BG,
    plot_bgcolor: DARK_BG,
    font: { color: '#aaa', size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: GRID_COLOR, zeroline: false },
    yaxis: { gridcolor: GRID_COLOR, zeroline: false },
};

// Peloton colors
const PELOTON_COLORS = [
    '#29b6f6', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc',
    '#26c6da', '#9ccc65', '#ffca28', '#ec407a', '#7e57c2',
];

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    loadRaceList();
    bindEvents();

    // Auto-load race from URL parameter (e.g. ?activity_id=12345)
    const params = new URLSearchParams(window.location.search);
    const urlActivityId = params.get('activity_id');
    if (urlActivityId) {
        document.getElementById('activity-input').value = urlActivityId;
        fetchRace(false);
    }
});

function bindEvents() {
    document.getElementById('fetch-btn').addEventListener('click', () => fetchRace(false));
    document.getElementById('refetch-btn').addEventListener('click', () => fetchRace(true));
    document.getElementById('load-btn').addEventListener('click', loadSelectedRace);
    document.getElementById('play-btn').addEventListener('click', togglePlay);
    document.getElementById('time-slider').addEventListener('input', onSliderInput);

    // Spacebar to toggle play/pause (only when not typing in an input)
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
            e.preventDefault();
            if (raceData) togglePlay();
        }
    });
    document.getElementById('speed-select').addEventListener('change', e => {
        playbackSpeed = parseInt(e.target.value);
    });

    // Power unit toggle (W / W/kg) — restore saved preference
    if (showWkg) {
        const wkgRadio = document.querySelector('input[name="power-unit"][value="wkg"]');
        if (wkgRadio) wkgRadio.checked = true;
    }
    document.querySelectorAll('input[name="power-unit"]').forEach(radio => {
        radio.addEventListener('change', e => {
            showWkg = e.target.value === 'wkg';
            localStorage.setItem('powerUnit', showWkg ? 'wkg' : 'watts');
            updateFrame();
        });
    });
    document.getElementById('check-all').addEventListener('change', e => {
        if (!raceData) return;
        if (e.target.checked) {
            raceData.riders.forEach(r => checkedRanks.add(r.rank));
        } else {
            checkedRanks.clear();
        }
        updateFrame();
    });

    // Zoom controls
    document.getElementById('zoom-slider').addEventListener('input', e => {
        viewWidth = parseFloat(e.target.value);
        document.getElementById('zoom-label').textContent = viewWidth.toFixed(1) + ' km';
        updateViewRange();
        updateFrame();
    });
    document.getElementById('follow-rider-select').addEventListener('change', e => {
        const val = e.target.value;
        if (val === '') {
            selectedRank = null;
            followRider = false;
        } else {
            selectedRank = parseInt(val);
            followRider = true;
            updateViewRange();
        }
        updateFrame();
    });

    // Sortable table headers
    document.querySelectorAll('#rider-table th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const field = th.dataset.sort;
            if (sortField === field) {
                sortAsc = !sortAsc;
            } else {
                sortField = field;
                sortAsc = true;
            }
            updateRiderTable();
        });
    });
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
async function checkAuth() {
    try {
        const resp = await fetch('/auth/status');
        const data = await resp.json();
        const text = document.getElementById('auth-status-text');
        const btn = document.getElementById('authBtn');
        isAuthenticated = data.authenticated;
        if (data.authenticated && data.method === 'oauth') {
            text.textContent = '✓ Logged in via Zwift';
            text.className = 'auth-status logged-in';
            btn.textContent = 'Logout';
            btn.className = 'btn-auth logout';
        } else if (data.authenticated && data.method === 'config') {
            text.textContent = 'Using config token (may be stale)';
            text.className = 'auth-status';
            btn.textContent = 'Login with Zwift';
            btn.className = 'btn-auth';
            isAuthenticated = false;
        } else {
            text.textContent = 'Not logged in';
            text.className = 'auth-status';
            btn.textContent = 'Login with Zwift';
            btn.className = 'btn-auth';
        }
        btn.style.display = 'inline-block';
    } catch (e) { /* ignore */ }
}

function handleAuth() {
    if (isAuthenticated) {
        window.location.href = '/auth/logout';
    } else {
        window.location.href = '/auth/login?next=/race-replay';
    }
}

// ---------------------------------------------------------------------------
// Race list
// ---------------------------------------------------------------------------
async function loadRaceList() {
    try {
        const resp = await fetch('/api/race/list');
        const data = await resp.json();
        const select = document.getElementById('race-select');
        select.innerHTML = '<option value="">-- Select a race --</option>';
        (data.races || []).forEach(race => {
            const opt = document.createElement('option');
            opt.value = race.race_id;
            const label = race.race_name || race.race_id.replace('race_data_', '');
            const riders = race.rider_count ? ` (${race.rider_count} riders)` : '';
            opt.textContent = label + riders;
            select.appendChild(opt);
        });
    } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Fetch race from Zwift API (with SSE progress streaming)
// ---------------------------------------------------------------------------
async function fetchRace(forceRefresh = false) {
    const input = document.getElementById('activity-input').value.trim();
    if (!input) return;

    showStatus('Fetching race data from Zwift API...', 'info');
    showLoading('Connecting to Zwift API...');
    showProgress(0, 0, '');

    let url = `/api/race/fetch_stream?activity_id=${encodeURIComponent(input)}`;
    if (forceRefresh) url += '&force_refresh=1';
    const evtSource = new EventSource(url);

    evtSource.onmessage = async function(event) {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        if (data.progress) {
            const pct = Math.round((data.current / data.total) * 100);
            showLoading(`Fetching rider ${data.current}/${data.total}: ${data.name}`);
            showProgress(data.current, data.total, data.name);
            return;
        }

        // Final message — either success or error
        evtSource.close();
        hideProgress();
        hideLoading();

        if (data.error) {
            console.error('Race fetch error:', data.error);
            showStatus(data.error, 'error');
            return;
        }

        if (data.success) {
            showStatus(data.message, 'success');
            await loadRaceList();
            document.getElementById('race-select').value = data.race_id;
            await loadRaceById(data.race_id);
        }
    };

    evtSource.onerror = function() {
        evtSource.close();
        hideProgress();
        hideLoading();
        showStatus('Connection lost while fetching race data.', 'error');
    };
}

// ---------------------------------------------------------------------------
// Load race data
// ---------------------------------------------------------------------------
async function loadSelectedRace() {
    const raceId = document.getElementById('race-select').value;
    if (!raceId) return;
    await loadRaceById(raceId);
}

async function loadRaceById(raceId) {
    showLoading('Loading and cleaning race data...');
    try {
        // Step 1: Load/clean on server
        const loadResp = await fetch('/api/race/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ race_id: raceId }),
        });
        const loadData = await loadResp.json();
        if (!loadResp.ok || loadData.error) {
            hideLoading();
            console.error('Race load error:', loadResp.status, loadData);
            showStatus(loadData.error || 'Failed to load race', 'error');
            return;
        }

        // Step 2: Fetch full data JSON
        showLoading('Downloading race data...');
        const dataResp = await fetch(`/api/race/data/${raceId}`);
        const fullData = await dataResp.json();
        if (fullData.error) {
            hideLoading();
            console.error('Race data error:', fullData);
            showStatus(fullData.error, 'error');
            return;
        }

        hideLoading();
        initRaceData(fullData);
    } catch (e) {
        hideLoading();
        console.error('Race load exception:', e);
        showStatus('Error loading race: ' + e.message, 'error');
    }
}

// ---------------------------------------------------------------------------
// Initialize race data and UI
// ---------------------------------------------------------------------------
function initRaceData(data) {
    raceData = data;
    currentTime = data.min_time;
    isPlaying = false;
    selectedRank = null;
    checkedRanks.clear();
    data.riders.forEach(r => checkedRanks.add(r.rank));
    riderLookup = {};

    // Initialize zoom to full course
    viewWidth = data.finish_line_km;
    viewXMin = 0;
    viewXMax = data.finish_line_km;
    const zoomSlider = document.getElementById('zoom-slider');
    zoomSlider.max = data.finish_line_km.toFixed(1);
    zoomSlider.value = viewWidth;
    document.getElementById('zoom-label').textContent = viewWidth.toFixed(1) + ' km';

    // Helper: convert array with possible nulls to Float64Array (null → NaN)
    function toF64(arr) {
        return new Float64Array(arr.map(v => v ?? NaN));
    }

    // Build typed-array lookup for each rider for fast interpolation
    data.riders.forEach(r => {
        riderLookup[r.rank] = {
            rank: r.rank,
            name: r.name,
            team: r.team,
            weight_kg: r.weight_kg || 75.0,
            is_finisher: r.is_finisher,
            finish_time_sec: r.finish_time_sec,
            time_sec: new Float64Array(r.time_sec),
            distance_km: toF64(r.distance_km),
            altitude_m: r.altitude_m.length ? toF64(r.altitude_m) : null,
            speed_kmh: r.speed_kmh.length ? toF64(r.speed_kmh) : null,
            power_watts: r.power_watts.length ? toF64(r.power_watts) : null,
            hr_bpm: r.hr_bpm.length ? toF64(r.hr_bpm) : null,
            lat: r.lat && r.lat.length ? toF64(r.lat) : null,
            lng: r.lng && r.lng.length ? toF64(r.lng) : null,
            min_time: r.time_sec[0],
            max_time: r.time_sec[r.time_sec.length - 1],
        };
    });

    // Show panels
    document.getElementById('race-info').style.display = 'flex';
    document.getElementById('playback-panel').style.display = 'block';
    document.getElementById('table-panel').style.display = 'block';
    document.getElementById('elevation-panel').style.display = 'block';
    document.getElementById('peloton-details-panel').style.display = chartMode === 'peloton' ? 'block' : 'none';
    document.getElementById('zoom-controls').style.display = 'flex';

    // Initialize map if map config is available
    if (data.map_config) {
        document.getElementById('map-panel').style.display = 'block';
        initMap(data.map_config, data.route_latlng);
    } else {
        document.getElementById('map-panel').style.display = 'none';
    }

    // Populate follow-rider dropdown
    const followSelect = document.getElementById('follow-rider-select');
    followSelect.innerHTML = '<option value="">None</option>';
    const sortedRiders = [...data.riders].sort((a, b) => a.name.localeCompare(b.name));
    for (const r of sortedRiders) {
        const opt = document.createElement('option');
        opt.value = r.rank;
        opt.textContent = r.name;
        followSelect.appendChild(opt);
    }
    followRider = false;

    // Auto-follow the source rider (whose activity ID was used to fetch)
    if (data.source_activity_id) {
        const sourceRider = data.riders.find(r => r.activity_id === data.source_activity_id);
        if (sourceRider) {
            selectedRank = sourceRider.rank;
            followSelect.value = String(sourceRider.rank);
            followRider = true;
        }
    }

    // Race info
    const routeEl = document.getElementById('info-route');
    const routeName = data.route_name || 'Unknown';
    if (data.route_slug) {
        routeEl.innerHTML = `<a href="https://zwiftinsider.com/route/${data.route_slug}/" target="_blank" rel="noopener">${routeName}</a>`;
    } else {
        routeEl.textContent = routeName;
    }
    document.getElementById('info-riders').textContent = data.riders.length;
    document.getElementById('info-finish').textContent = data.finish_line_km.toFixed(2) + ' km';
    document.getElementById('info-duration').textContent = formatTime(data.max_time - data.min_time);

    // Slider
    const slider = document.getElementById('time-slider');
    slider.min = Math.floor(data.min_time);
    slider.max = Math.floor(data.max_time);
    slider.value = slider.min;
    document.getElementById('time-end').textContent = formatTime(data.max_time - data.min_time);

    // Build elevation trace
    // Pre-compute gradient text for each elevation point
    const eDist = data.elevation_profile.distance_km;
    const eAlt = data.elevation_profile.altitude_m;
    const gradientText = eDist.map((d, i) => {
        const grad = calcGradient(eDist, eAlt, i);
        return `Gradient: ${grad >= 0 ? '+' : ''}${grad.toFixed(1)}%`;
    });

    elevationTrace = {
        x: eDist,
        y: eAlt,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(255,255,255,0.04)',
        line: { color: 'rgba(255,255,255,0.2)', width: 1 },
        text: gradientText,
        hoverinfo: 'text',
        hoverlabel: { bgcolor: '#1e2a3a', bordercolor: '#444', font: { color: '#fff', size: 12 } },
        name: 'Elevation',
    };

    // Update URL with activity ID so the link is shareable
    if (data.source_activity_id) {
        const url = new URL(window.location);
        url.searchParams.set('activity_id', data.source_activity_id);
        window.history.replaceState({}, '', url);
    }

    // Initial render
    initCharts();
    updateFrame();
    showStatus('Race loaded — use playback controls to replay.', 'success');

    // Fetch YouTube stream links (async, non-blocking)
    fetchStreamLinks(data.race_id);
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------
function togglePlay() {
    isPlaying = !isPlaying;
    document.getElementById('play-btn').textContent = isPlaying ? '⏸' : '▶';
    if (isPlaying) {
        lastFrameTs = performance.now();
        animFrameId = requestAnimationFrame(animationLoop);
    } else {
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
}

function animationLoop(timestamp) {
    if (!isPlaying || !raceData) return;

    const dt = (timestamp - lastFrameTs) / 1000; // seconds
    lastFrameTs = timestamp;

    currentTime += dt * playbackSpeed;
    if (currentTime >= raceData.max_time) {
        currentTime = raceData.max_time;
        isPlaying = false;
        document.getElementById('play-btn').textContent = '▶';
    }

    updateFrame();
    if (isPlaying) animFrameId = requestAnimationFrame(animationLoop);
}

function onSliderInput(e) {
    currentTime = parseInt(e.target.value);
    updateFrame();
}

// ---------------------------------------------------------------------------
// Per-frame update (called ~60fps while playing or on slider change)
// ---------------------------------------------------------------------------
function updateFrame() {
    const t = Math.round(currentTime);

    // Update slider & time display
    document.getElementById('time-slider').value = t;
    document.getElementById('time-display').textContent = formatTime(t - raceData.min_time);

    // Calculate rider positions at time t
    const positions = getRiderPositions(t);

    // Follow selected rider if zoomed in
    if (followRider && selectedRank !== null) {
        const sel = positions.find(p => p.rank === selectedRank);
        if (sel) updateViewRangeAroundRider(sel.distance_km);
    }

    // Calculate pelotons
    const pelotons = findPelotons(positions, t);

    updateRiderTable(positions);
    if (chartMode === 'riders') {
        updateElevationChart(positions);
    } else {
        updatePelotonChart(positions, pelotons);
        updatePelotonDetails(pelotons);
    }
    updateMap(positions);
    if (streamLinks.length) updateStreamLinks();
}

// ---------------------------------------------------------------------------
// Rider positions at time t
// ---------------------------------------------------------------------------
function getRiderPositions(t) {
    const positions = [];

    for (const r of raceData.riders) {
        const rl = riderLookup[r.rank];
        if (!rl || t < rl.min_time || t > rl.max_time) continue;

        // Binary search for the index
        const idx = binarySearch(rl.time_sec, t);
        if (idx < 0) continue;

        const dist = interpolate(rl.time_sec, rl.distance_km, t, idx);
        const noData = isNaN(dist);
        const pos = {
            rank: r.rank,
            name: r.name,
            team: r.team,
            weight_kg: rl.weight_kg,
            is_finisher: r.is_finisher,
            finish_time_sec: r.finish_time_sec,
            no_data: noData,
            distance_km: dist,
            speed_kmh: noData ? NaN : (rl.speed_kmh ? interpolate(rl.time_sec, rl.speed_kmh, t, idx) : 0),
            power_watts: noData ? NaN : (rl.power_watts ? stepInterpolate(rl.time_sec, rl.power_watts, t, idx) : 0),
            hr_bpm: noData ? NaN : (rl.hr_bpm ? interpolate(rl.time_sec, rl.hr_bpm, t, idx) : 0),
            altitude_m: noData ? NaN : (rl.altitude_m ? interpolate(rl.time_sec, rl.altitude_m, t, idx) : 0),
            lat: noData ? NaN : (rl.lat ? interpolate(rl.time_sec, rl.lat, t, idx) : NaN),
            lng: noData ? NaN : (rl.lng ? interpolate(rl.time_sec, rl.lng, t, idx) : NaN),
            finished: r.finish_time_sec != null && t >= r.finish_time_sec,
        };
        positions.push(pos);
    }

    // Sort by distance (descending = leader first), no-data riders last
    positions.sort((a, b) => {
        if (a.no_data && !b.no_data) return 1;
        if (!a.no_data && b.no_data) return -1;
        if (a.no_data && b.no_data) return 0;
        return b.distance_km - a.distance_km;
    });

    // Assign live position number (no-data riders get no position)
    let pos = 1;
    positions.forEach(p => { p.position = p.no_data ? '—' : pos++; });

    return positions;
}

// Binary search: find index i such that arr[i] <= t < arr[i+1]
function binarySearch(arr, t) {
    let lo = 0, hi = arr.length - 1;
    if (t < arr[lo] || t > arr[hi]) return -1;
    if (t >= arr[hi]) return hi;
    while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= t) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Linear interpolation between two data points
function interpolate(times, values, t, idx) {
    if (idx >= times.length - 1) return values[idx];
    const t0 = times[idx], t1 = times[idx + 1];
    const dt = t1 - t0;
    if (dt === 0) return values[idx];
    const frac = (t - t0) / dt;
    return values[idx] + frac * (values[idx + 1] - values[idx]);
}

// Step interpolation (for power — hold previous value)
function stepInterpolate(times, values, t, idx) {
    return values[idx];
}

// ---------------------------------------------------------------------------
// Zoom helpers
// ---------------------------------------------------------------------------
function updateViewRange() {
    if (!raceData) return;
    const maxDist = raceData.finish_line_km;
    // If a rider is selected and follow is on, center on them
    if (followRider && selectedRank !== null) {
        const positions = getRiderPositions(Math.round(currentTime));
        const sel = positions.find(p => p.rank === selectedRank);
        if (sel && !sel.no_data) { updateViewRangeAroundRider(sel.distance_km); return; }
    }
    // Otherwise keep current center
    const center = (viewXMin + viewXMax) / 2;
    viewXMin = Math.max(0, center - viewWidth / 2);
    viewXMax = Math.min(maxDist, viewXMin + viewWidth);
    if (viewXMax - viewXMin < viewWidth) viewXMin = Math.max(0, viewXMax - viewWidth);
}

function updateViewRangeAroundRider(riderDist) {
    const maxDist = raceData.finish_line_km;
    // Position rider at 25% from left
    viewXMin = riderDist - viewWidth * 0.25;
    viewXMax = viewXMin + viewWidth;
    if (viewXMin < 0) { viewXMin = 0; viewXMax = viewWidth; }
    if (viewXMax > maxDist) { viewXMax = maxDist; viewXMin = Math.max(0, maxDist - viewWidth); }
}

// ---------------------------------------------------------------------------
// Peloton detection (1-second gap rule using Union-Find)
// ---------------------------------------------------------------------------
function findPelotons(positions, t) {
    if (!positions || positions.length === 0) return [];

    // Exclude riders with no data from peloton calculations
    const withData = positions.filter(p => !p.no_data);
    if (withData.length === 0) return [];

    // Sort by distance descending (leader first)
    const sorted = [...withData].sort((a, b) => b.distance_km - a.distance_km);
    const n = sorted.length;

    // Union-Find
    const parent = Array.from({ length: n }, (_, i) => i);
    function find(x) {
        while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    function union(a, b) {
        const pa = find(a), pb = find(b);
        if (pa !== pb) parent[pa] = pb;
    }

    // Connect adjacent riders within 1-second gap
    for (let i = 0; i < n - 1; i++) {
        const leader = sorted[i];
        const follower = sorted[i + 1];
        const gapKm = leader.distance_km - follower.distance_km;
        const oneSecTravel = (follower.speed_kmh || 30) / 3600;
        if (gapKm <= oneSecTravel) union(i, i + 1);
    }

    // Group by root
    const groups = {};
    for (let i = 0; i < n; i++) {
        const root = find(i);
        if (!groups[root]) groups[root] = [];
        groups[root].push(sorted[i]);
    }

    // Build peloton objects
    const pelotons = Object.values(groups).map(riders => {
        const frontDist = Math.max(...riders.map(r => r.distance_km));
        const backDist = Math.min(...riders.map(r => r.distance_km));
        const avgPower = riders.reduce((s, r) => s + r.power_watts, 0) / riders.length;
        const avgSpeed = riders.reduce((s, r) => s + r.speed_kmh, 0) / riders.length;
        const avgWeight = riders.reduce((s, r) => s + (r.weight_kg || 75), 0) / riders.length;
        return {
            rider_ranks: riders.map(r => r.rank),
            front_distance_km: frontDist,
            back_distance_km: backDist,
            avg_power_watts: avgPower,
            avg_speed_kmh: avgSpeed,
            avg_weight_kg: avgWeight,
            size: riders.length,
            length_m: (frontDist - backDist) * 1000,
        };
    });

    // Sort pelotons by front distance (leaders first)
    pelotons.sort((a, b) => b.front_distance_km - a.front_distance_km);
    return pelotons;
}

// ---------------------------------------------------------------------------
// Normalized Power helper
// ---------------------------------------------------------------------------
function getNormalizedPower(rank, t) {
    const rl = riderLookup[rank];
    if (!rl || !rl.power_watts) return NaN;
    const startIdx = binarySearch(rl.time_sec, rl.min_time);
    const endIdx = binarySearch(rl.time_sec, t);
    if (startIdx < 0 || endIdx < 0 || endIdx - startIdx < 30) return NaN;

    // Collect valid power samples in the range
    const samples = [];
    for (let i = startIdx; i <= endIdx; i++) {
        const p = rl.power_watts[i];
        samples.push(isNaN(p) ? 0 : p);
    }
    if (samples.length < 30) return NaN;

    // 30-sample rolling average → raise to 4th power → mean → 4th root
    let np4Sum = 0;
    let np4Count = 0;
    let windowSum = 0;
    for (let i = 0; i < samples.length; i++) {
        windowSum += samples[i];
        if (i >= 30) windowSum -= samples[i - 30];
        if (i >= 29) {
            const avg = windowSum / 30;
            const avg4 = avg * avg * avg * avg;
            np4Sum += avg4;
            np4Count++;
        }
    }

    if (np4Count === 0) return NaN;
    return Math.pow(np4Sum / np4Count, 0.25);
}

// ---------------------------------------------------------------------------
// 1-minute average power helper
// ---------------------------------------------------------------------------
function get1MinPower(rank, t) {
    const rl = riderLookup[rank];
    if (!rl || !rl.power_watts) return NaN;
    const startT = Math.max(rl.min_time, t - 60);
    const startIdx = binarySearch(rl.time_sec, startT);
    const endIdx = binarySearch(rl.time_sec, t);
    if (startIdx < 0 || endIdx < 0) return NaN;
    let sum = 0, count = 0;
    for (let i = startIdx; i <= Math.min(endIdx, rl.power_watts.length - 1); i++) {
        if (!isNaN(rl.power_watts[i])) {
            sum += rl.power_watts[i];
            count++;
        }
    }
    return count > 0 ? sum / count : NaN;
}

// ---------------------------------------------------------------------------
// Rider Table
// ---------------------------------------------------------------------------
function updateRiderTable(positions) {
    if (!positions) {
        positions = getRiderPositions(Math.round(currentTime));
    }

    // Sort by chosen field
    const sorted = [...positions];
    sorted.sort((a, b) => {
        // No-data riders always sort to the bottom
        if (a.no_data && !b.no_data) return 1;
        if (!a.no_data && b.no_data) return -1;
        if (a.no_data && b.no_data) return 0;
        let va, vb;
        switch (sortField) {
            case 'position': va = a.position; vb = b.position; break;
            case 'name': va = a.name; vb = b.name; break;
            case 'gap': va = a.distance_km; vb = b.distance_km; return sortAsc ? vb - va : va - vb;
            case 'power':
                va = showWkg && a.weight_kg ? a.power_watts / a.weight_kg : a.power_watts;
                vb = showWkg && b.weight_kg ? b.power_watts / b.weight_kg : b.power_watts;
                break;
            case 'power1m':
                va = get1MinPower(a.rank, Math.round(currentTime));
                vb = get1MinPower(b.rank, Math.round(currentTime));
                if (showWkg) { va = a.weight_kg ? va / a.weight_kg : 0; vb = b.weight_kg ? vb / b.weight_kg : 0; }
                break;
            case 'np':
                va = getNormalizedPower(a.rank, Math.round(currentTime));
                vb = getNormalizedPower(b.rank, Math.round(currentTime));
                if (showWkg) { va = a.weight_kg ? va / a.weight_kg : 0; vb = b.weight_kg ? vb / b.weight_kg : 0; }
                break;
            case 'weight': va = a.weight_kg || 0; vb = b.weight_kg || 0; break;
            case 'hr': va = a.hr_bpm; vb = b.hr_bpm; break;
            case 'speed': va = a.speed_kmh; vb = b.speed_kmh; break;
            case 'distance': va = a.distance_km; vb = b.distance_km; break;
            default: va = a.position; vb = b.position;
        }
        if (typeof va === 'string') {
            return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        }
        return sortAsc ? va - vb : vb - va;
    });

    const withData = positions.filter(p => !p.no_data);
    const leaderDist = withData.length > 0 ?
        Math.max(...withData.map(p => p.distance_km)) : 0;
    const leaderSpeed = withData.length > 0 ?
        withData.find(p => p.position === 1)?.speed_kmh || 30 : 30;

    const tbody = document.getElementById('rider-tbody');
    // Build HTML in batch for performance
    let html = '';
    for (const p of sorted) {
        const isSelected = p.rank === selectedRank;
        const isChecked = checkedRanks.has(p.rank);
        const classes = [];
        if (isSelected) classes.push('selected');
        if (isChecked) classes.push('checked');
        if (p.finished) classes.push('finished');

        // Gap calculation
        const gapKm = leaderDist - p.distance_km;
        const gapSec = leaderSpeed > 0 ? (gapKm / (leaderSpeed / 3600)) : 0;
        const gapStr = p.no_data ? '—' : (p.position === 1 ? '—' : `+${gapSec.toFixed(1)}s`);

        // 1-min avg power
        const power1m = get1MinPower(p.rank, Math.round(currentTime));

        // Normalized power
        const np = getNormalizedPower(p.rank, Math.round(currentTime));

        // Display values — show '—' when no data available
        const nd = p.no_data;
        const pwrStr = nd ? '—' : formatPower(p.power_watts, p.weight_kg);
        const pwr1mStr = nd || isNaN(power1m) ? '—' : formatPower(power1m, p.weight_kg);
        const npStr = nd || isNaN(np) ? '—' : formatPower(np, p.weight_kg);
        const hrStr = nd || isNaN(p.hr_bpm) ? '—' : Math.round(p.hr_bpm);
        const spdStr = nd ? '—' : p.speed_kmh.toFixed(1);
        const distStr = nd ? '—' : p.distance_km.toFixed(2);

        html += `<tr class="${classes.join(' ')}${nd ? ' no-data' : ''}" data-rank="${p.rank}">
            <td class="col-check"><input type="checkbox" ${isChecked ? 'checked' : ''} data-rank="${p.rank}"></td>
            <td class="col-pos">${p.position}</td>
            <td class="col-name">${p.name}${p.finished ? ' 🏁' : ''}${nd ? ' <span class="no-data-badge">No data</span>' : ''}</td>
            <td class="col-gap">${gapStr}</td>
            <td class="col-power">${pwrStr}</td>
            <td class="col-power1m">${pwr1mStr}</td>
            <td class="col-np">${npStr}</td>
            <td class="col-weight">${nd || !p.weight_kg ? '—' : p.weight_kg.toFixed(1)}</td>
            <td class="col-hr">${hrStr}</td>
            <td class="col-speed">${spdStr}</td>
            <td class="col-dist">${distStr}</td>
        </tr>`;
    }
    tbody.innerHTML = html;

    // Re-bind click & checkbox events
    tbody.querySelectorAll('tr').forEach(tr => {
        tr.addEventListener('click', e => {
            if (e.target.type === 'checkbox') return;
            const rank = parseInt(tr.dataset.rank);
            selectedRank = selectedRank === rank ? null : rank;
            // Sync the follow dropdown
            const followSelect = document.getElementById('follow-rider-select');
            followSelect.value = selectedRank !== null ? String(selectedRank) : '';
            followRider = selectedRank !== null;
            if (followRider) updateViewRange();
            updateFrame();
        });
    });
    tbody.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', e => {
            const rank = parseInt(cb.dataset.rank);
            if (cb.checked) checkedRanks.add(rank);
            else checkedRanks.delete(rank);
            updateFrame();
        });
    });
}

// ---------------------------------------------------------------------------
// Charts — Elevation Profile
// ---------------------------------------------------------------------------
function initCharts() {
    Plotly.newPlot('elevation-chart', [elevationTrace], {
        ...PLOTLY_LAYOUT_BASE,
        hovermode: 'closest',
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Distance (km)', range: [viewXMin, viewXMax] },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Altitude (m)' },
        showlegend: false,
    }, { responsive: true, displayModeBar: false });

    // Chart mode toggle buttons
    document.getElementById('chart-mode-riders').addEventListener('click', () => setChartMode('riders'));
    document.getElementById('chart-mode-peloton').addEventListener('click', () => setChartMode('peloton'));
}

function setChartMode(mode) {
    chartMode = mode;
    document.getElementById('chart-mode-riders').classList.toggle('active', mode === 'riders');
    document.getElementById('chart-mode-peloton').classList.toggle('active', mode === 'peloton');
    document.getElementById('chart-heading').textContent = mode === 'riders' ? 'Individual Riders' : 'Peloton Groups';
    // Show/hide peloton details based on mode
    const detailsPanel = document.getElementById('peloton-details-panel');
    if (detailsPanel) detailsPanel.style.display = mode === 'peloton' ? 'block' : 'none';
    updateFrame();
}

function updateElevationChart(positions) {
    if (!positions || !raceData) return;

    // Rider dots on elevation — only show checked riders (+ followed rider)
    const showRanks = checkedRanks;

    const riderTraces = [];
    for (const p of positions) {
        if (p.no_data) continue;  // skip riders with no telemetry data
        if (!showRanks.has(p.rank) && p.rank !== selectedRank) continue;

        const isSelected = p.rank === selectedRank;
        const color = isSelected ? '#ff6b35' :
            (checkedRanks.has(p.rank) ? '#66bb6a' : '#29b6f6');
        const size = isSelected ? 12 : 8;

        riderTraces.push({
            x: [p.distance_km],
            y: [p.altitude_m],
            type: 'scatter',
            mode: 'markers+text',
            marker: { color, size, line: { color: '#fff', width: 1 } },
            text: [isSelected || checkedRanks.has(p.rank) ? p.name.split(' ')[0] : ''],
            textposition: 'top center',
            textfont: { size: 9, color },
            hovertext: `${p.name}<br>P${p.position} — ${p.distance_km.toFixed(2)} km<br>${formatPower(p.power_watts, p.weight_kg)}`,
            hoverinfo: 'text',
            hoverlabel: { bgcolor: '#1e2a3a', bordercolor: color, font: { color: '#fff', size: 12 } },
            showlegend: false,
            name: p.name,
        });
    }

    Plotly.react('elevation-chart', [elevationTrace, ...riderTraces], {
        ...PLOTLY_LAYOUT_BASE,
        hovermode: 'closest',
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Distance (km)', range: [viewXMin, viewXMax] },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Altitude (m)' },
        showlegend: false,
    });
}

// ---------------------------------------------------------------------------
// Charts — Peloton Elevation
// ---------------------------------------------------------------------------
function updatePelotonChart(positions, pelotons) {
    if (!positions || !raceData) return;

    const elevData = raceData.elevation_profile;
    const pelotonTraces = [];

    for (let i = 0; i < pelotons.length; i++) {
        const p = pelotons[i];
        const color = PELOTON_COLORS[i % PELOTON_COLORS.length];

        // Interpolate altitude at front and back
        const frontAlt = interpElevation(elevData, p.front_distance_km);
        const backAlt = interpElevation(elevData, p.back_distance_km);

        // Highlight peloton that contains selected rider
        const containsSelected = selectedRank !== null && p.rider_ranks.includes(selectedRank);
        const lineW = containsSelected ? 10 : 6;
        const markerS = containsSelected ? 10 : 6;

        pelotonTraces.push({
            x: [p.back_distance_km, p.front_distance_km],
            y: [backAlt, frontAlt],
            type: 'scatter',
            mode: 'lines+markers',
            line: { width: lineW, color },
            marker: { size: markerS, color },
            hovertext: `Peloton (${p.size} riders)<br>${formatPower(p.avg_power_watts, p.avg_weight_kg || 75)}<br>${p.avg_speed_kmh.toFixed(1)} km/h`,
            hoverinfo: 'text',
            hoverlabel: { bgcolor: '#1e2a3a', bordercolor: color, font: { color: '#fff', size: 12 } },
            showlegend: false,
        });
    }

    // Add a single dot for the followed rider (if any)
    if (selectedRank !== null) {
        const sel = positions.find(p => p.rank === selectedRank && !p.no_data);
        if (sel) {
            const selAlt = interpElevation(raceData.elevation_profile, sel.distance_km);
            pelotonTraces.push({
                x: [sel.distance_km],
                y: [selAlt],
                type: 'scatter',
                mode: 'markers+text',
                marker: { color: '#ff6b35', size: 12, line: { color: '#fff', width: 2 } },
                text: [sel.name.split(' ')[0]],
                textposition: 'top center',
                textfont: { size: 9, color: '#ff6b35' },
                hovertext: `${sel.name}<br>P${sel.position} — ${sel.distance_km.toFixed(2)} km<br>${formatPower(sel.power_watts, sel.weight_kg)}`,
                hoverinfo: 'text',
                hoverlabel: { bgcolor: '#1e2a3a', bordercolor: '#ff6b35', font: { color: '#fff', size: 12 } },
                showlegend: false,
            });
        }
    }

    Plotly.react('elevation-chart', [elevationTrace, ...pelotonTraces], {
        ...PLOTLY_LAYOUT_BASE,
        hovermode: 'closest',
        xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Distance (km)', range: [viewXMin, viewXMax] },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Altitude (m)' },
        showlegend: false,
    });
}

function interpElevation(elevData, distKm) {
    const dists = elevData.distance_km;
    const alts = elevData.altitude_m;
    if (distKm <= dists[0]) return alts[0];
    if (distKm >= dists[dists.length - 1]) return alts[alts.length - 1];
    // Binary search
    let lo = 0, hi = dists.length - 1;
    while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (dists[mid] <= distKm) lo = mid; else hi = mid;
    }
    const frac = (distKm - dists[lo]) / (dists[hi] - dists[lo]);
    return alts[lo] + frac * (alts[hi] - alts[lo]);
}

/**
 * Calculate gradient (%) at a given index in the elevation profile.
 * Uses a local window to smooth out noise.
 */
function calcGradient(dists, alts, idx) {
    const window = 5;  // look ±5 samples
    const lo = Math.max(0, idx - window);
    const hi = Math.min(dists.length - 1, idx + window);
    const dDist = (dists[hi] - dists[lo]) * 1000;  // km → m
    if (dDist === 0) return 0;
    const dAlt = alts[hi] - alts[lo];
    return (dAlt / dDist) * 100;
}

// ---------------------------------------------------------------------------
// Peloton Details Panel
// ---------------------------------------------------------------------------
function updatePelotonDetails(pelotons) {
    const container = document.getElementById('peloton-details-body');
    if (!pelotons || pelotons.length === 0) {
        container.innerHTML = '<p style="color:#888;">No peloton data</p>';
        return;
    }

    let html = `<div class="peloton-summary">${pelotons.length} group${pelotons.length > 1 ? 's' : ''}</div>`;
    html += '<div class="peloton-grid">';
    for (let i = 0; i < pelotons.length; i++) {
        const p = pelotons[i];
        const color = PELOTON_COLORS[i % PELOTON_COLORS.length];
        const containsSelected = selectedRank !== null && p.rider_ranks.includes(selectedRank);
        html += `<div class="peloton-card${containsSelected ? ' highlighted' : ''}" style="border-left: 4px solid ${color};">
            <strong>Group ${i + 1}</strong> <span class="peloton-size">(${p.size} rider${p.size > 1 ? 's' : ''})</span>
            <div class="peloton-stats">
                <span>⚡ ${formatPower(p.avg_power_watts, p.avg_weight_kg || 75)}</span>
                <span>🚴 ${p.avg_speed_kmh.toFixed(1)} km/h</span>
                <span>📏 ${p.length_m.toFixed(0)}m</span>
            </div>
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// YouTube Stream Links
// ---------------------------------------------------------------------------
async function fetchStreamLinks(raceId) {
    streamLinks = [];
    const panel = document.getElementById('stream-links-panel');
    panel.style.display = 'none';

    try {
        const resp = await fetch(`/api/race/streams/${raceId}`);
        const data = await resp.json();
        if (data.streams && data.streams.length > 0) {
            streamLinks = data.streams;
            panel.style.display = 'block';
            updateStreamLinks();
        }
    } catch (e) {
        // Stream links are optional — fail silently
    }
}

function updateStreamLinks() {
    const panel = document.getElementById('stream-links-body');
    if (!streamLinks.length || !raceData) {
        panel.innerHTML = '';
        return;
    }

    const elapsed = Math.round(currentTime - raceData.min_time);
    let html = '<span class="stream-label">📺 Streams:</span>';

    for (const s of streamLinks) {
        const streamSec = s.offset_seconds + elapsed;
        const t = Math.max(0, Math.round(streamSec));
        const url = `${s.youtube_url}&t=${t}s`;
        // Format as h:mm:ss for streams (can be long)
        const hrs = Math.floor(t / 3600);
        const mins = Math.floor((t % 3600) / 60);
        const secs = t % 60;
        const timeStr = hrs > 0
            ? `${hrs}:${mins.toString().padStart(2,'0')}:${secs.toString().padStart(2,'0')}`
            : `${mins}:${secs.toString().padStart(2,'0')}`;
        html += `<a href="${url}" target="_blank" rel="noopener" class="stream-link" title="${s.stream_title}">`;
        html += `<span class="yt-icon">▶</span> ${s.streamer_name} (${timeStr})`;
        html += `</a>`;
    }

    panel.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function formatTime(seconds) {
    const s = Math.abs(Math.round(seconds));
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function formatPower(watts, weightKg) {
    if (showWkg && weightKg > 0) {
        return (watts / weightKg).toFixed(1) + ' W/kg';
    }
    return Math.round(watts) + 'W';
}

function showStatus(msg, type) {
    const el = document.getElementById('race-status');
    el.style.display = 'block';
    el.className = 'status-msg ' + type;
    el.textContent = msg;
}

function showLoading(text) {
    document.getElementById('loading-text').textContent = text || 'Loading...';
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showProgress(current, total, name) {
    let bar = document.getElementById('fetch-progress-bar');
    if (!bar) {
        // Create progress bar container dynamically
        const overlay = document.getElementById('loading-overlay');
        const wrap = document.createElement('div');
        wrap.id = 'fetch-progress-wrap';
        wrap.style.cssText = 'width:320px; margin-top:12px; text-align:center;';
        wrap.innerHTML = `
            <div style="background:#333; border-radius:6px; overflow:hidden; height:22px; width:100%;">
                <div id="fetch-progress-bar" style="height:100%; width:0%; background:#4caf50; transition:width 0.3s; border-radius:6px;"></div>
            </div>
            <div id="fetch-progress-text" style="margin-top:6px; font-size:13px; color:#aaa;"></div>
        `;
        overlay.appendChild(wrap);
        bar = document.getElementById('fetch-progress-bar');
    }
    document.getElementById('fetch-progress-wrap').style.display = '';
    if (total > 0) {
        const pct = Math.round((current / total) * 100);
        bar.style.width = pct + '%';
        document.getElementById('fetch-progress-text').textContent = `${current} / ${total} riders`;
    }
}

function hideProgress() {
    const wrap = document.getElementById('fetch-progress-wrap');
    if (wrap) wrap.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Map panel — interactive canvas with world map, route, and rider dots
// ---------------------------------------------------------------------------
let mapState = null;  // null when no map available

function initMap(config, routeLatlng) {
    const canvas = document.getElementById('map-canvas');
    const container = document.getElementById('map-container');
    if (!canvas || !container) return;

    // Set canvas resolution to match container
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    // Load the map image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = config.image;

    // Build route points from ZwiftMap official route data
    let routePoints = null;
    if (routeLatlng && routeLatlng.length > 1) {
        routePoints = routeLatlng.map(pt => ({ lat: pt[0], lng: pt[1] }));
    }

    mapState = {
        config,
        canvas,
        ctx: canvas.getContext('2d'),
        img,
        imgLoaded: false,
        // View in GPS coordinates — start with full world bounds
        viewLat: (config.lat_min + config.lat_max) / 2,
        viewLng: (config.lng_min + config.lng_max) / 2,
        viewScale: 1.0,  // 1.0 = full world fits in canvas
        // Drag state
        dragging: false,
        dragStartX: 0,
        dragStartY: 0,
        dragStartLat: 0,
        dragStartLng: 0,
        // Official route polyline from ZwiftMap
        routePoints,
        // Background color
        bgColor: config.bg_color || '#0a1a2e',
    };

    img.onload = () => {
        mapState.imgLoaded = true;
        // Fit the map to show the full world
        fitMapToRoute();
        drawMap([]);
    };

    // Bind mouse events for zoom/pan
    canvas.addEventListener('wheel', onMapWheel, { passive: false });
    canvas.addEventListener('mousedown', onMapMouseDown);
    canvas.addEventListener('mousemove', onMapMouseMove);
    canvas.addEventListener('mouseup', onMapMouseUp);
    canvas.addEventListener('mouseleave', onMapMouseUp);

    // Map zoom buttons
    const zoomInBtn = document.getElementById('map-zoom-in');
    const zoomOutBtn = document.getElementById('map-zoom-out');
    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => {
            if (!mapState) return;
            mapState.viewScale = Math.min(50, mapState.viewScale * 1.4);
            drawMap(lastMapPositions);
        });
    }
    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => {
            if (!mapState) return;
            mapState.viewScale = Math.max(1.0, mapState.viewScale / 1.4);
            drawMap(lastMapPositions);
        });
    }

    // Handle resize
    window.addEventListener('resize', () => {
        if (!mapState) return;
        const r = container.getBoundingClientRect();
        canvas.width = r.width * window.devicePixelRatio;
        canvas.height = r.height * window.devicePixelRatio;
        canvas.style.width = r.width + 'px';
        canvas.style.height = r.height + 'px';
    });
}



function fitMapToRoute() {
    if (!mapState) return;
    const cfg = mapState.config;

    // If we have route points, fit to those; otherwise fit to world bounds
    let latMin, latMax, lngMin, lngMax;
    if (mapState.routePoints && mapState.routePoints.length > 2) {
        latMin = Infinity; latMax = -Infinity;
        lngMin = Infinity; lngMax = -Infinity;
        for (const p of mapState.routePoints) {
            if (p.lat < latMin) latMin = p.lat;
            if (p.lat > latMax) latMax = p.lat;
            if (p.lng < lngMin) lngMin = p.lng;
            if (p.lng > lngMax) lngMax = p.lng;
        }
        // Add 15% padding
        const latPad = (latMax - latMin) * 0.15 || 0.001;
        const lngPad = (lngMax - lngMin) * 0.15 || 0.001;
        latMin -= latPad;
        latMax += latPad;
        lngMin -= lngPad;
        lngMax += lngPad;
    } else {
        latMin = cfg.lat_min;
        latMax = cfg.lat_max;
        lngMin = cfg.lng_min;
        lngMax = cfg.lng_max;
    }

    // Center the view
    mapState.viewLat = (latMin + latMax) / 2;
    mapState.viewLng = (lngMin + lngMax) / 2;

    // Calculate scale to fit the route bounds in the canvas
    const canvas = mapState.canvas;
    const imgNatW = mapState.imgLoaded ? mapState.img.naturalWidth : 1000;
    const imgNatH = mapState.imgLoaded ? mapState.img.naturalHeight : 1000;
    const fitScale = Math.min(canvas.width / imgNatW, canvas.height / imgNatH);

    // Route extent in image pixels
    const routeImgW = (lngMax - lngMin) / (cfg.lng_max - cfg.lng_min) * imgNatW;
    const routeImgH = (latMax - latMin) / (cfg.lat_max - cfg.lat_min) * imgNatH;

    // We need: routeImgW * fitScale * viewScale <= canvas.width
    //          routeImgH * fitScale * viewScale <= canvas.height
    const scaleByW = canvas.width / (routeImgW * fitScale);
    const scaleByH = canvas.height / (routeImgH * fitScale);
    // Zoom in 2x beyond route-fit for a closer default view, but never below 1.0
    mapState.viewScale = Math.max(1.0, Math.min(scaleByW, scaleByH) * 1.8);
}

/**
 * Convert GPS coords to canvas pixel coords.
 * Uses the map image's natural dimensions for correct aspect ratio.
 */
function gpsToCanvas(lat, lng) {
    const ms = mapState;
    const cfg = ms.config;
    const canvas = ms.canvas;

    // GPS → image pixel (linear mapping)
    const imgNatW = ms.imgLoaded ? ms.img.naturalWidth : 1000;
    const imgNatH = ms.imgLoaded ? ms.img.naturalHeight : 1000;
    const imgX = (lng - cfg.lng_min) / (cfg.lng_max - cfg.lng_min) * imgNatW;
    const imgY = (cfg.lat_max - lat) / (cfg.lat_max - cfg.lat_min) * imgNatH;

    // View center in image pixels
    const cImgX = (ms.viewLng - cfg.lng_min) / (cfg.lng_max - cfg.lng_min) * imgNatW;
    const cImgY = (cfg.lat_max - ms.viewLat) / (cfg.lat_max - cfg.lat_min) * imgNatH;

    // Fit scale: at viewScale=1, the full image fits in the canvas
    const fitScale = Math.min(canvas.width / imgNatW, canvas.height / imgNatH);
    const s = fitScale * ms.viewScale;

    const x = canvas.width / 2 + (imgX - cImgX) * s;
    const y = canvas.height / 2 + (imgY - cImgY) * s;

    return { x, y };
}

/**
 * Convert canvas pixel coords back to GPS coords.
 */
function canvasToGps(px, py) {
    const ms = mapState;
    const cfg = ms.config;
    const canvas = ms.canvas;

    const imgNatW = ms.imgLoaded ? ms.img.naturalWidth : 1000;
    const imgNatH = ms.imgLoaded ? ms.img.naturalHeight : 1000;

    const cImgX = (ms.viewLng - cfg.lng_min) / (cfg.lng_max - cfg.lng_min) * imgNatW;
    const cImgY = (cfg.lat_max - ms.viewLat) / (cfg.lat_max - cfg.lat_min) * imgNatH;

    const fitScale = Math.min(canvas.width / imgNatW, canvas.height / imgNatH);
    const s = fitScale * ms.viewScale;

    const imgX = cImgX + (px - canvas.width / 2) / s;
    const imgY = cImgY + (py - canvas.height / 2) / s;

    const lng = cfg.lng_min + (imgX / imgNatW) * (cfg.lng_max - cfg.lng_min);
    const lat = cfg.lat_max - (imgY / imgNatH) * (cfg.lat_max - cfg.lat_min);

    return { lat, lng };
}

function drawMap(positions) {
    const ms = mapState;
    if (!ms) return;
    const { ctx, canvas, config: cfg } = ms;
    const dpr = window.devicePixelRatio;

    // Clear
    ctx.fillStyle = ms.bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw world map image
    if (ms.imgLoaded) {
        const topLeft = gpsToCanvas(cfg.lat_max, cfg.lng_min);
        const bottomRight = gpsToCanvas(cfg.lat_min, cfg.lng_max);
        const imgW = bottomRight.x - topLeft.x;
        const imgH = bottomRight.y - topLeft.y;
        ctx.drawImage(ms.img, topLeft.x, topLeft.y, imgW, imgH);
    }

    // Draw route polyline using Catmull-Rom spline for smooth curves
    if (ms.routePoints && ms.routePoints.length > 1) {
        // Convert GPS points to canvas pixel coordinates
        const pts = ms.routePoints.map(p => gpsToCanvas(p.lat, p.lng));

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 2 * dpr;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.moveTo(pts[0].x, pts[0].y);

        if (pts.length === 2) {
            ctx.lineTo(pts[1].x, pts[1].y);
        } else {
            // Catmull-Rom to cubic bezier conversion (alpha = 0.5 centripetal)
            const tension = 0.5;   // 0 = sharp, 1 = very loose
            for (let i = 0; i < pts.length - 1; i++) {
                const p0 = pts[Math.max(i - 1, 0)];
                const p1 = pts[i];
                const p2 = pts[i + 1];
                const p3 = pts[Math.min(i + 2, pts.length - 1)];

                // Control points derived from Catmull-Rom tangents
                const cp1x = p1.x + (p2.x - p0.x) * tension / 3;
                const cp1y = p1.y + (p2.y - p0.y) * tension / 3;
                const cp2x = p2.x - (p3.x - p1.x) * tension / 3;
                const cp2y = p2.y - (p3.y - p1.y) * tension / 3;

                ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
            }
        }
        ctx.stroke();
    }

    // Draw rider dots
    if (!positions || positions.length === 0) return;

    const dotRadius = 5 * dpr;
    const selectedDotRadius = 7 * dpr;

    for (const p of positions) {
        if (p.no_data || isNaN(p.lat) || isNaN(p.lng)) continue;
        if (!checkedRanks.has(p.rank)) continue;

        const { x, y } = gpsToCanvas(p.lat, p.lng);

        // Skip dots outside visible area (with margin)
        if (x < -20 || x > canvas.width + 20 || y < -20 || y > canvas.height + 20) continue;

        const isSelected = p.rank === selectedRank;
        const r = isSelected ? selectedDotRadius : dotRadius;

        // Color: selected = orange, others = cyan
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        if (isSelected) {
            ctx.fillStyle = '#f7931e';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2 * dpr;
            ctx.stroke();
        } else {
            ctx.fillStyle = p.finished ? '#81c784' : '#29b6f6';
            ctx.fill();
        }
    }

    // Draw selected rider name label
    if (selectedRank !== null) {
        const sel = positions.find(p => p.rank === selectedRank);
        if (sel && !sel.no_data && !isNaN(sel.lat) && !isNaN(sel.lng)) {
            const { x, y } = gpsToCanvas(sel.lat, sel.lng);
            const label = sel.name;
            ctx.font = `bold ${12 * dpr}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';

            // Text background
            const metrics = ctx.measureText(label);
            const pad = 4 * dpr;
            const labelY = y - selectedDotRadius - 4 * dpr;
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(x - metrics.width / 2 - pad, labelY - 12 * dpr - pad,
                         metrics.width + pad * 2, 12 * dpr + pad * 2);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x, labelY);
        }
    }
}

function updateMap(positions) {
    if (!mapState) return;
    lastMapPositions = positions;

    // Follow selected rider on the map
    if (followRider && selectedRank !== null) {
        const sel = positions.find(p => p.rank === selectedRank);
        if (sel && !sel.no_data && !isNaN(sel.lat) && !isNaN(sel.lng)) {
            mapState.viewLat = sel.lat;
            mapState.viewLng = sel.lng;
        }
    }

    drawMap(positions);
}

// --- Map mouse events ---
function onMapWheel(e) {
    if (!mapState) return;
    e.preventDefault();

    const rect = mapState.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio;
    const mx = (e.clientX - rect.left) * dpr;
    const my = (e.clientY - rect.top) * dpr;

    // GPS position under cursor before zoom
    const gpsBefore = canvasToGps(mx, my);

    const zoomFactor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    mapState.viewScale = Math.max(1.0, Math.min(50, mapState.viewScale * zoomFactor));

    // GPS position under cursor after zoom (with new scale)
    const gpsAfter = canvasToGps(mx, my);

    // Adjust center so the point under cursor stays fixed
    mapState.viewLat += gpsBefore.lat - gpsAfter.lat;
    mapState.viewLng += gpsBefore.lng - gpsAfter.lng;

    drawMap(lastMapPositions);
}

function onMapMouseDown(e) {
    if (!mapState) return;
    mapState.dragging = true;
    mapState.dragStartX = e.clientX;
    mapState.dragStartY = e.clientY;
    mapState.dragStartLat = mapState.viewLat;
    mapState.dragStartLng = mapState.viewLng;
}

function onMapMouseMove(e) {
    if (!mapState || !mapState.dragging) return;
    const dpr = window.devicePixelRatio;
    const dx = (e.clientX - mapState.dragStartX) * dpr;
    const dy = (e.clientY - mapState.dragStartY) * dpr;

    // Convert pixel delta to GPS delta using image-based scaling
    const cfg = mapState.config;
    const canvas = mapState.canvas;
    const imgNatW = mapState.imgLoaded ? mapState.img.naturalWidth : 1000;
    const imgNatH = mapState.imgLoaded ? mapState.img.naturalHeight : 1000;
    const fitScale = Math.min(canvas.width / imgNatW, canvas.height / imgNatH);
    const s = fitScale * mapState.viewScale;

    // dx in canvas pixels → dlng in GPS degrees
    const dImgX = dx / s;
    const dImgY = dy / s;
    const dlng = (dImgX / imgNatW) * (cfg.lng_max - cfg.lng_min);
    const dlat = (dImgY / imgNatH) * (cfg.lat_max - cfg.lat_min);

    mapState.viewLat = mapState.dragStartLat + dlat;
    mapState.viewLng = mapState.dragStartLng - dlng;

    drawMap(lastMapPositions);
}

function onMapMouseUp() {
    if (!mapState) return;
    mapState.dragging = false;
}

// Keep track of last positions for redraw during zoom/pan
let lastMapPositions = [];
