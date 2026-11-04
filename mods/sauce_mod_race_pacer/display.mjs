import * as Common from '/pages/src/common.mjs';
import { bucketSpans, spanIndexAt, BucketAverage, paceStatus } from './bucket-tracker.mjs';
import { DETAIL_DEFAULTS, detailHideClasses } from './details.mjs';

Common.enableSentry();

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let refTimes = null;
let refDistances = null;
let refPowers = null;
let refSpeeds = null;
let refWeightKg = null;
let refName = '';
let refRouteName = '';

// Bucketed TT plans: bucket spans from the CSV and the rider's running average
let refBucketSpans = [];
const bucketAvg = new BucketAverage();

// Anchor point set by the user (or auto-set on race start)
let anchorDistanceM = null;  // raw distance value at t=0
let anchorMs = null;         // wall-clock ms at t=0

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const filePanel = document.getElementById('file-panel');
const csvFileInput = document.getElementById('csv-file');
const fileStatus = document.getElementById('file-status');
const gapDisplay = document.getElementById('gap-display');
const routeNameEl = gapDisplay.querySelector('.route-name');
const gapTimeEl = gapDisplay.querySelector('.gap-time');
const gapLabelEl = gapDisplay.querySelector('.gap-label');
const refPowerEl = gapDisplay.querySelector('.ref-power');
const refSpeedEl = gapDisplay.querySelector('.ref-speed');
const rideNameEl = gapDisplay.querySelector('.ride-name');
const bucketAvgEl = gapDisplay.querySelector('.bucket-avg');
const bucketInfoEl = gapDisplay.querySelector('.bucket-info');
const resetBtn = document.getElementById('reset-btn');
const changeBtn = document.getElementById('change-btn');

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------
function parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();

    if (lines.length < 5) {
        throw new Error('File too short — need metadata header, values, data header, and data rows.');
    }

    const metaKeys = lines[0].split(',').map(s => s.trim());
    const metaVals = lines[1].split(',').map(s => s.trim());
    const meta = {};
    for (let i = 0; i < metaKeys.length; i++) meta[metaKeys[i]] = metaVals[i] ?? '';

    const dataHeader = lines[2].split(',').map(s => s.trim().toLowerCase());
    const timeIdx = dataHeader.indexOf('time_sec');
    const distIdx = dataHeader.indexOf('distance_m');
    const pwrIdx  = dataHeader.indexOf('power_watts');
    const spdIdx  = dataHeader.indexOf('speed_kmh');
    const bktIdx  = dataHeader.indexOf('bucket');
    if (timeIdx === -1) throw new Error('Missing column: time_sec');
    if (distIdx === -1) throw new Error('Missing column: distance_m');

    const times = [];
    const distances = [];
    const powers = pwrIdx !== -1 ? [] : null;
    const speeds = spdIdx !== -1 ? [] : null;
    const buckets = bktIdx !== -1 ? [] : null;
    for (let i = 3; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const cols = line.split(',');
        const t = parseFloat(cols[timeIdx]);
        const d = parseFloat(cols[distIdx]);
        if (isNaN(t) || isNaN(d)) throw new Error(`Bad data on line ${i + 1}`);
        times.push(t);
        distances.push(d);
        if (powers !== null) {
            const p = parseFloat(cols[pwrIdx]);
            powers.push(isNaN(p) ? null : p);
        }
        if (speeds !== null) {
            const s = parseFloat(cols[spdIdx]);
            speeds.push(isNaN(s) ? null : s);
        }
        if (buckets !== null) {
            const b = parseInt(cols[bktIdx], 10);
            buckets.push(isNaN(b) ? null : b);
        }
    }

    if (times.length < 2) throw new Error('Need at least 2 data rows.');
    return { meta, times, distances, powers, speeds, buckets };
}

function onFileSelected(ev) {
    const file = ev.target.files[0];
    if (!file) return;
    fileStatus.textContent = 'Reading…';

    const reader = new FileReader();
    reader.onload = e => {
        try {
            const { meta, times, distances, powers, speeds, buckets } = parseCSV(e.target.result);
            // Shift distances so the first data point is 0
            const d0 = distances[0] || 0;
            if (d0 !== 0) {
                for (let i = 0; i < distances.length; i++) distances[i] -= d0;
            }
            refTimes = times;
            refDistances = distances;
            refPowers = powers;
            refSpeeds = speeds;
            refBucketSpans = bucketSpans(distances, buckets, powers);
            bucketAvg.reset();
            clearBucketLine();
            refWeightKg = meta.weight_kg ? parseFloat(meta.weight_kg) : null;
            refName = meta.name || file.name;
            refRouteName = meta.route_id
                ? meta.route_id.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                : '';

            showGapPanel();
            routeNameEl.textContent = refRouteName;
            rideNameEl.textContent = refName;
            setWaiting('waiting for race start…');
        } catch (err) {
            fileStatus.textContent = `Error: ${err.message}`;
        }
    };
    reader.onerror = () => { fileStatus.textContent = 'Failed to read file.'; };
    reader.readAsText(file);
}

// ---------------------------------------------------------------------------
// Panel switching
// ---------------------------------------------------------------------------
function showFilePanel() {
    refTimes = null;
    refDistances = null;
    refPowers = null;
    refSpeeds = null;
    refWeightKg = null;
    refRouteName = '';
    refBucketSpans = [];
    bucketAvg.reset();
    clearBucketLine();
    anchorDistanceM = null;
    anchorMs = null;
    csvFileInput.value = '';
    fileStatus.textContent = '';
    filePanel.style.display = '';
    gapDisplay.style.display = 'none';
}

function showGapPanel() {
    filePanel.style.display = 'none';
    gapDisplay.style.display = '';
}

// ---------------------------------------------------------------------------
// Gap display helpers
// ---------------------------------------------------------------------------
function setWaiting(msg) {
    gapTimeEl.textContent = '—';
    gapLabelEl.textContent = msg;
    gapDisplay.className = 'waiting';
    clearBucketLine();
}

function clearBucketLine() {
    bucketAvgEl.textContent = '';
    bucketAvgEl.className = 'bucket-avg';
    bucketInfoEl.textContent = '';
}

/** Update the running bucket average and show it against the bucket's target. */
function updateBucketLine(distanceM, timeSec, powerW) {
    const idx = spanIndexAt(refBucketSpans, distanceM);
    if (idx === -1) { clearBucketLine(); return; }
    const span = refBucketSpans[idx];
    bucketAvg.update(span.bucket, timeSec, powerW);

    const avg = bucketAvg.average;
    const status = paceStatus(avg, span.target);
    if (avg == null) {
        bucketAvgEl.textContent = 'avg —';
    } else {
        const diff = span.target != null ? Math.round(avg - span.target) : null;
        const diffText = diff == null ? '' : ` (${diff > 0 ? '+' : diff < 0 ? '−' : '±'}${Math.abs(diff)})`;
        bucketAvgEl.textContent = `avg ${Math.round(avg)} W` +
            (span.target != null ? ` / ${Math.round(span.target)} W${diffText}` : '');
    }
    bucketAvgEl.className = status ? `bucket-avg ${status}` : 'bucket-avg';

    const leftKm = Math.max(0, span.end - distanceM) / 1000;
    bucketInfoEl.textContent = `bucket ${idx + 1}/${refBucketSpans.length} · ${leftKm.toFixed(1)} km left`;
}

/** Binary search: largest index i where arr[i] <= value */
function searchSorted(arr, value) {
    let lo = 0, hi = arr.length - 1;
    if (value <= arr[0]) return 0;
    if (value >= arr[hi]) return hi;
    while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        arr[mid] <= value ? (lo = mid) : (hi = mid);
    }
    return lo;
}

function refTimeAtDistance(distanceM) {
    if (!refDistances || refDistances.length < 2) return null;
    if (distanceM < refDistances[0] || distanceM > refDistances[refDistances.length - 1]) return null;
    const i = searchSorted(refDistances, distanceM);
    const d0 = refDistances[i], d1 = refDistances[i + 1];
    const t0 = refTimes[i], t1 = refTimes[i + 1];
    if (d1 === d0) return t0;
    return t0 + (distanceM - d0) / (d1 - d0) * (t1 - t0);
}

/**
 * Power is held from the last row at or before the distance (not interpolated),
 * so targets from a bucketed TT plan step cleanly at bucket boundaries.
 */
function refPowerAtDistance(distanceM) {
    if (!refPowers || refDistances.length < 2) return null;
    if (distanceM < refDistances[0] || distanceM > refDistances[refDistances.length - 1]) return null;
    const i = searchSorted(refDistances, distanceM);
    return refPowers[i] ?? refPowers[i + 1] ?? null;
}

function refSpeedAtDistance(distanceM) {
    if (!refSpeeds || refDistances.length < 2) return null;
    if (distanceM < refDistances[0] || distanceM > refDistances[refDistances.length - 1]) return null;
    const i = searchSorted(refDistances, distanceM);
    const s0 = refSpeeds[i], s1 = refSpeeds[i + 1];
    if (s0 == null || s1 == null) return s0 ?? s1 ?? null;
    const d0 = refDistances[i], d1 = refDistances[i + 1];
    if (d1 === d0) return s0;
    return s0 + (distanceM - d0) / (d1 - d0) * (s1 - s0);
}

function formatGap(seconds) {
    const abs = Math.abs(seconds);
    const m = Math.floor(abs / 60);
    const s = Math.floor(abs % 60);
    const sign = seconds >= 0 ? '+' : '-';
    return m > 0 ? `${sign}${m}:${String(s).padStart(2, '0')}` : `${sign}${s}s`;
}

// ---------------------------------------------------------------------------
// Athlete data handler
// ---------------------------------------------------------------------------
// Zwift course ID → world name (matches Sauce's state.courseId values)
const COURSE_NAMES = {
    1: 'Watopia', 2: 'Richmond', 3: 'London', 4: 'New York',
    5: 'Innsbruck', 6: 'Bologna', 7: 'Yorkshire', 8: 'Crit City',
    9: 'Makuri Islands', 10: 'France', 11: 'Paris', 13: 'Scotland',
    14: 'Gravel Mountain',
};
function courseName(id) { return COURSE_NAMES[id] ? `${COURSE_NAMES[id]} (${id})` : `course ${id}`; }

function getRawDistance(state) {
    // Prefer routeDistance (distance along the selected route from 0),
    // then eventDistance (race/event distance), then session distance.
    if (state.routeDistance != null && state.routeDistance > 0) return state.routeDistance;
    if (state.eventDistance != null && state.eventDistance > 0) return state.eventDistance;
    return state.distance ?? 0;
}

function onAthleteData(data) {
    if (!refTimes) return;

    const state = data?.state;
    if (!state) { setWaiting('waiting for athlete data…'); return; }

    const rawDistanceM = getRawDistance(state);

    // Auto-anchor: if Zwift gives us a real event elapsed time (race started),
    // use that directly. Otherwise use local wall-clock tracking.
    const eventTimeSec = state.time ?? 0;
    const isRaceMode = eventTimeSec > 0;

    if (isRaceMode) {
        // Race/event: Zwift tracks elapsed time from race start for us.
        anchorDistanceM = null;
        anchorMs = null;
    } else if (anchorMs === null && rawDistanceM > 0) {
        // Free ride: auto-set anchor the first time we see movement.
        anchorDistanceM = rawDistanceM;
        anchorMs = Date.now();
    }

    const distanceM = isRaceMode
        ? rawDistanceM
        : (anchorDistanceM != null ? rawDistanceM - anchorDistanceM : null);

    const currentTimeSec = isRaceMode
        ? eventTimeSec
        : (anchorMs != null ? (Date.now() - anchorMs) / 1000 : 0);

    if (distanceM == null || distanceM < 0) {
        setWaiting('press reset at the start line');
        return;
    }

    const refTime = refTimeAtDistance(distanceM);
    if (refTime === null) {
        setWaiting(distanceM > refDistances[refDistances.length - 1]
            ? 'beyond reference ride end'
            : 'waiting for movement…');
        return;
    }

    const gapSec = refTime - currentTimeSec;
    gapTimeEl.textContent = formatGap(gapSec);
    gapDisplay.className = gapSec > 0 ? 'ahead' : gapSec < 0 ? 'behind' : 'even';
    gapLabelEl.textContent = gapSec > 0 ? 'ahead' : gapSec < 0 ? 'behind' : 'on pace';

    const refPwr = refPowerAtDistance(distanceM);
    if (refPwr != null) {
        const watts = Math.round(refPwr);
        const wkg = refWeightKg ? (refPwr / refWeightKg).toFixed(2) : null;
        refPowerEl.textContent = wkg ? `${watts}W · ${wkg} W/kg` : `${watts}W`;
    } else {
        refPowerEl.textContent = '';
    }

    const refSpd = refSpeedAtDistance(distanceM);
    refSpeedEl.textContent = refSpd != null ? `${refSpd.toFixed(1)} km/h` : '';

    if (refBucketSpans.length) updateBucketLine(distanceM, currentTimeSec, Number(state.power ?? NaN));

    rideNameEl.textContent = refName;
}

// Store last known state so the reset button can anchor immediately
let lastState = null;

// Hide the detail lines switched off in display-settings.html (see display.css).
function applyDetailSettings() {
    const hide = detailHideClasses(key => Common.settingsStore.get(key));
    for (const [cls, on] of Object.entries(hide)) {
        document.documentElement.classList.toggle(cls, on);
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
export function main() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DETAIL_DEFAULTS});
    applyDetailSettings();
    Common.settingsStore.addEventListener('changed', applyDetailSettings);
    csvFileInput.addEventListener('change', onFileSelected);
    changeBtn.addEventListener('click', showFilePanel);
    resetBtn.addEventListener('click', () => {
        // Re-anchor to current position and wall clock
        if (lastState) {
            anchorDistanceM = getRawDistance(lastState);
            anchorMs = Date.now();
        }
        bucketAvg.reset();
        clearBucketLine();
    });
    Common.subscribe('athlete/watching', data => {
        lastState = data?.state ?? null;
        onAthleteData(data);
    });
}
