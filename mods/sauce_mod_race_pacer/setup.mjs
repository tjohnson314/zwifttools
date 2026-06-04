import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

const csvFileInput = document.getElementById('csv-file');
const statusEl = document.getElementById('status');
const rideInfoSection = document.getElementById('ride-info');
const clearBtn = document.getElementById('clear-btn');

function setStatus(msg, type = 'idle') {
    statusEl.textContent = msg;
    statusEl.className = `status ${type}`;
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function formatDistance(metres) {
    if (metres >= 1000) return `${(metres / 1000).toFixed(2)} km`;
    return `${metres.toFixed(0)} m`;
}

/**
 * Parse the race pacer CSV format:
 *
 * Line 1: metadata column names  →  course_id,route_id,name,date
 * Line 2: metadata values        →  12,1337403830,My Race,2026-06-01
 * Line 3: data column names      →  time_sec,distance_m,lat,lng
 * Lines 4+: data rows
 *
 * Returns { meta, times, distances } or throws on parse error.
 */
function parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');

    // Strip blank lines from the end
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();

    if (lines.length < 5) {
        throw new Error('CSV file too short — expected metadata header, metadata values, data header, and at least 2 data rows.');
    }

    // Parse metadata (lines 0 and 1)
    const metaKeys = lines[0].split(',').map(s => s.trim());
    const metaVals = lines[1].split(',').map(s => s.trim());
    const meta = {};
    for (let i = 0; i < metaKeys.length; i++) {
        meta[metaKeys[i]] = metaVals[i] ?? '';
    }

    // Parse data header (line 2)
    const dataHeader = lines[2].split(',').map(s => s.trim().toLowerCase());
    const timeIdx = dataHeader.indexOf('time_sec');
    const distIdx = dataHeader.indexOf('distance_m');
    if (timeIdx === -1) throw new Error('Data header missing required column: time_sec');
    if (distIdx === -1) throw new Error('Data header missing required column: distance_m');

    // Parse data rows (lines 3+)
    const times = [];
    const distances = [];
    for (let i = 3; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const cols = line.split(',');
        const t = parseFloat(cols[timeIdx]);
        const d = parseFloat(cols[distIdx]);
        if (isNaN(t) || isNaN(d)) {
            throw new Error(`Invalid data on line ${i + 1}: "${line}"`);
        }
        times.push(t);
        distances.push(d);
    }

    if (times.length < 2) {
        throw new Error('CSV must contain at least 2 data rows.');
    }

    return { meta, times, distances };
}

function populateRideInfo(meta, times, distances) {
    document.getElementById('info-name').textContent = meta.name || '—';
    document.getElementById('info-date').textContent = meta.date || '—';
    document.getElementById('info-course').textContent = meta.course_id || '—';
    document.getElementById('info-route').textContent = meta.route_id || '—';
    document.getElementById('info-duration').textContent = formatDuration(times[times.length - 1]);
    document.getElementById('info-distance').textContent = formatDistance(distances[distances.length - 1]);
    document.getElementById('info-points').textContent = times.length.toLocaleString();
    rideInfoSection.style.display = '';
}

function onFileSelected(ev) {
    const file = ev.target.files[0];
    if (!file) return;

    setStatus('Reading file…', 'loading');

    const reader = new FileReader();
    reader.onload = e => {
        try {
            const { meta, times, distances } = parseCSV(e.target.result);

            // Store in settingsStore so the display window picks it up
            Common.settingsStore.set('refRideData', {
                name: meta.name || file.name,
                date: meta.date || '',
                courseId: meta.course_id ? Number(meta.course_id) : null,
                routeId: meta.route_id ? Number(meta.route_id) : null,
                times,
                distances,
            });

            populateRideInfo(meta, times, distances);
            setStatus(`Loaded ${times.length.toLocaleString()} data points`, 'ok');
            clearBtn.disabled = false;
        } catch (err) {
            setStatus(`Error: ${err.message}`, 'error');
            rideInfoSection.style.display = 'none';
        }
    };
    reader.onerror = () => setStatus('Failed to read file', 'error');
    reader.readAsText(file);
}

function onClear() {
    Common.settingsStore.set('refRideData', null);
    csvFileInput.value = '';
    rideInfoSection.style.display = 'none';
    clearBtn.disabled = true;
    setStatus('No file loaded', 'idle');
}

function restoreState() {
    const data = Common.settingsStore.get('refRideData');
    if (!data || !data.times) return;
    populateRideInfo(
        { name: data.name, date: data.date, course_id: data.courseId, route_id: data.routeId },
        data.times,
        data.distances,
    );
    setStatus(`Loaded ${data.times.length.toLocaleString()} data points`, 'ok');
    clearBtn.disabled = false;
}

export function main() {
    Common.initInteractionListeners();
    csvFileInput.addEventListener('change', onFileSelected);
    clearBtn.addEventListener('click', onClear);
    restoreState();
}
