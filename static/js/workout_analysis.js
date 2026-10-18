'use strict';

let catalog = [];
let selectedWorkout = null;
let loadedActivity = null;
let powerChart = null;
let planRequest = 0;
let ftpEdited = false;

function typicalInterval(times) {
    const deltas = times.slice(1).map((time, index) => time - times[index]).sort((a, b) => a - b);
    return deltas.length ? deltas[Math.floor((deltas.length - 1) / 2)] : 1;
}

function targetBounds(segment, elapsed, scale) {
    if (segment.power_low === null || segment.power_high === null) return null;
    if (segment.ramp) {
        const fraction = (elapsed - segment.start_sec) / (segment.end_sec - segment.start_sec);
        const watts = (segment.power_low + fraction * (segment.power_high - segment.power_low)) * scale;
        const target = Math.round(watts / 5) * 5;
        return [target, target];
    }
    return [Math.round(Math.min(segment.power_low, segment.power_high) * scale / 5) * 5,
        Math.round(Math.max(segment.power_low, segment.power_high) * scale / 5) * 5];
}

function targetBreakpoints(segment, from, to, scale) {
    const times = [from, to];
    if (!segment.ramp || segment.power_low === null || segment.power_high === null) return times;
    const slope = (segment.power_high - segment.power_low) * scale / (segment.end_sec - segment.start_sec);
    if (slope === 0) return times;
    const initial = segment.power_low * scale;
    const first = initial + (from - segment.start_sec) * slope;
    const last = initial + (to - segment.start_sec) * slope;
    // A rounded ramp changes at the half-way points between 5 W targets.
    const thresholdStart = (Math.floor((Math.min(first, last) - 2.5) / 5) + 1) * 5 + 2.5;
    for (let threshold = thresholdStart; threshold < Math.max(first, last); threshold += 5) {
        const time = segment.start_sec + (threshold - initial) / slope;
        if (time > from && time < to) times.push(time);
    }
    return times.sort((a, b) => a - b);
}

function analyzeWorkout(activity, workout, ftp, intensity, offset) {
    const scale = (workout.ftp_override ?? ftp) * intensity / 100;
    const times = activity.time_sec;
    const cadence = typicalInterval(times);
    let cursor = 0;
    let totalCovered = 0;
    let totalScored = 0;
    let totalError = 0;
    const rows = workout.segments.map(segment => {
        const start = segment.start_sec + offset;
        const end = segment.end_sec + offset;
        while (cursor < times.length && times[cursor] + cadence <= start) cursor++;
        let covered = 0;
        let actualWork = 0;
        let targetWork = 0;
        const scored = segment.power_low !== null && segment.power_high !== null;
        for (let index = cursor; index < times.length && times[index] < end; index++) {
            const sampleEnd = Math.min(times[index] + cadence, times[index + 1] ?? Infinity);
            const from = Math.max(start, times[index]);
            const to = Math.min(end, sampleEnd);
            const watts = activity.power_watts[index];
            if (to <= from || watts === null || !Number.isFinite(watts)) continue;
            const duration = to - from;
            covered += duration;
            actualWork += watts * duration;
            if (scored) {
                const times = targetBreakpoints(segment, from - offset, to - offset, scale);
                for (let part = 1; part < times.length; part++) {
                    const bounds = targetBounds(segment, (times[part - 1] + times[part]) / 2, scale);
                    targetWork += (bounds[0] + bounds[1]) / 2 * (times[part] - times[part - 1]);
                }
            }
        }
        totalCovered += covered;
        if (scored) {
            totalScored += covered;
            totalError += actualWork - targetWork;
        }
        return {
            kind: segment.kind, start, end, covered,
            actual: covered ? actualWork / covered : null,
            target: scored && covered ? targetWork / covered : null,
            difference: scored && covered ? (actualWork - targetWork) / covered : null,
        };
    });
    return { rows, covered: totalCovered, scored: totalScored,
        bias: totalScored ? totalError / totalScored : null };
}

function plannedSeries(workout, scale, offset) {
    const lower = [];
    const upper = [];
    for (const segment of workout.segments) {
        const append = (time, bounds) => {
            lower.push({ x: (time + offset) / 60, y: bounds ? bounds[0] : null });
            upper.push({ x: (time + offset) / 60, y: bounds ? bounds[1] : null });
        };
        const times = targetBreakpoints(segment, segment.start_sec, segment.end_sec, scale);
        append(segment.start_sec, targetBounds(segment, segment.start_sec, scale));
        for (let part = 1; part < times.length; part++) {
            const bounds = targetBounds(segment, (times[part - 1] + times[part]) / 2, scale);
            append(times[part - 1], bounds);
            append(times[part], bounds);
        }
        append(segment.end_sec, targetBounds(segment, segment.end_sec, scale));
    }
    return { lower, upper };
}

function actualSeries(activity) {
    const cadence = typicalInterval(activity.time_sec);
    const points = [];
    activity.time_sec.forEach((time, index) => {
        if (index && time - activity.time_sec[index - 1] > cadence * 1.5) {
            points.push({ x: (activity.time_sec[index - 1] + cadence) / 60, y: null });
        }
        points.push({ x: time / 60, y: activity.power_watts[index] });
    });
    return points;
}

function formatTime(seconds) {
    const sign = seconds < 0 ? '-' : '';
    const value = Math.round(Math.abs(seconds));
    return `${sign}${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}

function displayNumber(value) {
    return value === null ? '-' : value.toFixed(1);
}

function renderComparison() {
    const results = document.getElementById('results');
    const status = document.getElementById('comparisonStatus');
    results.hidden = true;
    if (!selectedWorkout || !loadedActivity) {
        status.textContent = 'Load an activity and choose a workout to compare.';
        return;
    }
    const ftp = selectedWorkout.ftp_override ?? Number(document.getElementById('ftp').value);
    const intensity = Number(document.getElementById('intensity').value);
    const offsetField = document.getElementById('offset');
    const offset = Number(offsetField.value);
    if (!Number.isFinite(ftp) || ftp <= 0 || !Number.isFinite(intensity) || intensity <= 0
            || offsetField.value.trim() === '' || !Number.isFinite(offset)) {
        status.textContent = 'Enter a positive FTP and intensity, and a finite start offset.';
        return;
    }
    const analysis = analyzeWorkout(loadedActivity, selectedWorkout, ftp, intensity, offset);
    status.textContent = analysis.covered ? '' : 'The workout does not overlap recorded power. Adjust the start offset.';
    results.hidden = false;
    document.getElementById('resultTitle').textContent = `${loadedActivity.name} vs ${selectedWorkout.name}`;
    document.getElementById('summary').textContent =
        `Recorded coverage: ${formatTime(analysis.covered)} / ${formatTime(selectedWorkout.duration_sec)}. ` +
        `Scored time: ${formatTime(analysis.scored)}. Average actual minus target: ${displayNumber(analysis.bias)} W.`;
    const body = document.getElementById('intervals');
    body.replaceChildren();
    analysis.rows.forEach((row, index) => {
        const tr = document.createElement('tr');
        const values = [`${index + 1}. ${row.kind}`, `${formatTime(row.start)} - ${formatTime(row.end)}`,
            `${formatTime(row.covered)} / ${formatTime(row.end - row.start)}`,
            displayNumber(row.target), displayNumber(row.actual), displayNumber(row.difference)];
        for (const value of values) {
            const td = document.createElement('td');
            td.textContent = value;
            tr.appendChild(td);
        }
        body.appendChild(tr);
    });
    const chartError = document.getElementById('chartError');
    chartError.textContent = '';
    if (typeof Chart === 'undefined') {
        chartError.textContent = 'Chart library could not load. Interval statistics are still available; reload to retry the chart.';
        return;
    }
    const planned = plannedSeries(selectedWorkout, ftp * intensity / 100, offset);
    if (powerChart) powerChart.destroy();
    powerChart = new Chart(document.getElementById('powerChart'), {
        type: 'line',
        data: { datasets: [
            { label: 'Actual power (raw)', data: actualSeries(loadedActivity),
                borderColor: '#4fc3f7', borderWidth: 1, pointRadius: 0, spanGaps: false },
            { label: 'Planned target / lower bound', data: planned.lower, borderColor: '#f7931e',
                borderWidth: 2, pointRadius: 0, fill: '+1', backgroundColor: 'rgba(247,147,30,0.2)', spanGaps: false },
            { label: 'Planned upper bound', data: planned.upper, borderColor: '#f7931e',
                borderWidth: 1, pointRadius: 0, spanGaps: false },
        ] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
            interaction: { mode: 'nearest', intersect: false },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Activity elapsed time (minutes)', color: '#bbb' },
                    ticks: { color: '#bbb' }, grid: { color: '#ffffff15' } },
                y: { title: { display: true, text: 'Power (W)', color: '#bbb' },
                    ticks: { color: '#bbb' }, grid: { color: '#ffffff15' } },
            },
            plugins: { legend: { labels: { color: '#ddd' } } },
        },
    });
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        let message = `Request failed (HTTP ${response.status})`;
        try {
            const data = await response.json();
            message = data.error || message;
        } catch (error) {
            console.error('Non-JSON error response', error);
        }
        throw new Error(message);
    }
    return response.json();
}

function filterWorkouts() {
    const search = document.getElementById('workoutSearch').value.trim().toLowerCase();
    const select = document.getElementById('workoutSelect');
    const previous = select.value;
    const matches = catalog.filter(workout => `${workout.name} ${workout.source}`.toLowerCase().includes(search));
    select.replaceChildren(new Option('Choose a workout...', ''));
    for (const workout of matches) {
        select.add(new Option(`${workout.name} (${formatTime(workout.duration_sec)}) - ${workout.source}`, workout.id));
    }
    // Filtering must not silently switch the plan underneath an existing comparison.
    if (previous && !matches.some(workout => workout.id === previous)) {
        const existing = catalog.find(workout => workout.id === previous);
        select.add(new Option(`${existing.name} (selected; outside filter)`, existing.id));
    }
    select.value = previous;
    document.getElementById('catalogStatus').textContent = `${matches.length} matching workouts of ${catalog.length}.`;
}

async function chooseWorkout() {
    const requestId = ++planRequest;
    const id = document.getElementById('workoutSelect').value;
    selectedWorkout = null;
    document.getElementById('ftp').disabled = false;
    document.getElementById('planError').textContent = '';
    document.getElementById('planWarnings').textContent = '';
    renderComparison();
    if (!id) return;
    try {
        const workout = await fetchJson(`/api/workouts/${encodeURIComponent(id)}`);
        if (requestId !== planRequest) return;
        selectedWorkout = workout;
        document.getElementById('ftp').disabled = Number.isFinite(workout.ftp_override);
        document.getElementById('planWarnings').textContent = workout.warnings.join(' ');
        renderComparison();
    } catch (error) {
        if (requestId === planRequest) document.getElementById('planError').textContent = error.message;
    }
}

function parseActivityId(value) {
    const input = value.trim();
    if (/^\d+$/.test(input)) return input;
    try {
        const url = new URL(input);
        if (url.protocol !== 'https:' || !['zwift.com', 'www.zwift.com', 'my.zwift.com'].includes(url.hostname)) return null;
        return url.pathname.match(/^\/(?:activity|activities)\/(\d+)\/?$/)?.[1] || null;
    } catch {
        return null;
    }
}

async function loadActivity(event) {
    if (event) event.preventDefault();
    const errorField = document.getElementById('activityError');
    const id = parseActivityId(document.getElementById('activityId').value);
    errorField.textContent = '';
    if (!id) {
        errorField.textContent = 'Enter a numeric activity ID or a Zwift activity URL.';
        return;
    }
    const button = document.getElementById('loadActivity');
    button.disabled = true;
    loadedActivity = null;
    renderComparison();
    document.getElementById('activityStatus').textContent = 'Loading activity power...';
    try {
        loadedActivity = await fetchJson(`/api/workout_activity/${encodeURIComponent(id)}`);
        document.getElementById('activityStatus').textContent =
            `${loadedActivity.name}: ${loadedActivity.time_sec.length} recorded samples.`;
        renderComparison();
    } catch (error) {
        document.getElementById('activityStatus').textContent = 'Activity not loaded.';
        errorField.textContent = error.message;
    } finally {
        button.disabled = false;
    }
}

async function loadCatalog() {
    try {
        const data = await fetchJson('/api/workouts');
        catalog = data.workouts;
        filterWorkouts();
        document.getElementById('workoutSelect').disabled = false;
    } catch (error) {
        document.getElementById('planError').textContent = error.message;
    }
}

async function loadProfileFtp() {
    const status = document.getElementById('ftpStatus');
    try {
        const profile = await fetchJson('/api/my_profile');
        if (Number.isFinite(profile.ftp_w) && profile.ftp_w > 0) {
            if (!ftpEdited) document.getElementById('ftp').value = profile.ftp_w;
            status.textContent = `Current Zwift profile FTP: ${profile.ftp_w} W. You can override it for this analysis.`;
            renderComparison();
        } else {
            status.textContent = 'Zwift did not provide a valid current FTP. Enter the workout FTP manually.';
        }
    } catch (error) {
        status.textContent = `Could not load profile FTP: ${error.message}. Enter it manually.`;
    }
}

function init() {
    document.getElementById('activityForm').addEventListener('submit', loadActivity);
    document.getElementById('workoutSearch').addEventListener('input', filterWorkouts);
    document.getElementById('workoutSelect').addEventListener('change', chooseWorkout);
    for (const id of ['ftp', 'intensity', 'offset']) {
        document.getElementById(id).addEventListener('input', () => {
            if (id === 'ftp') ftpEdited = true;
            renderComparison();
        });
    }
    loadCatalog();
    loadProfileFtp();
    const activityId = new URLSearchParams(location.search).get('activity_id');
    if (activityId) {
        document.getElementById('activityId').value = activityId;
        loadActivity();
    }
}

document.addEventListener('DOMContentLoaded', init);
