/**
 * Power Dashboard — progressive loader.
 *
 * The dashboard first fetches the list of recent activities (fast), renders them
 * in a table, then loads each activity's peak-power curve one at a time. The
 * chart updates as each activity completes, so nothing depends on a single
 * long-running request.
 *
 * Instead of a time-range filter, the user inserts dividers between activities
 * in the list. The activities are partitioned by those dividers and a separate
 * power curve is plotted for each partition.
 */

'use strict';

const POWER_CURVE_X_TICKS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600, 7200];

// Distinct colours for the partition curves (cycled if there are more groups).
const PARTITION_COLORS = [
    '#f0f4ff', '#6adf71', '#ffd166', '#4fc3f7', '#ff7f95',
    '#c792ea', '#ffa94d', '#63e6be', '#f783ac', '#a0d911',
];

// ── State ─────────────────────────────────────────────────────────────────────
let durationsSec = [];
let durationsLabel = [];
let activities = [];        // ordered newest-first; each gets peak_watts + status
let dividers = new Set();   // boundary-before-index positions (1..activities.length-1)

let powerCurveChart = null;
let weeklyBarChart = null;
let weeklyDurationSec = 1200;

let selectedIndex = 0;
let pinnedIndex = null;

let chartRenderQueued = false;

// ── Formatting helpers ────────────────────────────────────────────────────────
function formatIntervalClock(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatTickInterval(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    }
    if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m`;
    }
    return `${Math.floor(seconds / 3600)}h`;
}

function formatDuration(sec) {
    if (!sec) {
        return '—';
    }
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    if (h > 0) {
        return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    return `${m}:${String(s).padStart(2, '0')}`;
}

function formatDateTime(value) {
    if (!value) {
        return '—';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return '—';
    }
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) +
        ' ' + date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function formatDateShort(value) {
    if (!value) {
        return '—';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return '—';
    }
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function formatWatts(value) {
    if (value === null || value === undefined) {
        return '—';
    }
    return `${Math.round(value)} W`;
}

function activityHref(activityId) {
    return activityId ? `https://www.zwift.com/activity/${activityId}` : '#';
}

function displayName(name) {
    return (name || '\u2014')
        .replace(/^Zwift - /i, '')
        .replace(/ in (Watopia|France|London|Richmond|New York|Innsbruck|Bologna|Yorkshire|Crit City|Makuri Islands|Paris|Scotland|Gravel Mountain)$/i, '');
}

// Table cell text for a per-activity metric: placeholder until loaded, then value or dash.
function metricCellText(activity, value, unit) {
    if (activity.status === 'pending' || activity.status === 'loading') {
        return '…';
    }
    if (value === null || value === undefined) {
        return '—';
    }
    return `${Math.round(value)}${unit}`;
}

function setStatusText(elementId, text, isError = false) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    element.textContent = text;
    element.classList.toggle('error', Boolean(isError));
}

// ── Partitioning ──────────────────────────────────────────────────────────────
function partitionBounds() {
    const bounds = [...dividers].filter((p) => p >= 1 && p < activities.length).sort((a, b) => a - b);
    const parts = [];
    let start = 0;
    for (const b of bounds) {
        parts.push([start, b]);
        start = b;
    }
    parts.push([start, activities.length]);
    return parts;
}

function partitionColor(index) {
    return PARTITION_COLORS[index % PARTITION_COLORS.length];
}

// Best (max) power for each interval across loaded activities in [startIdx, endIdx).
function partitionCurve(startIdx, endIdx) {
    return durationsSec.map((_, i) => {
        let best = null;
        for (let a = startIdx; a < endIdx; a++) {
            const pw = activities[a].peak_watts;
            if (!pw) {
                continue;
            }
            const v = pw[i];
            if (v === null || v === undefined) {
                continue;
            }
            if (best === null || v > best) {
                best = v;
            }
        }
        return best;
    });
}

// Best power + owning activity at a single interval index for a partition.
function partitionBestAt(startIdx, endIdx, i) {
    let bestPower = null;
    let bestActivity = null;
    for (let a = startIdx; a < endIdx; a++) {
        const pw = activities[a].peak_watts;
        if (!pw) {
            continue;
        }
        const v = pw[i];
        if (v === null || v === undefined) {
            continue;
        }
        if (bestPower === null || v > bestPower) {
            bestPower = v;
            bestActivity = activities[a];
        }
    }
    return { power: bestPower, activity: bestActivity };
}

function partitionLabel(startIdx, endIdx) {
    const first = activities[startIdx];
    const last = activities[endIdx - 1];
    if (!first || !last) {
        return 'Group';
    }
    if (startIdx === endIdx - 1) {
        return formatDateShort(first.start_date);
    }
    // Activities are newest-first, so the last row is the older date.
    return `${formatDateShort(last.start_date)} – ${formatDateShort(first.start_date)}`;
}

// ── Chart rendering ───────────────────────────────────────────────────────────
function buildChartDatasets() {
    return partitionBounds().map(([startIdx, endIdx], idx) => {
        const color = partitionColor(idx);
        const curve = partitionCurve(startIdx, endIdx);
        return {
            label: partitionLabel(startIdx, endIdx),
            data: durationsSec.map((sec, i) => ({ x: sec, y: curve[i] })),
            borderColor: color,
            backgroundColor: 'transparent',
            borderWidth: 3,
            pointRadius: 0,
            pointHoverRadius: 4,
            spanGaps: true,
            tension: 0.08,
        };
    });
}

function renderPowerCurveChart() {
    const canvas = document.getElementById('powerCurveChart');
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext('2d');

    powerCurveChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: buildChartDatasets() },
        options: {
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false },
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    min: 1,
                    max: 7200,
                    afterBuildTicks(scale) {
                        scale.ticks = POWER_CURVE_X_TICKS.map((value) => ({ value }));
                    },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    ticks: {
                        color: '#9fb0cc',
                        autoSkip: false,
                        maxRotation: 0,
                        minRotation: 0,
                        callback(value) {
                            return formatTickInterval(Number(value));
                        },
                    },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    ticks: {
                        color: '#9fb0cc',
                        callback(value) {
                            return `${value} W`;
                        },
                    },
                },
            },
            onHover(event, elements) {
                if (pinnedIndex !== null || !elements || elements.length === 0) {
                    return;
                }
                updateInspectPanel(elements[0].index);
            },
            onClick(event, elements) {
                if (pinnedIndex !== null) {
                    pinnedIndex = null;
                    updatePinStateLabel();
                    return;
                }
                if (!elements || elements.length === 0) {
                    return;
                }
                pinnedIndex = elements[0].index;
                updateInspectPanel(pinnedIndex);
                updatePinStateLabel();
            },
        },
    });
}

function refreshChart() {
    if (!powerCurveChart) {
        renderPowerCurveChart();
        return;
    }
    powerCurveChart.data.datasets = buildChartDatasets();
    powerCurveChart.update('none');
}

// Coalesce rapid updates during progressive loading into a single frame.
function queueChartRefresh() {
    if (chartRenderQueued) {
        return;
    }
    chartRenderQueued = true;
    requestAnimationFrame(() => {
        chartRenderQueued = false;
        refreshChart();
        renderPartitionLegend();
        updateInspectPanel(pinnedIndex !== null ? pinnedIndex : selectedIndex);
        updateWeeklyBars(weeklyDurationSec);
    });
}

// ── Partition legend + inspect panel ──────────────────────────────────────────
function renderPartitionLegend() {
    const container = document.getElementById('partitionLegend');
    if (!container) {
        return;
    }
    container.innerHTML = '';
    partitionBounds().forEach(([startIdx, endIdx], idx) => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const line = document.createElement('span');
        line.className = 'legend-line';
        line.style.borderTopColor = partitionColor(idx);
        const text = document.createElement('span');
        const count = endIdx - startIdx;
        text.textContent = `${partitionLabel(startIdx, endIdx)} (${count})`;
        item.appendChild(line);
        item.appendChild(text);
        container.appendChild(item);
    });
}

function updatePinStateLabel() {
    const pinState = document.getElementById('pinState');
    if (!pinState) {
        return;
    }
    pinState.textContent = pinnedIndex === null ? 'Hover to inspect' : 'Pinned (click chart to unpin)';
}

function updateInspectPanel(index) {
    if (durationsSec.length === 0) {
        return;
    }
    if (index === null || index === undefined || index < 0 || index >= durationsSec.length) {
        index = 0;
    }
    selectedIndex = index;

    const sec = durationsSec[index];
    document.getElementById('selectedInterval').textContent = formatIntervalClock(sec);
    document.getElementById('selectedIntervalLabel').textContent = durationsLabel[index] || '';

    const container = document.getElementById('partitionMetrics');
    container.innerHTML = '';

    const bounds = partitionBounds();
    bounds.forEach(([startIdx, endIdx], idx) => {
        const { power, activity } = partitionBestAt(startIdx, endIdx, index);

        const row = document.createElement('div');
        row.className = 'partition-metric-row';

        const swatch = document.createElement('span');
        swatch.className = 'partition-swatch';
        swatch.style.background = partitionColor(idx);

        const value = document.createElement('div');
        value.className = 'metric-value';
        value.textContent = formatWatts(power);

        const context = document.createElement('div');
        context.className = 'metric-context';

        const label = document.createElement('div');
        label.className = 'metric-label';
        label.textContent = bounds.length > 1 ? partitionLabel(startIdx, endIdx) : 'Best across all activities';

        const link = document.createElement('a');
        link.className = 'activity-link';
        link.href = activity ? activityHref(activity.activity_id) : '#';
        link.textContent = activity ? displayName(activity.name) : '—';
        if (activity) {
            link.target = '_blank';
            link.rel = 'noopener';
        }

        const date = document.createElement('div');
        date.className = 'activity-date';
        date.textContent = activity ? formatDateTime(activity.start_date) : '—';

        context.appendChild(label);
        context.appendChild(link);
        context.appendChild(date);

        const head = document.createElement('div');
        head.className = 'partition-metric-head';
        head.appendChild(swatch);
        head.appendChild(value);

        row.appendChild(head);
        row.appendChild(context);
        container.appendChild(row);
    });
}

// ── Weekly bars (computed client-side from loaded peaks) ───────────────────────
function weekStartOf(date) {
    const d = new Date(date);
    const day = (d.getDay() + 6) % 7; // Monday = 0
    d.setHours(0, 0, 0, 0);
    d.setDate(d.getDate() - day);
    return d;
}

function buildWeekBuckets() {
    const now = new Date();
    const lookbackStart = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
    const labels = [];
    const starts = [];
    let cursor = weekStartOf(lookbackStart);
    const lastWeek = weekStartOf(now);
    while (cursor <= lastWeek) {
        starts.push(new Date(cursor));
        labels.push(cursor.toLocaleDateString(undefined, { day: '2-digit', month: 'short' }));
        cursor.setDate(cursor.getDate() + 7);
    }
    return { labels, starts };
}

function computeWeeklySeries(durationSec) {
    const { labels, starts } = buildWeekBuckets();
    const durIdx = durationsSec.indexOf(durationSec);
    const data = new Array(starts.length).fill(null);
    if (durIdx === -1) {
        return { labels, data };
    }

    for (const activity of activities) {
        if (!activity.peak_watts) {
            continue;
        }
        const val = activity.peak_watts[durIdx];
        if (val === null || val === undefined) {
            continue;
        }
        const ws = weekStartOf(activity.start_date).getTime();
        const idx = starts.findIndex((s) => s.getTime() === ws);
        if (idx === -1) {
            continue;
        }
        if (data[idx] === null || val > data[idx]) {
            data[idx] = val;
        }
    }
    return { labels, data };
}

function renderWeeklyBarChart() {
    const canvas = document.getElementById('weeklyBarChart');
    if (!canvas) {
        return;
    }
    const { labels, data } = computeWeeklySeries(weeklyDurationSec);
    const ctx = canvas.getContext('2d');

    weeklyBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Weekly peak power',
                data,
                backgroundColor: '#72aee8',
                borderColor: '#72aee8',
                borderWidth: 1,
                barPercentage: 0.95,
                categoryPercentage: 1.0,
            }],
        },
        options: {
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#9fb0cc', maxTicksLimit: 10, maxRotation: 0, minRotation: 0 },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#9fb0cc',
                        callback(value) {
                            return `${value} W`;
                        },
                    },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                },
            },
        },
    });
}

function updateWeeklyBars(durationSec) {
    weeklyDurationSec = durationSec;
    if (!weeklyBarChart) {
        renderWeeklyBarChart();
        return;
    }
    const { labels, data } = computeWeeklySeries(durationSec);
    weeklyBarChart.data.labels = labels;
    weeklyBarChart.data.datasets[0].data = data;
    weeklyBarChart.update('none');
}

// ── Activity list + dividers ──────────────────────────────────────────────────
function partitionIndexOf(activityIndex) {
    const bounds = partitionBounds();
    for (let p = 0; p < bounds.length; p++) {
        if (activityIndex >= bounds[p][0] && activityIndex < bounds[p][1]) {
            return p;
        }
    }
    return 0;
}

function statusCellHtml(activity) {
    switch (activity.status) {
        case 'loaded':
            return '<span class="status-badge ok">&#10003; Loaded</span>';
        case 'loading':
            return '<span class="status-badge loading"><span class="mini-spinner"></span> Loading…</span>';
        case 'nodata':
            return '<span class="status-badge muted">No power data</span>';
        case 'error':
            return '<span class="status-badge err">Failed</span>';
        default:
            return '<span class="status-badge muted">Pending…</span>';
    }
}

function actionsCellHtml(activity) {
    if (!activity.activity_id) {
        return '';
    }
    let html =
        `<a class="action-link" href="/bike-comparison?activity_id=${activity.activity_id}" title="Bike Comparison">&#x1F6B2;</a>`;
    if (activity.is_race) {
        html +=
            `<a class="action-link race" href="/race-replay?activity_id=${activity.activity_id}" title="Race Replay">&#x1F3C1;</a>`;
    }
    html +=
        `<a class="action-link logo-link" href="${activityHref(activity.activity_id)}" target="_blank" rel="noopener" title="View on Zwift">` +
        `<img src="/static/img/zwift.ico" alt="Zwift" class="logo-icon"></a>`;
    if (activity.is_race && activity.event_id) {
        html +=
            `<a class="action-link logo-link" href="https://zwiftpower.com/events.php?zid=${activity.event_id}" target="_blank" rel="noopener" title="View on ZwiftPower">` +
            `<img src="/static/img/zwiftpower.svg" alt="ZwiftPower" class="logo-icon"></a>`;
    }
    return html;
}

function renderActivityList() {
    const tbody = document.getElementById('activityListBody');
    if (!tbody) {
        return;
    }
    tbody.innerHTML = '';

    activities.forEach((activity, i) => {
        // Divider control row before every activity except the first.
        if (i >= 1) {
            const dividerTr = document.createElement('tr');
            dividerTr.className = 'divider-control-row';
            const active = dividers.has(i);
            dividerTr.classList.toggle('active', active);
            const td = document.createElement('td');
            td.colSpan = 9;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'divider-btn' + (active ? ' active' : '');
            btn.textContent = active ? '✕ Remove divider' : '+ Divider';
            btn.addEventListener('click', () => toggleDivider(i));
            td.appendChild(btn);
            dividerTr.appendChild(td);
            tbody.appendChild(dividerTr);
        }

        const partIdx = partitionIndexOf(i);
        const color = partitionColor(partIdx);

        const tr = document.createElement('tr');
        tr.className = 'activity-row';
        tr.style.borderLeft = `4px solid ${color}`;

        const groupTd = document.createElement('td');
        groupTd.className = 'col-group';
        groupTd.innerHTML = `<span class="partition-swatch" style="background:${color}"></span>${partIdx + 1}`;

        const dateTd = document.createElement('td');
        dateTd.textContent = formatDateTime(activity.start_date);

        const nameTd = document.createElement('td');
        nameTd.className = 'col-name';
        nameTd.textContent = displayName(activity.name);
        if (activity.is_race) {
            nameTd.innerHTML += '<span class="badge-race">Race</span>';
        }

        const distTd = document.createElement('td');
        distTd.className = 'num';
        distTd.textContent = activity.distance_km ? `${activity.distance_km.toFixed(1)} km` : '—';

        const timeTd = document.createElement('td');
        timeTd.className = 'num';
        timeTd.textContent = formatDuration(activity.duration_sec);

        const avgPowerTd = document.createElement('td');
        avgPowerTd.className = 'num';
        avgPowerTd.textContent = metricCellText(activity, activity.avg_power, ' W');

        const avgHrTd = document.createElement('td');
        avgHrTd.className = 'num';
        avgHrTd.textContent = metricCellText(activity, activity.avg_hr, ' bpm');

        const statusTd = document.createElement('td');
        statusTd.className = 'col-status';
        statusTd.innerHTML = statusCellHtml(activity);

        const actionsTd = document.createElement('td');
        actionsTd.className = 'col-actions';
        actionsTd.innerHTML = actionsCellHtml(activity);

        tr.appendChild(groupTd);
        tr.appendChild(dateTd);
        tr.appendChild(nameTd);
        tr.appendChild(distTd);
        tr.appendChild(timeTd);
        tr.appendChild(avgPowerTd);
        tr.appendChild(avgHrTd);
        tr.appendChild(statusTd);
        tr.appendChild(actionsTd);
        tbody.appendChild(tr);
    });
}

// Update just one activity row's status/race cells without a full re-render.
function updateActivityRow(index) {
    // A full re-render keeps partition colouring consistent and is cheap here.
    renderActivityList();
}

function toggleDivider(position) {
    if (dividers.has(position)) {
        dividers.delete(position);
    } else {
        dividers.add(position);
    }
    document.getElementById('clearDividersBtn').style.display = dividers.size > 0 ? '' : 'none';
    renderActivityList();
    refreshChart();
    renderPartitionLegend();
    updateInspectPanel(pinnedIndex !== null ? pinnedIndex : selectedIndex);
}

function clearDividers() {
    dividers.clear();
    document.getElementById('clearDividersBtn').style.display = 'none';
    renderActivityList();
    refreshChart();
    renderPartitionLegend();
    updateInspectPanel(pinnedIndex !== null ? pinnedIndex : selectedIndex);
}

// ── Progressive loading ───────────────────────────────────────────────────────
function updateLoadSummary(loadedCount) {
    const el = document.getElementById('loadSummary');
    if (!el) {
        return;
    }
    el.style.display = '';
    if (loadedCount >= activities.length) {
        el.textContent = `All ${activities.length} activities loaded`;
        el.classList.add('done');
    } else {
        el.textContent = `Loading power data… ${loadedCount} / ${activities.length}`;
        el.classList.remove('done');
    }
}

async function fetchActivityCurve(activity) {
    const resp = await fetch(`/api/power_dashboard/activity/${encodeURIComponent(activity.activity_id)}`);
    if (resp.status === 401) {
        window.location.href = '/auth/login?next=/power';
        throw new Error('Not authenticated');
    }
    if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
    }
    return resp.json();
}

async function loadAllActivities() {
    let loaded = 0;
    for (let i = 0; i < activities.length; i++) {
        const activity = activities[i];
        activity.status = 'loading';
        updateActivityRow(i);

        try {
            const data = await fetchActivityCurve(activity);
            if (data.has_power && Array.isArray(data.peak_watts)) {
                activity.peak_watts = data.peak_watts;
                activity.avg_power = data.avg_power ?? null;
                activity.avg_hr = data.avg_hr ?? null;
                activity.is_race = Boolean(data.is_race);
                activity.event_id = data.event_id || null;
                activity.status = 'loaded';
            } else {
                activity.status = 'nodata';
            }
        } catch (err) {
            if (err && err.message === 'Not authenticated') {
                return;
            }
            activity.status = 'error';
        }

        loaded += 1;
        updateActivityRow(i);
        updateLoadSummary(loaded);
        queueChartRefresh();
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
function bindControls() {
    const durationSelect = document.getElementById('weeklyDurationSelect');
    if (durationSelect) {
        durationSelect.value = String(weeklyDurationSec);
        durationSelect.addEventListener('change', (event) => {
            updateWeeklyBars(Number(event.target.value));
        });
    }

    const clearBtn = document.getElementById('clearDividersBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearDividers);
    }
}

async function init() {
    const loading = document.getElementById('loadingState');
    const error = document.getElementById('errorState');
    const dashboard = document.getElementById('dashboard');
    const listPanel = document.getElementById('activityListPanel');

    setStatusText('loadingProgress', 'Fetching activity list…');

    try {
        const resp = await fetch('/api/power_dashboard/activities');
        if (resp.status === 401) {
            window.location.href = '/auth/login?next=/power';
            return;
        }
        if (!resp.ok) {
            const body = await resp.json().catch(() => ({}));
            throw new Error(body.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        durationsSec = data.durations_sec || [];
        durationsLabel = data.durations_label || [];
        activities = (data.activities || []).map((a) => ({
            activity_id: a.activity_id,
            name: a.name,
            start_date: a.start_date,
            end_date: a.end_date,
            duration_sec: a.duration_sec,
            distance_km: a.distance_km,
            peak_watts: null,
            avg_power: null,
            avg_hr: null,
            is_race: false,
            event_id: null,
            status: 'pending',
        }));

        loading.style.display = 'none';

        if (activities.length === 0) {
            error.style.display = '';
            error.textContent = 'No activities found in the last 90 days.';
            return;
        }

        dashboard.style.display = 'grid';
        listPanel.style.display = '';

        bindControls();
        renderPowerCurveChart();
        renderPartitionLegend();
        renderWeeklyBarChart();
        renderActivityList();
        updateInspectPanel(0);
        updatePinStateLabel();

        await loadAllActivities();
    } catch (err) {
        loading.style.display = 'none';
        error.style.display = '';
        error.textContent = `Unable to load dashboard: ${err.message}`;
    }
}

init();
