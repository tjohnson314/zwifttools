let dashboardPayload = null;
let powerCurveChart = null;
let weeklyBarChart = null;
let progressPollTimer = null;
const POWER_DASHBOARD_LOOKBACK_DAYS = 90;
const POWER_DASHBOARD_TIMEOUT_MS = 45000;

const POWER_CURVE_X_TICKS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600, 7200];

let selectedIndex = 0;
let pinnedIndex = null;
let currentRange = null;

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
        const m = Math.floor(seconds / 60);
        return `${m}m`;
    }
    const h = Math.floor(seconds / 3600);
    return `${h}h`;
}

function formatDate(value) {
    if (!value) {
        return '—';
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return '—';
    }

    return date.toLocaleDateString(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function setStatusText(elementId, text, isError = false) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }

    element.textContent = text;
    element.classList.toggle('error', Boolean(isError));
}

function destroyCharts() {
    if (powerCurveChart) {
        powerCurveChart.destroy();
        powerCurveChart = null;
    }

    if (weeklyBarChart) {
        weeklyBarChart.destroy();
        weeklyBarChart = null;
    }
}

function toIsoUtc(datetimeLocalValue) {
    const date = new Date(datetimeLocalValue);
    if (Number.isNaN(date.getTime())) {
        return null;
    }
    return date.toISOString();
}

function validateRangeInputs() {
    const startValue = document.getElementById('rangeStartInput')?.value || '';
    const endValue = document.getElementById('rangeEndInput')?.value || '';

    if (!startValue && !endValue) {
        return { rangeStartIso: null, rangeEndIso: null, error: null };
    }

    if (!startValue || !endValue) {
        return { rangeStartIso: null, rangeEndIso: null, error: 'Select both a start and end time.' };
    }

    const startDate = new Date(startValue);
    const endDate = new Date(endValue);
    if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
        return { rangeStartIso: null, rangeEndIso: null, error: 'Select a valid start and end time.' };
    }

    const now = new Date();
    const lookbackStart = new Date(now.getTime() - POWER_DASHBOARD_LOOKBACK_DAYS * 24 * 60 * 60 * 1000);

    if (startDate < lookbackStart || endDate < lookbackStart || startDate > now || endDate > now) {
        return { rangeStartIso: null, rangeEndIso: null, error: 'Range must be within the past 90 days.' };
    }

    if (startDate >= endDate) {
        return { rangeStartIso: null, rangeEndIso: null, error: 'Range start must be before range end.' };
    }

    return {
        rangeStartIso: toIsoUtc(startValue),
        rangeEndIso: toIsoUtc(endValue),
        error: null
    };
}

function formatWatts(value) {
    if (value === null || value === undefined) {
        return '—';
    }
    return `${Math.round(value)} watts`;
}

function activityHref(activityId) {
    if (!activityId) {
        return '#';
    }
    return `https://www.zwift.com/activity/${activityId}`;
}

function createActionLink({ href, title, className = 'action-link', innerHTML }) {
    const a = document.createElement('a');
    a.href = href;
    a.className = className;
    a.title = title;
    a.innerHTML = innerHTML;
    if (href.startsWith('http')) {
        a.target = '_blank';
        a.rel = 'noopener';
    }
    return a;
}

function renderActivityActions(containerId, activity) {
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }

    container.innerHTML = '';
    if (!activity || !activity.activity_id) {
        return;
    }

    container.appendChild(createActionLink({
        href: `/bike-comparison?activity_id=${activity.activity_id}`,
        title: 'Bike Comparison',
        innerHTML: '&#x1F6B2;'
    }));

    if (activity.is_race) {
        container.appendChild(createActionLink({
            href: `/race-replay?activity_id=${activity.activity_id}`,
            title: 'Race Replay',
            className: 'action-link race',
            innerHTML: '&#x1F3C1;'
        }));
    }

    container.appendChild(createActionLink({
        href: `https://www.zwift.com/activity/${activity.activity_id}`,
        title: 'View on Zwift',
        className: 'action-link logo-link',
        innerHTML: '<img src="/static/img/zwift.ico" alt="Zwift" class="logo-icon">'
    }));

    if (activity.is_race && activity.event_id) {
        container.appendChild(createActionLink({
            href: `https://zwiftpower.com/events.php?zid=${activity.event_id}`,
            title: 'View on ZwiftPower',
            className: 'action-link logo-link',
            innerHTML: '<img src="/static/img/zwiftpower.svg" alt="ZwiftPower" class="logo-icon">'
        }));
    }
}

function updatePinStateLabel() {
    const pinState = document.getElementById('pinState');
    if (pinnedIndex === null) {
        pinState.textContent = 'Hover to inspect';
    } else {
        pinState.textContent = 'Pinned (click chart to unpin)';
    }
}

function updateTopPanel(index) {
    if (!dashboardPayload || !dashboardPayload.durations_sec || dashboardPayload.durations_sec.length === 0) {
        return;
    }
    selectedIndex = index;

    const sec = dashboardPayload.durations_sec[index];
    const secKey = String(sec);

    const best = dashboardPayload.best_activity_by_duration[secKey] || null;
    const latestMeta = dashboardPayload.latest_activity || null;
    const latestPower = dashboardPayload.latest_curve_watts[index];
    const rangeMeta = getCustomRangeBestActivity(index);
    const rangePower = rangeMeta ? rangeMeta.power_watts : null;

    document.getElementById('selectedInterval').textContent = formatIntervalClock(sec);
    document.getElementById('selectedIntervalLabel').textContent = dashboardPayload.durations_label[index];

    document.getElementById('bestPowerValue').textContent = best ? formatWatts(best.power_watts) : '—';
    const bestLink = document.getElementById('bestActivityLink');
    bestLink.textContent = best ? best.name : '—';
    bestLink.href = best ? activityHref(best.activity_id) : '#';
    document.getElementById('bestActivityDate').textContent = best ? formatDate(best.start_date) : '—';

    document.getElementById('latestPowerValue').textContent = latestPower !== null ? formatWatts(latestPower) : '—';
    const latestLink = document.getElementById('latestActivityLink');
    latestLink.textContent = latestMeta ? latestMeta.name : '—';
    latestLink.href = latestMeta ? activityHref(latestMeta.activity_id) : '#';
    document.getElementById('latestActivityDate').textContent = latestMeta ? formatDate(latestMeta.start_date) : '—';

    renderActivityActions('bestActivityLinks', best);
    renderActivityActions('latestActivityLinks', latestMeta);

    document.getElementById('rangePowerValue').textContent = rangePower !== null ? formatWatts(rangePower) : '—';
    const rangeLink = document.getElementById('rangeActivityLink');
    rangeLink.textContent = rangeMeta ? rangeMeta.name : '—';
    rangeLink.href = rangeMeta ? activityHref(rangeMeta.activity_id) : '#';
    document.getElementById('rangeActivityDate').textContent = rangeMeta ? formatDate(rangeMeta.start_date) : '—';
    renderActivityActions('rangeActivityLinks', rangeMeta);
}

function getCustomRangeBestActivity(index) {
    if (!currentRange || !dashboardPayload || !Array.isArray(dashboardPayload.activities)) {
        return null;
    }

    const target = dashboardPayload.durations_sec[index];
    if (target === undefined) {
        return null;
    }

    const rangeStart = new Date(currentRange.start);
    const rangeEnd = new Date(currentRange.end);
    if (Number.isNaN(rangeStart.getTime()) || Number.isNaN(rangeEnd.getTime())) {
        return null;
    }

    let bestActivity = null;
    let bestPower = null;
    for (const activity of dashboardPayload.activities) {
        const activityStart = new Date(activity.start_date);
        const activityEnd = new Date(activity.end_date || activity.start_date);
        if (Number.isNaN(activityStart.getTime()) || Number.isNaN(activityEnd.getTime())) {
            continue;
        }
        if (activityStart > rangeEnd || activityEnd < rangeStart) {
            continue;
        }

        const power = activity.peak_watts ? activity.peak_watts[index] : null;
        if (power === null || power === undefined) {
            continue;
        }
        if (bestPower === null || power > bestPower) {
            bestPower = power;
            bestActivity = {
                activity_id: activity.activity_id,
                name: activity.name,
                start_date: activity.start_date,
                is_race: activity.is_race,
                event_id: activity.event_id,
                power_watts: power,
            };
        }
    }

    return bestActivity;
}

function renderPowerCurveChart() {
    if (!dashboardPayload || !dashboardPayload.durations_label || dashboardPayload.durations_label.length === 0) {
        return;
    }

    const hasCustomRange = Boolean(
        currentRange ||
        (Array.isArray(dashboardPayload.range_curve_watts) && dashboardPayload.range_curve_watts.some((value) => value !== null && value !== undefined))
    );
    const rangeData = hasCustomRange
        ? dashboardPayload.durations_sec.map((sec, idx) => ({
            x: sec,
            y: dashboardPayload.range_curve_watts ? dashboardPayload.range_curve_watts[idx] : null
        }))
        : dashboardPayload.durations_sec.map((sec) => ({ x: sec, y: null }));

    const ctx = document.getElementById('powerCurveChart').getContext('2d');

    powerCurveChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Last 90 days',
                    data: dashboardPayload.durations_sec.map((sec, idx) => ({
                        x: sec,
                        y: dashboardPayload.all_90_day_curve_watts[idx]
                    })),
                    borderColor: '#f0f4ff',
                    backgroundColor: 'transparent',
                    borderWidth: 3.5,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: true,
                    tension: 0.08
                },
                {
                    label: 'Latest activity',
                    data: dashboardPayload.durations_sec.map((sec, idx) => ({
                        x: sec,
                        y: dashboardPayload.latest_curve_watts[idx]
                    })),
                    borderColor: '#6adf71',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: true,
                    tension: 0.08
                },
                {
                    label: 'Custom range',
                    data: rangeData,
                    borderColor: '#ffd166',
                    backgroundColor: 'transparent',
                    borderDash: [6, 4],
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: true,
                    tension: 0.08
                }
            ]
        },
        options: {
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    min: 1,
                    max: 7200,
                    afterBuildTicks(scale) {
                        scale.ticks = POWER_CURVE_X_TICKS.map((value) => ({ value }));
                    },
                    grid: {
                        color: '#e6e6e6'
                    },
                    ticks: {
                        autoSkip: false,
                        maxRotation: 0,
                        minRotation: 0,
                        callback(value) {
                            return formatTickInterval(Number(value));
                        }
                    }
                },
                y: {
                    grid: {
                        color: '#e6e6e6'
                    },
                    ticks: {
                        callback(value) {
                            return `${value} W`;
                        }
                    }
                }
            },
            onHover(event, elements) {
                if (pinnedIndex !== null || !elements || elements.length === 0) {
                    return;
                }
                updateTopPanel(elements[0].index);
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

                const idx = elements[0].index;
                pinnedIndex = idx;
                updateTopPanel(idx);
                updatePinStateLabel();
            }
        }
    });
}

function renderWeeklyBarChart(initialDurationSec) {
    if (!dashboardPayload || !dashboardPayload.weekly) {
        return;
    }

    const ctx = document.getElementById('weeklyBarChart').getContext('2d');

    weeklyBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dashboardPayload.weekly.labels,
            datasets: [
                {
                    label: 'Weekly peak power',
                    data: dashboardPayload.weekly.series[String(initialDurationSec)] || [],
                    backgroundColor: '#72aee8',
                    borderColor: '#72aee8',
                    borderWidth: 1,
                    barPercentage: 0.95,
                    categoryPercentage: 1.0
                }
            ]
        },
        options: {
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10,
                        maxRotation: 0,
                        minRotation: 0
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback(value) {
                            return `${value} W`;
                        }
                    },
                    grid: {
                        color: '#e6e6e6'
                    }
                }
            }
        }
    });
}

function updateWeeklyBars(durationSec) {
    if (!weeklyBarChart) {
        return;
    }
    weeklyBarChart.data.datasets[0].data = dashboardPayload.weekly.series[String(durationSec)] || [];
    weeklyBarChart.update();
}

function computeCustomRangeCurve(rangeStartIso, rangeEndIso) {
    if (!dashboardPayload || !Array.isArray(dashboardPayload.activities)) {
        return { curve: null, count: 0 };
    }

    const rangeStart = new Date(rangeStartIso);
    const rangeEnd = new Date(rangeEndIso);
    if (Number.isNaN(rangeStart.getTime()) || Number.isNaN(rangeEnd.getTime())) {
        return { curve: null, count: 0 };
    }

    const rangeActivities = dashboardPayload.activities.filter((activity) => {
        const activityStart = new Date(activity.start_date);
        const activityEnd = new Date(activity.end_date || activity.start_date);
        if (Number.isNaN(activityStart.getTime()) || Number.isNaN(activityEnd.getTime())) {
            return false;
        }
        return activityStart <= rangeEnd && activityEnd >= rangeStart;
    });

    const curve = dashboardPayload.durations_sec.map((_, index) => {
        let best = null;
        for (const activity of rangeActivities) {
            const value = activity.peak_watts ? activity.peak_watts[index] : null;
            if (value === null || value === undefined) {
                continue;
            }
            if (best === null || value > best) {
                best = value;
            }
        }
        return best;
    });

    return { curve, count: rangeActivities.length };
}

function updateRangeStatusFromPayload() {
    const statusElement = document.getElementById('rangeStatus');
    if (!statusElement) {
        return;
    }

    if (!currentRange) {
        setStatusText('rangeStatus', 'No custom range selected.');
        return;
    }

    const start = formatDate(currentRange.start);
    const end = formatDate(currentRange.end);
    const count = dashboardPayload.range_activity_count || 0;
    const message = count > 0
        ? `Custom range applied: ${count} activities from ${start} to ${end}.`
        : `Custom range applied: no activities found from ${start} to ${end}.`;
    setStatusText('rangeStatus', message, false);
}

function applyCustomRange(rangeStartIso, rangeEndIso) {
    const result = computeCustomRangeCurve(rangeStartIso, rangeEndIso);
    dashboardPayload.range_start = rangeStartIso;
    dashboardPayload.range_end = rangeEndIso;
    dashboardPayload.range_curve_watts = result.curve;
    dashboardPayload.range_activity_count = result.count;
    currentRange = { start: rangeStartIso, end: rangeEndIso };

    if (powerCurveChart && powerCurveChart.data && powerCurveChart.data.datasets.length >= 3) {
        powerCurveChart.data.datasets[2].data = dashboardPayload.durations_sec.map((sec, idx) => ({
            x: sec,
            y: result.curve ? result.curve[idx] : null
        }));
        powerCurveChart.update();
    }

    if (dashboardPayload.durations_sec && dashboardPayload.durations_sec.length > 0) {
        updateTopPanel(selectedIndex || 0);
    }
    updateRangeStatusFromPayload();
}

function clearCustomRange() {
    dashboardPayload.range_start = null;
    dashboardPayload.range_end = null;
    dashboardPayload.range_curve_watts = dashboardPayload.durations_sec.map(() => null);
    dashboardPayload.range_activity_count = 0;
    currentRange = null;

    if (powerCurveChart && powerCurveChart.data && powerCurveChart.data.datasets.length >= 3) {
        powerCurveChart.data.datasets[2].data = dashboardPayload.durations_sec.map((sec) => ({ x: sec, y: null }));
        powerCurveChart.update();
    }

    if (dashboardPayload.durations_sec && dashboardPayload.durations_sec.length > 0) {
        updateTopPanel(selectedIndex || 0);
    }
    updateRangeStatusFromPayload();
}

function setRangeControlsDisabled(disabled) {
    const startInput = document.getElementById('rangeStartInput');
    const endInput = document.getElementById('rangeEndInput');
    const applyButton = document.getElementById('applyRangeBtn');
    const clearButton = document.getElementById('clearRangeBtn');

    [startInput, endInput, applyButton, clearButton].forEach((element) => {
        if (element) {
            element.disabled = disabled;
        }
    });
}

function bindRangeControls() {
    const applyButton = document.getElementById('applyRangeBtn');
    const clearButton = document.getElementById('clearRangeBtn');

    if (applyButton && !applyButton.dataset.bound) {
        applyButton.dataset.bound = '1';
        applyButton.addEventListener('click', async () => {
            const validation = validateRangeInputs();
            if (validation.error) {
                setStatusText('rangeStatus', validation.error, true);
                return;
            }

            applyCustomRange(validation.rangeStartIso, validation.rangeEndIso);
        });
    }

    if (clearButton && !clearButton.dataset.bound) {
        clearButton.dataset.bound = '1';
        clearButton.addEventListener('click', async () => {
            const startInput = document.getElementById('rangeStartInput');
            const endInput = document.getElementById('rangeEndInput');
            if (startInput) startInput.value = '';
            if (endInput) endInput.value = '';

            clearCustomRange();
        });
    }
}

function ensureWeeklyDurationSelect() {
    let durationSelect = document.getElementById('weeklyDurationSelect');
    if (durationSelect) {
        return durationSelect;
    }

    const headerRow = document.querySelector('.bars-header-row');
    if (!headerRow) {
        return null;
    }

    const control = document.createElement('div');
    control.id = 'weeklyDurationControl';
    control.className = 'duration-control';

    const label = document.createElement('label');
    label.className = 'duration-label-inline';
    label.setAttribute('for', 'weeklyDurationSelect');
    label.textContent = 'Interval';

    durationSelect = document.createElement('select');
    durationSelect.id = 'weeklyDurationSelect';
    durationSelect.setAttribute('aria-label', 'Weekly peak interval');

    const choices = [
        [1200, '20 minutes'],
        [600, '10 minutes'],
        [300, '5 minutes'],
        [60, '1 minute'],
        [30, '30 seconds'],
        [15, '15 seconds'],
        [5, '5 seconds']
    ];

    for (const [value, text] of choices) {
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = text;
        durationSelect.appendChild(option);
    }

    control.appendChild(label);
    control.appendChild(durationSelect);
    headerRow.appendChild(control);

    return durationSelect;
}

function buildProgressText(progress) {
    if (!progress) {
        return 'Starting...';
    }

    const status = progress.status || 'starting';
    const message = progress.message || '';
    const discovered = progress.discovered_activity_count;
    const selected = progress.selected_activity_count;
    const completed = progress.completed_activity_count;
    const successful = progress.successful_activity_count;

    if (status === 'telemetry_processing') {
        const left = Number.isFinite(completed) ? completed : 0;
        const right = Number.isFinite(selected) ? selected : 0;
        const ok = Number.isFinite(successful) ? successful : 0;
        return `${message} Processed ${left}/${right} activities (${ok} with telemetry).`;
    }

    if (Number.isFinite(discovered) && Number.isFinite(selected)) {
        return `${message} Found ${discovered} recent activities, selected ${selected}.`;
    }

    if (message) {
        return message;
    }

    return `Status: ${status}`;
}

function stopProgressPolling() {
    if (progressPollTimer !== null) {
        clearInterval(progressPollTimer);
        progressPollTimer = null;
    }
}

function startProgressPolling(progressId, statusElementId = 'loadingProgress') {
    stopProgressPolling();
    if (!progressId) {
        return;
    }

    const fetchProgress = async () => {
        try {
            const response = await fetch(`/api/power_dashboard_progress?progress_id=${encodeURIComponent(progressId)}`);
            if (!response.ok) {
                return;
            }

            const progress = await response.json();
            setStatusText(statusElementId, buildProgressText(progress), false);

            if (progress.status === 'completed' || progress.status === 'error') {
                stopProgressPolling();
            }
        } catch (_) {
            // Non-fatal: keep the dashboard request running.
        }
    };

    fetchProgress();
    progressPollTimer = setInterval(fetchProgress, 900);
}

async function loadDashboard(options = {}) {
    const {
        statusElementId = 'loadingProgress',
        showOverlay = true,
    } = options;

    const loading = document.getElementById('loadingState');
    const error = document.getElementById('errorState');
    const dashboard = document.getElementById('dashboard');

    if (showOverlay) {
        loading.style.display = '';
        dashboard.style.display = 'none';
    } else {
        loading.style.display = 'none';
        if (dashboardPayload) {
            dashboard.style.display = 'grid';
        }
    }
    error.style.display = 'none';
    setStatusText(statusElementId, showOverlay ? 'Starting...' : 'Refreshing custom range...');

    setRangeControlsDisabled(true);

    try {
        const progressId = (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function')
            ? globalThis.crypto.randomUUID()
            : `progress_${Date.now()}_${Math.floor(Math.random() * 100000)}`;

        startProgressPolling(progressId, statusElementId);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), POWER_DASHBOARD_TIMEOUT_MS);
        const response = await fetch(`/api/power_dashboard?progress_id=${encodeURIComponent(progressId)}`, { signal: controller.signal });
        clearTimeout(timeoutId);
        stopProgressPolling();

        if (response.status === 401) {
            window.location.href = '/auth/login?next=/power';
            return;
        }

        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.error || `HTTP ${response.status}`);
        }

        dashboardPayload = await response.json();
        dashboardPayload.range_curve_watts = dashboardPayload.range_curve_watts || dashboardPayload.durations_sec.map(() => null);
        dashboardPayload.range_activity_count = dashboardPayload.range_activity_count || 0;
        currentRange = null;

    loading.style.display = 'none';
        dashboard.style.display = 'grid';

        if (dashboardPayload.warning) {
            error.style.display = '';
            error.textContent = dashboardPayload.warning;
        }

        destroyCharts();
        renderPowerCurveChart();
        renderWeeklyBarChart(1200);

        if (dashboardPayload.durations_sec && dashboardPayload.durations_sec.length > 0) {
            updateTopPanel(0);
        }
        updatePinStateLabel();
        updateRangeStatusFromPayload();

        const durationSelect = ensureWeeklyDurationSelect();
        if (durationSelect) {
            durationSelect.value = '1200';
            durationSelect.addEventListener('change', (event) => {
                updateWeeklyBars(Number(event.target.value));
            });
        }
    } catch (err) {
        stopProgressPolling();
        if (showOverlay) {
            loading.style.display = 'none';
            error.style.display = '';
        } else {
            setStatusText(statusElementId, err && err.name === 'AbortError'
                ? 'Custom range refresh timed out after 45s. Please retry.'
                : `Custom range refresh failed: ${err.message}`,
            true);
        }
        if (err && err.name === 'AbortError') {
            if (showOverlay) {
                error.textContent = 'Unable to load dashboard: request timed out after 45s. Please retry.';
            }
        } else {
            if (showOverlay) {
                error.textContent = `Unable to load dashboard: ${err.message}`;
            }
        }
    } finally {
        setRangeControlsDisabled(false);
    }
}

bindRangeControls();
loadDashboard();
