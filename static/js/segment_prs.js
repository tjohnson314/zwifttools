'use strict';

const SEGMENT_CACHE_PREFIX = 'segmentPRs:';
const SEGMENT_PAGE_SIZE = 20;
let segmentContext = null;
let segmentActivities = [];
let selectedSegment = new URLSearchParams(window.location.search).get('segment') || '';
let segmentQuery = '';
let segmentPage = 0;
let segmentLoading = false;
let segmentSortKey = null;
let segmentSortDirection = 1;

function segmentCacheKey(activity) {
    return `${SEGMENT_CACHE_PREFIX}${segmentContext.profile_id}:${activity.activity_id}`;
}

function segmentCacheSignature(activity) {
    return JSON.stringify([
        segmentContext.catalog_version, segmentContext.weight_kg, segmentContext.height_cm,
        activity.end_date, activity.duration_sec, activity.distance_km,
    ]);
}

function validSegmentData(data, activity) {
    return data && data.activity_id === String(activity.activity_id) &&
        data.profile_id === segmentContext.profile_id &&
        data.catalog_version === segmentContext.catalog_version &&
        data.weight_kg === segmentContext.weight_kg && data.height_cm === segmentContext.height_cm &&
        Array.isArray(data.traversals) && data.traversals.every(row =>
            typeof row.segment_id === 'string' && typeof row.segment_name === 'string' &&
            Number.isInteger(row.pass) && row.pass > 0 &&
            Number.isFinite(row.start_time_sec) && Number.isFinite(row.end_time_sec) &&
            row.end_time_sec > row.start_time_sec && Number.isFinite(row.total_time_sec) &&
            Math.abs(row.total_time_sec - (row.end_time_sec - row.start_time_sec)) < 1e-6 &&
            row.total_time_sec > 0 && Number.isFinite(row.distance_m) &&
            ['elevation_change_m', 'avg_power', 'normalized_power', 'avg_draft_watts', 'initial_speed_kmh']
                .every(key => row[key] === null || Number.isFinite(row[key])));
}

function storageWarning(error) {
    console.warn('Segment PR cache unavailable:', error);
    const warning = document.getElementById('cacheWarning');
    warning.textContent = 'Browser cache is unavailable or full. Results are still loaded, but may need to be fetched again.';
    warning.hidden = false;
}

function readSegmentCache(activity) {
    try {
        const stored = localStorage.getItem(segmentCacheKey(activity));
        if (!stored) return null;
        const cached = JSON.parse(stored);
        if (cached.signature === segmentCacheSignature(activity) && validSegmentData(cached.data, activity)) {
            return cached.data;
        }
        localStorage.removeItem(segmentCacheKey(activity));
    } catch (error) {
        storageWarning(error);
    }
    return null;
}

function saveSegmentCache(activity, data) {
    try {
        localStorage.setItem(segmentCacheKey(activity), JSON.stringify({
            signature: segmentCacheSignature(activity), data,
        }));
    } catch (error) {
        storageWarning(error);
    }
}

function pruneSegmentCache(clear = false) {
    const prefix = `${SEGMENT_CACHE_PREFIX}${segmentContext.profile_id}:`;
    const active = new Set(segmentActivities.map(segmentCacheKey));
    try {
        for (let index = localStorage.length - 1; index >= 0; index--) {
            const key = localStorage.key(index);
            if (key && key.startsWith(prefix) && (clear || !active.has(key))) {
                localStorage.removeItem(key);
            }
        }
        return true;
    } catch (error) {
        storageWarning(error);
        return false;
    }
}

function allSegmentRows() {
    return segmentActivities.flatMap(activity => (activity.data?.traversals || []).map(row => ({
        ...row, activity,
    })));
}

function segmentTableRows() {
    let rows = allSegmentRows();
    if (selectedSegment) {
        rows = rows.filter(row => row.segment_id === selectedSegment);
    } else {
        const best = new Map();
        for (const row of rows) {
            const previous = best.get(row.segment_id);
            if (!previous || row.total_time_sec < previous.total_time_sec) best.set(row.segment_id, row);
        }
        rows = [...best.values()];
    }
    return rows.filter(row => row.segment_name.toLocaleLowerCase().includes(segmentQuery.toLocaleLowerCase()))
        .sort(compareSegmentRows);
}

function effectiveSegmentSortKey() {
    return segmentSortKey || (selectedSegment ? 'total_time_sec' : 'segment_name');
}

function segmentSortValue(row, key) {
    if (key === 'start_date') {
        const timestamp = Date.parse(row.activity.start_date);
        return Number.isFinite(timestamp) ? timestamp : null;
    }
    return row[key];
}

function compareSegmentRows(a, b) {
    const key = effectiveSegmentSortKey();
    const left = segmentSortValue(a, key);
    const right = segmentSortValue(b, key);
    const leftMissing = left === null || left === undefined;
    const rightMissing = right === null || right === undefined;
    if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
    const comparison = leftMissing ? 0 : typeof left === 'string'
        ? left.localeCompare(right) : left - right;
    return comparison * segmentSortDirection ||
        a.segment_name.localeCompare(b.segment_name) ||
        b.activity.start_date.localeCompare(a.activity.start_date) ||
        a.start_time_sec - b.start_time_sec ||
        a.segment_id.localeCompare(b.segment_id) ||
        String(a.activity.activity_id).localeCompare(String(b.activity.activity_id));
}

function toggleSegmentSort(key) {
    segmentSortDirection = key === effectiveSegmentSortKey() ? -segmentSortDirection : 1;
    segmentSortKey = key;
    segmentPage = 0;
    renderSegmentTable();
}

function updateSegmentSortHeaders() {
    for (const header of document.querySelectorAll('th[data-segment-sort]')) {
        const active = header.dataset.segmentSort === effectiveSegmentSortKey();
        const direction = segmentSortDirection === 1 ? 'ascending' : 'descending';
        header.setAttribute('aria-sort', active ? direction : 'none');
        header.querySelector('.sort-indicator').textContent = active
            ? segmentSortDirection === 1 ? '\u2191' : '\u2193' : '\u2195';
    }
}

function segmentClock(value) {
    const centiseconds = Math.round(value * 100);
    const minutes = Math.floor(centiseconds / 6000);
    return `${minutes}:${((centiseconds % 6000) / 100).toFixed(2).padStart(5, '0')}`;
}

function segmentMetric(value, unit, digits = 0) {
    return value === null || value === undefined ? '\u2014' : `${value.toFixed(digits)} ${unit}`;
}

function selectSegment(id) {
    selectedSegment = id;
    segmentPage = 0;
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('segment', id);
    else url.searchParams.delete('segment');
    window.history.replaceState({}, '', url);
    renderSegmentTable();
}

function renderSegmentTable() {
    updateSegmentSortHeaders();
    const names = new Map(allSegmentRows().map(row => [row.segment_id, row.segment_name]));
    const select = document.getElementById('segmentSelect');
    select.replaceChildren(new Option('Best time for each segment', ''));
    for (const [id, name] of [...names].sort((a, b) => a[1].localeCompare(b[1]))) {
        select.add(new Option(name, id));
    }
    if (selectedSegment && !names.has(selectedSegment)) select.add(new Option('Selected segment (no loaded attempts)', selectedSegment));
    select.value = selectedSegment;
    document.getElementById('tableTitle').textContent = selectedSegment
        ? `All attempts: ${names.get(selectedSegment) || 'selected segment'}` : 'Best times by segment';
    const rows = segmentTableRows();
    const pageCount = Math.max(1, Math.ceil(rows.length / SEGMENT_PAGE_SIZE));
    segmentPage = Math.min(segmentPage, pageCount - 1);
    const body = document.getElementById('segmentRows');
    body.replaceChildren();
    for (const row of rows.slice(segmentPage * SEGMENT_PAGE_SIZE, (segmentPage + 1) * SEGMENT_PAGE_SIZE)) {
        const tr = document.createElement('tr');
        const cell = (text, numeric = false) => {
            const td = document.createElement('td');
            td.textContent = text;
            if (numeric) td.className = 'num';
            tr.appendChild(td);
            return td;
        };
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'segment-name';
        button.textContent = row.segment_name;
        button.addEventListener('click', () => selectSegment(row.segment_id));
        cell('').appendChild(button);
        const ride = cell(row.activity.name);
        const date = document.createElement('span');
        date.className = 'ride-date';
        date.textContent = new Date(row.activity.start_date).toLocaleString();
        ride.appendChild(date);
        cell(segmentClock(row.total_time_sec), true);
        cell(segmentMetric(row.distance_m, 'm', 1), true);
        cell(segmentMetric(row.elevation_change_m, 'm', 1), true);
        cell(segmentMetric(row.avg_power, 'W', 1), true);
        cell(segmentMetric(row.normalized_power, 'W', 1), true);
        cell(segmentMetric(row.avg_draft_watts, 'W', 1), true);
        cell(segmentMetric(row.initial_speed_kmh, 'km/h', 1), true);
        const links = cell('');
        links.className = 'segment-actions';
        const link = (title, href, icon, className = 'action-link') => {
            const a = document.createElement('a');
            a.className = className;
            a.title = title;
            a.setAttribute('aria-label', title);
            a.innerHTML = icon;
            a.href = href;
            links.appendChild(a);
            return a;
        };
        const id = encodeURIComponent(row.activity.activity_id);
        link('Bike Comparison', `/bike-comparison?activity_id=${id}`, '&#x1F6B2;');
        if (row.activity.data.is_race) {
            link('Race Replay', `/race-replay?activity_id=${id}`, '&#x1F3C1;', 'action-link race');
        }
        const zwiftLink = link('View on Zwift', `https://www.zwift.com/activity/${id}`,
            '<img src="/static/img/zwift.ico" alt="Zwift" class="logo-icon">', 'action-link logo-link');
        zwiftLink.target = '_blank';
        zwiftLink.rel = 'noopener';
        body.appendChild(tr);
    }
    if (!rows.length) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 10;
        td.textContent = segmentLoading ? 'No matching completed traversals loaded yet.' : 'No matching complete traversals in the past 90 days.';
        tr.appendChild(td);
        body.appendChild(tr);
    }
    document.getElementById('pageStatus').textContent = `${rows.length} results \u00b7 Page ${segmentPage + 1} of ${pageCount}`;
    document.getElementById('previousBtn').disabled = segmentPage === 0;
    document.getElementById('nextBtn').disabled = segmentPage >= pageCount - 1;
}

async function segmentFetch(url) {
    const response = await fetch(url);
    if (response.status === 401) {
        window.location.href = '/auth/login?next=/segment-prs';
        throw new Error('Not authenticated');
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

function segmentLoadSummary() {
    const done = segmentActivities.filter(activity => activity.data || activity.error).length;
    const errors = segmentActivities.filter(activity => activity.error);
    const cached = segmentActivities.filter(activity => activity.cached).length;
    const warnings = [...new Set(segmentActivities.map(activity => activity.data?.warning).filter(Boolean))];
    const unavailable = segmentActivities.filter(activity => activity.data?.segment_status === 'unavailable').length;
    document.getElementById('loadStatus').textContent =
        `${segmentLoading ? 'Loading' : 'Loaded'} ${done} / ${segmentActivities.length} activities; ${cached} cached; ` +
        `${allSegmentRows().length} traversals; ${errors.length} failed; ${unavailable} without available segment data. Best times ${segmentLoading || errors.length ? 'are provisional' : 'are up to date'}.` +
        (warnings.length ? ` ${warnings.join(' ')}` : '');
    document.getElementById('retryBtn').hidden = !errors.length;
    document.getElementById('failureDetails').hidden = !errors.length;
    const list = document.getElementById('failureList');
    list.replaceChildren();
    for (const activity of errors) {
        const li = document.createElement('li');
        li.textContent = `${activity.name} (${activity.activity_id}): ${activity.error}`;
        list.appendChild(li);
    }
}

function segmentBusy(busy) {
    segmentLoading = busy;
    for (const id of ['refreshBtn', 'clearCacheBtn', 'retryBtn']) document.getElementById(id).disabled = busy;
}

async function loadSegmentActivities(retryOnly = false) {
    segmentBusy(true);
    try {
        for (const activity of segmentActivities) {
            if (retryOnly && !activity.error) continue;
            delete activity.error;
            try {
                const cached = retryOnly ? null : readSegmentCache(activity);
                const data = cached || await segmentFetch(`/api/segment_prs/activity/${encodeURIComponent(activity.activity_id)}`);
                if (!validSegmentData(data, activity)) throw new Error('Segment response changed or is invalid. Refresh activities to retry.');
                activity.data = data;
                activity.cached = Boolean(cached);
                if (!cached) saveSegmentCache(activity, data);
            } catch (error) {
                activity.error = error.message;
                console.error('Segment activity load failed:', activity.activity_id, error);
                if (error.message === 'Not authenticated') return;
            }
            renderSegmentTable();
            segmentLoadSummary();
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    } finally {
        segmentBusy(false);
        segmentLoadSummary();
        renderSegmentTable();
    }
}

async function initSegmentPRs() {
    segmentBusy(true);
    document.getElementById('errorState').hidden = true;
    try {
        const data = await segmentFetch('/api/segment_prs/activities');
        segmentContext = data;
        segmentActivities = data.activities;
        pruneSegmentCache();
        document.getElementById('dashboard').hidden = false;
        renderSegmentTable();
        await loadSegmentActivities();
    } catch (error) {
        console.error('Segment dashboard load failed:', error);
        const element = document.getElementById('errorState');
        element.textContent = error.message;
        element.hidden = false;
        document.getElementById('loadStatus').textContent = 'Could not load recent activities. Reload the page to try again.';
    } finally {
        segmentBusy(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    for (const header of document.querySelectorAll('th[data-segment-sort]')) {
        header.querySelector('button').addEventListener('click', () => toggleSegmentSort(header.dataset.segmentSort));
    }
    document.getElementById('segmentSearch').addEventListener('input', event => {
        segmentQuery = event.target.value;
        segmentPage = 0;
        renderSegmentTable();
    });
    document.getElementById('segmentSelect').addEventListener('change', event => selectSegment(event.target.value));
    document.getElementById('previousBtn').addEventListener('click', () => { segmentPage--; renderSegmentTable(); });
    document.getElementById('nextBtn').addEventListener('click', () => { segmentPage++; renderSegmentTable(); });
    document.getElementById('retryBtn').addEventListener('click', () => loadSegmentActivities(true));
    document.getElementById('refreshBtn').addEventListener('click', initSegmentPRs);
    document.getElementById('clearCacheBtn').addEventListener('click', () => {
        if (pruneSegmentCache(true)) initSegmentPRs();
    });
    initSegmentPRs();
});
