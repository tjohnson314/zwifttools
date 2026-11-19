'use strict';

let powerPlayData = null;
let powerPlayBusy = false;
const powerPlaySort = new Map();

function powerPlayRanks(rows, key) {
    const sorted = rows.filter(row => Number.isFinite(row[key]))
        .sort((a, b) => b[key] - a[key] || a.name.localeCompare(b.name));
    const ranks = new Map();
    let rank = 0;
    sorted.forEach((row, index) => {
        if (index === 0 || row[key] !== sorted[index - 1][key]) rank = index + 1;
        ranks.set(row, rank);
    });
    return ranks;
}

function powerPlayRows(rows, duration, key, direction = -1) {
    const filtered = rows.filter(row => row.duration_sec === duration);
    const rankKey = { watts_rank: 'avg_power', wkg_rank: 'watts_per_kg' }[key];
    const ranks = rankKey ? powerPlayRanks(filtered, rankKey) : null;
    return filtered.sort((a, b) => {
        const left = ranks ? ranks.get(a) : a[key];
        const right = ranks ? ranks.get(b) : b[key];
        if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
        return (Number.isFinite(left) ? (left - right) * direction : 0) || a.name.localeCompare(b.name);
    });
}

function powerPlayCell(row, value) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.appendChild(cell);
    return cell;
}

function renderPowerPlay() {
    if (!powerPlayData) return;
    for (const duration of powerPlayData.durations) {
        const { key, direction } = powerPlaySort.get(duration) || { key: 'avg_power', direction: -1 };
        const rows = powerPlayRows(powerPlayData.rows, duration, key, direction);
        const wattsRanks = powerPlayRanks(rows, 'avg_power');
        const wkgRanks = powerPlayRanks(rows, 'watts_per_kg');
        const body = document.getElementById(`efforts-${duration}`);
        body.replaceChildren();
        if (!rows.length) {
            const row = document.createElement('tr');
            powerPlayCell(row, 'No complete recorded efforts for this duration.').colSpan = 7;
            body.appendChild(row);
        }
        for (const effort of rows) {
            const row = document.createElement('tr');
            powerPlayCell(row, wattsRanks.get(effort) ?? '\u2014');
            powerPlayCell(row, wkgRanks.get(effort) ?? '\u2014');
            const link = document.createElement('a');
            link.textContent = effort.name;
            link.href = `https://www.zwift.com/activity/${encodeURIComponent(effort.activity_id)}`;
            link.target = '_blank';
            link.rel = 'noopener';
            powerPlayCell(row, '').appendChild(link);
            powerPlayCell(row, effort.avg_power.toFixed(1));
            powerPlayCell(row, Number.isFinite(effort.watts_per_kg) ? effort.watts_per_kg.toFixed(2) : '\u2014');
            powerPlayCell(row, Number.isFinite(effort.weight_kg) ? effort.weight_kg.toFixed(1) : '\u2014');
            const start = effort.start_time_sec.toFixed(2);
            powerPlayCell(row, `${effort.arch_name} @ ${start}s`).title = 'Start time in recorded activity telemetry';
            body.appendChild(row);
        }
        for (const header of document.querySelectorAll(`th[data-duration="${duration}"]`)) {
            const active = header.dataset.powerSort === key;
            header.setAttribute('aria-sort', active ? (direction === -1 ? 'descending' : 'ascending') : 'none');
            header.querySelector('.sort-indicator').textContent = active
                ? (direction === -1 ? '\u2193' : '\u2191') : '\u2195';
        }
    }
}

function sortPowerPlay(duration, key) {
    const previous = powerPlaySort.get(duration) || { key: 'avg_power', direction: -1 };
    powerPlaySort.set(duration, {
        key, direction: previous.key === key ? -previous.direction : (key.endsWith('_rank') ? 1 : -1),
    });
    renderPowerPlay();
}

function powerPlayStatus(message, error = false) {
    const status = document.getElementById('power-play-status');
    status.textContent = message;
    status.className = error ? 'error' : '';
}

async function powerPlayJson(url) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Server error: ${response.status}`);
    return data;
}

function fetchPowerPlayRace(activityId, forceRefresh) {
    const params = new URLSearchParams({ activity_id: activityId });
    if (forceRefresh) params.set('force_refresh', '1');
    return new Promise((resolve, reject) => {
        const stream = new EventSource(`/api/race/fetch_stream?${params}`);
        stream.onmessage = event => {
            try {
                const data = JSON.parse(event.data);
                if (data.error) throw new Error(data.error);
                if (data.progress) {
                    powerPlayStatus(`Fetching ${data.current}/${data.total}: ${data.name}`);
                } else if (data.success && data.race_id) {
                    stream.close();
                    resolve(data.race_id);
                } else {
                    throw new Error('Race download returned an unexpected response.');
                }
            } catch (error) {
                stream.close();
                reject(error);
            }
        };
        stream.onerror = () => {
            stream.close();
            reject(new Error('Race download failed. Check the Zwift service and try again.'));
        };
    });
}

async function loadPowerPlay(forceRefresh = false) {
    if (powerPlayBusy) return;
    const input = document.getElementById('event-input').value.trim();
    if (!/^[0-9]{1,20}$/.test(input) || /^0+$/.test(input)) {
        powerPlayStatus('Enter a positive Zwift event ID (not the pen ID).', true);
        return;
    }
    powerPlayBusy = true;
    document.getElementById('fetch-btn').disabled = true;
    document.getElementById('refetch-btn').disabled = true;
    document.getElementById('power-play-results').hidden = true;
    powerPlayData = null;
    const params = new URLSearchParams({ event_id: input });
    window.history.replaceState(null, '', `?${params}`);
    powerPlayStatus('Loading race data...');
    try {
        const resolved = await powerPlayJson(
            `/api/power-play/resolve/${encodeURIComponent(input)}${forceRefresh ? '?force_refresh=1' : ''}`);
        let raceId = resolved.race_id;
        if (forceRefresh || !raceId) {
            if (!resolved.source_activity_id) throw new Error('No source activity is available to fetch this event.');
            raceId = await fetchPowerPlayRace(resolved.source_activity_id, forceRefresh);
        }
        const data = await powerPlayJson(`/api/power-play/leaderboard/${encodeURIComponent(raceId)}`);
        powerPlayData = data;
        document.getElementById('race-title').textContent = data.race_name;
        document.getElementById('rider-count').textContent = `${data.rider_count} riders in race roster`;
        document.getElementById('replay-link').href = `/race-replay?activity_id=${encodeURIComponent(raceId)}`;
        const warnings = document.getElementById('warning-list');
        warnings.replaceChildren();
        for (const warning of data.warnings) {
            const item = document.createElement('li');
            item.textContent = warning;
            warnings.appendChild(item);
        }
        document.getElementById('power-play-warnings').hidden = data.warnings.length === 0;
        document.getElementById('warning-count').textContent = `${data.warnings.length} data warning(s)`;
        renderPowerPlay();
        document.getElementById('power-play-results').hidden = false;
        powerPlayStatus(data.rows.length ? 'Leaderboard loaded.' : 'No complete power-arch efforts were found. See data warnings below.');
    } catch (error) {
        console.error('Power Play load failed:', error);
        powerPlayStatus(error.message, true);
    } finally {
        powerPlayBusy = false;
        document.getElementById('fetch-btn').disabled = false;
        document.getElementById('refetch-btn').disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('power-play-form').addEventListener('submit', event => {
        event.preventDefault();
        loadPowerPlay();
    });
    document.getElementById('refetch-btn').addEventListener('click', () => loadPowerPlay(true));
    document.querySelectorAll('th[data-power-sort] button').forEach(button => {
        button.addEventListener('click', () => {
            const header = button.closest('th');
            sortPowerPlay(Number(header.dataset.duration), header.dataset.powerSort);
        });
    });
    const params = new URLSearchParams(window.location.search);
    const eventId = params.get('event_id');
    if (eventId) {
        document.getElementById('event-input').value = eventId;
        loadPowerPlay();
    }
});
