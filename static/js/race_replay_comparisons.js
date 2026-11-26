'use strict';

let replayComparisonRace = null;
let replayComparisons = null;
let replayComparisonRequest = 0;
let replaySegmentSort = 'total_time_sec';
let replaySegmentAscending = true;
let replayCurveUnit = 'W';
let replaySegmentUnit = 'W';
let replayPowerDurationIndex = null;

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-race-power-unit]').forEach(button => {
        button.addEventListener('click', () => {
            replayCurveUnit = button.dataset.racePowerUnit;
            document.querySelectorAll('[data-race-power-unit]').forEach(option => {
                const active = option.dataset.racePowerUnit === replayCurveUnit;
                option.classList.toggle('active', active);
                option.setAttribute('aria-pressed', String(active));
            });
            renderRacePowerCurve();
        });
    });
    document.querySelectorAll('[data-race-segment-unit]').forEach(button => {
        button.addEventListener('click', () => {
            replaySegmentUnit = button.dataset.raceSegmentUnit;
            document.querySelectorAll('[data-race-segment-unit]').forEach(option => {
                const active = option.dataset.raceSegmentUnit === replaySegmentUnit;
                option.classList.toggle('active', active);
                option.setAttribute('aria-pressed', String(active));
            });
            renderRaceSegmentTable();
        });
    });
    document.getElementById('race-segment-select').addEventListener('change', renderRaceSegmentTable);
    document.getElementById('race-comparison-retry').addEventListener('click', () => {
        initRaceComparisons(replayComparisonRace);
    });
    document.querySelectorAll('[data-race-segment-sort]').forEach(header => {
        header.querySelector('button').addEventListener('click', () => {
            const field = header.dataset.raceSegmentSort;
            replaySegmentAscending = replaySegmentSort === field ? !replaySegmentAscending : true;
            replaySegmentSort = field;
            renderRaceSegmentTable();
        });
    });
});

function replayOccurrenceLabel(segment) {
    return `${segment.name} - pass ${segment.pass}, lap ${segment.lap}`;
}

async function initRaceComparisons(race) {
    const request = ++replayComparisonRequest;
    replayComparisonRace = race;
    replayComparisons = null;
    replaySegmentSort = 'total_time_sec';
    replaySegmentAscending = true;
    replayPowerDurationIndex = null;
    document.getElementById('race-power-inspect').hidden = true;
    document.getElementById('race-power-rows').replaceChildren();
    document.getElementById('race-power-panel').style.display = 'block';
    document.getElementById('race-segment-panel').style.display = 'block';
    const status = document.getElementById('race-comparison-status');
    status.textContent = 'Calculating rider power curves and segment times...';
    status.className = 'status-msg info';
    document.getElementById('race-comparison-retry').hidden = true;
    document.getElementById('race-comparison-warnings').hidden = true;
    const select = document.getElementById('race-segment-select');
    select.replaceChildren();
    select.disabled = true;
    document.getElementById('race-segment-rows').replaceChildren();
    Plotly.purge('race-power-chart');
    try {
        const response = await fetch(`/api/race/comparisons/${encodeURIComponent(race.race_id)}`);
        const data = await response.json();
        if (request !== replayComparisonRequest) return;
        if (!response.ok) throw new Error(data.error || `Server error: ${response.status}`);
        replayComparisons = data;
        for (const segment of data.segments) {
            const option = document.createElement('option');
            option.value = segment.id;
            option.textContent = replayOccurrenceLabel(segment);
            select.appendChild(option);
        }
        select.disabled = data.segments.length === 0;
        if (data.segments.length) select.value = data.segments[0].id;
        const warnings = document.getElementById('race-comparison-warning-list');
        warnings.replaceChildren();
        for (const warning of data.warnings) {
            const item = document.createElement('li');
            item.textContent = warning;
            warnings.appendChild(item);
        }
        document.getElementById('race-comparison-warnings').hidden = data.warnings.length === 0;
        status.textContent = `${data.riders.length} riders; ${data.segments.length} segment occurrences.`;
        await renderRacePowerCurve();
        if (request !== replayComparisonRequest) return;
        document.getElementById('race-power-chart').on('plotly_click', event => {
            const duration = event.points?.[0]?.x;
            const index = replayComparisons.durations_sec.indexOf(duration);
            if (index < 0) return;
            replayPowerDurationIndex = index;
            renderRacePowerTable();
        });
        renderRaceSegmentTable();
    } catch (error) {
        if (request !== replayComparisonRequest) return;
        console.error('Race comparisons failed:', error);
        replayComparisons = null;
        status.textContent = `Could not load race comparisons: ${error.message}`;
        status.className = 'status-msg error';
        document.getElementById('race-comparison-retry').hidden = false;
    }
}

function replayPowerValue(value, rider, unit = replayCurveUnit) {
    if (value === null || value === undefined) return null;
    if (unit === 'W') return value;
    return Number.isFinite(rider.weight_kg) && rider.weight_kg > 0 ? value / rider.weight_kg : null;
}

function renderRacePowerCurve() {
    if (!replayComparisons) return;
    const unit = replayCurveUnit;
    const riders = new Map(replayComparisonRace.riders.map(rider => [rider.rank, rider]));
    const traces = replayComparisons.riders.map((row, index) => {
        const rider = riders.get(row.rank);
        return {
            x: replayComparisons.durations_sec,
            y: row.peak_watts.map(value => replayPowerValue(value, rider)),
            text: replayComparisons.durations_label,
            name: rider.name,
            type: 'scatter',
            mode: 'lines',
            connectgaps: false,
            line: { color: `hsl(${index * 137.508 % 360}, 70%, 65%)`, width: 2 },
            hovertemplate: `%{text}: %{y:.2f} ${unit}<extra>%{fullData.name}</extra>`,
        };
    });
    const ticks = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600, 7200];
    renderRacePowerTable();
    return Plotly.react('race-power-chart', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#aaa', size: 11 },
        margin: { l: 65, r: 20, t: 10, b: 55 },
        xaxis: {
            type: 'log', title: 'Duration', tickvals: ticks,
            ticktext: ['1s', '2s', '5s', '10s', '30s', '1m', '2m', '5m', '10m', '20m', '1h', '2h'],
            gridcolor: 'rgba(255,255,255,0.08)', zeroline: false,
        },
        yaxis: { title: unit, gridcolor: 'rgba(255,255,255,0.08)', zeroline: false },
        uirevision: `${replayComparisonRace.race_id}:${unit}`,
    }, { responsive: true, displaylogo: false });
}

function renderRacePowerTable() {
    if (!replayComparisons || replayPowerDurationIndex === null) return;
    const index = replayPowerDurationIndex;
    const riders = new Map(replayComparisonRace.riders.map(rider => [rider.rank, rider]));
    const rows = replayComparisons.riders.map(result => {
        const rider = riders.get(result.rank);
        return { name: rider.name, power: replayPowerValue(result.peak_watts[index], rider) };
    }).sort((a, b) => {
        if (a.power === null) return b.power === null ? a.name.localeCompare(b.name) : 1;
        if (b.power === null) return -1;
        return b.power - a.power || a.name.localeCompare(b.name);
    });
    document.getElementById('race-power-duration').textContent =
        `Power at ${replayComparisons.durations_label[index]}`;
    document.getElementById('race-power-value-heading').textContent = `Power (${replayCurveUnit})`;
    const body = document.getElementById('race-power-rows');
    body.replaceChildren();
    for (const [position, row] of rows.entries()) {
        const tr = document.createElement('tr');
        const number = document.createElement('td');
        number.textContent = String(position + 1);
        const name = document.createElement('td');
        name.textContent = row.name;
        const power = document.createElement('td');
        power.textContent = row.power === null ? '\u2014'
            : row.power.toFixed(replayCurveUnit === 'W' ? 1 : 2);
        tr.appendChild(number);
        tr.appendChild(name);
        tr.appendChild(power);
        body.appendChild(tr);
    }
    document.getElementById('race-power-inspect').hidden = false;
}

function replaySegmentRows() {
    if (!replayComparisons) return [];
    const occurrence = document.getElementById('race-segment-select').value;
    const riders = new Map(replayComparisonRace.riders.map(rider => [rider.rank, rider]));
    const rows = replayComparisons.riders.flatMap(result => result.traversals
        .filter(row => row.occurrence_id === occurrence)
        .map(row => {
            const rider = riders.get(result.rank);
            return {
                ...row, name: rider.name,
                avg_power: replayPowerValue(row.avg_power, rider, replaySegmentUnit),
                normalized_power: replayPowerValue(row.normalized_power, rider, replaySegmentUnit),
                avg_draft_watts: replayPowerValue(row.avg_draft_watts, rider, replaySegmentUnit),
            };
        }));
    const field = replaySegmentSort;
    return rows.sort((a, b) => {
        const av = a[field], bv = b[field];
        if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
        if (bv === null || bv === undefined) return -1;
        const comparison = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
        return comparison * (replaySegmentAscending ? 1 : -1) || a.name.localeCompare(b.name);
    });
}

function renderRaceSegmentTable() {
    if (!replayComparisons) return;
    const body = document.getElementById('race-segment-rows');
    body.replaceChildren();
    document.querySelectorAll('[data-race-segment-sort]').forEach(header => {
        header.setAttribute('aria-sort', header.dataset.raceSegmentSort === replaySegmentSort
            ? (replaySegmentAscending ? 'ascending' : 'descending') : 'none');
    });
    const metric = (value, unit) => value === null || value === undefined
        ? '\u2014' : `${value.toFixed(unit === 'W/kg' ? 2 : 1)} ${unit}`;
    for (const [index, row] of replaySegmentRows().entries()) {
        const tr = document.createElement('tr');
        const cell = text => {
            const td = document.createElement('td');
            td.textContent = text;
            tr.appendChild(td);
            return td;
        };
        cell(String(index + 1));
        cell(row.name);
        const centiseconds = Math.round(row.total_time_sec * 100);
        cell(`${Math.floor(centiseconds / 6000)}:${((centiseconds % 6000) / 100).toFixed(2).padStart(5, '0')}`);
        for (const [field, unit] of [
            ['distance_m', 'm'], ['elevation_change_m', 'm'], ['avg_power', replaySegmentUnit],
            ['normalized_power', replaySegmentUnit], ['avg_draft_watts', replaySegmentUnit],
            ['initial_speed_kmh', 'km/h'],
        ]) cell(metric(row[field], unit));
        body.appendChild(tr);
    }
    if (!body.children.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 9;
        cell.textContent = replayComparisons.segments.length
            ? 'No complete recorded traversals for this segment pass.'
            : 'No complete timed segments are available for this race route.';
        row.appendChild(cell);
        body.appendChild(row);
    }
}
