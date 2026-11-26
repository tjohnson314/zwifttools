const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay_comparisons.js'), 'utf8');

function setup(fetch) {
    const elements = new Map();
    const plots = [];
    const purges = [];
    function makeUnitButtons(key) {
        return ['W', 'W/kg'].map(unit => {
            const button = element();
            button.dataset = { [key]: unit };
            button.classList = {
                toggle(name, active) { button.active = active; },
            };
            return button;
        });
    }
    const unitButtons = makeUnitButtons('racePowerUnit');
    const segmentUnitButtons = makeUnitButtons('raceSegmentUnit');
    const headers = ['name', 'total_time_sec', 'avg_draft_watts', 'avg_power', 'normalized_power'].map(field => {
        const header = element();
        header.dataset = { raceSegmentSort: field };
        header.button = element();
        header.querySelector = () => header.button;
        return header;
    });
    function element() {
        return {
            children: [], style: {}, value: '', hidden: false, disabled: false,
            textContent: '', events: {}, attributes: {},
            appendChild(child) { this.children.push(child); return child; },
            replaceChildren() { this.children = []; },
            addEventListener(name, callback) { this.events[name] = callback; },
            on(name, callback) { this.events[name] = callback; },
            setAttribute(name, value) { this.attributes[name] = value; },
        };
    }
    let ready;
    const context = vm.createContext({
        document: {
            addEventListener(name, callback) { ready = callback; },
            querySelectorAll: selector => {
                if (selector === '[data-race-power-unit]') return unitButtons;
                if (selector === '[data-race-segment-unit]') return segmentUnitButtons;
                return headers;
            },
            createElement: element,
            createTextNode: text => ({ textContent: text }),
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, element());
                return elements.get(id);
            },
        },
        Plotly: {
            react: (...args) => { plots.push(args); return Promise.resolve(); },
            purge: id => purges.push(id),
        },
        fetch, console: { error() {} },
    });
    vm.runInContext(source, context);
    ready();
    const get = id => context.document.getElementById(id);
    return { context, get, plots, purges, headers, unitButtons, segmentUnitButtons };
}

function fixture() {
    return {
        race: {
            race_id: 'race_data_123', race_name: 'Test Race',
            riders: [
                { rank: 1, name: '<Heavy>', weight_kg: 100, activity_id: '9007199254740993' },
                { rank: 2, name: 'Light', weight_kg: 50, activity_id: '2' },
                { rank: 3, name: 'Unknown', weight_kg: null, activity_id: '3' },
            ],
        },
        data: {
            durations_sec: [1, 5, 7200], durations_label: ['1s', '5s', '120m'],
            segments: [
                { id: '0', name: 'Sprint', pass: 1, lap: 1 },
                { id: '1', name: 'Sprint', pass: 2, lap: 1 },
                { id: '2', name: 'Sprint', pass: 1, lap: 2 },
            ],
            riders: [1, 2, 3].map(rank => ({
                rank, activity_start_time: '2026-11-25T12:00:00Z',
                peak_watts: rank === 1 ? [3000, 1500, null] : rank === 2 ? [-50, -100, null] : [null, null, null],
                traversals: [{
                    occurrence_id: '0', segment_name: 'Sprint', total_time_sec: 10 + rank,
                    start_time_sec: 100, end_time_sec: 110 + rank,
                    distance_m: 200, elevation_change_m: -10, avg_power: 1500,
                    normalized_power: 1500, avg_draft_watts: rank === 3 ? null : -50 * rank,
                    initial_speed_kmh: 100,
                }, ...(rank === 2 ? [{
                    occurrence_id: '1', segment_name: 'Sprint', total_time_sec: 15,
                    distance_m: 200, elevation_change_m: null, avg_power: -10,
                    normalized_power: null, avg_draft_watts: null, initial_speed_kmh: 30,
                }] : [])],
            })),
            warnings: ['Unknown: no recorded race power samples.'],
        },
    };
}

test('power graph includes every rider, preserves raw values, and switches units without inventing weights', async () => {
    const { race, data } = fixture();
    const { context, get, plots, unitButtons } = setup(async url => {
        assert.equal(url, '/api/race/comparisons/race_data_123');
        return { ok: true, json: async () => data };
    });
    await context.initRaceComparisons(race);
    const [id, traces, layout] = plots.at(-1);
    assert.equal(id, 'race-power-chart');
    assert.equal(traces.length, 3);
    assert.deepEqual(Array.from(traces[0].x), data.durations_sec);
    assert.deepEqual(Array.from(traces[0].y), [3000, 1500, null]);
    assert.deepEqual(Array.from(traces[1].y), [-50, -100, null]);
    assert.equal(layout.xaxis.type, 'log');
    assert.equal(layout.yaxis.range, undefined);
    unitButtons[1].events.click();
    const converted = plots.at(-1)[1];
    assert.deepEqual(Array.from(converted[0].y), [30, 15, null]);
    assert.deepEqual(Array.from(converted[1].y), [-1, -2, null]);
    assert.deepEqual(Array.from(converted[2].y), [null, null, null]);
    assert.equal(unitButtons[1].attributes['aria-pressed'], 'true');
    assert.equal(unitButtons[0].attributes['aria-pressed'], 'false');
    assert.equal(unitButtons[1].active, true);
    unitButtons[0].events.click();
    assert.deepEqual(Array.from(plots.at(-1)[1][0].y), [3000, 1500, null]);
    assert.equal(get('race-comparison-warnings').hidden, false);
    assert.equal(get('race-comparison-warning-list').children[0].textContent, data.warnings[0]);
});

test('segment dropdown distinguishes passes and laps without redundant table columns or action links', async () => {
    const { race, data } = fixture();
    const { context, get, headers } = setup(async () => ({ ok: true, json: async () => data }));
    await context.initRaceComparisons(race);
    assert.deepEqual(get('race-segment-select').children.map(option => option.textContent), [
        'Sprint - pass 1, lap 1', 'Sprint - pass 2, lap 1', 'Sprint - pass 1, lap 2',
    ]);
    let cells = get('race-segment-rows').children[0].children;
    assert.equal(cells.length, 9);
    assert.equal(cells[0].textContent, '1');
    assert.equal(cells[1].textContent, '<Heavy>');
    assert.equal(cells[2].textContent, '0:11.00');
    assert.equal(cells[4].textContent, '-10.0 m');
    assert.equal(cells[7].textContent, '-50.0 W');
    assert.equal(cells[8].textContent, '100.0 km/h');
    assert.ok(cells.every(cell => cell.children.every(child => !child.href)));
    headers[2].button.events.click();
    assert.equal(get('race-segment-rows').children[0].children[1].textContent, 'Light');
    assert.deepEqual(get('race-segment-rows').children.map(row => row.children[0].textContent), ['1', '2', '3']);
    headers[2].button.events.click();
    assert.equal(get('race-segment-rows').children[0].children[1].textContent, '<Heavy>');
    assert.equal(get('race-segment-rows').children[2].children[1].textContent, 'Unknown');
    assert.deepEqual(get('race-segment-rows').children.map(row => row.children[0].textContent), ['1', '2', '3']);
    assert.equal(headers[2].attributes['aria-sort'], 'descending');

    get('race-segment-select').value = '1';
    get('race-segment-select').events.change();
    assert.equal(get('race-segment-rows').children.length, 1);
    cells = get('race-segment-rows').children[0].children;
    assert.equal(cells[0].textContent, '1');
    assert.equal(cells[1].textContent, 'Light');
    assert.equal(cells[5].textContent, '-10.0 W');
    assert.equal(cells[6].textContent, '\u2014');
    get('race-segment-select').value = '2';
    get('race-segment-select').events.change();
    assert.match(get('race-segment-rows').children[0].children[0].textContent, /No complete recorded/);
    assert.equal(get('race-segment-rows').children[0].children[0].colSpan, 9);
});

test('segment power units convert power, NP, and draft, sort displayed values, and leave curves unchanged', async () => {
    const { race, data } = fixture();
    data.riders[0].traversals[0].avg_power = 2000;
    data.riders[0].traversals[0].normalized_power = 2100;
    const { context, get, headers, plots, segmentUnitButtons } =
        setup(async () => ({ ok: true, json: async () => data }));
    await context.initRaceComparisons(race);
    const powerHeader = headers.find(header => header.dataset.raceSegmentSort === 'avg_power');
    powerHeader.button.events.click();
    powerHeader.button.events.click();
    assert.equal(get('race-segment-rows').children[0].children[1].textContent, '<Heavy>');
    const plotCount = plots.length;
    segmentUnitButtons[1].events.click();
    const rows = get('race-segment-rows').children;
    assert.deepEqual(rows.map(row => row.children[0].textContent), ['1', '2', '3']);
    assert.deepEqual(rows.map(row => row.children[1].textContent), ['Light', '<Heavy>', 'Unknown']);
    assert.deepEqual(rows[1].children.slice(5, 8).map(cell => cell.textContent),
        ['20.00 W/kg', '21.00 W/kg', '-0.50 W/kg']);
    assert.deepEqual(rows[2].children.slice(5, 8).map(cell => cell.textContent),
        ['\u2014', '\u2014', '\u2014']);
    assert.equal(rows[1].children[3].textContent, '200.0 m');
    assert.equal(rows[1].children[8].textContent, '100.0 km/h');
    assert.equal(plots.length, plotCount);
    assert.equal(plots.at(-1)[2].yaxis.title, 'W');
    assert.equal(segmentUnitButtons[1].active, true);
    assert.equal(segmentUnitButtons[1].attributes['aria-pressed'], 'true');
    assert.equal(segmentUnitButtons[0].attributes['aria-pressed'], 'false');
    assert.equal(powerHeader.attributes['aria-sort'], 'descending');
    assert.equal(get('race-segment-select').value, '0');

    segmentUnitButtons[0].events.click();
    assert.equal(get('race-segment-rows').children[0].children[5].textContent, '2000.0 W');
    segmentUnitButtons[1].events.click();
    get('race-segment-select').value = '1';
    get('race-segment-select').events.change();
    const cells = get('race-segment-rows').children[0].children;
    assert.equal(cells[5].textContent, '-0.20 W/kg');
    assert.equal(cells[6].textContent, '\u2014');
    assert.equal(cells[7].textContent, '\u2014');
});

test('clicking a power curve lists every rider at that duration, preserving negative and missing power', async () => {
    const { race, data } = fixture();
    const { context, get } = setup(async () => ({ ok: true, json: async () => data }));
    await context.initRaceComparisons(race);
    assert.equal(get('race-power-inspect').hidden, true);
    get('race-power-chart').events.plotly_click({ points: [{ x: 5 }] });
    assert.equal(get('race-power-inspect').hidden, false);
    assert.equal(get('race-power-duration').textContent, 'Power at 5s');
    assert.equal(get('race-power-value-heading').textContent, 'Power (W)');
    assert.deepEqual(get('race-power-rows').children.map(row => row.children.map(cell => cell.textContent)), [
        ['1', '<Heavy>', '1500.0'], ['2', 'Light', '-100.0'], ['3', 'Unknown', '\u2014'],
    ]);
    get('race-power-chart').events.plotly_click({ points: [{ x: 1 }] });
    assert.equal(get('race-power-duration').textContent, 'Power at 1s');
    assert.equal(get('race-power-rows').children[0].children[0].textContent, '1');
    assert.equal(get('race-power-rows').children[0].children[2].textContent, '3000.0');
});

test('selected duration table follows unit changes and reorders riders by displayed power', async () => {
    const { race, data } = fixture();
    data.riders[1].peak_watts[0] = 2000;
    const { context, get, unitButtons } = setup(async () => ({ ok: true, json: async () => data }));
    await context.initRaceComparisons(race);
    get('race-power-chart').events.plotly_click({ points: [{ x: 1 }] });
    unitButtons[1].events.click();
    assert.equal(get('race-power-duration').textContent, 'Power at 1s');
    assert.equal(get('race-power-value-heading').textContent, 'Power (W/kg)');
    assert.deepEqual(get('race-power-rows').children.map(row => row.children.map(cell => cell.textContent)), [
        ['1', 'Light', '40.00'], ['2', '<Heavy>', '30.00'], ['3', 'Unknown', '\u2014'],
    ]);
    unitButtons[0].events.click();
    assert.deepEqual(get('race-power-rows').children.map(row => row.children.map(cell => cell.textContent)), [
        ['1', '<Heavy>', '3000.0'], ['2', 'Light', '2000.0'], ['3', 'Unknown', '\u2014'],
    ]);
    await context.initRaceComparisons(race);
    assert.equal(get('race-power-inspect').hidden, true);
    assert.equal(get('race-power-rows').children.length, 0);
});

test('loading a new race clears stale comparisons and reports errors with retry', async () => {
    const { race, data } = fixture();
    let fail = false;
    const { context, get, purges } = setup(async () => fail
        ? { ok: false, status: 422, json: async () => ({ error: 'Invalid segment geometry' }) }
        : { ok: true, json: async () => data });
    await context.initRaceComparisons(race);
    fail = true;
    await context.initRaceComparisons(race);
    assert.match(get('race-comparison-status').textContent, /Invalid segment geometry/);
    assert.equal(get('race-comparison-retry').hidden, false);
    assert.equal(get('race-segment-select').disabled, true);
    assert.equal(get('race-segment-rows').children.length, 0);
    assert.equal(purges.length, 2);
    fail = false;
    await get('race-comparison-retry').events.click();
    // The event starts an asynchronous retry; let its response render.
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(get('race-comparison-retry').hidden, true);
    assert.equal(get('race-segment-rows').children.length, 3);
});

test('segment durations roll over minutes and rounded centiseconds correctly', async () => {
    const { race, data } = fixture();
    const { context, get } = setup(async () => ({ ok: true, json: async () => data }));
    for (const [seconds, expected] of [[69.53, '1:09.53'], [59.995, '1:00.00'], [3601.23, '60:01.23']]) {
        data.riders[0].traversals[0].total_time_sec = seconds;
        await context.initRaceComparisons(race);
        const row = get('race-segment-rows').children.find(row => row.children[1].textContent === '<Heavy>');
        assert.equal(row.children[2].textContent, expected);
    }
});

test('an earlier race response cannot overwrite a newer race', async () => {
    const { race, data } = fixture();
    const pending = [];
    const { context, get } = setup(() => new Promise(resolve => pending.push(resolve)));
    const first = context.initRaceComparisons(race);
    const second = context.initRaceComparisons({ ...race, race_id: 'race_data_456' });
    pending[1]({ ok: true, json: async () => ({ ...data, warnings: ['Newest race'] }) });
    await second;
    pending[0]({ ok: false, status: 500, json: async () => ({ error: 'Old race failure' }) });
    await first;
    assert.equal(get('race-comparison-warning-list').children[0].textContent, 'Newest race');
    assert.equal(get('race-comparison-status').className, 'status-msg info');
});
