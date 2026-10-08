const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function plannerContext() {
    const storage = {};
    const context = vm.createContext({
        document: {
            addEventListener() {},
            getElementById() { return { value: '', checked: false }; },
        },
        localStorage: { setItem(key, value) { storage[key] = value; } },
    });
    const source = fs.readFileSync(path.join(__dirname, '../static/js/ttt_pacing.js'), 'utf8');
    vm.runInContext(source, context);
    return { context, storage };
}

function profile() {
    return {
        distance_km: [0, 1, 2, 3, 4],
        power_w: { A: [100, null, 0, -50, 5000], B: [200, null, 0, -100, 10000] },
        peak_power_w: { A: [150], B: [300] },
        braking_w: { A: [25], B: [50] },
    };
}

test('TTT power defaults to watts without modifying the profile', () => {
    const { context } = plannerContext();
    const raw = profile();
    const before = JSON.stringify(raw);
    const config = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    assert.deepEqual(Array.from(config.data.datasets[0].data), raw.power_w.A);
    assert.equal(config.options.scales.y.title.text, 'Power (W)');
    assert.equal(JSON.stringify(raw), before);
});

test('TTT watt chart and tooltips round to nearest watts without altering raw data', () => {
    const { context } = plannerContext();
    const raw = profile();
    raw.power_w.A = [100.4, null, 0, -50.6, 5000.8];
    raw.power_w.B = [100.6, null, 0, -100.4, 10000.2];
    raw.peak_power_w.A = [150.6];
    raw.braking_w.A = [0.4];
    const before = JSON.stringify(raw);
    const config = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    assert.deepEqual(Array.from(config.data.datasets[0].data), [100, null, 0, -51, 5001]);
    assert.deepEqual(Array.from(config.data.datasets[1].data), [101, null, 0, -100, 10000]);
    const tooltip = { dataset: { label: 'A' }, dataIndex: 0, parsed: { y: 100.4 } };
    assert.equal(config.options.plugins.tooltip.callbacks.label(tooltip), 'A: 100 W');
    assert.deepEqual(Array.from(config.options.plugins.tooltip.callbacks.afterLabel(tooltip)),
        ['Peak: 151 W', 'Speed-matching dissipation: 0 W']);
    vm.runInContext('powerUnit = "W/kg";', context);
    const perKg = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    assert.equal(perKg.data.datasets[0].data[0], 100.4 / 50);
    tooltip.parsed.y = 100.4 / 50;
    assert.equal(perKg.options.plugins.tooltip.callbacks.label(tooltip), 'A: 2.01 W/kg');
    assert.equal(JSON.stringify(raw), before);
});

test('TTT pull table rounds watts while keeping raw schedule and CSV precision', () => {
    const { context } = plannerContext();
    const tbody = { children: [], appendChild(child) { this.children.push(child); } };
    context.document.querySelector = () => tbody;
    context.document.createElement = () => ({
        children: [], appendChild(child) { this.children.push(child); },
    });
    const pulls = [{ rider: 'A', start_time_s: 0, duration_s: 59.123456,
        power_w: 321.4, power_wkg: 4.285333333, start_km: 0, end_km: 0.7 },
    { rider: 'B', start_time_s: 59.123456, duration_s: 10,
        power_w: 322.6, power_wkg: 4.301333333, start_km: 0.7, end_km: 0.8 }];
    const before = JSON.stringify(pulls);
    context.renderPullSchedule(pulls);
    assert.equal(String(tbody.children[0].children[4].textContent), '321');
    assert.equal(String(tbody.children[1].children[4].textContent), '323');
    assert.equal(tbody.children[0].children[5].textContent, '4.29');
    assert.equal(JSON.stringify(pulls), before);
    assert.ok(context.buildPullsCsv(pulls).includes('"321.4","4.285333333"'));
});

test('TTT W/kg scales each rider and all tooltip powers without clipping or filling gaps', () => {
    const { context } = plannerContext();
    vm.runInContext('powerUnit = "W/kg";', context);
    const raw = profile();
    const config = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    assert.deepEqual(Array.from(config.data.datasets[0].data), [2, null, 0, -1, 100]);
    assert.deepEqual(Array.from(config.data.datasets[1].data), [2, null, 0, -1, 100]);
    assert.equal(config.options.scales.y.title.text, 'Power (W/kg)');
    const tooltip = { dataset: { label: 'A' }, dataIndex: 0, parsed: { y: 2 } };
    assert.equal(config.options.plugins.tooltip.callbacks.label(tooltip), 'A: 2.00 W/kg');
    assert.deepEqual(Array.from(config.options.plugins.tooltip.callbacks.afterLabel(tooltip)),
        ['Peak: 3.00 W/kg', 'Speed-matching dissipation: 0.50 W/kg']);
    assert.deepEqual(raw.power_w.A, [100, null, 0, -50, 5000]);
});

test('TTT W/kg does not invent a weight for unavailable riders', () => {
    const { context } = plannerContext();
    vm.runInContext('powerUnit = "W/kg";', context);
    const config = context.buildPowerChartConfig(profile(), { B: 0 });
    assert.deepEqual(Array.from(config.data.datasets[0].data), [null, null, null, null, null]);
    assert.deepEqual(Array.from(config.data.datasets[1].data), [null, null, null, null, null]);
});

test('TTT unit toggle updates the existing chart, uses plan weights, and saves the selection', () => {
    const { context, storage } = plannerContext();
    const raw = profile();
    const config = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    const updates = [];
    const chart = { data: config.data, options: config.options, update(mode) { updates.push(mode); } };
    chart.data.datasets[0].hidden = true;
    const dataset = chart.data.datasets[0];
    context.chart = chart;
    context.raw = raw;
    vm.runInContext(`
        charts.powerChart = chart;
        powerChartData = { profile: raw, weights: { A: 50, B: 100 } };
        riders = [{ name: 'A', weight_kg: 500 }];
    `, context);
    context.setPowerUnit('W/kg');
    assert.equal(chart.data.datasets[0], dataset);
    assert.equal(dataset.hidden, true);
    assert.deepEqual(Array.from(dataset.data), [2, null, 0, -1, 100]);
    assert.equal(JSON.parse(storage.tttSettings).powerUnit, 'W/kg');
    context.setPowerUnit('W');
    assert.deepEqual(Array.from(dataset.data), raw.power_w.A);
    assert.equal(chart.options.scales.y.title.text, 'Power (W)');
    assert.equal(JSON.parse(storage.tttSettings).powerUnit, 'W');
    assert.deepEqual(updates, ['none', 'none']);
});

test('TTT charts share surface-colored elevation and identical plot margins', () => {
    const { context } = plannerContext();
    context.document.getElementById = () => null;
    const configs = {};
    context.drawChart = (id, config) => { configs[id] = config; };
    const raw = { ...profile(), altitude_m: [5, 10, -5, 30, 20],
        surface_type: ['Tarmac', 'Dirt', 'Cobbles', 'Unknown', 'Grass'],
        crr: [0.004, 0.016, 0.0065, 0.004, 0.025],
        wbal_j: { A: [20000, 18000, 15000, -30, 17000] },
        speed_kph: [35, 33, 31, 40, 35], leader: ['A', 'A', 'A', 'A', 'A'],
        gradient_pct: [0, 1, -1, 2, -1] };
    const before = JSON.stringify(raw);
    context.renderCharts(raw, { A: 50, B: 100 });
    for (const config of Object.values(configs)) {
        const elevation = config.data.datasets.find(dataset => dataset.yAxisID === 'yElev');
        assert.deepEqual(Array.from(elevation.data), raw.altitude_m);
        assert.equal(elevation.segment.borderColor({ p0DataIndex: 1 }), '#b79562');
        assert.equal(elevation.segment.borderColor({ p0DataIndex: 2 }), '#d4a373');
        assert.equal(elevation.segment.borderColor({ p0DataIndex: 3 }), '#9e9e9e');
        const left = { width: 10 };
        const right = { width: 20 };
        config.options.scales.y.afterFit(left);
        config.options.scales.yElev.afterFit(right);
        assert.equal(left.width, 80);
        assert.equal(right.width, 80);
        const tooltip = { dataset: elevation, dataIndex: 1, parsed: { y: 10 } };
        assert.deepEqual(Array.from(config.options.plugins.tooltip.callbacks.afterLabel(tooltip)),
            ['Surface: Dirt', 'Crr: 0.016']);
    }
    assert.deepEqual(Array.from(configs.wbalChart.data.datasets[0].data), [20, 18, 15, -0.03, 17]);
    const power = configs.powerChart;
    const elevation = power.data.datasets.at(-1);
    assert.equal(power.options.plugins.tooltip.callbacks.label(
        { dataset: elevation, parsed: { y: -5 } }), 'Elevation: -5 m');
    vm.runInContext('powerUnit = "W/kg";', context);
    const perKg = context.buildPowerChartConfig(raw, { A: 50, B: 100 });
    assert.deepEqual(Array.from(perKg.data.datasets.at(-1).data), raw.altitude_m);
    assert.equal(JSON.stringify(raw), before);
});

test('TTT lookup preserves known names and identifies saved stats after a failed fetch', async () => {
    const { context } = plannerContext();
    context.document.getElementById = () => ({ value: '111 222' });
    context.fetch = async () => ({ ok: true, status: 200, json: async () => ({ riders: [
        { zwift_id: '111', name: 'Rider 111', height_cm: 175, weight_kg: 68, ftp_w: 290,
            warning: 'Name unavailable from Zwift' },
        { zwift_id: '222', name: 'Rider 222', error: 'Profile lookup failed (404)' },
    ] }) });
    vm.runInContext(`
        riders = [
            { zwift_id: '111', name: 'Ann', cp_w: 300, w_prime_kj: 22 },
            { zwift_id: '222', name: 'Bob', height_cm: 180, weight_kg: 75, cp_w: 280 },
        ];
        renderRiderTable = () => {};
        saveSettings = () => {};
    `, context);
    await context.fetchRiders();
    const rows = vm.runInContext('riders', context);
    assert.equal(rows[0].name, 'Ann');
    assert.equal(rows[0].weight_kg, 68);
    assert.equal(rows[0].cp_w, 300);
    assert.equal(rows[0].error, undefined);
    assert.match(rows[0].warning, /Name unavailable/);
    assert.equal(rows[1].name, 'Bob');
    assert.equal(rows[1].weight_kg, 75);
    assert.equal(rows[1].error, undefined);
    assert.match(rows[1].warning, /saved rider data retained/);
});

test('TTT pull CSV keeps numeric precision and quotes rider names correctly', () => {
    const { context } = plannerContext();
    const pulls = [{ rider: 'Ann, "A"\nTeam', start_time_s: 0, duration_s: 59.123456,
        power_w: 321.123456, power_wkg: 4.28164608, start_km: 0, end_km: 0.7654321 }];
    const csv = context.buildPullsCsv(pulls);
    assert.ok(csv.startsWith('"Pull","Rider","Start (s)","Duration (s)","Power (W)","Power (W/kg)"'));
    assert.ok(csv.includes('"Ann, ""A""\nTeam"'));
    assert.ok(csv.includes('"59.123456","321.123456","4.28164608"'));
    assert.ok(csv.endsWith('"0","0.7654321"\r\n'));
    vm.runInContext('powerUnit = "W/kg";', context);
    assert.equal(context.buildPullsCsv(pulls), csv);
});

test('TTT pull CSV protects spreadsheet formulas in rider names', () => {
    const { context } = plannerContext();
    const csv = context.buildPullsCsv([{ rider: '=1+1', start_time_s: 0, duration_s: 10,
        power_w: 300, power_wkg: 4, start_km: 0, end_km: 0.1 }]);
    assert.ok(csv.includes('"\'=1+1"'));
});

const headingPlan = {
    route_name: 'Flat', feasible: true, total_time_formatted: '1:30',
    total_distance_km: 1, total_ascent_m: 0, avg_speed_kph: 40,
    scoring_rider_count: 3, team_size: 4, flat_rotation: { speed_kph: 40 },
    riders: [], phases: [], profile: {},
};

function planningContext(response) {
    const { context } = plannerContext();
    const elements = new Map();
    const headings = [];
    context.document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, {
                value: '25', checked: false, validity: { valid: true }, style: {},
                classList: { add() {}, remove() {} }, scrollIntoView() {},
            });
            return elements.get(id);
        },
        querySelector() { return { innerHTML: '' }; },
    };
    const heading = context.document.getElementById('resultsRouteName');
    Object.defineProperty(heading, 'textContent', {
        get() { return headings.at(-1); },
        set(value) { headings.push(value); },
    });
    context.fetch = async () => response;
    context.TextDecoder = TextDecoder;
    context.setInterval = () => 1;
    context.clearInterval = () => {};
    context.setTimeout = () => {};
    vm.runInContext(`
        selectedRoute = { id: 'flat', name: 'Flat', world: 'watopia' };
        bikeSelector = { getConfig: () => ({}), isReady: () => true };
        riders = Array.from({ length: 4 }, (_, i) => ({
            name: 'R' + i, height_cm: 180, weight_kg: 75, cp_w: 300, w_prime_kj: 20,
        }));
        renderCharts = () => {};
        renderPullSchedule = () => {};
    `, context);
    return { context, elements, headings };
}

function headingStream(complete) {
    const updates = [{ type: 'progress', iteration: 1, stage: 'Team together',
        step: 'power search', candidate_iteration: 1, candidate_iterations: 1,
        improved: true, plan: headingPlan }];
    if (complete) updates.push({ type: 'complete', plan: headingPlan });
    return new Response(updates.map(update => 'data: ' + JSON.stringify(update) + '\n\n').join(''),
        { headers: { 'Content-Type': 'text/event-stream' } });
}

test('TTT streamed completion removes the provisional heading', async () => {
    const { context, elements, headings } = planningContext(headingStream(true));
    await context.runPlan();
    assert.deepEqual(headings, ['Flat (provisional)', 'Flat']);
    assert.match(elements.get('planHint').textContent, /^Finished in /);
});

test('TTT interrupted optimization keeps its provisional heading', async () => {
    const { context, elements, headings } = planningContext(headingStream(false));
    await context.runPlan();
    assert.deepEqual(headings, ['Flat (provisional)']);
    assert.equal(elements.get('planHint').textContent, 'Stopped - showing best feasible plan so far');
});

test('TTT JSON completion renders a final heading', async () => {
    const response = new Response(JSON.stringify(headingPlan),
        { headers: { 'Content-Type': 'application/json' } });
    const { context, elements, headings } = planningContext(response);
    await context.runPlan();
    assert.deepEqual(headings, ['Flat']);
    assert.match(elements.get('planHint').textContent, /^Finished in /);
});