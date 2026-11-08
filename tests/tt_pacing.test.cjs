const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function plannerContext() {
    const legend = { children: [], replaceChildren() { this.children = []; },
        appendChild(child) { this.children.push(child); } };
    const configs = [];
    const context = vm.createContext({
        window: { addEventListener() {} },
        document: {
            addEventListener() {},
            getElementById(id) { return id === 'surfaceLegend' ? legend : { getContext() {} }; },
            createElement() { return { style: {}, children: [], append(...children) { this.children.push(...children); } }; },
            createTextNode(text) { return { textContent: text }; },
        },
        Chart: class {
            constructor(ctx, config) { configs.push(config); }
            destroy() {}
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/tt_pacing.js'), 'utf8'), context);
    return { context, configs, legend };
}

function profile() {
    return { distance_km: [0, 1, 2, 3], altitude_m: [-10, 20, 15, 30],
        power_w: [250, 300, 200, 260], speed_kph: [35, 32, 40, 34],
        gradient_pct: [0, 2, -1, 1], surface_type: ['Tarmac', 'Dirt', 'Cobbles', 'Dirt'] };
}

test('TT elevation uses TTT surface colors and a deduplicated legend without mutating data', () => {
    const { context, configs, legend } = plannerContext();
    const raw = profile();
    const before = JSON.stringify(raw);
    context.renderChart(raw);
    const config = configs[0];
    const elevation = config.data.datasets[1];
    assert.deepEqual(Array.from(elevation.data), raw.altitude_m);
    assert.equal(elevation.segment.borderColor({ p0DataIndex: 0 }), '#78b4ff');
    assert.equal(elevation.segment.borderColor({ p0DataIndex: 1 }), '#b79562');
    assert.equal(elevation.segment.borderColor({ p0DataIndex: 2 }), '#d4a373');
    assert.equal(elevation.segment.backgroundColor({ p0DataIndex: 1 }), '#b7956226');
    assert.deepEqual(legend.children.map(item => item.children[1].textContent), ['Tarmac', 'Dirt', 'Cobbles']);
    assert.equal(config.options.plugins.tooltip.callbacks.afterLabel(
        { dataset: elevation, dataIndex: 1 }), 'Surface: Dirt');
    assert.equal(JSON.stringify(raw), before);
});

test('TT imperial elevation retains surface colors and metadata', () => {
    const { context, configs } = plannerContext();
    vm.runInContext('currentUnit = "imperial";', context);
    context.renderChart(profile());
    const elevation = configs[0].data.datasets[1];
    assert.equal(elevation.label, 'Elevation (ft)');
    assert.deepEqual(Array.from(elevation.data), [-33, 66, 49, 98]);
    assert.equal(elevation.segment.borderColor({ p0DataIndex: 1 }), '#b79562');
    assert.equal(configs[0].options.plugins.tooltip.callbacks.afterLabel(
        { dataset: elevation, dataIndex: 2 }), 'Surface: Cobbles');
});

test('TT missing surface metadata is shown as unknown instead of invented tarmac', () => {
    const { context, configs, legend } = plannerContext();
    const raw = profile();
    delete raw.surface_type;
    context.renderChart(raw);
    const elevation = configs[0].data.datasets[1];
    assert.equal(elevation.segment.borderColor({ p0DataIndex: 0 }), '#9e9e9e');
    assert.equal(configs[0].options.plugins.tooltip.callbacks.afterLabel(
        { dataset: elevation, dataIndex: 0 }), 'Surface: Unknown');
    assert.deepEqual(legend.children.map(item => item.children[1].textContent), ['Unknown']);
});
function bucketedPlan() {
    // 10 m/s everywhere; one divider at 1 km between a 300 W and a 250 W bucket.
    return {
        bucketed: true,
        dividers_km: [1.0],
        sections: [{ power_w: 300 }, { power_w: 250 }],
        profile: { distance_km: [0.25, 0.75, 1.25, 1.75], speed_kph: [36, 36, 36, 36],
            power_w: [300, 300, 250, 250] },
    };
}

function csvRows(csv) {
    return csv.split('\n').slice(3).map(line => line.split(','));
}

test('Sauce plan export includes weight, target power and speed columns', () => {
    const { context } = plannerContext();
    const csv = context.buildSaucePlanCsv(bucketedPlan(), {
        courseId: 6, routeId: 123, name: 'Big, Loop', date: '2026-11-03', weightKg: 72.44,
        gpsAt: d => [d / 1000, -d / 1000],
    });
    const lines = csv.split('\n');
    assert.equal(lines[0], 'course_id,route_id,name,date,weight_kg');
    assert.equal(lines[1], '6,123,Big  Loop,2026-11-03,72.4');
    assert.equal(lines[2], 'time_sec,distance_m,lat,lng,power_watts,speed_kmh,bucket');
    assert.deepEqual(csvRows(csv)[0], ['0.000', '0.000', '', '', '300', '36.00', '1']);
    assert.deepEqual(csvRows(csv)[1], ['25.000', '250.000', '0.250000', '-0.250000', '300', '36.00', '1']);
});

test('Sauce plan export adds a row at each bucket divider carrying the next bucket power', () => {
    const { context } = plannerContext();
    const rows = csvRows(context.buildSaucePlanCsv(bucketedPlan()));
    assert.deepEqual(rows.map(r => r[1]), ['0.000', '250.000', '750.000', '1000.000', '1250.000', '1750.000']);
    assert.deepEqual(rows.map(r => r[0]), ['0.000', '25.000', '75.000', '100.000', '125.000', '175.000']);
    assert.deepEqual(rows.map(r => r[4]), ['300', '300', '300', '250', '250', '250']);
    assert.deepEqual(rows.map(r => r[6]), ['1', '1', '1', '2', '2', '2']);
});

test('Sauce plan export numbers a single-bucket plan as bucket 1', () => {
    const { context } = plannerContext();
    const plan = { bucketed: true, dividers_km: [], sections: [{ power_w: 270 }],
        profile: { distance_km: [0.5, 1.0], speed_kph: [36, 36], power_w: [270, 270] } };
    const rows = csvRows(context.buildSaucePlanCsv(plan));
    assert.deepEqual(rows.map(r => r[6]), ['1', '1', '1']);
    assert.deepEqual(rows.map(r => r[4]), ['270', '270', '270']);
});

test('Sauce plan export uses profile power unchanged for an unbucketed plan', () => {
    const { context } = plannerContext();
    const plan = { bucketed: false, dividers_km: [], sections: [{ power_w: 999 }],
        profile: { distance_km: [0.5, 1.0], speed_kph: [36, 18], power_w: [280, 320] } };
    const csv = context.buildSaucePlanCsv(plan, { weightKg: null });
    assert.equal(csv.split('\n')[1].split(',')[4], '');
    const rows = csvRows(csv);
    assert.deepEqual(rows.map(r => r[4]), ['280', '280', '320']);
    assert.deepEqual(rows.map(r => r[0]), ['0.000', '50.000', '150.000']);
    assert.deepEqual(rows.map(r => r[6]), ['', '', '']);
});

test('Sauce plan export returns null without a usable profile', () => {
    const { context } = plannerContext();
    assert.equal(context.buildSaucePlanCsv({ profile: { distance_km: [1], speed_kph: [30] } }), null);
});
