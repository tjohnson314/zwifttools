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