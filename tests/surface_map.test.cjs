const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function render(escalators) {
    let result;
    const listeners = {};
    const chart = { on(name, handler) { listeners[name] = handler; } };
    const context = vm.createContext({
        document: {
            addEventListener() {},
            getElementById() { return chart; },
        },
        Plotly: {
            react(element, traces, layout) { result = { traces, layout }; },
            relayout(element, update) { result.update = update; },
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/surface_map.js'), 'utf8'), context);
    context.paint = () => {};
    const leg = { d: [0, 100, 200], x: [0, 1, 2], y: [0, 0, 0],
        alt: [-10, 100, 20], surface: ['Tarmac', 'Tarmac', 'Tarmac'] };
    const data = { route: leg, leadin_distance_m: 50, escalators,
        segments: [{ name: 'Sprint', type: 'sprint', start_distance_m: 200, end_distance_m: 240 }] };
    const before = JSON.stringify(data);
    context.buildCombinedPath(data);
    context.renderElevation(data);
    assert.equal(JSON.stringify(data), before);
    return { result, listeners };
}

test('escalator intervals and hover details render without changing elevation', () => {
    const { result, listeners } = render([
        { name: 'SubwayB_Forward', start_distance_m: 80, end_distance_m: 120,
            bonus_watts: 15, trainer_slope_scale: 0.2000000030 },
        { name: 'SubwayC_Forward', start_distance_m: 180, end_distance_m: 220,
            bonus_watts: 15, trainer_slope_scale: 0.2 },
    ]);
    assert.deepEqual(Array.from(result.traces[0].y), [-10, 100, 20]);
    const markers = result.traces.filter(trace => trace.name === 'Escalator');
    assert.equal(markers.length, 2);
    assert.deepEqual(Array.from(markers[0].x), [0.08, 0.12]);
    assert.equal(markers[0].yaxis, 'y2');
    assert.equal(markers[0].showlegend, true);
    assert.equal(markers[1].showlegend, false);
    assert.match(markers[0].hovertemplate, /Bonus Watts parameter: 15/);
    assert.match(markers[0].hovertemplate, /Trainer slope: 20%/);
    assert.equal(result.layout.shapes.filter(shape => shape.fillcolor === '#f4bd50').length, 2);
    // Clicking elevation must retain both sprint and escalator shading.
    listeners.plotly_click({ points: [{ customdata: 0 }] });
    assert.equal(result.update.shapes.filter(shape => shape.type === 'rect').length, 3);
});

test('profiles without escalator metadata retain existing traces', () => {
    const { result } = render(undefined);
    assert.equal(result.traces.length, 1);
    assert.equal(result.layout.shapes.filter(shape => shape.type === 'rect').length, 1);
});
