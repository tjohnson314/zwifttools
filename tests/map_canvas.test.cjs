const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function load(context, name) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js', name), 'utf8'), context);
}

test('road widths are thin at world fit and grow gently within readable limits', () => {
    const context = vm.createContext({});
    load(context, 'map_canvas.js');
    const widths = [0.01, 0.25, 1, 2, 4, 64].map(zoom =>
        context.mapStrokeWidth(1.5, zoom, 1));
    assert.deepEqual(widths, [0.75, 0.75, 1.5, 1.5 * Math.sqrt(2), 3, 3]);
    for (const zoom of [0.01, 0.25, 1, 2, 4, 64]) {
        assert.equal(context.mapStrokeWidth(1.5, zoom, 2),
            2 * context.mapStrokeWidth(1.5, zoom, 1));
    }
});

test('sprint and KOM overlays stay narrower than the selected route at close zoom', () => {
    const strokes = [];
    const radii = [];
    const context = vm.createContext({
        document: { addEventListener() {} },
        g: {
            beginPath() {}, moveTo() {}, lineTo() {}, setLineDash() {},
            fill() {}, strokeText() {}, fillText() {},
            arc(x, y, radius) { radii.push(radius); },
            stroke() { strokes.push(this.lineWidth); },
        },
    });
    load(context, 'map_canvas.js');
    load(context, 'surface_map.js');
    for (const dpr of [1, 2]) {
        for (const zoom of [1, 4, 64]) {
            strokes.length = radii.length = 0;
            vm.runInContext(`
                state.dpr = ${dpr};
                drawRouteSegments(g, [{
                    name: 'Sprint', type: 'sprint',
                    path: [{ x: 0, y: 0 }, { x: 100, y: 100 }],
                }], ${zoom});
            `, context);
            assert.ok(strokes[0] < context.mapStrokeWidth(3.5, zoom, dpr));
            assert.ok(strokes[0] / dpr <= 4);
            assert.ok(radii.every(radius => radius / dpr >= 4 && radius / dpr <= 8));
        }
    }
});

for (const script of ['surface_map.js', 'surface_map_dev.js']) {
    test(`${script}: rendered roads and selections scale with zoom, DPI and resize`, () => {
        const strokes = [];
        const g = {
            clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, setLineDash() {},
            stroke() { strokes.push({ width: this.lineWidth, color: this.strokeStyle }); },
        };
        const context = vm.createContext({
            document: { addEventListener() {} },
            requestAnimationFrame() {},
            g,
        });
        load(context, 'map_canvas.js');
        load(context, script);
        vm.runInContext(`
            canvas = { width: 1000, height: 1000 };
            baseCanvas = canvas;
            baseCtx = g;
            const leg = { x: [0, 100], y: [0, 100] };
            state.world = {
                bounds: { min_x: 0, max_x: 100, min_y: 0, max_y: 100 },
                colors: { Tarmac: '#888' },
                segments: [{ ...leg, surface: 'Tarmac' }],
                roads: [{ ...leg, id: 1 }],
            };
            state.route = { leadin: leg, route: leg };
            state.road = { full: leg };
            state.highlight = { ...leg, color: '#00f' };
        `, context);

        function render(zoom, dpr = 1, size = 1000) {
            strokes.length = 0;
            vm.runInContext(`
                state.dpr = ${dpr};
                canvas.width = canvas.height = ${size * dpr};
                state.scale = ${size * dpr / 100 * 0.94 * zoom};
                renderBase();
            `, context);
            return strokes.map(stroke => stroke.width / dpr);
        }

        for (const zoom of [0.25, 1, 4, 64]) {
            const normal = render(zoom);
            assert.equal(normal[0], 1.5 * Math.max(0.5, Math.min(2, Math.sqrt(zoom))));
            assert.deepEqual(render(zoom, 2), normal);
            assert.deepEqual(render(zoom, 1, 600), normal);
            assert.deepEqual(render(zoom, 2, 600), normal);
            assert.ok(normal[2] > normal[0], 'selected path stands out from network');
            assert.ok(normal[2] <= 7, 'selected path does not grow without bound');
            assert.equal(normal[1] - normal[2], 2, 'casing stays two CSS pixels wider');
        }

        vm.runInContext('state.showNetwork = false;', context);
        render(1);
        assert.equal(strokes.length, 4, 'hiding network preserves selections');
        vm.runInContext('state.route = null; state.road = null; state.highlight = null;', context);
        render(1);
        assert.equal(strokes.length, 0);
    });
}
