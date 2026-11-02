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
    assert.doesNotMatch(markers[0].hovertemplate, /bonus watts/i);
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

test('video routes only display their own independent road network', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/surface_map.js'), 'utf8'), context);
    const segments = [{ road_id: 27060 }, { road_id: 27061 }];
    const world = { segments, disconnected_routes: true };
    const route = { road_nodes: [{ road_id: 27060 }] };
    assert.deepEqual(Array.from(context.networkSegments(world, route)), [segments[0]]);
    assert.deepEqual(Array.from(context.networkSegments(world, null)), []);
    assert.equal(context.networkSegments({ segments }, route), segments);
});

test('video GPS projection interpolates the nearest road without changing source coordinates', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/surface_map.js'), 'utf8'), context);
    const route = { road_nodes: [{ nodes: [
        { x: 0, z: 0, lat: 56.63, lng: -4.89 },
        { x: 200, z: 30, lat: 56.62, lng: -4.98 },
    ] }] };
    const before = JSON.stringify(route);
    const project = context.videoGpsProjection(route);
    const point = project(100, -15);
    assert.ok(Math.abs(point.lat - 56.625) < 1e-9);
    assert.ok(Math.abs(point.lng + 4.935) < 1e-9);
    assert.equal(JSON.stringify(route), before);
    assert.equal(context.videoGpsProjection({ road_nodes: [] }), null);
});

test('video GPS projection preserves off-road offsets and native GPS nodes', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/surface_map.js'), 'utf8'), context);
    const project = context.videoGpsProjection({ road_nodes: [{ nodes: [
        { x: 0, z: 0, lat: 0, lng: 0 },
        { x: 100, z: 0, lat: 0, lng: 0.001 },
        { x: 100, z: 100, lat: 0.001, lng: 0.001 },
    ] }] });
    assert.deepEqual({ ...project(100, -100) }, { lat: 0.001, lng: 0.001 });
    const offset = project(-10, -5);
    assert.ok(Math.abs(offset.lat - 0.00005) < 1e-12);
    assert.ok(Math.abs(offset.lng + 0.0001) < 1e-12);
});

function satelliteContext(key = 'test-key') {
    const elements = {
        'satellite-map': { hidden: true },
        status: { textContent: '', classList: { toggle() {} } },
    };
    const scripts = [];
    const maps = {
        Map: class {
            constructor(element, options) { this.options = options; }
            addListener() {}
            fitBounds(bounds, padding) { this.bounds = bounds; this.padding = padding; }
            getZoom() { return 12; }
        },
        LatLngBounds: class {
            constructor() { this.points = []; }
            extend(point) { this.points.push(point); }
            isEmpty() { return this.points.length === 0; }
        },
        Polyline: class {
            constructor(options) { this.options = options; }
            setMap(map) { this.options.map = map; }
        },
        Circle: class {
            setMap(map) { this.map = map; }
            setOptions(options) { this.options = options; this.map = options.map; }
        },
        event: { trigger() {} },
    };
    const context = vm.createContext({
        document: {
            addEventListener() {},
            getElementById(id) { return elements[id]; },
            querySelector() { return { content: key }; },
            createElement() { return { remove() {} }; },
            head: { appendChild(script) { scripts.push(script); } },
        },
        window: { google: { maps } }, URL, URLSearchParams,
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/surface_map.js'), 'utf8'), context);
    vm.runInContext(`
        canvas = { hidden: false };
        state.world = { disconnected_routes: true, colors: { Tarmac: '#888' },
            segments: [{ road_id: 1, x: [0, 100], y: [0, 0], surface: 'Tarmac' }] };
        state.route = { road_nodes: [{ road_id: 1, nodes: [
            { x: 0, z: 0, lat: 56, lng: -4 },
            { x: 100, z: 0, lat: 56, lng: -3.999 },
        ] }], route: { x: [0, 100], y: [0, 0] } };
        renderBase = () => { if (state.satelliteActive) renderSatellite(); };
    `, context);
    return { context, elements, scripts, maps,
        current: () => vm.runInContext('state', context) };
}

test('satellite mode uses GPS for native overlays, fit, markers and network toggle', async () => {
    const { context, elements, current } = satelliteContext();
    const state = current();
    await context.showSatellite(state.route);
    assert.equal(state.satelliteActive, true);
    assert.equal(elements['satellite-map'].hidden, false);
    assert.equal(state.satellite.map.options.mapTypeId, 'satellite');
    assert.equal(state.satellite.lines.length, 3);
    assert.equal(state.satellite.map.bounds.points.length, 2);
    assert.equal(state.satellite.lines[2].options.path[0].lat, 56);
    state.hoverPoint = { x: 50, y: 0 };
    context.paintSatelliteMarkers();
    assert.ok(Math.abs(state.satellite.hover.options.center.lng + 3.9995) < 1e-9);
    assert.ok(state.satellite.hover.options.radius > 0);
    const previous = [...state.satellite.lines];
    state.showNetwork = false;
    context.renderSatellite();
    assert.equal(state.satellite.lines.length, 2);
    assert.ok(previous.every(line => line.options.map === null));
    context.hideSatellite();
    assert.equal(state.satelliteActive, false);
    assert.equal(elements['satellite-map'].hidden, true);
    assert.equal(state.satellite.hover.map, null);
});

test('missing Google API key leaves the original route map available', async () => {
    const { context, elements, current, scripts } = satelliteContext('');
    await context.showSatellite(current().route);
    assert.equal(current().satelliteActive, false);
    assert.match(elements.status.textContent, /API key is not configured/);
    assert.equal(scripts.length, 0);
});

test('Google script loads once and late completion cannot reopen a previous route', async () => {
    const { context, scripts, current, maps } = satelliteContext();
    context.window.google = undefined;
    const pending = context.showSatellite(current().route);
    const sameLoad = context.loadGoogleMaps('test-key');
    assert.equal(scripts.length, 1);
    assert.match(scripts[0].src, /^https:\/\/maps.googleapis.com\/maps\/api\/js\?/);
    current().routeRequest++;
    context.window.google = { maps };
    context.window.initSurfaceSatelliteMaps();
    await Promise.all([pending, sameLoad]);
    assert.equal(current().satelliteActive, false);
});

test('Google authentication and network failures retain the canvas fallback', async () => {
    for (const failure of ['network', 'authorization']) {
        const { context, scripts, elements, current } = satelliteContext();
        context.window.google = undefined;
        const pending = context.showSatellite(current().route);
        if (failure === 'network') scripts[0].onerror();
        else context.window.gm_authFailure();
        await pending;
        assert.equal(current().satelliteActive, false);
        assert.equal(elements['satellite-map'].hidden, true);
        assert.match(elements.status.textContent, /Satellite imagery unavailable/);
    }
});
