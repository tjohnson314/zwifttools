const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const SEGMENT_LOCATION_TOLERANCE_M = 20;

function render(escalators, overrides = {}) {
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
        segments: [{ name: 'Sprint', type: 'sprint', start_distance_m: 200, end_distance_m: 240 }],
        ...overrides };
    const before = JSON.stringify(data);
    context.buildCombinedPath(data);
    context.renderElevation(data);
    assert.equal(JSON.stringify(data), before);
    return { result, listeners, combined: vm.runInContext('state.combined', context) };
}

function exportedSegmentRoutes() {
    const directory = path.join(__dirname, '../zwift_routes');
    return fs.readdirSync(directory).filter(name => /^world_\d+\.json$/.test(name)).flatMap(name => {
        const world = JSON.parse(fs.readFileSync(path.join(directory, name), 'utf8'),
            (key, value, context) => key === 'hash' && typeof value === 'number' ? context.source : value);
        return world.routes.filter(route => route.segments?.length).map(route => ({ mapID: world.mapID, route }));
    });
}

function checkpointPosition(route, fraction) {
    const profile = route.segment_path;
    assert.ok(profile?.x?.length >= 2, `${route.name}: missing checkpoint-index segment path`);
    assert.ok(Number.isFinite(fraction), `${route.name}: invalid segment boundary percentage`);
    const positions = profile.d || profile.x.map((value, index) => index);
    const position = fraction * positions.at(-1);
    let index = 0;
    while (index < positions.length - 2 && positions[index + 1] < position) index++;
    const span = positions[index + 1] - positions[index];
    const weight = span ? (position - positions[index]) / span : 0;
    const point = {
        x: profile.x[index] + weight * (profile.x[index + 1] - profile.x[index]),
        z: profile.z[index] + weight * (profile.z[index + 1] - profile.z[index]),
    };
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.z), `${route.name}: invalid segment coordinates`);
    return point;
}

function displayedBoundaryError(rendered, distanceKm, expected) {
    let nearest = Infinity;
    for (const trace of rendered.result.traces) {
        for (let index = 1; index < trace.x.length; index++) {
            const start = trace.x[index - 1];
            const end = trace.x[index];
            if (distanceKm < start - 0.00002 || distanceKm > end + 0.00002) continue;
            const first = rendered.combined[trace.customdata[index - 1]];
            const last = rendered.combined[trace.customdata[index]];
            const weight = end === start ? 0 : (distanceKm - start) / (end - start);
            const position = {
                x: first.x + weight * (last.x - first.x),
                z: -(first.y + weight * (last.y - first.y)),
            };
            nearest = Math.min(nearest, Math.hypot(position.x - expected.x, position.z - expected.z));
        }
    }
    return nearest;
}

test('directional segment endpoints agree across every exported route within 20 metres', context => {
    const groups = new Map();
    const routes = exportedSegmentRoutes();
    for (const { mapID, route } of routes) {
        for (const segment of route.segments) {
            const key = `${mapID}:${segment.hash}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push({ route, segment });
        }
    }
    const failures = new Map();
    let comparisons = 0;
    for (const [key, occurrences] of groups) {
        for (let first = 0; first < occurrences.length; first++) {
            for (let second = first + 1; second < occurrences.length; second++) {
                const left = occurrences[first];
                const right = occurrences[second];
                if (left.route.nameHash === right.route.nameHash) continue;
                for (const boundary of ['start', 'end']) {
                    const leftPoint = checkpointPosition(left.route, left.segment[`percent_${boundary}`]);
                    const rightPoint = checkpointPosition(right.route, right.segment[`percent_${boundary}`]);
                    const distance = Math.hypot(leftPoint.x - rightPoint.x, leftPoint.z - rightPoint.z);
                    comparisons++;
                    if (distance > SEGMENT_LOCATION_TOLERANCE_M) {
                        const failureKey = `${key}:${boundary}`;
                        if (!failures.has(failureKey) || distance > failures.get(failureKey).distance) {
                            failures.set(failureKey, { distance,
                                message: `${key} ${left.segment.name} ${boundary}: ${left.route.name} / ${right.route.name}: ${distance.toFixed(3)} m` });
                        }
                    }
                }
            }
        }
    }
    assert.ok(comparisons > 0, 'No shared directional segments were compared');
    context.diagnostic(`${routes.length} routes, ${groups.size} directional segments, ${comparisons} endpoint comparisons`);
    assert.equal(failures.size, 0,
        `${failures.size} inconsistent directional segment boundaries; largest 20 shown:\n` +
        [...failures.values()].sort((left, right) => right.distance - left.distance).slice(0, 20)
            .map(failure => failure.message).join('\n'));
});

test('every displayed segment boundary aligns with its route profile within 20 metres', context => {
    const failures = [];
    let checked = 0;
    const pack = leg => leg ? { ...leg, y: leg.z.map(value => -value) } : null;
    for (const { mapID, route } of exportedSegmentRoutes()) {
        const omitted = route.leadin_view?.paddock_distance_m || 0;
        const segments = route.segments.map(segment => ({ ...segment,
            start_distance_m: Math.round((segment.start_distance_m - omitted) * 100) / 100,
            end_distance_m: Math.round((segment.end_distance_m - omitted) * 100) / 100,
        }));
        const rendered = render(undefined, {
            ...route, escalators: [], segments,
            leadin: pack(route.leadin_view?.profile || route.leadin), route: pack(route.route),
        });
        const shapes = rendered.result.layout.shapes.filter(shape => shape.type === 'rect');
        assert.equal(shapes.length, segments.length, `${route.name}: missing segment shading`);
        for (let index = 0; index < segments.length; index++) {
            const segment = segments[index];
            for (const [boundary, edge] of [['start', 'x0'], ['end', 'x1']]) {
                const expected = checkpointPosition(route, segment[`percent_${boundary}`]);
                const error = displayedBoundaryError(rendered, shapes[index][edge], expected);
                checked++;
                if (error > SEGMENT_LOCATION_TOLERANCE_M) {
                    failures.push({ error,
                        message: `${mapID}:${segment.hash} ${route.name}: ${segment.name} ${boundary}: ${error.toFixed(3)} m` });
                }
            }
        }
    }
    assert.ok(checked > 0, 'No displayed segment boundaries were checked');
    context.diagnostic(`${checked} rendered segment boundaries checked; ` +
        `${failures.filter(failure => !Number.isFinite(failure.error)).length} outside the plotted profile`);
    assert.equal(failures.length, 0,
        `${failures.length} misaligned rendered boundaries; largest 20 shown:\n` +
        failures.sort((left, right) => right.error - left.error).slice(0, 20)
            .map(failure => failure.message).join('\n'));
});

test('boundary checks detect displaced map paths and phantom lead-in display offsets', () => {
    const profile = { d: [0, 500, 1000], x: [0, 500, 1000], z: [0, 0, 0],
        y: [0, 0, 0], alt: [0, 0, 0], surface: ['Tarmac', 'Tarmac', 'Tarmac'] };
    const source = { name: 'Straight route', segment_path: profile };
    const expected = checkpointPosition(source, 0.4);
    assert.deepEqual(expected, { x: 400, z: 0 });
    const displaced = checkpointPosition({ ...source,
        segment_path: { ...profile, x: profile.x.map(value => value + 40) } }, 0.4);
    assert.ok(Math.hypot(displaced.x - expected.x, displaced.z - expected.z) > SEGMENT_LOCATION_TOLERANCE_M);
    const rendered = render(undefined, { route: profile,
        segments: [{ name: 'Sprint', start_distance_m: 200, end_distance_m: 400 }] });
    const edge = rendered.result.layout.shapes.find(shape => shape.type === 'rect').x1;
    assert.equal(displayedBoundaryError(rendered, edge, expected), 0);
    for (const trace of rendered.result.traces) {
        trace.x = trace.x.map(value => value + 0.2483);
    }
    const error = displayedBoundaryError(rendered, edge, expected);
    assert.ok(Math.abs(error - 248.3) < 1e-8);
    assert.ok(error > SEGMENT_LOCATION_TOLERANCE_M);
});

test('segment coordinate checks interpolate through nonuniform checkpoint positions', () => {
    const route = { name: 'Adaptive bends', segment_path: {
        d: [0, 1.5, 2, 3.5, 4], x: [0, 100, 100, 250, 300], z: [0, 50, 100, 100, 100],
    } };
    assert.deepEqual(checkpointPosition(route, 0.375), { x: 100, z: 50 });
    assert.deepEqual(checkpointPosition(route, 0.875), { x: 250, z: 100 });
});

test('routes without lead-in geometry align every rendered boundary within two centimetres', () => {
    const routes = exportedSegmentRoutes().filter(({ route }) => !route.leadin?.d?.length);
    assert.ok(routes.length > 0);
    for (const { route } of routes) {
        const rendered = render(undefined, {
            ...route, escalators: [], route: { ...route.route, y: route.route.z.map(value => -value) },
        });
        const shapes = rendered.result.layout.shapes.filter(shape => shape.type === 'rect');
        assert.equal(shapes.length, route.segments.length);
        for (let index = 0; index < route.segments.length; index++) {
            const segment = route.segments[index];
            for (const [boundary, edge] of [['start', 'x0'], ['end', 'x1']]) {
                const expected = checkpointPosition(route, segment[`percent_${boundary}`]);
                const error = displayedBoundaryError(rendered, shapes[index][edge], expected);
                assert.ok(error <= 0.02, `${route.name}: ${segment.name} ${boundary}: ${error} m`);
            }
        }
    }
});

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

test('Outer Scotland finishes at Champions Sprint without an unrepresented lead-in offset', () => {
    const world = JSON.parse(fs.readFileSync(path.join(__dirname, '../zwift_routes/world_13.json'), 'utf8'));
    const route = world.routes.find(entry => entry.nameHash === 2826340040);
    assert.equal(route.leadin, null);
    assert.equal(route.leadin_distance_m, 248.3);
    const { result, combined } = render(undefined, {
        ...route,
        route: { ...route.route, y: route.route.z.map(value => -value) },
    });
    const sprint = route.segments.find(segment => segment.name === "Champion's Sprint");
    const finishKm = sprint.end_distance_m / 1000;
    assert.equal(result.traces.at(-1).x.at(-1), finishKm);
    assert.equal(combined.at(-1).dist_m, sprint.end_distance_m);
    const sprintShape = result.layout.shapes.find(shape => shape.type === 'rect');
    assert.equal(sprintShape.x1, finishKm);
    assert.equal(result.layout.shapes.filter(shape => shape.type === 'line').length, 0);
});

test('missing, null and empty lead-ins do not shift available geometry or mark a phantom boundary', () => {
    for (const leadin of [undefined, null, { d: [], x: [], y: [], alt: [], surface: [] }]) {
        const { result, combined } = render(undefined, { leadin });
        assert.deepEqual(Array.from(result.traces[0].x), [0, 0.1, 0.2]);
        assert.equal(combined[0].dist_m, 0);
        assert.equal(result.layout.shapes.filter(shape => shape.type === 'line').length, 0);
    }
});

test('profiles with lead-in geometry retain their published main-route offset', () => {
    const leadin = { d: [0, 40], x: [-1, 0], y: [0, 0],
        alt: [0, 0], surface: ['Tarmac', 'Tarmac'] };
    const { result, combined } = render(undefined, { leadin });
    assert.deepEqual(Array.from(result.traces[1].x), [0.05, 0.15, 0.25]);
    assert.equal(combined[leadin.d.length].dist_m, 50);
    assert.equal(result.layout.shapes.find(shape => shape.type === 'line').x0, 0.05);
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
