const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function plannerContext() {
    const legend = { children: [], replaceChildren() { this.children = []; },
        appendChild(child) { this.children.push(child); } };
    const configs = [];
    const elements = new Map();
    const storage = new Map();
    const browserWindow = {
        addEventListener() {},
        location: { href: 'http://localhost/tt-pacing' },
        history: { replaceState(state, title, url) { browserWindow.location.href = String(url); } },
    };
    const context = vm.createContext({
        window: browserWindow, URL, URLSearchParams,
        clearTimeout() {}, setTimeout() {},
        localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
        document: {
            addEventListener() {},
            getElementById(id) {
                if (id === 'surfaceLegend') return legend;
                if (!elements.has(id)) elements.set(id, {
                    value: '', checked: true, style: {}, getContext() {}, checkValidity() { return true; },
                    classList: { toggle() {}, add() {}, remove() {} }, setAttribute() {}, appendChild() {},
                });
                return elements.get(id);
            },
            createElement() { return { style: {}, children: [], append(...children) { this.children.push(...children); } }; },
            createTextNode(text) { return { textContent: text }; },
        },
        Chart: class {
            constructor(ctx, config) { configs.push(config); }
            destroy() {}
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/tt_pacing.js'), 'utf8'), context);
    return { context, configs, legend, elements };
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

test('TT portion inputs convert together and retain empty defaults', () => {
    const { context } = plannerContext();
    vm.runInContext(`
        document.getElementById('startDistance').value = '2';
        document.getElementById('customDistance').value = '5';
        convertInputs('imperial'); currentUnit = 'imperial';
    `, context);
    assert.ok(Math.abs(context.getStartDistanceKm() - 2) < 0.001);
    assert.ok(Math.abs(context.getCustomDistanceKm() - 5) < 0.001);
    vm.runInContext(`document.getElementById('customDistance').value = '';`, context);
    assert.equal(context.getCustomDistanceKm(), null);
    assert.equal(context.portionError(), '');
});

test('TT segment mode hides route-only controls and ignores invalid portion settings', () => {
    const { context, elements } = plannerContext();
    vm.runInContext(`
        document.getElementById('startDistance').value = '10';
        document.getElementById('customDistance').value = '2';
    `, context);
    assert.match(context.portionError(), /End distance/);
    vm.runInContext(`planMode = 'segment'; updateCourseUI(); updateLapUI();`, context);
    assert.equal(context.portionError(), '');
    assert.equal(elements.get('courseLabel').textContent, 'Segment');
    for (const id of ['leadinGroup', 'portionGroup', 'lapGroup', 'routeStartModeGroup']) {
        assert.equal(elements.get(id).style.display, 'none');
    }
    vm.runInContext(`planMode = 'route'; updateCourseUI();`, context);
    assert.equal(elements.get('portionGroup').style.display, '');
});

test('TT map markers and GPS export offset portion distances before wrapping laps', () => {
    const { context } = plannerContext();
    vm.runInContext(`
        lastPlanData = { start_distance_km: 0.15 };
        routeGeometry = { legs: [{ name: 'route', pts: [{d: 0}, {d: 200}] }],
            combined: [{ d: 0, lat: 1, lng: 2 }, { d: 200, lat: 3, lng: 4 }] };
    `, context);
    assert.equal(context.mapDistanceToGeometry(0), 150);
    assert.equal(context.mapDistanceToGeometry(100), 50);
    assert.deepEqual(Array.from(context.interpolateRouteGps(0)), [2.5, 3.5]);
});

for (const mode of ['route', 'segment']) {
    test(`TT ${mode} requests send the applicable course and distance settings`, async () => {
        const { context } = plannerContext();
        let body;
        context.fetch = async (url, options) => {
            body = JSON.parse(options.body);
            return { ok: true, json: async () => ({}) };
        };
        vm.runInContext(`
            planMode = '${mode}';
            selectedRoute = { id: '1:123', name: 'Course', world: 'WATOPIA', is_loop: true };
            laps = 3;
            bikeSelector = { getConfig: () => ({frameId: 'bike', level: 0}), isReady: () => true };
            document.getElementById('riderWeight').value = '75';
            document.getElementById('riderHeight').value = '180';
            document.getElementById('avgPower').value = '250';
            document.getElementById('startDistance').value = '2';
            document.getElementById('customDistance').value = '5';
            displayResults = () => {};
        `, context);
        await context.requestPlan();
        if (mode === 'route') {
            assert.equal(body.segment_id, undefined);
            assert.equal(body.start_distance_km, 2);
            assert.equal(body.custom_distance_km, 5);
            assert.equal(body.include_leadin, true);
            assert.equal(body.laps, 3);
            assert.equal(body.route_mode, 'race');
        } else {
            assert.equal(body.segment_id, '1:123');
            assert.equal(body.start_distance_km, undefined);
            assert.equal(body.custom_distance_km, undefined);
            assert.equal(body.include_leadin, false);
            assert.equal(body.laps, 1);
            assert.equal(body.route_mode, undefined);
        }
    });
}

function selectStartRoute(context) {
            vm.runInContext(`
                selectedRoute = {
                    id: '123', name: 'Course', world: 'WATOPIA', distance_km: 10, ascent_m: 100,
                    available_modes: ['race', 'free_ride'],
                    race_leadin_distance_km: 3, race_leadin_ascent_m: 20,
                    free_ride_leadin_distance_km: 1, free_ride_leadin_ascent_m: 5
                };
                allRoutes = [selectedRoute];
                document.getElementById('routeSelect').value = '123';
                bikeSelector = { getConfig: () => ({frameId: 'bike'}), isReady: () => true };
                document.getElementById('riderWeight').value = '75';
                document.getElementById('riderHeight').value = '180';
                document.getElementById('avgPower').value = '250';
            `, context);
        }

        test('TT start toggle updates totals, invalidates old results, saves selection and shares mode in links', () => {
            const { context, elements } = plannerContext();
            selectStartRoute(context);
            vm.runInContext(`lastPlanData = {}; showRouteStats(selectedRoute);`, context);
            assert.equal(elements.get('statDist').textContent, '13.0 km');
            context.setRouteMode('free_ride');
            assert.equal(elements.get('statDist').textContent, '11.0 km');
            assert.equal(elements.get('statAscent').textContent, '105 m');
            assert.equal(elements.get('resultsSection').style.display, 'none');
            assert.equal(vm.runInContext('lastPlanData', context), null);
            assert.equal(context.loadSettings().routeMode, 'free_ride');
            assert.equal(new URL(context.window.location.href).searchParams.get('mode'), 'free_ride');
            assert.equal(elements.get('planBtn').disabled, false);
            vm.runInContext(`planMode = 'segment'; setRouteUrlParam(selectedRoute);`, context);
            assert.equal(new URL(context.window.location.href).searchParams.get('mode'), null);
        });

        test('TT unavailable starts are disabled, never silently substituted, and block planning', () => {
            const { context, elements } = plannerContext();
            selectStartRoute(context);
            vm.runInContext(`selectedRoute.available_modes = ['race']; updateRouteModeUI();`, context);
            assert.equal(elements.get('freeRideStart').disabled, true);
            context.setRouteMode('free_ride');
            assert.equal(vm.runInContext('routeMode', context), 'race');
            vm.runInContext(`routeMode = 'free_ride'; updateRouteModeUI(); updatePlanButton();`, context);
            assert.match(elements.get('routeModeHint').textContent, /Free Ride is unavailable/);
            assert.equal(elements.get('planBtn').disabled, true);
            assert.equal(vm.runInContext('routeMode', context), 'free_ride');
        });

        test('TT missing free-ride lead-ins do not borrow racing totals and can be excluded explicitly', () => {
            const { context, elements } = plannerContext();
            selectStartRoute(context);
            vm.runInContext(`selectedRoute.free_ride_leadin_distance_km = null;`, context);
            context.setRouteMode('free_ride');
            assert.equal(elements.get('statDist').textContent, 'Unavailable');
            assert.equal(elements.get('planBtn').disabled, true);
            vm.runInContext(`
                document.getElementById('includeLeadin').checked = false;
                updateRouteModeUI(); updatePlanButton(); showRouteStats(selectedRoute);
            `, context);
            assert.equal(elements.get('statDist').textContent, '10.0 km');
            assert.equal(elements.get('routeModeHint').textContent, '');
            assert.equal(elements.get('planBtn').disabled, false);
        });

        test('TT mode changes discard pending plans and subsequent requests send the new mode', async () => {
            const { context } = plannerContext();
            selectStartRoute(context);
            let resolve;
            let rendered = 0;
            const bodies = [];
            context.displayResults = () => rendered++;
            context.fetch = async (url, options) => {
                bodies.push(JSON.parse(options.body));
                if (bodies.length === 1) return new Promise(done => { resolve = done; });
                return { ok: true, json: async () => ({}) };
            };
            const pending = context.requestPlan();
            context.setRouteMode('free_ride');
            resolve({ ok: true, json: async () => ({}) });
            await pending;
            assert.equal(rendered, 0);
            await context.requestPlan();
            assert.deepEqual(bodies.map(body => body.route_mode), ['race', 'free_ride']);
            assert.equal(rendered, 1);
        });

        test('TT map requests and cache distinguish starts and discard stale geometry', async () => {
            const { context } = plannerContext();
            selectStartRoute(context);
            context.drawRouteMap = () => {};
            let resolveRace;
            const urls = [];
            const response = distance => ({ ok: true, json: async () => ({
                route: { x: [0, 1], y: [0, 1], d: [0, distance] },
            }) });
            context.fetch = async url => {
                urls.push(url);
                if (new URL(url, 'http://localhost').searchParams.get('mode') === 'race') {
                    return new Promise(resolve => { resolveRace = resolve; });
                }
                return response(200);
            };
            const pending = context.loadRouteMap();
            context.setRouteMode('free_ride');
            await context.loadRouteMap();
            resolveRace(response(100));
            await pending;
            assert.match(vm.runInContext('routeGeometry.key', context), /free_ride$/);
            assert.equal(vm.runInContext('routeGeometry.combined.at(-1).d', context), 200);
            await context.loadRouteMap();
            assert.equal(urls.length, 2);
            assert.deepEqual(urls.map(url => new URL(url, 'http://localhost').searchParams.get('mode')),
                ['race', 'free_ride']);
        });

        test('TT world filtering clears unavailable selections and re-enables start choices', () => {
            const { context, elements } = plannerContext();
            selectStartRoute(context);
            vm.runInContext(`
                selectedRoute.available_modes = ['race'];
                updateRouteModeUI();
                document.getElementById('worldFilter').value = 'OTHER';
                filterRoutes();
            `, context);
            assert.equal(vm.runInContext('selectedRoute', context), null);
            assert.equal(elements.get('freeRideStart').disabled, false);
            assert.equal(elements.get('planBtn').disabled, true);
        });