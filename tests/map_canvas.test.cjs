const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function load(context, name) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js', name), 'utf8'), context);
}

function loadingContext(script) {
    const elements = new Map();
    const pending = [];
    const createElement = () => ({
        children: [], dataset: {}, attributes: {}, style: {}, textContent: '', listeners: {}, open: false,
        classList: { add() {}, toggle() {} }, addEventListener(name, handler) { this.listeners[name] = handler; },
        showModal() { this.open = true; }, close() { this.open = false; },
        set innerHTML(value) { this.html = value; this.children = []; },
        get innerHTML() { return this.html || ''; },
        setAttribute(name, value) { this.attributes[name] = value; },
        appendChild(child) { this.children.push(child); },
    });
    const context = vm.createContext({
        document: {
            addEventListener() {}, createElement, querySelectorAll() { return []; },
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, createElement());
                return elements.get(id);
            },
        },
        fetch: () => new Promise((resolve, reject) => pending.push({ resolve, reject })),
    });
    load(context, script);
    for (const name of ['hideSatellite', 'buildCombinedPath', 'renderRouteDetail',
        'renderRoadDetail', 'fitToBounds', 'renderBase', 'updateUrl']) context[name] = () => {};
    vm.runInContext(`state.world = { mapID: 1, routes: [1, 2].map(nameHash => ({
        nameHash, name: 'Route ' + nameHash, distance_m: 100, ascent_m: 0,
        leadin_distance_m: 0, leadin_ascent_m: 0,
    })), roads: [1, 2].map(id => ({ id, surface: 'Tarmac', length_m: 100, node_count: 2 })) };`, context);
    const dev = script === 'surface_map_dev.js';
    const renderList = dev ? context.renderRoadList : context.renderRouteList;
    renderList();
    return { context, pending, renderList, elements,
        list: elements.get(dev ? 'road-list' : 'route-list'),
        loadItem: dev ? context.loadRoad : context.loadRoute,
        clearItem: dev ? context.clearRoad : context.clearRoute,
        status: () => elements.get('status').textContent,
    };
}

function roadHit(road_id, x = 50, y = 0) {
    return {road_id, point: {x, y}};
}

for (const script of ['surface_map.js', 'surface_map_dev.js']) {
    for (const fails of [false, true]) {
        test(`${script}: row spinner survives rebuilds and stale ${fails ? 'errors' : 'responses'}`, async () => {
            const { context, pending, renderList, list, loadItem, status } = loadingContext(script);
            const earlier = loadItem(1);
            const clicked = list.children[1];
            const latest = loadItem(2);
            assert.equal(list.children[1], clicked, 'starting a load leaves the clicked row in place');
            assert.deepEqual(list.children.map(item => item.attributes['aria-busy']), ['false', 'true']);
            assert.equal(status(), '');
            renderList();
            assert.deepEqual(list.children.map(item => item.attributes['aria-busy']), ['false', 'true']);
            if (fails) pending[0].reject(new Error('Stale failure'));
            else pending[0].resolve({ ok: true, json: async () => ({ id: 1 }) });
            await earlier;
            assert.equal(list.children[1].attributes['aria-busy'], 'true');
            assert.equal(status(), '');
            if (fails) pending[1].reject(new Error('Latest failure'));
            else pending[1].resolve({ ok: true, json: async () => ({ id: 2 }) });
            await latest;
            assert.ok(list.children.every(item => item.attributes['aria-busy'] === 'false'));
            assert.equal(status(), fails ? 'Latest failure' : '');
            assert.equal(vm.runInContext(script === 'surface_map_dev.js'
                ? 'state.loadingRoadId' : 'state.loadingHash', context), null);
        });
    }

    test(`${script}: clearing a selection removes its pending spinner immediately`, async () => {
        const { pending, list, loadItem, clearItem, status } = loadingContext(script);
        const loading = loadItem(1);
        assert.equal(list.children[0].attributes['aria-busy'], 'true');
        clearItem();
        assert.ok(list.children.every(item => item.attributes['aria-busy'] === 'false'));
        pending[0].reject(new Error('Cancelled failure'));
        await loading;
        assert.equal(status(), '');
        assert.ok(list.children.every(item => item.attributes['aria-busy'] === 'false'));
    });
}

test('main road clicks do nothing for zero matches, auto-select one, and offer every matching route', async () => {
    const { context, pending, elements } = loadingContext('surface_map.js');
    vm.runInContext(`state.world.routes[0].road_ids = { race: [0, 1], free_ride: [0] };
        state.world.routes[1].road_ids = { race: [0], free_ride: [0] };
        state.world.road_match_tolerance = 1;
        const path = [{x:[0,100],y:[0,0]}];
        const approach = [{x:[0,100],y:[20,20]}];
        state.world.routes[0].road_paths = {race:{0:path,1:approach},free_ride:{0:path}};
        state.world.routes[1].road_paths = {race:{0:path},free_ride:{0:path}};`, context);
    await context.showRoadRoutes(roadHit(99,200));
    assert.equal(pending.length, 0);
    const single = context.showRoadRoutes(roadHit(1,50,20));
    assert.equal(pending.length, 1);
    pending[0].resolve({ ok: true, json: async () => ({ name: 'Route 1' }) });
    await single;
    assert.equal(vm.runInContext('state.activeHash', context), 1);
    vm.runInContext('state.routeFilter = "No matches"; state.sportFilter = "running";', context);
    await context.showRoadRoutes(roadHit(0));
    assert.equal(elements.get('road-route-picker').open, true);
    const list = elements.get('road-picker-list');
    assert.deepEqual(list.children.map(item => item.dataset.hash), ['1', '2']);
    elements.get('road-picker-search').oninput({target:{value:'Route 2'}});
    assert.deepEqual(list.children.map(item => item.hidden), [true, false]);
    const choosing = list.children[1].children[0].listeners.click();
    assert.equal(list.children[1].attributes['aria-busy'], 'true');
    pending[1].resolve({ ok: true, json: async () => ({ name: 'Route 2' }) });
    await choosing;
    assert.equal(vm.runInContext('state.activeHash', context), 2);
    assert.equal(elements.get('road-route-picker').open, false);
    vm.runInContext('state.routeMode = "free_ride";', context);
    const count = pending.length;
    await context.showRoadRoutes(roadHit(1,50,20));
    assert.equal(pending.length, count, 'race-only approaches do not match free-ride mode');
});

test('cancelling the road picker removes its spinner and rejects the pending selection', async () => {
    const { context, pending, elements } = loadingContext('surface_map.js');
    vm.runInContext('state.world.routes.forEach(route => route.road_paths = {race:{0:[{x:[0,100],y:[0,0]}]}});', context);
    await context.showRoadRoutes(roadHit(0));
    const choosing = elements.get('road-picker-list').children[0].children[0].listeners.click();
    context.closeRoadPicker();
    pending[0].resolve({ ok: true, json: async () => ({ name: 'Cancelled route' }) });
    await choosing;
    assert.equal(vm.runInContext('state.activeHash', context), null);
    assert.equal(vm.runInContext('state.loadingHash', context), null);
    assert.equal(elements.get('road-route-picker').open, false);
});

test('a failed picker reload keeps the list and local error visible for retry', async () => {
    const { context, pending, elements } = loadingContext('surface_map.js');
    vm.runInContext(`state.world.routes.forEach(route => route.road_paths = {race:{0:[{x:[0,100],y:[0,0]}]}});
        state.activeHash = 1; state.route = {name:'Existing route'};`, context);
    await context.showRoadRoutes(roadHit(0));
    const choosing = elements.get('road-picker-list').children[0].children[0].listeners.click();
    pending[0].resolve({ok:false,json:async()=>({error:'Unavailable'})});
    await choosing;
    assert.equal(elements.get('road-route-picker').open, true);
    assert.equal(elements.get('road-picker-status').textContent, 'Unavailable');
    assert.equal(elements.get('road-picker-list').children[0].attributes['aria-busy'], 'false');
});

test('main road picking distinguishes native IDs at crossings and ignores invisible geometry', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    load(context, 'surface_map.js');
    vm.runInContext(`canvas = {getBoundingClientRect(){return {left:10,top:20,width:300,height:300};}};
        state.dpr = 2; state.scale = 4; state.offsetX = state.offsetY = 40;
        state.world = {segments:[
            {road_id:0,x:[0,100],y:[10,10]},
            {road_id:1,x:[40,40],y:[0,100]},
        ]};`, context);
    assert.equal(context.pickRoad(70, 60).road_id, 0);
    assert.equal(context.pickRoad(110, 120).road_id, 1);
    assert.equal(context.pickRoad(9, 60), null);
    vm.runInContext('state.showNetwork = false;', context);
    assert.equal(context.pickRoad(70, 60), null);
    vm.runInContext('state.route = {route:{x:[0,100],y:[10,10]}};', context);
    assert.equal(context.pickRoad(70, 60).road_id, 0);
});

test('main clicks match the clicked section, not another part of the same road', async () => {
    const {context, pending} = loadingContext('surface_map.js');
    vm.runInContext(`state.world.road_match_tolerance = 1;
        state.world.routes[0].road_paths = {race:{0:[{x:[0,40],y:[0,0]}]}};
        state.world.routes[1].road_paths = {race:{0:[{x:[60,100],y:[0,0]}]}};`, context);
    await context.showRoadRoutes(roadHit(0, 50));
    assert.equal(pending.length, 0);
    const left = context.showRoadRoutes(roadHit(0, 20));
    pending[0].resolve({ok:true,json:async()=>({name:'Left section'})});
    await left;
    assert.equal(vm.runInContext('state.activeHash', context), 1);
    const right = context.showRoadRoutes(roadHit(0, 80));
    pending[1].resolve({ok:true,json:async()=>({name:'Right section'})});
    await right;
    assert.equal(vm.runInContext('state.activeHash', context), 2);
});

test('a clicked junction includes paths traversing that position on different roads', async () => {
    const {context,elements} = loadingContext('surface_map.js');
    vm.runInContext(`state.world.road_match_tolerance = 1;
        state.world.routes[0].road_paths = {race:{0:[{x:[0,40],y:[0,0]}]}};
        state.world.routes[1].road_paths = {race:{1:[{x:[20,20],y:[-20,20]}]}};`,context);
    await context.showRoadRoutes(roadHit(0,20,0));
    assert.deepEqual(elements.get('road-picker-list').children.map(item=>item.dataset.hash),['1','2']);
});

test('independent video coordinate frames do not match unrelated roads at the same coordinates', async () => {
    const {context,pending,elements} = loadingContext('surface_map.js');
    vm.runInContext(`state.world.disconnected_routes = true; state.world.road_match_tolerance = 1;
        state.world.routes[0].road_paths = {race:{0:[{x:[0,40],y:[0,0]}]}};
        state.world.routes[1].road_paths = {race:{1:[{x:[0,40],y:[0,0]}]}};`,context);
    const choosing = context.showRoadRoutes(roadHit(0,20,0));
    assert.equal(pending.length,1);
    assert.equal(elements.get('road-route-picker').open,false);
    pending[0].resolve({ok:true,json:async()=>({name:'Displayed video'})});
    await choosing;
    assert.equal(vm.runInContext('state.activeHash',context),1);
});

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
            document: { addEventListener() {}, getElementById() { return null; } },
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

test('dev road picking uses segment distance and a constant CSS-pixel tolerance across zoom and DPI', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { getBoundingClientRect() { return { left: 10, top: 20, width: 400, height: 300 }; } };
        state.world = { roads: [{ id: 0, x: [0, 100], y: [0, 0] }] };
    `, context);
    for (const dpr of [1, 2]) {
        for (const zoom of [0.5, 4]) {
            vm.runInContext(`state.dpr = ${dpr}; state.scale = ${zoom * dpr};
                state.offsetX = ${40 * dpr}; state.offsetY = ${60 * dpr};`, context);
            const clientX = 10 + 40 + 50 * zoom;
            assert.equal(context.pickRoad(clientX, 80).id, 0, 'the middle of a segment is selectable');
            assert.equal(context.pickRoad(clientX, 87.9).id, 0);
            assert.equal(context.pickRoad(clientX, 88.1), null);
            assert.equal(context.pickRoad(9, 80), null, 'clicks outside the canvas are ignored');
        }
    }
});

test('dev road picking chooses the closest displayed curved polyline', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 300 }; } };
        state.world = { roads: [
            { id: 2, x: [0, 100], y: [12, 12] },
            { id: 7, x: [0, 50, 100], y: [0, 30, 0] },
            { id: 9, x: [], y: [] },
        ] };
    `, context);
    assert.equal(context.pickRoad(25, 15).id, 7);
    assert.equal(context.pickRoad(25, 12).id, 2);
    assert.equal(context.pickRoad(200, 200), null);
    vm.runInContext('state.world = null;', context);
    assert.equal(context.pickRoad(25, 15), null);
});

test('hidden network roads cannot be picked but visible selected and spawn-highlight roads can', () => {
    const context = vm.createContext({ document: { addEventListener() {} } });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 300 }; } };
        state.world = { roads: [
            { id: 7, x: [0, 100], y: [0, 0] },
            { id: 8, x: [0, 100], y: [80, 80] },
        ] };
        state.showNetwork = false;
        state.activeRoadId = 7;
        state.road = { full: { x: [0, 100], y: [30, 30] } };
    `, context);
    assert.equal(context.pickRoad(50, 0), null);
    assert.equal(context.pickRoad(50, 30).id, 7);
    assert.equal(context.pickRoad(50, 80), null);
    vm.runInContext(`state.spawn = { road_id: 8 };
        state.highlight = { x: [0, 100], y: [80, 80] };`, context);
    assert.equal(context.pickRoad(50, 80).id, 8);
});

test('dev canvas clicks select roads while marker priority, drag, and cancellation remain intact', () => {
    const events = {}, selections = [];
    const elements = new Map();
    const element = id => ({ addEventListener(name, handler) { events[`${id}:${name}`] = handler; } });
    const surfaceCanvas = { ...element('canvas'), setPointerCapture() {},
        getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 300 }; } };
    const context = vm.createContext({
        document: { addEventListener() {}, getElementById(id) {
            if (!elements.has(id)) elements.set(id, element(id));
            return elements.get(id);
        } },
        window: element('window'), surfaceCanvas,
    });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`canvas = surfaceCanvas; state.world = {
        roads: [{ id: 0, x: [0, 100], y: [40, 40] }], arches: [], spawn_markers: [],
    };`, context);
    context.setBrowserMode = mode => selections.push(mode);
    context.loadRoad = id => selections.push(id);
    context.selectArch = arch => selections.push(arch.id);
    context.selectSpawn = spawn => selections.push(spawn.id);
    context.renderBase = () => {};
    context.bindEvents();
    const click = (clientX, clientY) => {
        events['canvas:pointerdown']({ button: 0, pointerId: 1, clientX, clientY });
        events['window:pointerup']({ clientX, clientY });
    };
    click(50, 40);
    assert.deepEqual(selections, ['roads', 0]);
    click(250, 250);
    assert.equal(selections.length, 2);
    events['canvas:pointerdown']({ button: 0, pointerId: 1, clientX: 20, clientY: 40 });
    events['window:pointermove']({ clientX: 50, clientY: 80 });
    events['window:pointerup']({ clientX: 50, clientY: 80 });
    assert.equal(selections.length, 2, 'panning does not select a road');
    events['canvas:pointerdown']({ button: 0, pointerId: 1, clientX: 50, clientY: 40 });
    events['window:pointercancel']();
    events['window:pointerup']({ clientX: 50, clientY: 40 });
    assert.equal(selections.length, 2);
    vm.runInContext(`state.offsetX = state.offsetY = 0;
        state.world.arches = [{ id: 'arch', start: { x: 50, y: 40 } }];`, context);
    click(50, 40);
    assert.deepEqual(selections.slice(-2), ['arches', 'arch']);
    vm.runInContext(`state.world.spawn_markers = [{ id: 'spawn', position: { x: 50, y: 40 } }];`, context);
    click(50, 40);
    assert.deepEqual(selections.slice(-2), ['spawns', 'spawn']);
});

test('dev timing markers skip unlocated boundaries and keep exact hashes when hit-tested at high DPI', () => {
    const radii = [];
    const context = vm.createContext({
        document: { addEventListener() {} },
        g: {
            beginPath() {}, fill() {}, stroke() {}, strokeText() {}, fillText() {},
            measureText() { return { width: 100 }; },
            arc(screenX, screenY, radius) { radii.push(radius); },
        },
    });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { width: 400, height: 300, getBoundingClientRect() { return { left: 10, top: 20 }; } };
        state.dpr = 2;
        state.scale = 2;
        state.arch = {
            id: '1:R', name: 'Sprint Reverse', direction: 'R', hash: '-9223372036854775807',
            start: { x: 10, y: 20 }, finish: { x: 100, y: 120 },
        };
        state.world = { arches: [{ id: '2:F', start: null, finish: null }, state.arch] };
        drawArchMarkers(g);
    `, context);
    assert.deepEqual(radii, [16, 10]);
    assert.equal(context.pickArch(20, 40).arch.hash, '-9223372036854775807');
    assert.equal(context.pickArch(110, 140).boundary, 'finish');
    assert.equal(context.pickArch(300, 300), null);
    vm.runInContext('state.showArches = false;', context);
    assert.equal(context.pickArch(20, 40), null);
});

test('shared timing finishes draw once, omit direction, and keep the selected reverse record on clicks', () => {
    const labels = [];
    const radii = [];
    const context = vm.createContext({
        document: { addEventListener() {} },
        g: {
            beginPath() {}, fill() {}, stroke() {}, strokeText() {},
            fillText(text) { labels.push(text); },
            measureText() { return { width: 100 }; },
            arc(screenX, screenY, radius) { radii.push(radius); },
        },
    });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { width: 400, height: 300, getBoundingClientRect() { return { left: 0, top: 0 }; } };
        const finish = { x: 100, y: 100 };
        const forward = { id: '1:F', name: 'Sprint', direction: 'F', road_id: 4, end_time: 0.5,
            start: { x: 20, y: 20 }, finish };
        const reverse = { id: '1:R', name: 'Sprint Reverse', direction: 'R', road_id: 4, end_time: 0.5,
            start: { x: 200, y: 20 }, finish };
        state.world = { arches: [forward, reverse] };
        state.arch = reverse;
        drawArchMarkers(g);
    `, context);
    assert.equal(radii.length, 3);
    assert.deepEqual(labels, ['Sprint Reverse (R) start', 'Sprint finish']);
    assert.equal(context.pickArch(100, 100).arch.direction, 'R');
    assert.equal(context.pickArch(100, 100).boundary, 'finish');
    vm.runInContext('state.world.arches[0].finish = { x: 110, y: 100 };', context);
    assert.equal(vm.runInContext('sharedTimingFinish(state.arch)', context), null);
});

test('dev spawn labels avoid collisions, preserve selection, and pick at high DPI', () => {
    const labels = [];
    const radii = [];
    const context = vm.createContext({
        document: { addEventListener() {} },
        g: {
            beginPath() {}, fill() {}, stroke() {}, strokeText() {},
            fillText(text, left, top) { labels.push({ text, left, top }); },
            measureText() { return { width: 100 }; },
            arc(screenX, screenY, radius) { radii.push(radius); },
        },
    });
    load(context, 'surface_map_dev.js');
    vm.runInContext(`
        canvas = { width: 400, height: 300, getBoundingClientRect() { return { left: 10, top: 20 }; } };
        state.dpr = 2;
        state.scale = 2;
        const spawn = { id: '130', name: 'AdZ_Paddock', road_id: 130,
            position: { x: 10, y: 20 }, position_source: 'road_geometry_midpoint' };
        state.spawn = spawn;
        state.world = { spawn_markers: [
            { id: '2', name: 'Overlapping', position: { x: 11, y: 21 } },
            { id: '3', name: 'Unlocated', position: null }, spawn,
        ] };
        drawSpawnMarkers(g);
    `, context);
    assert.deepEqual(radii, [8, 14]);
    assert.deepEqual(labels.map(label => label.text), ['AdZ_Paddock']);
    assert.ok(labels[0].left >= 12 && labels[0].left + 100 <= 388);
    assert.ok(labels[0].top >= 32 && labels[0].top <= 288);
    assert.equal(context.pickSpawn(20, 40).name, 'AdZ_Paddock');
    assert.equal(context.pickSpawn(300, 300), null);
    vm.runInContext('state.showSpawns = false;', context);
    assert.equal(context.pickSpawn(20, 40), null);
});

for (const fails of [false, true]) {
    test(`a stale road ${fails ? 'failure' : 'success'} cannot replace a selected spawn marker`, async () => {
        let finish;
        const elements = new Map();
        const context = vm.createContext({
            document: {
                addEventListener() {},
                getElementById(id) {
                    if (!elements.has(id)) elements.set(id, { children: [], style: {}, classList: { toggle() {} } });
                    return elements.get(id);
                },
            },
            fetch: () => new Promise((resolve, reject) => {
                finish = () => fails ? reject(new Error('Stale failure')) : resolve({
                    ok: true, json: async () => ({ id: 4 }),
                });
            }),
        });
        load(context, 'surface_map_dev.js');
        vm.runInContext(`
            state.world = { mapID: 9, roads: [] };
            renderRoadList = () => {};
            renderBase = () => {};
        `, context);
        const loading = context.loadRoad(4);
        vm.runInContext("selectSpawn({ id: '130', name: 'AdZ_Paddock', road_id: 130, xml: '<ent />' });", context);
        finish();
        await loading;
        assert.equal(vm.runInContext('state.spawn.id', context), '130');
        assert.equal(vm.runInContext('state.road', context), null);
        assert.equal(vm.runInContext('state.loadingRoadId', context), null);
        assert.equal(elements.get('status').textContent, '');
    });
}

for (const fails of [false, true]) {
    test(`a stale road ${fails ? 'failure' : 'success'} cannot replace a selected timing arch`, async () => {
        let finish;
        const elements = new Map();
        const context = vm.createContext({
            document: {
                addEventListener() {},
                getElementById(id) {
                    if (!elements.has(id)) elements.set(id, { children: [], style: {}, classList: { toggle() {} } });
                    return elements.get(id);
                },
            },
            fetch: () => new Promise((resolve, reject) => {
                finish = () => fails ? reject(new Error('Stale failure')) : resolve({
                    ok: true, json: async () => ({ id: 4 }),
                });
            }),
        });
        load(context, 'surface_map_dev.js');
        vm.runInContext(`
            state.world = { mapID: 9 };
            renderRoadList = () => {};
            renderBase = () => {};
        `, context);
        const loading = context.loadRoad(4);
        vm.runInContext(`
            selectArch({ id: '1:F', name: 'Sprint', direction: 'F', xml: '<ent />' });
        `, context);
        finish();
        await loading;
        assert.equal(vm.runInContext('state.arch.id', context), '1:F');
        assert.equal(vm.runInContext('state.road', context), null);
        assert.equal(vm.runInContext('state.loadingRoadId', context), null);
        assert.equal(elements.get('status').textContent, '');
    });
}
