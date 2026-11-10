const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function pageContext(fetch = async () => { throw new Error('Unexpected request'); }, storage = new Map()) {
    const elements = new Map();
    const charts = [];
    const listeners = {};
    const warnings = [];
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            value: '', hidden: false, disabled: false, textContent: '', checked: true,
            children: [], files: [], listeners: {},
            get options() { return this.children; },
            appendChild(child) {
                this.children.push(child);
                if (this.children.length === 1 && child.value !== undefined) this.value = child.value;
            },
            replaceChildren(...children) {
                this.children = children;
                this.value = children[0]?.value || '';
            },
            addEventListener(name, callback) {
                (this.listeners[name] ||= []).push(callback);
            },
            reportValidity() { return true; },
        });
        return elements.get(id);
    }
    const context = vm.createContext({
        document: { getElementById: element, createElement: () => ({}),
            addEventListener(name, callback) { listeners[name] = callback; } },
        Option: function(text, value) { this.text = text; this.value = value; },
        Chart: class { constructor(_canvas, config) { charts.push(config); } destroy() {} },
        FormData: class { append() {} },
        BikeSelector: class {
            constructor(options) { this.onChange = options.onChange; this.config = { frameId: '', wheelId: null, level: 0 }; }
            async load() { this.onChange(); }
            async setConfig(config) {
                for (const [key, value] of Object.entries(config)) {
                    if (value != null) this.config[key] = value;
                }
                this.onChange();
            }
            getConfig() { return this.config; }
            isReady() { return true; }
        },
        localStorage: {
            getItem(key) { return storage.get(key) ?? null; },
            setItem(key, value) { storage.set(key, value); },
        },
        console: { warn(...args) { warnings.push(args); } },
        fetch,
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/workout_route.js'), 'utf8'), context);
    vm.runInContext(`bikeSelector = {
        isReady: () => true,
        getConfig: () => ({ frameId: 'bike', wheelId: 'wheel', level: 2 }),
    };`, context);
    for (const [id, value] of Object.entries({
        riderWeight: '75', riderHeight: '175', ftp: '250', routeSelect: 'route', untargetedPercent: '80',
    })) element(id).value = value;
    return { context, element, charts, storage, warnings, initialize() {
        element('routeSelect').value = '';
        element('worldFilter').children = [{ value: '' }];
        return listeners.DOMContentLoaded();
    } };
}

function plan() {
    return { id: 'plan', name: 'Test', source: 'Test.zwo', duration_sec: 60, ftp_override: null,
        warnings: [], segments: [{ power_low: null, power_high: null }] };
}

test('input panels follow the TT planner order: route, rider, bike', () => {
    const html = fs.readFileSync(path.join(__dirname, '../templates/workout_route.html'), 'utf8');
    const headings = [...html.matchAll(/<h2>(.*?)<\/h2>/g)].slice(0, 3)
        .map(match => match[1].replace(/<span aria-hidden="true">.*?<\/span>\s*/, ''));
    assert.deepEqual(headings, ['Route', 'Rider', 'Bike']);
});

test('workout route reuses TT planner styles and control structure', () => {
    const html = fs.readFileSync(path.join(__dirname, '../templates/workout_route.html'), 'utf8');
    assert.ok(html.includes('href="/static/css/tt_pacing.css"'));
    assert.ok(!html.includes('href="/static/css/workout_analysis.css"'));
    for (const className of ['panels', 'form-group form-check', 'bike-summary', 'bike-summary-row',
        'bike-summary-value weight', 'bike-summary-value cda', 'btn-simulate']) {
        assert.ok(html.includes(`class="${className}"`), `Missing TT planner style: ${className}`);
    }
});
test('built-in and custom plans use the same inputs and mutually exclusive payloads', () => {
    const { context } = pageContext();
    context.showWorkout(plan());
    const builtIn = JSON.parse(JSON.stringify(context.buildRequest()));
    assert.equal(builtIn.workout_id, 'plan');
    assert.ok(!('custom_workout' in builtIn));
    assert.equal(builtIn.ftp_w, 250);
    assert.equal(builtIn.untargeted_percent, 80);
    assert.equal(builtIn.frame_id, 'bike');
    assert.equal(builtIn.upgrade_level, 2);
    context.showWorkout(plan(), true);
    const custom = JSON.parse(JSON.stringify(context.buildRequest()));
    assert.deepEqual(custom.custom_workout, plan());
    assert.ok(!('workout_id' in custom));
});

test('fixed FTP reference and untargeted effort assumptions are visible', () => {
    const { context, element } = pageContext();
    const workout = plan();
    workout.ftp_override = 100;
    context.showWorkout(workout);
    assert.equal(element('untargetedGroup').hidden, false);
    assert.equal(element('ftp').disabled, false);
    assert.match(element('workoutWarnings').children[0].textContent, /fixed 100 W/);
    workout.segments = [{ power_low: .5, power_high: .5 }];
    context.showWorkout(workout);
    assert.equal(element('untargetedGroup').hidden, true);
    assert.equal(element('untargetedPercent').disabled, true);
    element('untargetedPercent').value = '-10';
    assert.ok(!('untargeted_percent' in context.buildRequest()));
});

test('profile FTP loads automatically but does not overwrite edits made during fetch', async () => {
    let resolve;
    const { context, element } = pageContext(() => new Promise(done => { resolve = done; }));
    const pending = context.loadProfile();
    element('ftp').value = '321';
    vm.runInContext("editedProfileFields.add('ftp');", context);
    resolve({ ok: true, json: async () => ({ ftp_w: 240, height_cm: 180, weight_kg: 80 }) });
    await pending;
    assert.equal(element('ftp').value, '321');
    assert.equal(element('riderWeight').value, 80);
    assert.equal(element('riderHeight').value, 180);
    const loaded = pageContext(async () => ({
        ok: true, json: async () => ({ ftp_w: 275, height_cm: 175, weight_kg: 75 }),
    }));
    await loaded.context.loadProfile();
    assert.equal(loaded.element('ftp').value, 275);
});

test('upload replaces the selected built-in plan and uses the existing uploader', async () => {
    const calls = [];
    const workout = plan();
    const { context, element } = pageContext(async (url, options) => {
        calls.push({ url, options });
        return { ok: true, json: async () => workout };
    });
    element('workoutSelect').value = 'old';
    element('workoutFile').files = [{}];
    await context.uploadWorkout();
    assert.equal(calls[0].url, '/api/workouts/upload');
    assert.equal(calls[0].options.method, 'POST');
    assert.equal(element('workoutSelect').value, '');
    assert.equal(context.buildRequest().custom_workout.id, 'plan');
    assert.equal(element('uploadBtn').disabled, false);
});

test('late workout fetches cannot replace a newer selection', async () => {
    const pending = [];
    const { context, element } = pageContext(() => new Promise(resolve => pending.push(resolve)));
    element('workoutSelect').value = 'first';
    const first = context.selectBuiltIn();
    element('workoutSelect').value = 'second';
    const second = context.selectBuiltIn();
    pending[1]({ ok: true, json: async () => ({ ...plan(), id: 'second' }) });
    await second;
    pending[0]({ ok: true, json: async () => ({ ...plan(), id: 'first' }) });
    await first;
    assert.equal(context.buildRequest().workout_id, 'second');
});

test('failed uploads clear stale plans and show the server error', async () => {
    const { context, element } = pageContext(async () => ({
        ok: false, json: async () => ({ error: 'Invalid workout file' }),
    }));
    context.showWorkout(plan());
    element('workoutFile').files = [{}];
    await context.uploadWorkout();
    assert.equal(element('error').textContent, 'Invalid workout file');
    assert.equal(element('estimateBtn').disabled, true);
});

test('results describe completion and retain raw chart data without explicit clipping', () => {
    const { context, element, charts } = pageContext();
    const result = {
        completed: true, finish_time_sec: 30, workout_time_remaining_sec: 30,
        workout_duration_sec: 60, route_distance_km: .2, route_name: 'Route',
        workout_name: 'Workout', rider_ftp_w: 250, reference_ftp_w: 250,
        include_leadin: true, warnings: [],
        profile: { time_sec: [0, 30], power_w: [0, 5000], distance_km: [0, .2] },
    };
    context.displayResult(result);
    assert.match(element('resultSummary').textContent, /0:00:30/);
    assert.match(element('resultTitle').textContent, /Yes/);
    assert.equal(charts[0].data.datasets[0].data[1].y, 5000);
    assert.equal(charts[0].data.datasets[0].stepped, 'after');
    assert.ok(!('max' in charts[0].options.scales.power));
    assert.ok(!('min' in charts[0].options.scales.power));
    context.displayResult({ ...result, completed: false, distance_covered_km: .1, distance_remaining_km: .1 });
    assert.match(element('resultSummary').textContent, /0.10 km still to go/);
    context.invalidateEstimate();
    assert.equal(element('results').hidden, true);
});

test('responses for inputs edited while calculating are discarded', async () => {
    let resolve;
    const { context, element, charts } = pageContext(() => new Promise(done => { resolve = done; }));
    context.showWorkout(plan());
    const pending = context.runEstimate({ preventDefault() {} });
    context.invalidateEstimate();
    resolve({ ok: true, json: async () => ({}) });
    await pending;
    assert.equal(charts.length, 0);
    assert.equal(element('estimateBtn').disabled, false);
});

function cachedSettings() {
    return {
        riderWeight: '81.25', riderHeight: '183', ftp: '321.5',
        untargetedPercent: '92', includeLeadin: false,
        world: 'WATOPIA', routeId: 'route', frameId: 'saved-bike', wheelId: 'saved-wheel',
        level: 4, workoutSearch: 'Test', workoutId: 'plan',
    };
}

function catalogResponse(url) {
    const data = {
        '/api/my_profile': { weight_kg: 70, height_cm: 170, ftp_w: 200 },
        '/api/tt_pacing/routes?mode=free_ride': { routes: [
            { id: 'route', world: 'WATOPIA', world_name: 'Watopia', name: 'Test route',
                distance_km: 10, ascent_m: 50, leadin_distance_km: 1, leadin_ascent_m: 5 },
        ] },
        '/api/workouts': { workouts: [plan()] },
        '/api/workouts/plan': plan(),
        '/api/workouts/upload': plan(),
    }[url];
    assert.ok(data, `Unexpected request: ${url}`);
    return { ok: true, json: async () => data };
}

test('reload restores rider overrides, route, bike, built-in workout and power assumptions', async () => {
    const settings = cachedSettings();
    const storage = new Map([['workoutRouteSettings', JSON.stringify(settings)]]);
    const { context, element, initialize } = pageContext(catalogResponse, storage);
    await initialize();
    for (const id of ['riderWeight', 'riderHeight', 'ftp', 'untargetedPercent', 'workoutSearch']) {
        assert.equal(element(id).value, settings[id]);
    }
    assert.equal(element('includeLeadin').checked, false);
    assert.equal(element('worldFilter').value, 'WATOPIA');
    assert.equal(element('routeSelect').value, 'route');
    assert.equal(element('workoutSelect').value, 'plan');
    const body = context.buildRequest();
    assert.equal(body.ftp_w, 321.5);
    assert.equal(body.frame_id, 'saved-bike');
    assert.equal(body.wheel_id, 'saved-wheel');
    assert.equal(body.upgrade_level, 4);
    assert.equal(body.workout_id, 'plan');
    assert.equal(body.untargeted_percent, 92);
    assert.equal(element('estimateBtn').disabled, false);
    assert.deepEqual(JSON.parse(storage.get('workoutRouteSettings')), settings);
});

test('input and bike changes are saved automatically and survive another page load', async () => {
    const storage = new Map([['workoutRouteSettings', JSON.stringify(cachedSettings())]]);
    const first = pageContext(catalogResponse, storage);
    await first.initialize();
    first.element('ftp').value = '9999';
    for (const callback of first.element('estimateForm').listeners.input) callback();
    await vm.runInContext("bikeSelector.setConfig({ frameId: 'changed-bike', level: 5 });", first.context);
    const next = pageContext(catalogResponse, storage);
    await next.initialize();
    assert.equal(next.element('ftp').value, '9999');
    assert.equal(next.context.buildRequest().frame_id, 'changed-bike');
    assert.equal(next.context.buildRequest().upgrade_level, 5);
});

test('custom workout contents and results are never cached and require re-upload after reload', async () => {
    const storage = new Map([['workoutRouteSettings', JSON.stringify(cachedSettings())]]);
    const first = pageContext(catalogResponse, storage);
    await first.initialize();
    first.element('workoutFile').files = [{}];
    await first.context.uploadWorkout();
    const stored = JSON.parse(storage.get('workoutRouteSettings'));
    assert.equal(stored.workoutId, '');
    assert.ok(!storage.get('workoutRouteSettings').includes('segments'));
    assert.ok(!storage.get('workoutRouteSettings').includes('Test.zwo'));
    assert.ok(!('custom_workout' in stored));
    assert.ok(!('profile' in stored));
    const next = pageContext(catalogResponse, storage);
    await next.initialize();
    assert.equal(vm.runInContext('selectedWorkout', next.context), null);
    assert.equal(next.element('estimateBtn').disabled, true);
    assert.equal(next.element('ftp').value, '321.5');
});

test('catalog failures do not erase cached choices for the unavailable data', async () => {
    const settings = cachedSettings();
    const storage = new Map([['workoutRouteSettings', JSON.stringify(settings)]]);
    const page = pageContext(url => {
        if (url === '/api/tt_pacing/routes?mode=free_ride' || url === '/api/workouts') {
            return { ok: false, json: async () => ({ error: 'Catalog unavailable' }) };
        }
        return catalogResponse(url);
    }, storage);
    await page.initialize();
    const cached = JSON.parse(storage.get('workoutRouteSettings'));
    assert.equal(cached.routeId, settings.routeId);
    assert.equal(cached.workoutId, settings.workoutId);
    assert.match(page.element('error').textContent, /Catalog unavailable/);
});

test('a failed restored workout request remains visible after initialization', async () => {
    const storage = new Map([['workoutRouteSettings', JSON.stringify(cachedSettings())]]);
    const page = pageContext(url => url === '/api/workouts/plan'
        ? { ok: false, json: async () => ({ error: 'Workout detail unavailable' }) }
        : catalogResponse(url), storage);
    await page.initialize();
    assert.match(page.element('error').textContent, /Workout detail unavailable/);
    assert.equal(page.element('estimateBtn').disabled, true);
});

test('removed route and workout IDs are not selected from stale storage', async () => {
    const settings = { ...cachedSettings(), routeId: 'removed', workoutId: 'removed' };
    const storage = new Map([['workoutRouteSettings', JSON.stringify(settings)]]);
    const page = pageContext(catalogResponse, storage);
    await page.initialize();
    assert.equal(page.element('routeSelect').value, '');
    assert.equal(page.element('workoutSelect').value, '');
    assert.equal(page.element('estimateBtn').disabled, true);
});

test('corrupt or unavailable storage reports a warning but the page still initializes', async () => {
    for (const value of ['not json', 'null', '[]']) {
        const page = pageContext(catalogResponse, new Map([['workoutRouteSettings', value]]));
        await page.initialize();
        assert.match(page.element('storageStatus').textContent, /could not be loaded or saved/);
        assert.equal(page.warnings.length, 1);
    }
    const page = pageContext(catalogResponse);
    page.context.localStorage.getItem = () => { throw new Error('Storage disabled'); };
    page.context.localStorage.setItem = () => { throw new Error('Quota exceeded'); };
    await page.initialize();
    assert.equal(page.element('ftp').value, 200);
    assert.match(page.element('storageStatus').textContent, /could not be loaded or saved/);
    assert.equal(page.warnings.length, 2);
});

test('initial catalog loading does not overwrite cached choices or in-flight edits', async () => {
    const settings = cachedSettings();
    const serialized = JSON.stringify(settings);
    const storage = new Map([['workoutRouteSettings', serialized]]);
    let resolve;
    const page = pageContext(url => url === '/api/tt_pacing/routes?mode=free_ride'
        ? new Promise(done => { resolve = done; }) : catalogResponse(url), storage);
    const pending = page.initialize();
    page.element('ftp').value = '350';
    for (const callback of page.element('ftp').listeners.input) callback();
    for (const callback of page.element('estimateForm').listeners.input) callback();
    assert.equal(storage.get('workoutRouteSettings'), serialized);
    resolve(catalogResponse('/api/tt_pacing/routes?mode=free_ride'));
    await pending;
    assert.equal(page.element('ftp').value, '350');
    assert.equal(JSON.parse(storage.get('workoutRouteSettings')).ftp, '350');
});
