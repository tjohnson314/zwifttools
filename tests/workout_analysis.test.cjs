const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function context(globals = {}) {
    const scope = vm.createContext({ document: { addEventListener() {} }, URL, console, ...globals });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/workout_analysis.js'), 'utf8'), scope);
    return scope;
}

function segment(start, end, low, high = low, ramp = false) {
    return { kind: 'Test', start_sec: start, end_sec: end, power_low: low, power_high: high, ramp };
}

test('exact interval boundaries, negative offsets, intensity, and raw outliers are preserved', () => {
    const scope = context();
    const workout = { segments: [segment(0, 2, 1), segment(2, 4, .5)] };
    const activity = { time_sec: [0, 1, 2, 3], power_watts: [-100, 1000, 0, 100] };
    const analysis = scope.analyzeWorkout(activity, workout, 200, 50, 0);
    assert.equal(analysis.rows[0].actual, 450);
    assert.equal(analysis.rows[0].target, 100);
    assert.equal(analysis.rows[1].actual, 50);
    assert.equal(analysis.rows[1].target, 50);
    assert.equal(analysis.bias, 175);
    assert.equal(analysis.covered, 4);
    const shifted = scope.analyzeWorkout(activity, workout, 200, 100, -1);
    assert.equal(shifted.rows[0].covered, 1);
    assert.equal(shifted.rows[0].actual, -100);
    assert.equal(activity.power_watts[0], -100);
});

test('ramps integrate target over only the observed part of the block', () => {
    const scope = context();
    const analysis = scope.analyzeWorkout({ time_sec: [5, 6], power_watts: [200, 200] },
        { segments: [segment(0, 10, 0, 1, true)] }, 200, 100, 0);
    assert.equal(analysis.rows[0].covered, 2);
    assert.equal(analysis.rows[0].target, 120);
    assert.equal(analysis.rows[0].difference, 80);
    assert.equal(scope.targetBounds(segment(0, 10, 1, .5, true), 10, 200)[0], 100);
});

test('fixed-reference FTP tests do not scale by the rider profile FTP', () => {
    const scope = context();
    const workout = { ftp_override: 100, segments: [segment(0, 2, 1.2)] };
    const result = scope.analyzeWorkout({ time_sec: [0, 1], power_watts: [120, 120] },
        workout, 300, 100, 0);
    assert.equal(result.rows[0].target, 120);
    assert.equal(result.bias, 0);
});

test('ranges use midpoint; free ride remains unscored and includes raw actual power', () => {
    const scope = context();
    const analysis = scope.analyzeWorkout({ time_sec: [0, 1, 2, 3], power_watts: [100, 100, 8000, 0] },
        { segments: [segment(0, 2, .4, .6), segment(2, 4, null)] }, 200, 100, 0);
    assert.equal(analysis.rows[0].target, 100);
    assert.equal(analysis.rows[1].actual, 4000);
    assert.equal(analysis.rows[1].difference, null);
    assert.equal(analysis.scored, 2);
    assert.equal(analysis.covered, 4);
    assert.equal(analysis.bias, 0);
});

test('missing samples and recording gaps do not extend power across missing time', () => {
    const scope = context();
    const activity = { time_sec: [0, 1, 2, 10, 11], power_watts: [100, null, 0, 100, 100] };
    const result = scope.analyzeWorkout(activity, { segments: [segment(0, 12, .5)] }, 200, 100, 0);
    assert.equal(result.covered, 4);
    assert.equal(result.rows[0].actual, 75);
    const points = scope.actualSeries(activity);
    assert.equal(points.length, 6);
    assert.equal(points[3].y, null);
    assert.equal(points[4].x, 10 / 60);
});

test('no overlap is reported as missing rather than zero watts', () => {
    const scope = context();
    const result = scope.analyzeWorkout({ time_sec: [0, 1], power_watts: [200, 200] },
        { segments: [segment(0, 2, 1)] }, 200, 100, 100);
    assert.equal(result.covered, 0);
    assert.equal(result.bias, null);
    assert.equal(result.rows[0].actual, null);
});

test('planned chart includes exact ramp endpoints and gaps for free ride', () => {
    const scope = context();
    const result = scope.plannedSeries({ segments: [
        segment(0, 60, .3, 1, true), segment(60, 90, null), segment(90, 120, 1, 1.2),
    ] }, 200, 30);
    assert.equal(result.lower[0].x, .5);
    assert.equal(result.lower[0].y, 60);
    assert.equal(result.lower.find(point => point.x === 1.5).y, 200);
    assert.ok(result.lower.some(point => point.x === 1.5 && point.y === null));
    assert.equal(result.upper.find(point => point.x === 2 && point.y !== null).y, 240);
    assert.ok(result.lower.every(point => point.y === null || point.y % 5 === 0));
    assert.ok(result.upper.every(point => point.y === null || point.y % 5 === 0));
});

test('targets round to nearest five only after FTP and intensity scaling', () => {
    const scope = context();
    const workout = { segments: [segment(0, 1, 1.1), segment(1, 2, .5, .6)] };
    const activity = { time_sec: [0, 1], power_watts: [247.3, -12.2] };
    const result = scope.analyzeWorkout(activity, workout, 250, 90, 0);
    assert.equal(result.rows[0].target, 250);
    assert.equal(result.rows[0].actual, 247.3);
    assert.ok(Math.abs(result.rows[0].difference + 2.7) < 1e-10);
    assert.equal(scope.targetBounds(workout.segments[1], 1, 253 * .9).join(','), '115,135');
    assert.equal(scope.targetBounds(segment(0, 1, 1), 0, 252.49)[0], 250);
    assert.equal(scope.targetBounds(segment(0, 1, 1), 0, 252.5)[0], 255);
    assert.equal(scope.targetBounds(segment(0, 1, 1), 0, 252.51)[0], 255);
    assert.equal(scope.targetBounds(segment(0, 1, null), 0, 250), null);
    assert.equal(activity.power_watts[1], -12.2);
    assert.equal(workout.segments[0].power_low, 1.1);
    const fixed = scope.analyzeWorkout(activity,
        { ftp_override: 100, segments: [segment(0, 1, 1.23)] }, 300, 100, 0);
    assert.equal(fixed.rows[0].target, 125);
});

test('rounded ramp integration splits at thresholds, including partial coverage and cooldowns', () => {
    const scope = context();
    const activity = { time_sec: [0, 1.2], power_watts: [10, null] };
    for (const [low, high, work] of [[0, 1, 3.5], [1, 0, 8.5]]) {
        const workout = { segments: [segment(0, 2, low, high, true)] };
        const result = scope.analyzeWorkout(activity, workout, 10, 100, 0);
        assert.equal(result.covered, 1.2);
        assert.ok(Math.abs(result.rows[0].target - work / 1.2) < 1e-10);
        const chart = scope.plannedSeries(workout, 10, 0);
        const crossing = .5 / 60;
        const values = chart.lower.filter(point => point.x === crossing).map(point => point.y);
        assert.equal(values.join(','), low === 0 ? '0,5' : '10,5');
    }
});

test('activity URLs accept only numeric IDs on Zwift hosts', () => {
    const scope = context();
    for (const input of ['123', ' 123 ', 'https://www.zwift.com/activity/123?x=1',
        'https://my.zwift.com/activities/123/']) {
        assert.equal(scope.parseActivityId(input), '123');
    }
    for (const input of ['a123', 'https://example.org/activity/123', 'https://zwift.com.evil.test/activity/123',
        'https://www.zwift.com/activity/+']) {
        assert.equal(scope.parseActivityId(input), null);
    }
});

test('late profile response never overwrites a manually edited FTP', async () => {
    const elements = { ftp: { value: '250' }, ftpStatus: { textContent: '' } };
    let finish;
    const scope = context({
        document: { addEventListener() {}, getElementById(id) { return elements[id]; } },
        fetch: () => new Promise(resolve => { finish = resolve; }),
    });
    vm.runInContext('renderComparison = () => {};', scope);
    const loading = scope.loadProfileFtp();
    vm.runInContext('ftpEdited = true;', scope);
    elements.ftp.value = '310';
    finish({ ok: true, json: async () => ({ ftp_w: 275 }) });
    await loading;
    assert.equal(elements.ftp.value, '310');
    assert.match(elements.ftpStatus.textContent, /275 W/);
});

test('profile FTP prefills and missing FTP explicitly requests manual entry', async () => {
    const elements = { ftp: { value: '' }, ftpStatus: { textContent: '' } };
    let ftp = 280;
    const scope = context({
        document: { addEventListener() {}, getElementById(id) { return elements[id]; } },
        fetch: async () => ({ ok: true, json: async () => ({ ftp_w: ftp }) }),
    });
    vm.runInContext('renderComparison = () => {};', scope);
    await scope.loadProfileFtp();
    assert.equal(elements.ftp.value, 280);
    ftp = null;
    await scope.loadProfileFtp();
    assert.match(elements.ftpStatus.textContent, /manually/);
});

function uploadContext(fetch) {
    const elements = Object.fromEntries([
        'ftp', 'planError', 'planWarnings', 'uploadStatus', 'uploadWorkout',
        'workoutSource', 'workoutFile', 'builtinControls', 'customControls', 'workoutSelect',
    ].map(id => [id, { textContent: '', value: '', disabled: false, hidden: false }]));
    elements.workoutSource.value = 'custom';
    elements.workoutFile.files = [{ name: 'my-workout.zwo', size: 100 }];
    const scope = context({
        document: { addEventListener() {}, getElementById(id) { return elements[id]; } },
        FormData: class {
            constructor() { this.values = {}; }
            append(key, value) { this.values[key] = value; }
        },
        fetch,
    });
    vm.runInContext('renderComparison = () => {};', scope);
    return { scope, elements };
}

test('custom upload uses shared plan state and preserves FTP, intensity and rounding behavior', async () => {
    const workout = { name: 'Custom', duration_sec: 10, ftp_override: null, warnings: ['Test warning'],
        segments: [segment(0, 10, .731)] };
    const { scope, elements } = uploadContext(async (url, options) => {
        assert.equal(url, '/api/workouts/upload');
        assert.equal(options.method, 'POST');
        assert.equal(options.body.values.file.name, 'my-workout.zwo');
        return { ok: true, json: async () => workout };
    });
    await scope.uploadWorkout({ preventDefault() {} });
    assert.equal(vm.runInContext('selectedWorkout.name', scope), 'Custom');
    assert.equal(elements.ftp.disabled, false);
    assert.match(elements.uploadStatus.textContent, /Custom \(0:10\) loaded/);
    assert.equal(elements.planWarnings.textContent, 'Test warning');
    assert.equal(elements.uploadWorkout.disabled, false);
    const result = scope.analyzeWorkout({ time_sec: [0], power_watts: [201.7] }, workout, 250, 100, 0);
    assert.equal(result.rows[0].target, 185);
    assert.equal(result.rows[0].actual, 201.7);
});

test('custom upload failures clear the previous plan and show the server error', async () => {
    const { scope, elements } = uploadContext(async () => ({
        ok: false, status: 400, json: async () => ({ error: 'Invalid workout XML' }),
    }));
    vm.runInContext('customWorkout = { name: "Old" }; selectedWorkout = customWorkout;', scope);
    await scope.uploadWorkout({ preventDefault() {} });
    assert.equal(vm.runInContext('selectedWorkout', scope), null);
    assert.equal(vm.runInContext('customWorkout', scope), null);
    assert.equal(elements.planError.textContent, 'Invalid workout XML');
    assert.equal(elements.uploadWorkout.disabled, false);
});

test('custom upload rejects oversized and non-zwo files without a request', async () => {
    const { scope, elements } = uploadContext(() => { throw new Error('Unexpected upload'); });
    elements.workoutFile.files = [{ name: 'custom.fit', size: 20 }];
    await scope.uploadWorkout({ preventDefault() {} });
    assert.match(elements.planError.textContent, /\.zwo/);
    elements.workoutFile.files = [{ name: 'custom.zwo', size: 1024 * 1024 + 1 }];
    await scope.uploadWorkout({ preventDefault() {} });
    assert.match(elements.planError.textContent, /1 MiB/);
});

test('switching to custom invalidates pending built-in requests and restores the last uploaded plan', async () => {
    let finishBuiltin;
    const { scope, elements } = uploadContext(url => {
        assert.match(url, /^\/api\/workouts\//);
        return new Promise(resolve => { finishBuiltin = resolve; });
    });
    elements.workoutSelect.value = 'official-id';
    const pending = scope.chooseWorkout();
    vm.runInContext('customWorkout = {name:"Custom", ftp_override:100, warnings:[]};', scope);
    scope.changeWorkoutSource();
    assert.equal(elements.builtinControls.hidden, true);
    assert.equal(elements.customControls.hidden, false);
    assert.equal(elements.ftp.disabled, true);
    finishBuiltin({ ok: true, json: async () => ({ name: 'Official', ftp_override: null, warnings: [] }) });
    await pending;
    assert.equal(vm.runInContext('selectedWorkout.name', scope), 'Custom');
});
