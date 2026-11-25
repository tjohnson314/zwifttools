const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

async function harness() {
    const {MODEL} = await import('../mods/sauce_mod_power_breakdown/bike-model.mjs');
    const {createBikeResolver} = await import('../mods/sauce_mod_power_breakdown/bike-setup.mjs');
    const source = (await fs.readFile(path.join(__dirname, '..', 'mods',
        'sauce_mod_power_breakdown', 'power-breakdown.mjs'), 'utf8'))
        .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
    function element() {
        const children = new Map();
        return {textContent: '', style: {}, appendChild() {},
            querySelector(selector) {
                if (!children.has(selector)) children.set(selector, element());
                return children.get(selector);
            }};
    }
    const fields = element();
    let update;
    const timers = new Map();
    const cleared = [];
    let pagehide;
    const rpc = {
        async getAthlete() { return ride().athlete; },
        async getGameConnectionStatus() { return {connected: true}; },
        async getWorkoutStatus() { return null; },
        async getGroupsData() { return [{athletes: [{athleteId: 1}]}]; },
    };
    const sandbox = vm.createContext({
        MODEL, createBikeResolver,
        Surface: {preload() {}, lookupStyle() { return {style: 'NORMAL'}; }},
        Common: {enableSentry() {}, settingsStore: {
            setDefault() {}, get(key) { return key === 'physicalSpeed' ? false : undefined; },
            addEventListener() {},
        }, subscribe(event, callback) { update = callback; }, rpc},
        document: {querySelector: selector => fields.querySelector(selector), createElement: element},
        setInterval(callback, delay) { timers.set(delay, callback); return delay; },
        clearInterval(id) { cleared.push(id); },
        addEventListener(event, callback) { if (event === 'pagehide') pagehide = callback; },
        console: {error() {}},
    });
    vm.runInContext(source, sandbox);
    vm.runInContext('main()', sandbox);
    await new Promise(resolve => setImmediate(resolve));
    return {
        sandbox, rpc, update, fields, timers, cleared, pagehide,
        detects: vm.runInContext('isSupertucking', sandbox),
        refresh: vm.runInContext('refreshSupertuckContext', sandbox),
        values: () => vm.runInContext('({...smVals})', sandbox),
        draftLabel: () => vm.runInContext('rowEls.draft.label.textContent', sandbox),
    };
}

function ride(overrides = {}) {
    return {athleteId: 1, athlete: {weight: 75, height: 183, equipment: {cycling: {
        frame: {name: 'Specialized Tarmac SL8'},
        wheelFront: {name: 'Princeton Wake 6560 Lava', imageName: 'Wheel_PrincetonCarbonWorksWake6560Red'},
        wheelRear: {name: 'Princeton Wake 6560 Lava', imageName: 'Wheel_PrincetonCarbonWorksWake6560Red'},
    }}}, state: {worldTime: 1000, courseId: 6, speed: 58, grade: -0.03,
        power: 10, draft: 0, altitude: 10, workoutZone: null, ...overrides}};
}

function context(overrides = {}) {
    return {eventId: null, workoutStatus: null, groups: [{athletes: [{athleteId: 1}]}],
        banded: false, ...overrides};
}

test('supertuck thresholds use raw game speed, grade, power, and exactly zero draft', async () => {
    const {detects} = await harness();
    const setup = {frame: {type: 'Standard'}};
    assert.equal(detects(ride(), setup, context()), true);
    assert.equal(detects(ride({power: 10.999}), setup, context()), true);
    for (const overrides of [
        {speed: 57.999}, {grade: -0.029999}, {power: 11}, {draft: 0.001}, {draft: -1},
        {speed: undefined}, {grade: undefined}, {power: undefined}, {draft: undefined},
        {speed: NaN}, {grade: NaN}, {power: NaN},
    ]) {
        assert.equal(detects(ride(overrides), setup, context()), false, JSON.stringify(overrides));
    }
});

test('equipment profiles refresh every five seconds and update on the next rider sample', async () => {
    const h = await harness();
    const original = ride();
    h.update(original);
    assert.match(h.fields.querySelector('.bike-summary').textContent, /Specialized Tarmac SL8/);
    let calls = 0;
    let resolveProfile;
    const pending = new Promise(resolve => { resolveProfile = resolve; });
    h.rpc.getAthlete = async (id, options) => {
        calls++;
        assert.equal(id, 'self');
        assert.deepEqual(Array.from(options.required), ['equipment']);
        assert.equal(options.refresh, true);
        return pending;
    };
    const refreshed = ride({worldTime: 2000});
    refreshed.athlete.equipment.cycling.frame = {name: 'Zwift TT'};
    const refresh = h.timers.get(5000)();
    await h.timers.get(5000)();
    assert.equal(calls, 1, 'overlapping profile requests are skipped');
    resolveProfile(refreshed.athlete);
    await refresh;
    h.update(refreshed);
    assert.match(h.fields.querySelector('.bike-summary').textContent, /Zwift TT/);
    assert.equal(h.fields.querySelector('.equipment-status').textContent, '');
    h.pagehide();
    assert.deepEqual(h.cleared.sort(), [1000, 5000]);
});

test('equipment refresh failures and missing profiles are visible and recover on retry', async () => {
    const h = await harness();
    h.update(ride());
    h.rpc.getAthlete = async () => { throw new Error('network down'); };
    await h.timers.get(5000)();
    assert.match(h.fields.querySelector('.equipment-status').textContent, /network down/);
    assert.match(h.fields.querySelector('.bike-summary').textContent, /Specialized Tarmac SL8/);
    h.rpc.getAthlete = async () => null;
    await h.timers.get(5000)();
    assert.match(h.fields.querySelector('.equipment-status').textContent, /Rider equipment unavailable/);
    h.rpc.getAthlete = async () => ride().athlete;
    await h.timers.get(5000)();
    assert.equal(h.fields.querySelector('.equipment-status').textContent, '');
});

test('TT/MTB frames, workouts including free-ride blocks, and unbanded packs cannot supertuck', async () => {
    const {detects} = await harness();
    const setup = {frame: {type: 'Standard'}};
    for (const type of ['TT', 'MTB']) {
        assert.equal(detects(ride(), {frame: {type}}, context()), false);
    }
    for (const workoutStatus of [{info: {}}, {state: {isFinished: false}},
        {state: {isPaused: true, isFinished: false}}]) {
        assert.equal(detects(ride({workoutZone: null}), setup, context({workoutStatus})), false);
    }
    assert.equal(detects(ride({workoutZone: 1}), setup, context()), false);
    assert.equal(detects(ride(), setup, context({workoutStatus: {state: {isFinished: true}}})), true);
    const pack = [{athletes: [{athleteId: 1}, {athleteId: 2}]}];
    assert.equal(detects(ride(), setup, context({groups: pack})), false);
    assert.equal(detects(ride(), setup, context({groups: pack, banded: true})), true);
    assert.equal(detects(ride({draft: 1}), setup, context({groups: pack, banded: true})), false);
    assert.equal(detects(ride(), setup, context({groups: []})), false);
    assert.equal(detects(ride(), setup, null), false);
});

test('supertuck draft is one third of aero drag, balances the residual, and resets on entry/exit', async () => {
    const h = await harness();
    h.update(ride({draft: 70}));
    assert.equal(h.values().draft, 70);
    assert.equal(h.draftLabel(), 'Draft');
    h.update(ride({worldTime: 2000}));
    assert.equal(h.draftLabel(), 'Draft (supertuck)');
    for (const speed of [58, 65]) {
        h.update(ride({worldTime: speed * 1000, speed}));
        const values = h.values();
        assert.ok(values.draft > 0);
        assert.ok(Math.abs(values.draft + values.aero / 3) < 1e-9);
        assert.ok(Math.abs(Object.values(values).reduce((sum, value) => sum + value, 0)) < 1e-9);
    }
    h.update(ride({worldTime: 66000, power: 11, draft: 0}));
    assert.equal(h.draftLabel(), 'Draft');
    assert.equal(h.values().draft, 0);
    h.update(ride({worldTime: 67000, draft: 90}));
    assert.equal(h.draftLabel(), 'Draft');
    assert.equal(h.values().draft, 90 * (1 - Math.exp(-1 / 3)));
});

test('banded event rules are loaded and stale or failed context never produces a supertuck', async () => {
    const h = await harness();
    const data = {...ride(), eventSubgroupId: 123};
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft');
    h.rpc.getGroupsData = async () => [{athletes: [{athleteId: 1}, {athleteId: 2}]}];
    h.rpc.getEventSubgroup = async id => {
        assert.equal(id, 123);
        return {eventId: 456, rulesSet: []};
    };
    h.rpc.getEvent = async id => {
        assert.equal(id, 456);
        return {rulesSet: ['RUBBERBANDING']};
    };
    await h.refresh();
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft (supertuck)');
    h.rpc.getWorkoutStatus = async () => { throw new Error('Game connection unavailable'); };
    await h.refresh();
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft');
    assert.equal(h.values().draft, 0);
    assert.match(h.fields.querySelector('.supertuck-status').textContent, /Game connection unavailable/);
});

test('disabled or disconnected Game Connection reports the remedy and recovers when connected', async () => {
    const h = await harness();
    const data = ride();
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft (supertuck)');

    h.rpc.getGameConnectionStatus = async () => ({connected: false, state: 'disabled'});
    h.rpc.getWorkoutStatus = async () => { throw new Error('Invalid handler name: getWorkoutStatus'); };
    await h.refresh();
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft');
    assert.equal(h.values().draft, 0);
    assert.match(h.fields.querySelector('.supertuck-status').textContent,
        /Enable Game Connection in Sauce Settings > General, then restart Sauce/);

    h.rpc.getGameConnectionStatus = async () => ({connected: false, state: 'waiting'});
    await h.refresh();
    assert.match(h.fields.querySelector('.supertuck-status').textContent, /Game Connection not connected/);

    h.rpc.getGameConnectionStatus = async () => ({connected: true});
    h.rpc.getWorkoutStatus = async () => null;
    await h.refresh();
    h.update(data);
    assert.equal(h.draftLabel(), 'Draft (supertuck)');
    assert.equal(h.fields.querySelector('.supertuck-status').textContent, '');
    assert.ok(Math.abs(h.values().draft + h.values().aero / 3) < 1e-9);
});
