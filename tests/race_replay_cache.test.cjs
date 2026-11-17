const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('shared replay controls initialize on Ride Compare without optional race-analysis panels', () => {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
    const rideCompare = fs.readFileSync(path.join(__dirname, '../templates/ride_compare.html'), 'utf8');
    const ids = new Set([...rideCompare.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
    const context = vm.createContext({
        document: {
            addEventListener() {},
            getElementById: id => ids.has(id) ? { addEventListener() {} } : null,
            querySelectorAll: () => [],
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(source, context);
    assert.doesNotThrow(() => context.bindEvents());
    assert.doesNotThrow(() => context.initAnalysisPanel({}, []));
});

for (const internal of [false, true]) {
    test(`saved replay requests roster refresh unless just fetched (internal=${internal})`, async () => {
        const requests = [];
        const context = vm.createContext({
            raceOpInProgress: false,
            setRaceBusy() {},
            showLoading() {},
            hideLoading() {},
            showStatus() {},
            async fetch(url, options) {
                requests.push({ url, body: JSON.parse(options.body) });
                return { ok: false, status: 502, async json() { return { error: 'Stop after load' }; } };
            },
        });
        const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
        vm.runInContext(source.slice(
            source.indexOf('async function loadRaceById('),
            source.indexOf('function initRaceData('),
        ), context);
        await context.loadRaceById('race_data_456', internal);
        assert.deepEqual(requests, [{
            url: '/api/race/load',
            body: { race_id: 'race_data_456', refresh_participants: !internal },
        }]);
    });
}

test('Ride Compare loads its fetched data without refreshing a race roster and preserves partial warning', async () => {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/ride_compare.js'), 'utf8');
    const events = 'data: {"success":true,"race_id":"ride_compare_test","message":"Loaded: 1 rides for comparison; 1 unavailable."}\n\n';
    const statuses = [];
    const calls = [];
    const context = vm.createContext({
        document: { getElementById: () => ({ value: '1,2' }) },
        window: { location: 'http://localhost/ride-compare', history: { replaceState() {} } },
        URL, TextDecoder,
        showStatus: message => statuses.push(message),
        showLoading() {}, hideLoading() {}, showProgress() {}, hideProgress() {},
        async loadRaceById(...args) { calls.push(args); return true; },
        async fetch() {
            let sent = false;
            return { ok: true, body: { getReader: () => ({
                async read() {
                    if (sent) return { done: true };
                    sent = true;
                    return { done: false, value: new TextEncoder().encode(events) };
                },
            }) } };
        },
        TextEncoder,
    });
    vm.runInContext(source.slice(source.indexOf('async function fetchRideCompare()')), context);
    await context.fetchRideCompare();
    assert.deepEqual(calls, [['ride_compare_test', true]]);
    assert.match(statuses.at(-1), /1 unavailable/);
});

test('race analysis only applies a saved model matching the analyzed rider ID', () => {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
    const elements = new Map();
    const models = new Map([
        ['criticalPowerModelV1', JSON.stringify({ cp: 999, wPrime: 20000 })],
        ['criticalPowerModelV2:9007199254740993', JSON.stringify({ cp: 310, wPrime: 21000 })],
        ['criticalPowerModelV2:2', JSON.stringify({ cp: 220, wPrime: 15000 })],
    ]);
    function element(id) {
        if (!elements.has(id)) elements.set(id, { value: '', style: {}, addEventListener() {} });
        return elements.get(id);
    }
    const context = vm.createContext({
        document: { getElementById: element, addEventListener() {}, querySelectorAll: () => [] },
        window: { addEventListener() {} },
        localStorage: { getItem: key => models.get(key) ?? null },
        console,
    });
    vm.runInContext(source, context);
    vm.runInContext(`
        raceData = { source_activity_id: 'other' };
        riderLookup = {
            1: { zwift_id: '9007199254740993', activity_id: 'other' },
            2: { zwift_id: '2', activity_id: 'other2' },
            3: { zwift_id: null, activity_id: 'other' },
            4: { zwift_id: '3', activity_id: 'other' }
        };
        runAnalysis = () => {};
    `, context);
    context.selectAnalysisRider(1);
    assert.equal(element('analysis-cp').value, 310);
    assert.equal(element('analysis-wprime').value, '21.0');
    context.selectAnalysisRider(2);
    assert.equal(element('analysis-cp').value, 220);
    assert.equal(element('analysis-use-saved-cp').style.display, '');
    context.selectAnalysisRider(3);
    assert.equal(element('analysis-cp').value, '');
    assert.equal(element('analysis-use-saved-cp').style.display, 'none');
    context.selectAnalysisRider(4);
    assert.equal(element('analysis-cp').value, '');
    context.fillSavedCp();
    assert.equal(element('analysis-cp').value, '');
});
