const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function dashboardContext(globals = {}) {
    const context = vm.createContext({ document: { addEventListener() {} }, ...globals });
    const source = fs.readFileSync(path.join(__dirname, '../static/js/power_dashboard.js'), 'utf8');
    vm.runInContext(source.replace(/\ninit\(\);\s*$/, '\n'), context);
    return context;
}

test('Morton fit recovers known parameters with a fixed peak power', () => {
    const context = dashboardContext();
    const cp = 300;
    const wPrime = 20000;
    const peakPower = 1200;
    const timeConstant = wPrime / (peakPower - cp);
    const points = [300, 600, 1200].map(duration => ({
        duration, power: cp + wPrime / (duration + timeConstant),
    }));
    const fit = context.fitMortonModel(points, peakPower);
    assert.ok(Math.abs(fit.cp - cp) < 0.001);
    assert.ok(Math.abs(fit.wPrime - wPrime) < 0.1);
    assert.equal(fit.peakPower, peakPower);
});

test('Morton overlay uses the visible 1-second peak and covers one second through one hour', () => {
    const context = dashboardContext();
    vm.runInContext(`
        durationsSec = [1, 5, 60, 300, 600, 1200, 3600, 7200];
        activities = [
            { activity_id: 'visible', peak_watts: durationsSec.map(duration => duration === 1 ? 1200 : 300 + 20000 / (duration + 20000 / 900)) },
            { activity_id: 'hidden', peak_watts: durationsSec.map(() => 2000) },
        ];
        dividers = new Set([1]);
        hiddenPartitions.add(partitionKey(1));
        showMortonModel = true;
    `, context);
    const dataset = context.buildMortonDataset();
    assert.equal(dataset.label, 'Morton model (1s peak; 5/10/20 min fit)');
    assert.equal(dataset.data.length, 8);
    assert.ok(Math.abs(dataset.data[0].y - (300 + 20000 / (1 + 20000 / 900))) < 0.001);
    assert.ok(Math.abs(dataset.data[1].y - (300 + 20000 / (5 + 20000 / 900))) < 0.001);
    assert.ok(Math.abs(dataset.data[6].y - (300 + 20000 / (3600 + 20000 / 900))) < 0.001);
    assert.equal(dataset.data[7].y, null);
    assert.equal(vm.runInContext('activities[0].peak_watts[0]', context), 1200);
    vm.runInContext('powerUnit = "W/kg"; activities[0].weight_kg = 75;', context);
    assert.ok(Math.abs(context.buildMortonDataset().data[6].y - dataset.data[6].y / 75) < 0.001);
    vm.runInContext('activities[0].peak_watts[0] = null;', context);
    assert.equal(context.buildMortonDataset(), null);
});

test('Morton fit omits unavailable or incompatible inputs without clipping raw data', () => {
    const context = dashboardContext();
    assert.equal(context.fitMortonModel([{ duration: 300, power: 400 }, { duration: 600, power: null }], 1200), null);
    assert.equal(context.fitMortonModel([{ duration: 300, power: 400 }, { duration: 600, power: 350 }], null), null);
    assert.equal(context.fitMortonModel([{ duration: 300, power: 1400 }, { duration: 600, power: 350 }], 1200), null);
});

test('progressive loading updates only the changed row and preserves partition cells', () => {
    const cells = Array.from({ length: 9 }, () => ({
        textContent: 'unchanged', innerHTML: '',
        insertAdjacentHTML(position, html) { this.innerHTML += html; },
    }));
    const row = { cells };
    const context = dashboardContext({
        document: { getElementById(id) {
            assert.equal(id, 'power-activity-1');
            return row;
        } },
    });
    vm.runInContext(`
        activities = [{}, {
            activity_id: 'race', name: 'Zwift - Test Race in Watopia',
            status: 'cached', avg_power: 250, avg_hr: 160,
            is_race: true, event_id: 'event',
        }];
        renderActivityList = () => { throw new Error('Full table rebuild'); };
    `, context);
    context.updateActivityRow(1);
    assert.equal(cells[0].textContent, 'unchanged');
    assert.equal(cells[1].textContent, 'unchanged');
    assert.equal(cells[2].textContent, 'Test Race');
    assert.match(cells[2].innerHTML, /badge-race/);
    assert.equal(cells[5].textContent, '250 W');
    assert.equal(cells[6].textContent, '160 bpm');
    assert.match(cells[7].innerHTML, /Cached/);
    assert.match(cells[8].innerHTML, /race-replay/);
    assert.match(cells[8].innerHTML, /events.php\?zid=event/);
});

test('large cached loads yield between bounded batches without fetching or changing raw values', async () => {
    const timers = [];
    const updated = [];
    const summaries = [];
    const frames = [];
    let now = 0;
    const context = dashboardContext({
        setTimeout(callback) { timers.push(callback); },
        performance: { now() { return now++; } },
        requestAnimationFrame(callback) { frames.push(callback); },
        localStorage: { getItem() {
            return JSON.stringify({ durations_sec: [1, 300], data: {
                has_power: true, peak_watts: [2500, -20], avg_power: -10,
                avg_hr: 160, weight_kg: 75, is_race: true, event_id: 'event',
            } });
        } },
        fetch() { throw new Error('Cached activities must not fetch'); },
        updated, summaries,
    });
    vm.runInContext(`
        durationsSec = [1, 300];
        activities = Array.from({ length: 250 }, (_, index) => ({ activity_id: String(index), status: 'pending' }));
        updateActivityRow = index => updated.push(index);
        updateLoadSummary = loaded => summaries.push(loaded);
    `, context);
    const loading = context.loadAllActivities();
    assert.equal(updated.length, 0);
    let yields = 0;
    while (timers.length) {
        const before = updated.length;
        timers.shift()();
        await Promise.resolve();
        assert.ok(updated.length - before <= 50);
        yields += 1;
    }
    await loading;
    assert.ok(yields > 1);
    assert.equal(updated.length, 250);
    assert.equal(new Set(updated).size, 250);
    assert.equal(summaries.at(-1), 250);
    assert.equal(frames.length, 1);
    assert.equal(vm.runInContext('activities.every(activity => activity.status === "cached")', context), true);
    assert.deepEqual(Array.from(vm.runInContext('activities[249].peak_watts', context)), [2500, -20]);
    assert.equal(vm.runInContext('activities[249].avg_power', context), -10);
});

test('progressive loading still fetches cache misses and continues after failed activities', async () => {
    const requests = [];
    const writes = [];
    const updates = [];
    const summaries = [];
    const context = dashboardContext({
        setTimeout(callback) { callback(); },
        performance: { now() { return 0; } },
        requestAnimationFrame() {},
        localStorage: {
            getItem(key) {
                if (key.endsWith('nodata')) {
                    return JSON.stringify({ durations_sec: [1], data: { has_power: false, peak_watts: [null] } });
                }
                return key.endsWith('invalid') ? '{broken' : null;
            },
            setItem(key, value) { writes.push([key, JSON.parse(value)]); },
        },
        async fetch(url) {
            requests.push(url);
            return {
                status: url.endsWith('failed') ? 500 : 200,
                ok: !url.endsWith('failed'),
                async json() { return { has_power: true, peak_watts: [900], avg_power: 250 }; },
            };
        },
        updates, summaries,
    });
    vm.runInContext(`
        durationsSec = [1];
        activities = ['nodata', 'miss', 'failed', 'invalid'].map(activity_id => ({ activity_id, status: 'pending' }));
        updateActivityRow = index => updates.push([index, activities[index].status]);
        updateLoadSummary = loaded => summaries.push(loaded);
    `, context);
    await context.loadAllActivities();
    assert.deepEqual(Array.from(vm.runInContext('activities.map(activity => activity.status)', context)),
        ['cached-nodata', 'loaded', 'error', 'loaded']);
    assert.equal(requests.length, 3);
    assert.equal(writes.length, 2);
    assert.deepEqual(updates.map(update => Array.from(update)), [
        [0, 'cached-nodata'], [1, 'loading'], [1, 'loaded'],
        [2, 'loading'], [2, 'error'], [3, 'loading'], [3, 'loaded'],
    ]);
    assert.deepEqual(summaries, [4]);
});