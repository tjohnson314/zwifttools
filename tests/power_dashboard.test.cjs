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

function inspectContext() {
    function element() {
        return {
            children: [], style: {}, textContent: '',
            set innerHTML(html) { this.html = html; this.children = []; },
            get innerHTML() { return this.html || ''; },
            appendChild(child) { this.children.push(child); },
            addEventListener(event, callback) { this[event] = callback; },
        };
    }
    const elements = Object.fromEntries(['partitionMetrics', 'selectedInterval', 'selectedIntervalLabel',
        'criticalPowerToggle'].map(id => [id, element()]));
    const context = dashboardContext({
        document: {
            getElementById(id) { return elements[id]; },
            createElement: element,
            querySelectorAll() { return []; },
        },
    });
    return { context, elements };
}

test('inspect panel reuses activity actions for the winning point in each visible group', () => {
    const { context, elements } = inspectContext();
    vm.runInContext(`
        durationsSec = [1, 300];
        activities = [
            { activity_id: 'ride', name: 'Ride', peak_watts: [900, 200] },
            { activity_id: 'race', name: 'Race', peak_watts: [800, 300], is_race: true, event_id: 'event' },
            { activity_id: 'hidden', name: 'Hidden', peak_watts: [2000, 1000] },
            { activity_id: 'empty', name: 'Empty', peak_watts: [null, null] },
        ];
        dividers = new Set([2, 3]);
        hiddenPartitions.add(partitionKey(2));
    `, context);
    context.updateInspectPanel(1);
    const rows = elements.partitionMetrics.children;
    assert.equal(rows.length, 2);
    const details = rows[0].children[1];
    assert.equal(details.children[1].textContent, 'Race');
    assert.equal(details.children[3].innerHTML, context.actionsCellHtml({
        activity_id: 'race', is_race: true, event_id: 'event',
    }));
    assert.equal(rows[1].children[1].children.length, 3);

    context.updateInspectPanel(0);
    const actions = elements.partitionMetrics.children[0].children[1].children[3].innerHTML;
    assert.match(actions, /bike-comparison\?activity_id=ride/);
    assert.match(actions, /www.zwift.com\/activity\/ride/);
    assert.doesNotMatch(actions, /race-replay|zwiftpower/);
});

test('inspect panel shows the plotted Morton prediction and three parameters in the selected units', () => {
    const { context, elements } = inspectContext();
    vm.runInContext(`
        durationsSec = [1, 300, 600, 1200, 7200];
        activities = [{ activity_id: 'ride', weight_kg: 75,
            peak_watts: durationsSec.map(duration => duration === 1 ? 1200 : 300 + 20000 / (duration + 20000 / 900)) }];
        showMortonModel = true;
        powerCurveChart = { data: { datasets: buildChartDatasets() } };
    `, context);
    context.updateInspectPanel(1);
    function modelDetails() {
        const row = elements.partitionMetrics.children.at(-1);
        assert.equal(row.className, 'partition-metric-row morton-metric-row');
        return {
            prediction: row.children[0].children[1].textContent,
            parameters: row.children[1].children[1].children.flatMap(parameter => {
                assert.equal(parameter.className, 'morton-parameter');
                assert.equal(parameter.children.length, 2);
                return parameter.children.map(child => child.textContent);
            }),
        };
    }
    assert.equal(modelDetails().prediction, context.formatPower(context.buildMortonDataset().data[1].y));
    assert.deepEqual(modelDetails().parameters, ['CP', '300 W', "W'", '20.0 kJ', 'Pmax', '1200 W']);
    context.updateInspectPanel(4);
    assert.equal(modelDetails().prediction, '\u2014');

    vm.runInContext('powerUnit = "W/kg"; powerCurveChart.data.datasets = buildChartDatasets();', context);
    context.updateInspectPanel(1);
    assert.deepEqual(modelDetails().parameters, ['CP', '4.00 W/kg', "W'", '266.7 J/kg', 'Pmax', '16.00 W/kg']);
    vm.runInContext('activities[0].peak_watts[0] = null; powerCurveChart.data.datasets = buildChartDatasets();', context);
    context.updateInspectPanel(1);
    assert.equal(elements.partitionMetrics.children.length, 1);
});

test('Morton toggle immediately refreshes the pinned interval in the inspect panel', () => {
    const { context, elements } = inspectContext();
    vm.runInContext(`
        durationsSec = [1, 300, 600, 1200];
        activities = [{ activity_id: 'ride', peak_watts: [1200, 360, 330, 315] }];
        pinnedIndex = 2;
        powerCurveChart = { data: { datasets: [] }, update() {} };
    `, context);
    context.bindControls();
    elements.criticalPowerToggle.checked = true;
    elements.criticalPowerToggle.change();
    assert.equal(elements.partitionMetrics.children.length, 2);
    assert.equal(elements.selectedInterval.textContent, '00:10:00');
    elements.criticalPowerToggle.checked = false;
    elements.criticalPowerToggle.change();
    assert.equal(elements.partitionMetrics.children.length, 1);
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