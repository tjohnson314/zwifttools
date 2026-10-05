const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function dashboardContext() {
    const context = vm.createContext({ document: { addEventListener() {} } });
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