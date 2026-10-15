const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '../templates/ttt_analysis.html'), 'utf8');
const source = template.match(/    function averageRiderGap\(rider, lo, hi\) \{[\s\S]*?\n    \}/)[0];
const averageRiderGap = vm.runInNewContext(`(${source})`);

test('average gap excludes leaders and gaps above 25 m without changing raw data', () => {
    const rider = {
        gap_ahead: {
            distance_km: [0, 0.1, 0.2, 0.3, 0.4],
            gap_m: [null, 10, 25, 25.01, 100],
        },
    };
    const before = JSON.stringify(rider);
    assert.equal(averageRiderGap(rider, 0, 1), 17.5);
    assert.equal(JSON.stringify(rider), before);
});

test('average gap follows the selected distance range including its boundaries', () => {
    const rider = {
        gap_ahead: { distance_km: [1, 2, 3, 4], gap_m: [1, 10, 20, 24] },
    };
    assert.equal(averageRiderGap(rider, 2, 3), 15);
    assert.equal(averageRiderGap(rider, 3, 3), 20);
    assert.equal(averageRiderGap(rider, 5, 6), null);
});

test('average gap retains per-second weighting while leader distance stalls', () => {
    const rider = {
        gap_ahead: { distance_km: [0.1, 0.1, 0.1, 0.11], gap_m: [10, 8, 6, 10] },
    };
    assert.equal(averageRiderGap(rider, 0, 1), 8.5);
});

test('average gap is unavailable when no samples qualify or gap data is absent', () => {
    const rider = {
        gap_ahead: {
            distance_km: [0, 1, 2, 3],
            gap_m: [null, 26, Number.NaN, Number.POSITIVE_INFINITY],
        },
    };
    assert.equal(averageRiderGap(rider, 0, 3), null);
    assert.equal(averageRiderGap({}, 0, 3), null);
    assert.equal(averageRiderGap({ gap_ahead: { distance_km: [], gap_m: [] } }, 0, 3), null);
});

const boundsSource = template.match(/    function positionYAxisBounds\(value\) \{[\s\S]*?\n    \}/)[0];
const positionYAxisBounds = vm.runInNewContext(`(${boundsSource})`);

test('position axis range shows the requested maximum gap below the leader', () => {
    assert.equal(positionYAxisBounds('25').min, -25);
    assert.equal(positionYAxisBounds('25').max, 0);
    assert.equal(positionYAxisBounds('12.5').min, -12.5);
});

test('empty or invalid position axis ranges leave automatic scaling enabled', () => {
    for (const value of ['', '0', '-10', 'invalid', 'Infinity']) {
        assert.equal(positionYAxisBounds(value).min, undefined);
        assert.equal(positionYAxisBounds(value).max, undefined);
    }
});

test('position axis updates preserve raw dropped-rider data and restore auto scaling', () => {
    const chart = {
        data: { datasets: [{ data: [{ x: 1, y: -1000 }] }] },
        options: { scales: { y: { title: { text: 'Position' } } } },
        update(mode) { this.lastUpdate = mode; },
    };
    const input = {
        value: '25',
        setCustomValidity(message) { this.validationMessage = message; },
        reportValidity() { return !this.validationMessage; },
    };
    const context = vm.createContext({
        positionDeviationChartInstance: chart,
        positionYAxisBounds,
        document: { getElementById() { return input; } },
    });
    const updateSource = template.match(/    function updatePositionYAxisRange\(\) \{[\s\S]*?\n    \}/)[0];
    vm.runInContext(updateSource, context);
    const before = JSON.stringify(chart.data);

    context.updatePositionYAxisRange();
    assert.equal(chart.options.scales.y.min, -25);
    assert.equal(chart.options.scales.y.max, 0);
    assert.equal(chart.lastUpdate, 'none');
    assert.equal(chart.options.scales.y.title.text, 'Position');

    for (const value of ['0', '-10']) {
        input.value = value;
        context.updatePositionYAxisRange();
        assert.ok(input.validationMessage);
        assert.equal(chart.options.scales.y.min, -25);
    }

    input.value = '';
    context.updatePositionYAxisRange();
    assert.equal(chart.options.scales.y.min, undefined);
    assert.equal(chart.options.scales.y.max, undefined);
    assert.equal(JSON.stringify(chart.data), before);
});