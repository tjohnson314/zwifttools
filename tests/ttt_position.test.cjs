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