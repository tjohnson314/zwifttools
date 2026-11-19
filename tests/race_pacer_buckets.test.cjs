const assert = require('node:assert/strict');
const test = require('node:test');

function tracker() {
    return import('../mods/sauce_mod_race_pacer/bucket-tracker.mjs');
}

test('bucket spans run from each bucket start to the next, with the bucket target', async () => {
    const { bucketSpans } = await tracker();
    const spans = bucketSpans([0, 250, 750, 1000, 1250, 1750], [1, 1, 1, 2, 2, 2],
        [300, 300, 300, 250, 250, 250]);
    assert.deepEqual(spans, [
        { bucket: 1, start: 0, end: 1000, target: 300 },
        { bucket: 2, start: 1000, end: 1750, target: 250 },
    ]);
    assert.deepEqual(bucketSpans([0, 1], null), []);
    assert.deepEqual(bucketSpans([0, 1], [null, null]), []);
});

test('span lookup switches exactly at the divider and is -1 past the end', async () => {
    const { bucketSpans, spanIndexAt } = await tracker();
    const spans = bucketSpans([0, 1000, 2000], [1, 2, 2], [300, 250, 250]);
    assert.equal(spanIndexAt(spans, 0), 0);
    assert.equal(spanIndexAt(spans, 999.9), 0);
    assert.equal(spanIndexAt(spans, 1000), 1);
    assert.equal(spanIndexAt(spans, 2000), 1);
    assert.equal(spanIndexAt(spans, 2000.1), -1);
    assert.equal(spanIndexAt([], 10), -1);
});

test('bucket average is time-weighted and holds each sample until the next', async () => {
    const { BucketAverage } = await tracker();
    const avg = new BucketAverage();
    avg.update(1, 0, 200);
    assert.equal(avg.average, null);
    avg.update(1, 1, 300);   // 200 W for 1 s
    avg.update(1, 4, 250);   // 300 W for 3 s
    assert.equal(avg.average, (200 * 1 + 300 * 3) / 4);
});

test('bucket average restarts on a new bucket or when time goes backwards', async () => {
    const { BucketAverage } = await tracker();
    const avg = new BucketAverage();
    avg.update(1, 0, 400);
    avg.update(1, 2, 400);
    avg.update(2, 3, 200);
    assert.equal(avg.average, null);
    avg.update(2, 4, 200);
    assert.equal(avg.average, 200);
    avg.update(2, 1, 300);   // reset pressed: clock went back
    avg.update(2, 2, 300);
    assert.equal(avg.average, 300);
});

test('bucket average skips missing power and long gaps between samples', async () => {
    const { BucketAverage, MAX_SAMPLE_GAP_SEC } = await tracker();
    const avg = new BucketAverage();
    avg.update(1, 0, 250);
    avg.update(1, 1, NaN);                        // counts 250 W for 1 s
    avg.update(1, 2, 100);                        // missing power: interval not counted
    avg.update(1, 2 + MAX_SAMPLE_GAP_SEC + 1, 100); // paused: gap not counted
    assert.equal(avg.average, 250);
});

test('switching watched athletes resets bucket power and status before accepting samples', async () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const vm = require('node:vm');
    const elements = new Map();
    function element(selector) {
        if (!elements.has(selector)) {
            elements.set(selector, {
                textContent: '', className: '',
                querySelector: element,
                addEventListener() {},
            });
        }
        return elements.get(selector);
    }
    let onWatching;
    const context = vm.createContext({
        ...await tracker(),
        ...await details(),
        ...await import('../mods/sauce_mod_race_pacer/reference.mjs'),
        Common: {
            enableSentry() {},
            initInteractionListeners() {},
            settingsStore: { setDefault() {}, get() {}, addEventListener() {} },
            subscribe(event, callback) {
                assert.equal(event, 'athlete/watching');
                onWatching = callback;
            },
        },
        document: {
            getElementById: element,
            documentElement: { classList: { toggle() {} } },
        },
    });
    const source = fs.readFileSync(path.join(__dirname, '../mods/sauce_mod_race_pacer/display.mjs'), 'utf8');
    vm.runInContext(source.replace(/^import .*;\r?\n/gm, '').replace('export function main()', 'function main()'), context);
    vm.runInContext(`
        refTimes = [0, 100];
        refDistances = [0, 1000];
        refBucketSpans = [{ bucket: 1, start: 0, end: 1000, target: 200 }];
        main();
    `, context);
    function sample(athleteId, time, power) {
        onWatching({ athleteId, state: { time, power, routeDistance: 100 } });
    }
    const avgEl = element('.bucket-avg');
    sample(1, 10, 400);
    sample(1, 11, 400);
    assert.equal(avgEl.textContent, 'avg 400 W / 200 W (+200)');
    assert.equal(avgEl.className, 'bucket-avg over');

    sample(2, 12, 200);
    assert.equal(avgEl.textContent, 'avg —');
    assert.equal(avgEl.className, 'bucket-avg');
    sample(2, 13, 200);
    assert.equal(avgEl.textContent, 'avg 200 W / 200 W (±0)');
    assert.equal(avgEl.className, 'bucket-avg on');

    onWatching({ athleteId: 1 });
    assert.equal(avgEl.textContent, '');
    sample(1, 14, 100);
    assert.equal(avgEl.textContent, 'avg —');
    sample(1, 15, 100);
    assert.equal(avgEl.textContent, 'avg 100 W / 200 W (−100)');
    assert.equal(avgEl.className, 'bucket-avg under');
});

test('pace status allows max(2%, 3 W) either side of target', async () => {
    const { paceStatus } = await tracker();
    assert.equal(paceStatus(295, 300), 'on');
    assert.equal(paceStatus(293, 300), 'under');
    assert.equal(paceStatus(307, 300), 'over');
    assert.equal(paceStatus(98, 100), 'on');      // 3 W floor beats 2 W
    assert.equal(paceStatus(96, 100), 'under');
    assert.equal(paceStatus(null, 300), null);
});

function details() {
    return import('../mods/sauce_mod_race_pacer/details.mjs');
}

test('all detail lines are shown by default', async () => {
    const { detailHideClasses } = await details();
    const hide = detailHideClasses(() => undefined);
    assert.deepEqual(Object.values(hide), [false, false, false, false, false]);
});

test('switching a detail off hides only that line', async () => {
    const { detailHideClasses } = await details();
    const settings = { showBucketInfo: false, showNames: false, showRefSpeed: true };
    assert.deepEqual(detailHideClasses(key => settings[key]), {
        'hide-ref-power': false,
        'hide-ref-speed': false,
        'hide-bucket-avg': false,
        'hide-bucket-info': true,
        'hide-names': true,
    });
});

test('every hide class has a matching CSS rule and every setting a checkbox', async () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const { DETAIL_CLASSES } = await details();
    const dir = path.join(__dirname, '../mods/sauce_mod_race_pacer');
    const css = fs.readFileSync(path.join(dir, 'display.css'), 'utf8');
    const settingsHtml = fs.readFileSync(path.join(dir, 'display-settings.html'), 'utf8');
    for (const [key, cls] of Object.entries(DETAIL_CLASSES)) {
        assert.match(css, new RegExp(`\\.${cls} \\.`), `${cls} has no CSS rule`);
        assert.match(settingsHtml, new RegExp(`type="checkbox" name="${key}"`), `${key} has no checkbox`);
    }
});
