const assert = require('node:assert/strict');
const test = require('node:test');

const modulePromise = import('../mods/sauce_mod_segment_times/tracker.mjs');
const forward = {id: 'forward', name: 'Sprint', roadId: 18, roadStart: 0.2, roadFinish: 0.4};
const reverse = {id: 'reverse', name: 'Sprint Reverse', roadId: 18, roadStart: 0.795, roadFinish: 0.6855761013, reverse: true};

function entry(id, time, percent, options = {}) {
    return {
        athleteId: id,
        state: {
            courseId: 13, roadId: 18, roadTime: 5000 + percent * 1e6,
            worldTime: time, distance: 0, reverse: false, ...options,
        },
    };
}

test('interpolates both crossings independently for multiple riders and ignores duplicate updates', async () => {
    const {SegmentTracker} = await modulePromise;
    const tracker = new SegmentTracker();
    tracker.setCourseSegments(13, [forward]);
    tracker.update(entry(1, 1000, 0.1));
    tracker.update(entry(2, 1000, 0.1));
    tracker.update(entry(1, 3000, 0.3));
    tracker.update(entry(2, 4000, 0.3));
    assert.deepEqual(tracker.update(entry(1, 3000, 0.3)), []);
    const first = tracker.update(entry(1, 5000, 0.5));
    const second = tracker.update(entry(2, 7000, 0.5));
    assert.equal(first[0].name, 'Sprint');
    assert.ok(Math.abs(first[0].timeMs - 2000) < 1e-6);
    assert.ok(Math.abs(second[0].timeMs - 3000) < 1e-6);
    assert.deepEqual(tracker.update(entry(1, 5000, 0.5)), []);
});

test('tracks the reverse Castle Park segment and does not start mid-segment', async () => {
    const {SegmentTracker} = await modulePromise;
    const tracker = new SegmentTracker();
    tracker.setCourseSegments(13, [reverse]);
    const rev = {reverse: true};
    tracker.update(entry('mid', 1000, 0.75, rev));
    assert.deepEqual(tracker.update(entry('mid', 3000, 0.68, rev)), []);
    tracker.update(entry('rider', 1000, 0.81, rev));
    tracker.update(entry('rider', 3000, 0.78, rev));
    const result = tracker.update(entry('rider', 11000, 0.68, rev));
    const expectedStart = 2000;
    const expectedFinish = 3000 + (0.78 - reverse.roadFinish) / 0.1 * 8000;
    assert.equal(result[0].name, 'Sprint Reverse');
    assert.ok(Math.abs(result[0].timeMs - (expectedFinish - expectedStart)) < 1e-6);
});

test('a single update can cross both boundaries in order', async () => {
    const {SegmentTracker} = await modulePromise;
    const tracker = new SegmentTracker();
    tracker.setCourseSegments(13, [forward]);
    tracker.update(entry(1, 1000, 0.1));
    const result = tracker.update(entry(1, 5000, 0.5));
    assert.equal(result.length, 1);
    assert.ok(Math.abs(result[0].timeMs - 2000) < 1e-6);
});

test('discards incomplete attempts after gaps, road changes, direction changes, or course changes', async () => {
    const {SegmentTracker} = await modulePromise;
    for (const [change, resumed] of [
        [entry(1, 20000, 0.3), entry(1, 21000, 0.3)],
        [entry(1, 4000, 0.3, {roadId: 19}), entry(1, 5000, 0.3)],
        [entry(1, 4000, 0.3, {reverse: true}), entry(1, 5000, 0.3)],
        [entry(1, 4000, 0.3, {courseId: 6}), entry(1, 5000, 0.3)],
    ]) {
        const tracker = new SegmentTracker();
        tracker.setCourseSegments(13, [forward]);
        tracker.setCourseSegments(6, [forward]);
        tracker.update(entry(1, 1000, 0.1));
        tracker.update(entry(1, 3000, 0.3));
        tracker.update(change);
        tracker.update(resumed);
        assert.deepEqual(tracker.update(entry(1, resumed.state.worldTime + 1000, 0.5)), []);
    }
});

test('the same rider can complete the same segment again', async () => {
    const {SegmentTracker} = await modulePromise;
    const tracker = new SegmentTracker();
    tracker.setCourseSegments(13, [forward]);
    tracker.update(entry(1, 1000, 0.1));
    assert.equal(tracker.update(entry(1, 5000, 0.5)).length, 1);
    tracker.update(entry(1, 6000, 0.1, {roadId: 19}));
    tracker.update(entry(1, 7000, 0.1));
    assert.equal(tracker.update(entry(1, 11000, 0.5)).length, 1);
});

test('loop completion requires a full lap and can cross other roads', async () => {
    const {SegmentTracker} = await modulePromise;
    const tracker = new SegmentTracker();
    tracker.setCourseSegments(13, [{
        id: 'loop', name: 'Loop', roadId: 18, roadStart: 0.5, roadFinish: 0.5,
        distance: 1000, loop: true,
    }]);
    tracker.update(entry(1, 1000, 0.4, {distance: 0}));
    tracker.update(entry(1, 2000, 0.6, {distance: 20}));
    tracker.update(entry(1, 3000, 0.7, {roadId: 19, distance: 300}));
    tracker.update(entry(1, 4000, 0.4, {distance: 1000}));
    const result = tracker.update(entry(1, 5000, 0.6, {distance: 1020}));
    assert.equal(result.length, 1);
    assert.equal(result[0].timeMs, 3000);

    tracker.update(entry(1, 6000, 0.4, {distance: 1050}));
    assert.deepEqual(tracker.update(entry(1, 7000, 0.6, {distance: 1070})), []);
});

test('ignores malformed catalog entries and formats times across minute boundaries', async () => {
    const {SegmentTracker, formatTime} = await modulePromise;
    const tracker = new SegmentTracker();
    assert.equal(tracker.setCourseSegments(13, [
        forward, {...forward, id: 'bad', roadStart: 0.5}, {name: 'Missing'},
    ]), 1);
    assert.equal(formatTime(16149), '16.15 s');
    assert.equal(formatTime(59999), '1:00.00');
    assert.equal(formatTime(62123), '1:02.12');
});

test('overlay appends safe text for a nearby rider after loading Sauce segment data', async () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const dir = path.join(__dirname, '../mods/sauce_mod_segment_times');
    const asModule = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
    const trackerUrl = asModule(fs.readFileSync(path.join(dir, 'tracker.mjs'), 'utf8'));
    const commonUrl = asModule('export const {enableSentry, subscribe, rpc} = globalThis.__segmentTest;');
    const source = fs.readFileSync(path.join(dir, 'segment-times.mjs'), 'utf8')
        .replace("'/pages/src/common.mjs'", `'${commonUrl}'`)
        .replace("'./tracker.mjs'", `'${trackerUrl}'`);
    const subscriptions = {};
    const elements = {
        status: {textContent: '', classList: {add() {}, remove() {}}},
        results: {
            children: [], scrollHeight: 0, scrollTop: 0, clientHeight: 100,
            appendChild(child) { this.children.push(child); },
        },
    };
    globalThis.__segmentTest = {
        enableSentry() {},
        subscribe(event, callback) { subscriptions[event] = callback; },
        rpc: {async getCourseSegments() { return [forward]; }},
    };
    globalThis.document = {
        getElementById(id) { return elements[id]; },
        createElement() {
            return {children: [], appendChild(child) { this.children.push(child); }};
        },
    };
    try {
        const overlay = await import(asModule(source));
        overlay.main();
        const update = (time, pos) => {
            const rider = entry(1, time, pos);
            rider.athlete = {sanitizedFullname: '<Rider>'};
            subscriptions.nearby([rider]);
        };
        update(1000, 0.1);
        update(3000, 0.3);
        update(5000, 0.5);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(elements.status.textContent, 'Tracking nearby riders.');
        assert.equal(elements.results.children.length, 1);
        assert.deepEqual(elements.results.children[0].children.map(cell => cell.textContent),
            ['<Rider>', 'Sprint', '2.00 s']);
    } finally {
        delete globalThis.__segmentTest;
        delete globalThis.document;
    }
});
