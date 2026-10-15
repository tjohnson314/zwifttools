const assert = require('node:assert/strict');
const test = require('node:test');

async function modelContext() {
    return import('../mods/sauce_mod_nearby_draft/draft-model.mjs');
}

function rider(athleteId, gap, overrides = {}) {
    return {athleteId, gap, athlete: {weight: 75, height: 183},
        state: {worldTime: 1000, courseId: 6, speed: 36, power: 200, draft: 100,
            altitude: 10, ...overrides}};
}

test('nearby selection includes both 15-second boundaries and self, excluding unknown gaps', async () => {
    const context = await modelContext();
    const entries = [rider(1, 0), rider(2, -15), rider(3, 15), rider(4, -15.01),
        rider(5, 15.01), rider(6, null), rider(7, NaN)];
    assert.deepEqual(Array.from(context.selectNearby(entries, 1), entry => entry.athleteId), [2, 1, 3]);
    assert.equal(context.selectNearby(entries, null).length, 0);
});

test('gaps are anchored to the user even while watching another rider', async () => {
    const context = await modelContext();
    const entries = [rider(1, 40), rider(2, 25), rider(3, 55), rider(4, 0)];
    const selected = context.selectNearby(entries, 1, 4);
    assert.deepEqual(Array.from(selected, entry => entry.athleteId), [2, 1, 3]);
    assert.deepEqual(Array.from(selected, entry => entry.gap), [-15, 0, 15]);
    assert.equal(context.selectNearby([rider(4, 0)], 1, 4).length, 0);
});

test('summary retains negative effective draft and percentages above 100 without clipping', async () => {
    const context = await modelContext();
    const negative = context.summarizeDraft({aero: -200, draft: 60, residual: -90});
    assert.equal(negative.effective, -30);
    assert.equal(negative.residual, 90);
    const high = context.summarizeDraft({aero: -200, draft: 150, residual: 20});
    assert.equal(high.percent, 150);
    for (const summary of [negative, high]) {
        const geometry = context.draftGeometry(summary);
        assert.ok(geometry.greenWidth > 0);
        assert.ok(geometry.greenLeft + geometry.greenWidth <= 100);
        assert.ok(geometry.blueLeft + geometry.blueWidth <= 100);
    }
});

test('rider histories are independent, repeated timestamps do not resmooth, and leaving prunes state', async () => {
    const context = await modelContext();
    const settings = {...context.DEFAULTS, physicalSpeed: false};
    const together = new context.DraftTracker();
    const alone = new context.DraftTracker();
    together.update(rider(1, 0), settings);
    alone.update(rider(1, 0), settings);
    together.update(rider(2, 1, {speed: 70, altitude: 900, draft: 500}), settings);
    const next = rider(1, 0, {worldTime: 2000, altitude: 11, speed: 38});
    assert.deepEqual(together.update(next, settings), alone.update(next, settings));
    const summary = together.update(next, settings);
    assert.equal(together.update(next, settings), summary);
    together.retain([1]);
    assert.equal(together.riders.size, 1);
    assert.equal(together.riders.has('2'), false);
});

test('course changes and stale samples reset derivatives; athlete dimensions affect the estimate', async () => {
    const context = await modelContext();
    const tracker = new context.DraftTracker();
    const settings = {...context.DEFAULTS, physicalSpeed: false};
    tracker.update(rider(1, 0), settings);
    const changed = rider(1, 0, {courseId: 13, worldTime: 2000, altitude: 1000});
    assert.deepEqual(tracker.update(changed, settings), new context.DraftTracker().update(changed, settings));
    const stale = rider(1, 0, {courseId: 13, worldTime: 10000, altitude: 2000});
    assert.deepEqual(tracker.update(stale, settings), new context.DraftTracker().update(stale, settings));
    const heavy = rider(2, 1);
    heavy.athlete = {weight: 110, height: 200};
    assert.notEqual(tracker.update(heavy, settings).maxDraft,
        new context.DraftTracker().update(rider(2, 1), settings).maxDraft);
});

test('spatial selection uses a 50-metre radius in native centimetres, not time gaps', async () => {
    const context = await modelContext();
    const self = rider(1, 100, {x: 10000, y: 20000});
    const entries = [rider(2, null, {x: 13000, y: 24000}),
        rider(3, 0, {x: 13001, y: 24000}),
        rider(4, 0, {courseId: 13, x: 10000, y: 20000}), rider(5, 0)];
    const selected = context.selectSpatialNearby(entries, self);
    assert.deepEqual(selected.map(entry => entry.athleteId), [1, 2]);
    assert.deepEqual(selected[1].position, {x: 30, y: 40});
    assert.equal(selected[1].distance, 50);
    assert.equal(selected[0].isSelf, true);
    assert.equal(context.selectSpatialNearby(entries, rider(1, 0)).length, 0);
});

test('spatial selection falls back to GPS and de-duplicates the user', async () => {
    const context = await modelContext();
    const self = rider(1, 0, {latlng: [0, 0]});
    const entries = [self, self, rider(2, null, {latlng: [0.0001, 0]}),
        rider(3, 0, {latlng: [0.001, 0]}), rider(4, 0, {latlng: [NaN, 0]})];
    const selected = context.selectSpatialNearby(entries, self);
    assert.deepEqual(selected.map(entry => entry.athleteId), [1, 2]);
    assert.ok(Math.abs(selected[1].distance - 11.11949) < 0.001);
    assert.ok(selected[1].position.y < 0);
});

test('donuts use effective, wasted, and remaining max draft without discarding signed data', async () => {
    const context = await modelContext();
    const normal = context.donutSegments({effective: 40, residual: 20, maxDraft: 100});
    assert.deepEqual(normal.values, [40, 20, 40]);
    assert.deepEqual(normal.fractions, [0.4, 0.2, 0.4]);
    assert.equal(normal.signed, false);
    for (const summary of [{effective: -30, residual: 90, maxDraft: 100},
        {effective: 150, residual: 20, maxDraft: 100}]) {
        const signed = context.donutSegments(summary);
        assert.equal(signed.values[0], summary.effective);
        assert.equal(signed.values[1], summary.residual);
        assert.equal(signed.values.reduce((sum, value) => sum + value, 0), summary.maxDraft);
        assert.equal(signed.signed, true);
        assert.ok(Math.abs(signed.fractions.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
    }
    assert.deepEqual(context.donutSegments({effective: 0, residual: 0, maxDraft: 0}).fractions, [0, 0, 1]);
});

test('initials omit team tags and retain Unicode names', async () => {
    const {riderInitials} = await modelContext();
    assert.equal(riderInitials({sanitizedFullname: 'Alex Rider [RtB]'}, 1), 'AR');
    assert.equal(riderInitials({sanitizedFullname: 'A. Rider (Team)'}, 1), 'AR');
    assert.equal(riderInitials({sanitizedFullname: 'Rider'}, 1), 'R');
    assert.equal(riderInitials({}, 12345), '45');
    assert.equal(riderInitials({sanitizedFullname: '\u00c5sa \u00d6berg'}, 1), '\u00c5\u00d6');
});

test('configurable radius and rider limit select the closest riders while counting everyone in range', async () => {
    const {DEFAULTS, selectDisplayedRiders} = await modelContext();
    const self = rider(1, 0, {x: 0, y: 0});
    const entries = [rider(5, 0, {x: 10001, y: 0}), rider(4, 0, {x: 10000, y: 0}),
        rider(3, 0, {x: 7500, y: 0}), rider(2, 0, {x: 2000, y: 0})];
    const limited = selectDisplayedRiders(entries, self, {...DEFAULTS, radiusM: 100, maxRiders: 2});
    assert.deepEqual(limited.riders.map(entry => entry.athleteId), [1, 2]);
    assert.equal(limited.actual, 4);
    const unlimited = selectDisplayedRiders(entries, self, {...DEFAULTS, radiusM: 100});
    assert.deepEqual(unlimited.riders.map(entry => entry.athleteId), [1, 2, 3, 4]);
    assert.equal(unlimited.actual, 4);
    const smaller = selectDisplayedRiders(entries, self, {...DEFAULTS, radiusM: 20, maxRiders: 10});
    assert.deepEqual(smaller.riders.map(entry => entry.athleteId), [1, 2]);
    assert.equal(smaller.actual, 2);
    assert.equal(selectDisplayedRiders(entries, null).actual, 0);
});

test('a one-rider cap retains self even when another rider has the same coordinates', async () => {
    const {DEFAULTS, selectDisplayedRiders} = await modelContext();
    const self = rider(1, 0, {x: 0, y: 0});
    const selection = selectDisplayedRiders([rider(2, 0, {x: 0, y: 0})], self,
        {...DEFAULTS, maxRiders: 1});
    assert.deepEqual(selection.riders.map(entry => entry.athleteId), [1]);
    assert.equal(selection.actual, 2);
});

test('assigned frame and wheels affect the draft model equally for every rider', async () => {
    const {DEFAULTS, DraftTracker} = await modelContext();
    const {MODEL} = await import('../mods/sauce_mod_nearby_draft/bike-model.mjs');
    const defaultFrame = MODEL.frames.find(frame => frame.id === DEFAULTS.frameId);
    const otherFrame = MODEL.frames.find(frame =>
        frame.cda[DEFAULTS.upgradeLevel] !== defaultFrame.cda[DEFAULTS.upgradeLevel]);
    const wheel = MODEL.wheels.find(candidate => Number.isFinite(candidate.cdaTt) && candidate.cdaTt !== 0);
    assert.ok(otherFrame);
    assert.ok(wheel);
    const baseline = new DraftTracker().update(rider(1, 0), DEFAULTS).maxDraft;
    for (const settings of [{...DEFAULTS, frameId: otherFrame.id}, {...DEFAULTS, wheelId: wheel.id}]) {
        const tracker = new DraftTracker();
        const first = tracker.update(rider(1, 0), settings);
        const second = tracker.update(rider(2, 1), settings);
        assert.notEqual(first.maxDraft, baseline);
        assert.deepEqual(first, second);
    }
});

test('position interpolation moves continuously between telemetry samples without extrapolating', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    const first = {courseId: 6, worldTime: 1000, x: 0, y: 0};
    const next = {...first, worldTime: 2000, x: 1000, y: -2000};
    motion.update(1, first, 0);
    motion.update(1, next, 1000);
    assert.deepEqual(motion.sample(1, 1000), {...next, worldTime: 1000, x: 0, y: 0, latlng: undefined});
    assert.equal(motion.sample(1, 1500).x, 500);
    assert.equal(motion.sample(1, 1500).y, -1000);
    assert.equal(motion.isAnimating(1500), true);
    assert.equal(motion.sample(1, 5000).x, 1000);
    assert.equal(motion.isAnimating(5000), false);
    assert.equal(first.x, 0);
    assert.equal(next.x, 1000);
});

test('duplicate updates do not restart motion and mid-transition targets remain continuous', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    motion.update(1, {courseId: 6, worldTime: 1000, x: 0, y: 0}, 0);
    const next = {courseId: 6, worldTime: 2000, x: 1000, y: 0};
    motion.update(1, next, 1000);
    motion.update(1, next, 1250);
    assert.equal(motion.sample(1, 1500).x, 500);
    motion.update(1, {...next, worldTime: 2500, x: 2000}, 1500);
    assert.equal(motion.sample(1, 1500).x, 500);
    assert.equal(motion.sample(1, 1750).x, 750);
    assert.equal(motion.sample(1, 2000).x, 1000);
    assert.equal(motion.sample(1, 2500).x, 2000);
});

test('position interpolation snaps on course changes, stale feeds, and time resets', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    motion.update(1, {courseId: 6, worldTime: 1000, x: 0, y: 0}, 0);
    motion.update(1, {courseId: 13, worldTime: 2000, x: 90000, y: 0}, 1000);
    assert.equal(motion.sample(1, 1000).x, 90000);
    motion.update(1, {courseId: 13, worldTime: 6000, x: 100000, y: 0}, 5000);
    assert.equal(motion.sample(1, 5000).x, 100000);
    motion.update(1, {courseId: 13, worldTime: 500, x: 20, y: 0}, 5500);
    assert.equal(motion.sample(1, 5500).x, 20);
    assert.equal(motion.isAnimating(5500), false);
    motion.retain([]);
    assert.equal(motion.sample(1, 6000), null);
});

test('GPS interpolation supports missing timestamps and keeps rider histories independent', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    motion.update(1, {courseId: 6, latlng: [0, 0]}, 0);
    motion.update(2, {courseId: 6, latlng: [1, 1]}, 0);
    motion.update(1, {courseId: 6, latlng: [0.001, -0.002]}, 1000);
    motion.update(2, {courseId: 6, latlng: [1, 1]}, 1000);
    assert.deepEqual(motion.sample(1, 1500).latlng, [0.0005, -0.001]);
    assert.deepEqual(motion.sample(2, 1500).latlng, [1, 1]);
    motion.retain([1]);
    assert.equal(motion.sample(2, 1500), null);
});

test('fast self updates and slower delayed riders retain their true relative positions', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    const state = (time, offset = 0) => ({courseId: 6, worldTime: time, x: time + offset, y: 0});
    motion.update(1, state(1000), 0);
    motion.update(2, state(1000, 500), 100);
    motion.update(2, state(2000, 500), 1100);
    for (let time = 1100; time <= 2500; time += 100) motion.update(1, state(time), time - 1000);
    for (const now of [1500, 1750, 2000, 3000]) {
        const self = motion.sample(1, now);
        const other = motion.sample(2, now);
        assert.equal(self.worldTime, other.worldTime);
        assert.equal(other.x - self.x, 500);
        assert.ok(self.worldTime <= 2000);
    }
});

test('only the rider without overlapping history waits, without mixing unmatched timestamps', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    motion.update(1, {courseId: 6, worldTime: 2000, x: 2000, y: 0}, 0);
    motion.update(2, {courseId: 6, worldTime: 1000, x: 1500, y: 0}, 0);
    assert.equal(motion.sample(1, 0).worldTime, 2000);
    assert.equal(motion.sample(2, 0), null);
    motion.update(2, {courseId: 6, worldTime: 2000, x: 2500, y: 0}, 1000);
    assert.equal(motion.sample(2, 1000).x - motion.sample(1, 1000).x, 500);
});

test('displayed-position latency includes the buffer and grows when the displayed frame becomes stale', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    assert.equal(motion.latencyMs(0), null);
    motion.update(1, {courseId: 6, worldTime: 1000, x: 0, y: 0}, 0);
    motion.update(1, {courseId: 6, worldTime: 2000, x: 1000, y: 0}, 1000);
    assert.equal(motion.latencyMs(1500), 1000);
    const displayedTime = motion.timeline(1500).time;
    assert.equal(motion.latencyMs(2000, displayedTime), 1500);
    assert.equal(motion.latencyMs(5000), 4000);
    assert.equal(motion.latencyMs(5000, null), null);
});

test('reference latency remains available while another rider is synchronizing', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    motion.update(1, {courseId: 6, worldTime: 2000, x: 2000, y: 0}, 0);
    motion.update(2, {courseId: 6, worldTime: 1000, x: 1500, y: 0}, 0);
    assert.equal(motion.latencyMs(500), 500);
});

test('a newly arriving rider cannot blank the reference rider and synchronized group', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    const state = (time, offset = 0) => ({courseId: 6, worldTime: time, x: time + offset, y: 0});
    for (const time of [1000, 2000]) {
        motion.update(1, state(time), time - 1000);
        motion.update(2, state(time, 500), time - 1000);
    }
    motion.update(3, state(2500, 1000), 1500);
    const timeline = motion.timeline(1500, 1);
    assert.ok(timeline, 'a joining rider must not invalidate the group timeline');
    assert.equal(motion.sample(1, 1500, timeline).worldTime, 1500);
    assert.equal(motion.sample(2, 1500, timeline).x - motion.sample(1, 1500, timeline).x, 500);
    assert.equal(motion.sample(3, 1500, timeline), null, 'never extrapolate a joining rider backwards');
});

test('stale riders and their history resets do not block playback, and rejoin at the shared time', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    const state = (time, offset = 0) => ({courseId: 6, worldTime: time, x: time + offset, y: 0});
    for (const time of [1000, 2000]) {
        motion.update(1, state(time), time - 1000);
        motion.update(2, state(time, 500), time - 1000);
    }
    for (let time = 3000; time <= 8000; time += 1000) {
        motion.update(1, state(time), time - 1000);
    }
    motion.update(2, state(8000, 500), 7000);
    let timeline = motion.timeline(7000, 1);
    assert.equal(timeline.time, 7000);
    assert.equal(motion.sample(1, 7000, timeline).x, 7000);
    assert.equal(motion.sample(2, 7000, timeline), null);
    motion.update(1, state(9000), 8000);
    motion.update(2, state(9000, 500), 8000);
    timeline = motion.timeline(8000, 1);
    assert.equal(motion.sample(1, 8000, timeline).worldTime, 8000);
    assert.equal(motion.sample(2, 8000, timeline).worldTime, 8000);
    assert.equal(motion.sample(2, 8000, timeline).x - motion.sample(1, 8000, timeline).x, 500);
});

test('future rider histories wait without invalidating the reference timeline', async () => {
    const {PositionInterpolator} = await modelContext();
    const motion = new PositionInterpolator();
    for (const time of [1000, 2000]) {
        motion.update(1, {courseId: 6, worldTime: time, x: time, y: 0}, time - 1000);
    }
    for (const time of [3000, 4000]) {
        motion.update(2, {courseId: 6, worldTime: time, x: time + 500, y: 0}, time - 1000);
    }
    const timeline = motion.timeline(3500, 1);
    assert.equal(timeline.time, 2000);
    assert.equal(motion.sample(1, 3500, timeline).worldTime, 2000);
    assert.equal(motion.sample(2, 3500, timeline), null);
    assert.equal(motion.sample(1, 3500, {time: 999}), null);
    assert.equal(motion.sample(1, 3500, {time: 2001}), null);
});