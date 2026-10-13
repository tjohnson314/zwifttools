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