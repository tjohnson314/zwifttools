const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function setup(payload, storageOverrides = {}) {
    const stored = new Map();
    const warning = { hidden: true, textContent: '' };
    const elements = new Map([['cacheWarning', warning]]);
    let rider = '7';
    const calls = [];
    const localStorage = {
        getItem: key => stored.get(key) ?? null,
        setItem: (key, value) => stored.set(key, value),
        removeItem: key => stored.delete(key),
        ...storageOverrides,
    };
    const state = vm.createContext({
        console, localStorage,
        window: { ZwiftRider: { ready: new Promise(() => {}), get: () => ({ id: rider }) } },
        document: {
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, { style: {} });
                return elements.get(id);
            },
            createElement() { return {}; },
        },
        fetch: async url => {
            calls.push(url);
            return { ok: true, json: async () => payload };
        },
    });
    const html = fs.readFileSync(path.join(__dirname, '../templates/my_activities.html'), 'utf8');
    const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
    vm.runInContext(script, state);
    function row(id = '11', signature = '[1,2,3]') {
        const cells = {
            '.col-np': {}, '.col-hr': {}, '.col-name': { innerHTML: '' },
            '.col-actions': {
                links: [],
                querySelector: () => null,
                appendChild(link) { this.links.push(link); },
            },
        };
        return {
            dataset: { activityId: id, cacheSignature: signature },
            querySelector: selector => cells[selector],
            querySelectorAll: () => [],
            cells,
        };
    }
    return { state, stored, calls, warning, row, setRider: value => { rider = value; } };
}

test('caches successful enrichment per rider and activity, including race links and missing HR', async () => {
    const env = setup({ telemetry_available: true, normalized_power: 301.4, avg_hr: null, is_race: true, event_id: 123 });
    const first = env.row();
    await env.state.enrichRow(first);
    const second = env.row();
    await env.state.enrichRow(second);
    assert.equal(env.calls.length, 1);
    assert.equal(second.cells['.col-np'].textContent, '301 W');
    assert.equal(second.cells['.col-hr'].textContent, '—');
    assert.match(second.cells['.col-name'].innerHTML, /badge-race/);
    assert.deepEqual(second.cells['.col-actions'].links.map(link => link.href),
        ['/race-replay?activity_id=11', 'https://zwiftpower.com/events.php?zid=123']);

    await env.state.enrichRow(env.row('12'));
    env.setRider('8');
    await env.state.enrichRow(env.row());
    assert.equal(env.calls.length, 3);
    assert.ok(env.stored.has('myActivitiesEnrichmentV1:7:11'));
    assert.ok(env.stored.has('myActivitiesEnrichmentV1:8:11'));
});

test('refreshes changed or invalid entries and preserves zero values', async () => {
    const env = setup({ telemetry_available: true, normalized_power: 0, avg_hr: 0, is_race: false, event_id: null });
    await env.state.enrichRow(env.row());
    const zero = env.row();
    await env.state.enrichRow(zero);
    assert.equal(zero.cells['.col-np'].textContent, '0 W');
    assert.equal(zero.cells['.col-hr'].textContent, '0 bpm');
    await env.state.enrichRow(env.row('11', '[1,3,3]'));
    env.stored.set('myActivitiesEnrichmentV1:7:11', '{bad');
    await env.state.enrichRow(env.row());
    assert.equal(env.calls.length, 3);
});

test('does not cache unavailable telemetry and retries it', async () => {
    const env = setup({ telemetry_available: false, normalized_power: null, avg_hr: null, is_race: true, event_id: 123 });
    const row = env.row();
    await env.state.enrichRow(row);
    await env.state.enrichRow(env.row());
    assert.equal(env.calls.length, 2);
    assert.equal(env.stored.size, 0);
    assert.equal(row.cells['.col-np'].textContent, 'Unavailable');
    assert.equal(row.cells['.col-hr'].textContent, 'Unavailable');
    assert.equal(row.cells['.col-actions'].links.length, 2);
});

test('storage failures show a warning and do not block enrichment', async () => {
    const env = setup(
        { telemetry_available: true, normalized_power: 200, avg_hr: 150, is_race: false, event_id: null },
        { getItem() { throw new Error('denied'); }, setItem() { throw new Error('full'); } },
    );
    const row = env.row();
    await env.state.enrichRow(row);
    assert.equal(row.cells['.col-np'].textContent, '200 W');
    assert.equal(row.cells['.col-hr'].textContent, '150 bpm');
    assert.equal(env.warning.hidden, false);
    assert.match(env.warning.textContent, /unavailable or full/);
});
