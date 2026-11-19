const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/power_play.js'), 'utf8');

function setup(fetch) {
    const elements = new Map();
    const historyUrls = [];
    function element() {
        return {
            value: '', checked: false, hidden: true, disabled: false, children: [],
            textContent: '', className: '',
            appendChild(child) { this.children.push(child); return child; },
            replaceChildren() { this.children = []; },
        };
    }
    const context = vm.createContext({
        document: {
            addEventListener() {},
            querySelectorAll: () => [],
            createElement: element,
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, element());
                return elements.get(id);
            },
        },
        window: { history: { replaceState: (state, title, url) => historyUrls.push(url) } },
        URLSearchParams, console: { error() {} }, fetch,
    });
    vm.runInContext(source, context);
    return { context, get: id => context.document.getElementById(id), historyUrls };
}

const rows = [
    { duration_sec: 5, name: '<Heavy>', avg_power: 1500, watts_per_kg: 15, weight_kg: 100, activity_id: '9007199254740993', arch_name: '5 Sec', start_time_sec: .5 },
    { duration_sec: 5, name: 'Light', avg_power: 1000, watts_per_kg: 20, weight_kg: 50, activity_id: '2', arch_name: '5 Sec', start_time_sec: .5 },
    { duration_sec: 5, name: 'Unknown', avg_power: 1000, watts_per_kg: null, weight_kg: null, activity_id: '3', arch_name: '5 Sec', start_time_sec: .5 },
    { duration_sec: 10, name: 'Other duration', avg_power: 2000, watts_per_kg: 30 },
];

test('watts and W/kg give distinct rankings with ties and missing weights', () => {
    const { context } = setup();
    const watts = context.powerPlayRows(rows, 5, 'avg_power');
    assert.deepEqual(Array.from(watts, row => row.name), ['<Heavy>', 'Light', 'Unknown']);
    const ranks = context.powerPlayRanks(watts, 'avg_power');
    assert.deepEqual(Array.from(watts, row => ranks.get(row)), [1, 2, 2]);
    const wkg = context.powerPlayRows(rows, 5, 'watts_per_kg');
    assert.deepEqual(Array.from(wkg, row => row.name), ['Light', '<Heavy>', 'Unknown']);
    assert.equal(context.powerPlayRanks(wkg, 'watts_per_kg').has(rows[2]), false);
});

test('event loader renders raw values with sortable power columns and no category', async () => {
    const { context, get, historyUrls } = setup(async url => {
        if (url === '/api/power-play/resolve/456') {
            return { ok: true, json: async () => ({ race_id: 'race_data_123' }) };
        }
        assert.equal(url, '/api/power-play/leaderboard/race_data_123');
        return { ok: true, json: async () => ({
            race_id: 'race_data_123', race_name: 'Stage 1 - Zwift Power Play Test', rider_count: 3,
            durations: [5, 10, 20], rows: rows.filter(row => row.duration_sec === 5),
            warnings: ['Unknown: event weight unavailable.'],
        }) };
    });
    get('event-input').value = '456';
    await context.loadPowerPlay();
    assert.equal(get('power-play-results').hidden, false);
    assert.deepEqual(historyUrls, ['?event_id=456']);
    assert.equal(get('efforts-5').children[0].children.length, 7);
    assert.equal(get('efforts-5').children[0].children[3].textContent, '1500.0');
    const riderLink = get('efforts-5').children[0].children[2].children[0];
    assert.equal(riderLink.textContent, '<Heavy>');
    assert.equal(riderLink.href, 'https://www.zwift.com/activity/9007199254740993');
    assert.equal(get('efforts-20').children[0].children[0].colSpan, 7);
    assert.equal(get('power-play-warnings').hidden, false);
    context.sortPowerPlay(5, 'watts_per_kg');
    assert.equal(get('efforts-5').children[0].children[2].children[0].textContent, 'Light');
    assert.equal(get('efforts-5').children[2].children[1].textContent, '\u2014');
    context.sortPowerPlay(5, 'watts_per_kg');
    assert.equal(get('efforts-5').children[0].children[2].children[0].textContent, '<Heavy>');
    context.sortPowerPlay(5, 'avg_power');
    assert.equal(get('efforts-5').children[0].children[3].textContent, '1500.0');
    context.sortPowerPlay(5, 'avg_power');
    assert.equal(get('efforts-5').children[0].children[3].textContent, '1000.0');
    context.sortPowerPlay(5, 'watts_rank');
    assert.equal(get('efforts-5').children[0].children[0].textContent, 1);
    context.sortPowerPlay(5, 'watts_rank');
    assert.equal(get('efforts-5').children[0].children[0].textContent, 2);
    context.sortPowerPlay(5, 'wkg_rank');
    assert.equal(get('efforts-5').children[0].children[2].children[0].textContent, 'Light');
    context.sortPowerPlay(5, 'wkg_rank');
    assert.equal(get('efforts-5').children[0].children[2].children[0].textContent, '<Heavy>');
    assert.equal(get('efforts-5').children[2].children[1].textContent, '\u2014');
});

test('failed race loads hide old results and restore controls', async () => {
    const { context, get } = setup(async () => ({
        ok: false, status: 400, json: async () => ({ error: 'Not a Power Play race.' }),
    }));
    get('event-input').value = '456';
    get('power-play-results').hidden = false;
    await context.loadPowerPlay();
    assert.equal(get('power-play-results').hidden, true);
    assert.equal(get('power-play-status').textContent, 'Not a Power Play race.');
    assert.equal(get('fetch-btn').disabled, false);
});

test('event ID resolves its cached pen without another download', async () => {
    const calls = [];
    const { context, get } = setup(async url => {
        calls.push(url);
        return { ok: true, json: async () => calls.length === 1
            ? { race_id: 'race_data_123' }
            : { race_name: 'Stage 1 - Zwift Power Play', rider_count: 0, durations: [5, 10, 20], rows: [], warnings: [] } };
    });
    get('event-input').value = '456';
    await context.loadPowerPlay();
    assert.equal(calls[0], '/api/power-play/resolve/456');
    assert.equal(calls[1], '/api/power-play/leaderboard/race_data_123');
});

test('re-fetching an event uses its source activity and only pen', async () => {
    const { context, get } = setup(async () => ({
        ok: true, json: async () => ({
            source_activity_id: '123', race_name: 'Stage 1 - Zwift Power Play',
            rider_count: 0, durations: [5, 10, 20], rows: [], warnings: [],
        }),
    }));
    const downloads = [];
    context.fetchPowerPlayRace = async (...args) => { downloads.push(args); return 'race_data_123'; };
    get('event-input').value = '456';
    await context.loadPowerPlay(true);
    assert.deepEqual(downloads, [['123', true]]);
});

test('race download streams progress and closes on success', async () => {
    const { context, get } = setup();
    let stream;
    context.EventSource = class {
        constructor(url) { this.url = url; this.closed = false; stream = this; }
        close() { this.closed = true; }
    };
    const result = context.fetchPowerPlayRace('123', true);
    assert.doesNotMatch(stream.url, /all_subgroups/);
    assert.match(stream.url, /force_refresh=1/);
    stream.onmessage({ data: JSON.stringify({ progress: true, current: 1, total: 2, name: 'Rider' }) });
    assert.match(get('power-play-status').textContent, /1\/2/);
    stream.onmessage({ data: JSON.stringify({ success: true, race_id: 'race_data_123' }) });
    assert.equal(await result, 'race_data_123');
    assert.equal(stream.closed, true);
});

test('race download errors are surfaced and stop automatic retries', async () => {
    const { context } = setup();
    let stream;
    context.EventSource = class {
        constructor() { stream = this; }
        close() { this.closed = true; }
    };
    const result = context.fetchPowerPlayRace('123', false);
    stream.onmessage({ data: JSON.stringify({ error: 'Zwift service unavailable.' }) });
    await assert.rejects(result, /Zwift service unavailable/);
    assert.equal(stream.closed, true);
});

test('replay offers Power Play link only for matching race titles and clears stale links', () => {
    const replay = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
    const link = { hidden: true, removeAttribute(key) { delete this[key]; } };
    const context = vm.createContext({ document: { getElementById: () => link } });
    vm.runInContext(replay.slice(replay.indexOf('function updatePowerPlayLink('), replay.indexOf('function initRaceData(')), context);
    context.updatePowerPlayLink({ race_name: 'Series: STAGE 1 - ZWIFT POWER PLAY Test', race_id: 'race_data_123', event_id: 456 });
    assert.equal(link.hidden, false);
    assert.equal(link.href, '/power-play?event_id=456');
    for (const race_name of ['Stage 2 - Zwift Power Play', 'Stage 10 - Zwift Power Play', 'Zwift Power Play', 'Other race']) {
        context.updatePowerPlayLink({ race_name, race_id: 'race_data_124', event_id: 457 });
        assert.equal(link.hidden, true);
        assert.equal(link.href, undefined);
    }
    context.updatePowerPlayLink({ race_name: 'Stage 1 - Zwift Power Play' });
    assert.equal(link.hidden, true);
});
