const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function context(extra = {}) {
    const store = new Map();
    const state = vm.createContext({
        console, URLSearchParams, URL, setTimeout,
        window: { location: { search: '', href: 'http://localhost/segment-prs' }, history: { replaceState() {} } },
        document: { addEventListener() {}, getElementById() { return {}; }, querySelectorAll() { return []; } },
        localStorage: {
            getItem: key => store.get(key) ?? null,
            setItem: (key, value) => store.set(key, value),
            removeItem: key => store.delete(key),
            get length() { return store.size; },
            key: index => [...store.keys()][index],
        },
        ...extra,
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/segment_prs.js'), 'utf8'), state);
    vm.runInContext(`
        segmentContext = { profile_id: '7', catalog_version: 'v1', weight_kg: 75, height_cm: 180 };
        segmentActivities = [{ activity_id: '11', name: 'Ride', start_date: '2026-11-01', end_date: '2026-11-01', duration_sec: 100 }];
    `, state);
    return { state, store };
}

function row(id, time, start = 0) {
    return {
        segment_id: id, segment_name: `Segment ${id}`, pass: 1, total_time_sec: time,
        start_time_sec: start, end_time_sec: start + time, distance_m: 100,
        elevation_change_m: -25, avg_power: 1500, normalized_power: 1600,
        avg_draft_watts: -900, initial_speed_kmh: 120,
    };
}

function payload(rows) {
    return { activity_id: '11', profile_id: '7', catalog_version: 'v1', weight_kg: 75, height_cm: 180, traversals: rows };
}

function fakeElement() {
    return {
        children: [], listeners: {}, textContent: '', hidden: false,
        appendChild(child) { this.children.push(child); },
        replaceChildren(...children) { this.children = children; },
        add(child) { this.children.push(child); },
        addEventListener(event, handler) { this.listeners[event] = handler; },
        setAttribute(name, value) { this[name] = value; },
    };
}

test('rendering paginates segments and drills into repeated attempts with ride analysis links', () => {
    const elements = new Map();
    const { state } = context({
        Option: function(text, value) { return { textContent: text, value }; },
        document: {
            addEventListener() {},
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, fakeElement());
                return elements.get(id);
            },
            createElement: fakeElement,
            createTextNode: text => ({ textContent: text }),
            querySelectorAll() { return []; },
        },
    });
    state.data = { ...payload([
        ...Array.from({ length: 25 }, (_, index) => row(String(index).padStart(2, '0'), 50)),
        { ...row('00', 40, 100), pass: 2 },
    ]), is_race: true };
    vm.runInContext('segmentActivities[0].data = data;', state);
    state.renderSegmentTable();
    const body = elements.get('segmentRows');
    assert.equal(body.children.length, 20);
    assert.equal(elements.get('pageStatus').textContent, '25 results \u00b7 Page 1 of 2');
    assert.equal(body.children[0].children.length, 10);
    const links = body.children[0].children[9].children.filter(element => element.href);
    assert.deepEqual(links.map(element => element.href), [
        '/bike-comparison?activity_id=11', '/race-replay?activity_id=11', 'https://www.zwift.com/activity/11',
    ]);
    assert.deepEqual(links.map(element => element.title), ['Bike Comparison', 'Race Replay', 'View on Zwift']);
    assert.deepEqual(links.map(element => element['aria-label']), links.map(element => element.title));
    assert.deepEqual(links.map(element => element.className), ['action-link', 'action-link race', 'action-link logo-link']);
    assert.equal(links[0].innerHTML, '&#x1F6B2;');
    assert.equal(links[1].innerHTML, '&#x1F3C1;');
    assert.match(links[2].innerHTML, /src="\/static\/img\/zwift.ico".*alt="Zwift".*class="logo-icon"/);
    assert.equal(links[2].target, '_blank');
    assert.equal(links[2].rel, 'noopener');
    assert.ok(links.every(element => element.textContent === ''));
    vm.runInContext('segmentPage = 1;', state);
    state.renderSegmentTable();
    assert.equal(body.children.length, 5);
    assert.equal(elements.get('nextBtn').disabled, true);
    state.selectSegment('1:not-present');
    assert.equal(body.children.length, 1);
    assert.equal(body.children[0].children[0].colSpan, 10);
    state.selectSegment('00');
    assert.equal(body.children.length, 2);
    assert.equal(body.children[0].children[2].textContent, '0:40.00');
    assert.equal(body.children[1].children[2].textContent, '0:50.00');
    assert.equal(elements.get('tableTitle').textContent, 'All attempts: Segment 00');
    assert.equal(state.allSegmentRows().length, 26);
    vm.runInContext('segmentActivities[0].data.is_race = false;', state);
    state.renderSegmentTable();
    assert.equal(body.children[0].children[9].children.length, 2);
    state.allAttempts = payload(Array.from({length:26}, (_, index) => row('00', 50 + index, index * 100)));
    vm.runInContext('segmentActivities[0].data = allAttempts;', state);
    state.renderSegmentTable();
    assert.equal(body.children.length, 20);
    assert.equal(elements.get('pageStatus').textContent, '26 results \u00b7 Page 1 of 2');
    vm.runInContext('segmentPage = 1;', state);
    state.renderSegmentTable();
    assert.equal(body.children.length, 6);
    assert.equal(body.children[0].children[2].textContent, '1:10.00');
});

test('table includes duration but no entry-time column', () => {
    const template = fs.readFileSync(path.join(__dirname, '../templates/segment_prs.html'), 'utf8');
    assert.match(template, /data-segment-sort="total_time_sec"[^>]*><button[^>]*>Duration /);
    assert.doesNotMatch(template, /Attempt \/ Entry|Entry time|data-segment-sort="start_time_sec"|<th class="num">Time<\/th>/);
    assert.equal((template.match(/data-segment-sort=/g) || []).length, 9);
    assert.equal((template.match(/class="sort-button"/g) || []).length, 9);
    assert.match(template, /<th>Analyze<\/th>/);
});

test('every numeric data column sorts raw numbers ascending and descending', () => {
    const fields = [
        'total_time_sec', 'distance_m', 'elevation_change_m',
        'avg_power', 'normalized_power', 'avg_draft_watts', 'initial_speed_kmh',
    ];
    for (const field of fields) {
        const { state } = context();
        state.data = payload([20, 3, 100].map((value, index) => ({
            ...row(String(index), 10), [field]: value,
        })));
        vm.runInContext('segmentActivities[0].data = data;', state);
        state.field = field;
        vm.runInContext('segmentSortKey = field;', state);
        assert.deepEqual(Array.from(state.segmentTableRows(), item => item[field]), [3, 20, 100], field);
        vm.runInContext('segmentSortDirection = -1;', state);
        assert.deepEqual(Array.from(state.segmentTableRows(), item => item[field]), [100, 20, 3], field);
    }
});

test('sorting preserves signed metrics and puts missing values last in either direction', () => {
    const { state } = context();
    state.data = payload([null, 0, -900, 200].map((value, index) => ({
        ...row(String(index), 10), avg_draft_watts: value,
    })));
    vm.runInContext('segmentActivities[0].data = data; segmentSortKey = "avg_draft_watts";', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.avg_draft_watts), [-900, 0, 200, null]);
    vm.runInContext('segmentSortDirection = -1;', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.avg_draft_watts), [200, 0, -900, null]);
});

test('date sorting is chronological across rides and survives segment drilldown', () => {
    const { state } = context();
    state.records = [
        {activity_id:'1', start_date:'2026-11-01T10:00:00+02:00', data:payload([row('A', 10)])},
        {activity_id:'2', start_date:'2026-11-01T07:30:00Z', data:payload([row('A', 20)])},
        {activity_id:'3', start_date:'2026-11-01T09:00:00Z', data:payload([row('A', 30)])},
    ];
    vm.runInContext('segmentActivities = records; selectedSegment = "A"; segmentSortKey = "start_date";', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.activity.activity_id), ['2', '1', '3']);
    vm.runInContext('segmentSortDirection = -1;', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.activity.activity_id), ['3', '1', '2']);
});

test('sorting keeps only each segment PR and applies to the entire filtered result set', () => {
    const { state } = context();
    state.data = payload([
        ...Array.from({length:25}, (_, index) => ({ ...row(String(index).padStart(2, '0'), 50), avg_power:index })),
        {...row('00', 100), avg_power:9999},
    ]);
    vm.runInContext('segmentActivities[0].data = data; segmentSortKey = "avg_power"; segmentSortDirection = -1;', state);
    const rows = state.segmentTableRows();
    assert.equal(rows.length, 25);
    assert.equal(rows[0].segment_id, '24');
    assert.equal(rows[24].segment_id, '00');
    assert.equal(rows[24].total_time_sec, 50);
    vm.runInContext('segmentQuery = "Segment 0";', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.avg_power), [9,8,7,6,5,4,3,2,1,0]);
});

test('sort toggles reset pagination, update accessible headers and preserve default ordering', () => {
    const { state } = context();
    const headers = ['segment_name', 'total_time_sec', 'avg_power'].map(key => {
        const header = fakeElement();
        const indicator = fakeElement();
        header.dataset = {segmentSort:key};
        header.querySelector = () => indicator;
        header.indicator = indicator;
        return header;
    });
    state.document.querySelectorAll = () => headers;
    state.updateSegmentSortHeaders();
    assert.equal(headers[0]['aria-sort'], 'ascending');
    assert.equal(headers[0].indicator.textContent, '\u2191');
    vm.runInContext('renderSegmentTable = () => updateSegmentSortHeaders(); segmentPage = 2;', state);
    state.toggleSegmentSort('avg_power');
    assert.equal(vm.runInContext('segmentPage', state), 0);
    assert.equal(headers[2]['aria-sort'], 'ascending');
    assert.equal(headers[0]['aria-sort'], 'none');
    state.toggleSegmentSort('avg_power');
    assert.equal(headers[2]['aria-sort'], 'descending');
    assert.equal(headers[2].indicator.textContent, '\u2193');
    vm.runInContext('segmentSortKey = null; segmentSortDirection = 1; selectedSegment = "A";', state);
    state.updateSegmentSortHeaders();
    assert.equal(headers[1]['aria-sort'], 'ascending');
    assert.equal(headers[2]['aria-sort'], 'none');
});

test('best table chooses fastest traversal, including repeat attempts in one activity', () => {
    const { state } = context();
    state.data = payload([row('B', 25), row('A', 20), { ...row('A', 15, 50), pass: 2 }]);
    vm.runInContext('segmentActivities[0].data = data;', state);
    const rows = state.segmentTableRows();
    assert.equal(rows.length, 2);
    assert.equal(rows[0].segment_id, 'A');
    assert.equal(rows[0].total_time_sec, 15);
    assert.equal(rows[0].pass, 2);
    assert.equal(rows[0].avg_draft_watts, -900);
});

test('single-segment drilldown retains every attempt from every ride and search is case-insensitive', () => {
    const { state } = context();
    state.data = payload([row('B', 25), row('A', 20), row('A', 15, 50)]);
    vm.runInContext('segmentActivities[0].data = data; selectedSegment = "A";', state);
    state.secondData = {...payload([row('A', 18), row('A', 12, 50)]), activity_id:'12'};
    vm.runInContext('segmentActivities.push({activity_id:"12",start_date:"2026-11-02",data:secondData});', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.total_time_sec), [12, 15, 18, 20]);
    assert.equal(state.allSegmentRows().length, 5);
    vm.runInContext('segmentQuery = "sEgMeNt a";', state);
    assert.equal(state.segmentTableRows().length, 4);
    vm.runInContext('segmentQuery = "unknown";', state);
    assert.equal(state.segmentTableRows().length, 0);
});

test('distinct directional IDs are not merged even with identical display names', () => {
    const { state } = context();
    state.data = payload([
        {...row('13:18228230436', 44.76), segment_name:'Breakaway Brae Reverse', distance_m:399.5},
        {...row('13:18228230436', 55, 100), segment_name:'Breakaway Brae Reverse', distance_m:399.5},
        {...row('13:-9223372018626545372', 73.05, 200), segment_name:'Breakaway Brae Reverse', distance_m:623.3},
    ]);
    vm.runInContext('segmentActivities[0].data = data;', state);
    assert.equal(state.segmentTableRows().length, 2);
    vm.runInContext('selectedSegment = "13:18228230436";', state);
    const rows = state.segmentTableRows();
    assert.equal(rows.length, 2);
    assert.equal(rows[0].total_time_sec, 44.76);
    assert.equal(rows[1].total_time_sec, 55);
    assert.equal(rows[0].distance_m, 399.5);
    assert.equal(state.allSegmentRows().length, 3);
});

test('drilldown metric sorting retains all attempts and resolves tied durations consistently', () => {
    const { state } = context();
    state.data = payload([
        {...row('A', 40, 100), avg_power:300},
        {...row('A', 50, 0), avg_power:9000},
        {...row('A', 40, 50), avg_power:200},
    ]);
    vm.runInContext('segmentActivities[0].data = data; selectedSegment = "A"; segmentSortKey = "avg_power"; segmentSortDirection = -1;', state);
    const rows = state.segmentTableRows();
    assert.deepEqual(Array.from(rows, item => item.avg_power), [9000, 300, 200]);
    vm.runInContext('segmentSortKey = "total_time_sec"; segmentSortDirection = 1;', state);
    assert.deepEqual(Array.from(state.segmentTableRows(), item => item.start_time_sec), [50, 100, 0]);
});

test('per-activity cache is scoped to user, profile measurements, geometry and ride summary', () => {
    const { state } = context();
    const activity = vm.runInContext('segmentActivities[0]', state);
    const data = payload([row('A', 15)]);
    state.saveSegmentCache(activity, data);
    assert.equal(state.readSegmentCache(activity).traversals[0].avg_power, 1500);
    vm.runInContext('segmentContext.profile_id = "8";', state);
    assert.equal(state.readSegmentCache(activity), null);
    vm.runInContext('segmentContext.profile_id = "7"; segmentContext.weight_kg = 76;', state);
    assert.equal(state.readSegmentCache(activity), null);
    vm.runInContext('segmentContext.weight_kg = 75;', state);
    state.saveSegmentCache(activity, data);
    vm.runInContext('segmentContext.catalog_version = "v2";', state);
    assert.equal(state.readSegmentCache(activity), null);
    vm.runInContext('segmentContext.catalog_version = "v1";', state);
    state.saveSegmentCache(activity, data);
    activity.duration_sec++;
    assert.equal(state.readSegmentCache(activity), null);
});

test('empty complete results cache correctly while malformed rows do not', () => {
    const { state, store } = context();
    const activity = vm.runInContext('segmentActivities[0]', state);
    state.saveSegmentCache(activity, payload([]));
    assert.equal(state.readSegmentCache(activity).traversals.length, 0);
    const data = payload([row('A', 15)]);
    data.traversals[0].end_time_sec = -1;
    state.saveSegmentCache(activity, data);
    assert.equal(state.readSegmentCache(activity), null);
    assert.equal(store.size, 0);
});

test('cached loads skip fetches and failed requests remain explicit and retryable', async () => {
    const elements = new Map();
    const requested = [];
    const { state } = context({
        document: {
            addEventListener() {},
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, fakeElement());
                return elements.get(id);
            },
            createElement: fakeElement,
        },
        fetch: async url => {
            requested.push(url);
            return { status: 502, ok: false, json: async () => ({ error: 'Telemetry unavailable' }) };
        },
        console: { warn() {}, error() {} },
    });
    vm.runInContext('renderSegmentTable = () => {}; segmentActivities.push({activity_id: "12", name: "Unavailable"});', state);
    state.saveSegmentCache(vm.runInContext('segmentActivities[0]', state), payload([row('A', 15)]));
    await state.loadSegmentActivities();
    assert.deepEqual(requested, ['/api/segment_prs/activity/12']);
    assert.equal(vm.runInContext('segmentActivities[1].error', state), 'Telemetry unavailable');
    assert.equal(elements.get('retryBtn').hidden, false);
    assert.match(elements.get('loadStatus').textContent, /1 cached.*1 failed.*provisional/);
    assert.equal(elements.get('failureList').children[0].textContent, 'Unavailable (12): Telemetry unavailable');
    state.fetch = async url => ({
        status: 200, ok: true, json: async () => ({ ...payload([]), activity_id: '12' }),
    });
    await state.loadSegmentActivities(true);
    assert.equal(elements.get('retryBtn').hidden, true);
    assert.equal(elements.get('failureDetails').hidden, true);
    assert.match(elements.get('loadStatus').textContent, /0 failed.*up to date/);
});

test('video activities cache explicit unavailable status and display every warning', () => {
    const elements = new Map();
    const { state } = context({
        document: {
            addEventListener() {},
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, fakeElement());
                return elements.get(id);
            },
            createElement: fakeElement,
        },
    });
    const activity = vm.runInContext('segmentActivities[0]', state);
    const data = {
        ...payload([]), segment_status: 'unavailable',
        warning: 'No timed segments are currently available in the catalog for video routes.',
    };
    state.saveSegmentCache(activity, data);
    activity.data = state.readSegmentCache(activity);
    state.otherData = { ...payload([]), warning: 'Draft unavailable: current height or weight is missing.' };
    vm.runInContext('segmentActivities.push({activity_id:"12", data:otherData});', state);
    state.segmentLoadSummary();
    assert.equal(activity.data.segment_status, 'unavailable');
    assert.match(elements.get('loadStatus').textContent, /0 failed; 1 without available segment data/);
    assert.match(elements.get('loadStatus').textContent, /No timed segments.*video routes/);
    assert.match(elements.get('loadStatus').textContent, /Draft unavailable/);
    assert.equal(elements.get('retryBtn').hidden, true);
});

test('pruning and clearing cache preserves other users and unrelated dashboards', () => {
    const { state, store } = context();
    store.set('segmentPRs:7:old', '{}');
    store.set('segmentPRs:8:old', '{}');
    store.set('powerDashboardCurveV1:11', '{}');
    state.saveSegmentCache(vm.runInContext('segmentActivities[0]', state), payload([]));
    state.pruneSegmentCache();
    assert.equal(store.has('segmentPRs:7:old'), false);
    assert.equal(store.has('segmentPRs:7:11'), true);
    state.pruneSegmentCache(true);
    assert.equal(store.has('segmentPRs:7:11'), false);
    assert.equal(store.has('segmentPRs:8:old'), true);
    assert.equal(store.has('powerDashboardCurveV1:11'), true);
});

test('raw signed metrics are displayed without clipping', () => {
    const { state } = context();
    assert.equal(state.segmentMetric(-900, 'W', 1), '-900.0 W');
    assert.equal(state.segmentMetric(1500, 'W'), '1500 W');
    assert.equal(state.segmentMetric(0, 'm'), '0 m');
    assert.equal(state.segmentMetric(null, 'W'), '\u2014');
    assert.equal(state.segmentClock(10.5), '0:10.50');
    assert.equal(state.segmentClock(59.999), '1:00.00');
});
