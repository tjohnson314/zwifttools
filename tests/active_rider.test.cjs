const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function page({ saved, href = 'http://localhost/power', fetch, unavailableStorage = false } = {}) {
    const storage = new Map(saved ? [['zwifttools.activeRider', JSON.stringify(saved)]] : []);
    const elements = new Map();
    const events = {};
    const navigations = [];
    const requests = [];
    const warnings = [];
    const context = vm.createContext({
        URL, console: { warn(...args) { warnings.push(args); } },
        document: { getElementById(id) {
            if (!elements.has(id)) elements.set(id, {
                value: '', hidden: false, disabled: false, textContent: '', listeners: {},
                addEventListener(name, handler) { this.listeners[name] = handler; },
            });
            return elements.get(id);
        } },
        window: {
            location: { href, assign(value) { navigations.push(value); } },
            addEventListener(name, handler) { events[name] = handler; },
        },
        localStorage: {
            getItem(key) {
                if (unavailableStorage) throw new Error('Storage disabled');
                return storage.get(key) ?? null;
            },
            setItem(key, value) {
                if (unavailableStorage) throw new Error('Storage disabled');
                storage.set(key, value);
            },
            removeItem(key) {
                if (unavailableStorage) throw new Error('Storage disabled');
                storage.delete(key);
            },
        },
        fetch: async (url, options) => {
            requests.push({ url, options });
            if (!fetch) throw new Error('Unexpected request');
            return fetch(url, options);
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/active_rider.js'), 'utf8'), context);
    return { api: context.window.ZwiftRider, elements, storage, requests, navigations, events, warnings,
        submit(id) {
            elements.get('activeRiderId').value = id;
            return elements.get('activeRiderForm').listeners.submit({ preventDefault() {} });
        },
    };
}

const rider = { id: '7', name: 'Test Rider' };
const ok = data => ({ ok: true, json: async () => data });

test('restores confirmed rider and scopes API requests without storing credentials or measurements', async () => {
    const scope = page({ saved: rider });
    await scope.api.ready;
    assert.deepEqual(JSON.parse(JSON.stringify(scope.api.get())), rider);
    assert.equal(scope.elements.get('activeRiderStatus').textContent, 'Test Rider \u00b7 7');
    assert.equal(scope.api.url('/api/my_activities?start=30'), '/api/my_activities?start=30&zwift_id=7');
    assert.equal(scope.api.url('/api/my_profile', '8'), '/api/my_profile?zwift_id=8');
    assert.equal(scope.requests.length, 0);
    scope.api.get().id = '999';
    assert.equal(scope.api.get().id, '7');
});

test('setting a rider validates via server and saves only name and ID before reloading', async () => {
    const scope = page({ href: 'http://localhost/my-activities?start=30#details',
        fetch: () => ok({ id: 8, name: 'New Rider', weight_kg: 80, ftp_w: 300 }) });
    await scope.submit('8');
    assert.deepEqual(JSON.parse(scope.storage.get('zwifttools.activeRider')), { id: '8', name: 'New Rider' });
    assert.equal(scope.requests[0].url, '/api/profiles/8');
    assert.equal(scope.requests[0].options.cache, 'no-store');
    assert.equal(scope.navigations[0], 'http://localhost/my-activities?start=30&zwift_id=8#details');
});

test('failed or invalid lookup preserves current rider and reports the failure', async () => {
    const scope = page({ saved: rider, fetch: () => ({
        ok: false, status: 403, json: async () => ({ error: 'Profile is private' }),
    }) });
    await scope.submit('8');
    assert.equal(scope.api.get().id, '7');
    assert.match(scope.elements.get('activeRiderStatus').textContent, /Profile is private.*Test Rider/);
    assert.equal(scope.navigations.length, 0);
    await scope.submit('abc');
    assert.match(scope.elements.get('activeRiderStatus').textContent, /positive number/);
    assert.equal(scope.requests.length, 1);
    assert.deepEqual(JSON.parse(scope.storage.get('zwifttools.activeRider')), rider);
});

test('clearing invalidates an in-flight lookup instead of restoring the old rider later', async () => {
    let finish;
    const scope = page({ saved: rider, fetch: () => new Promise(resolve => { finish = resolve; }) });
    const setting = scope.submit('8');
    await Promise.resolve();
    scope.elements.get('clearActiveRider').listeners.click();
    finish(ok({ id: '8', name: 'Other' }));
    await setting;
    assert.equal(scope.storage.has('zwifttools.activeRider'), false);
    assert.deepEqual(scope.navigations, ['http://localhost/power?zwift_id=']);
});

test('shared URL rider takes precedence over saved rider and is remembered for other pages', async () => {
    const scope = page({ saved: rider, href: 'http://localhost/power?zwift_id=8',
        fetch: () => ok({ id: '8', name: 'Shared Rider' }) });
    await scope.api.ready;
    assert.equal(scope.api.get().id, '8');
    assert.equal(JSON.parse(scope.storage.get('zwifttools.activeRider')).id, '8');
    assert.equal(scope.api.url('/api/segment_prs/activities'), '/api/segment_prs/activities?zwift_id=8');
});

test('missing selection never defaults to a server-account rider', async () => {
    const scope = page();
    await scope.api.ready;
    assert.equal(scope.api.get(), null);
    assert.throws(() => scope.api.url('/api/my_profile'), /Enter a Zwift ID/);
    await assert.rejects(scope.api.profile(), /Enter a Zwift ID/);
    assert.equal(scope.requests.length, 0);
});

test('cross-tab selection changes reload context but unrelated settings do not', () => {
    const scope = page({ saved: rider, href: 'http://localhost/power?zwift_id=7&other=1' });
    scope.events.storage({ key: 'bikeSettings' });
    assert.equal(scope.navigations.length, 0);
    scope.events.storage({ key: 'zwifttools.activeRider' });
    assert.equal(scope.navigations[0], 'http://localhost/power?other=1');
});

test('blocked storage has an explicit warning and selection still works through the URL', async () => {
    const scope = page({ unavailableStorage: true, fetch: () => ok(rider) });
    await scope.submit('7');
    assert.equal(scope.navigations[0], 'http://localhost/power?zwift_id=7');
    assert.ok(scope.warnings.length);
});
