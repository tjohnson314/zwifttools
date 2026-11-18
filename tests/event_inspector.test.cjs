const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

async function inspector({ search = '', responses = [] } = {}) {
    const elements = {};
    const requests = [];
    const revoked = [];
    const blobs = [];
    const copied = [];
    const window = {
        location: { search, href: `http://localhost/event-inspector${search}` },
        history: { replaceState(state, title, url) { window.location.href = String(url); } },
    };
    class BrowserURL extends URL {
        static createObjectURL(blob) { blobs.push(blob); return `blob:${blobs.length}`; }
        static revokeObjectURL(url) { revoked.push(url); }
    }
    const context = vm.createContext({
        URL: BrowserURL, URLSearchParams, Blob, window,
        navigator: { clipboard: { writeText: async text => copied.push(text) } },
        document: {
            getElementById(id) {
                return elements[id] ||= {
                    value: '', textContent: '', disabled: false, hidden: true, dataset: {},
                    listeners: {},
                    addEventListener(name, callback) { this.listeners[name] = callback; },
                    removeAttribute(name) { delete this[name]; },
                };
            },
        },
        fetch: async (url, options) => {
            requests.push({ url, options });
            const response = responses.shift();
            if (response instanceof Error) throw response;
            return response;
        },
    });
    vm.runInContext(fs.readFileSync(
        path.join(__dirname, '..', 'static', 'js', 'event_inspector.js'), 'utf8'), context);
    await new Promise(resolve => setImmediate(resolve));
    return {
        elements, requests, revoked, blobs, copied, window,
        async load(value) {
            elements['event-id'].value = value;
            await elements['event-form'].listeners.submit({ preventDefault() {} });
        },
    };
}

test('shared event link preserves full JSON text, large IDs, copy, and download', async () => {
    const text = '{\n  "id": 5795670,\n  "hash": 9223372036854775807,\n  "name": "<script>unsafe()</script>"\n}';
    const ui = await inspector({
        search: '?event_id=005795670',
        responses: [{ ok: true, json: async () => ({ raw_json: text, summary_html: '<article>Pen C: Anvil</article>' }) }],
    });
    assert.equal(ui.requests[0].url, '/api/events/5795670?view=readable');
    assert.equal(ui.requests[0].options.cache, 'no-store');
    assert.equal(ui.elements['event-json'].textContent, text);
    assert.equal(ui.elements['event-result'].hidden, false);
    assert.equal(ui.elements['event-summary'].innerHTML, '<article>Pen C: Anvil</article>');
    assert.equal(await ui.blobs[0].text(), text);
    assert.equal(ui.elements['download-json'].download, 'zwift-event-5795670.json');
    await ui.elements['copy-json'].listeners.click();
    assert.deepEqual(ui.copied, [text]);
    assert.equal(new URL(ui.window.location.href).searchParams.get('event_id'), '5795670');
});

test('invalid input is reported without fetching', async () => {
    const ui = await inspector();
    for (const value of ['0', '-1', 'abc', '1.5', '1'.repeat(21)]) {
        await ui.load(value);
        assert.equal(ui.elements['event-status'].dataset.error, 'true');
        assert.equal(ui.elements['event-result'].hidden, true);
    }
    assert.equal(ui.requests.length, 0);
});

test('a failed lookup clears previous results and download and reports upstream error', async () => {
    const ui = await inspector({ responses: [
        { ok: true, json: async () => ({ raw_json: '{"id":1}', summary_html: '<article>Previous event</article>' }) },
        { ok: false, status: 403, json: async () => ({ error: 'Event inaccessible.' }) },
    ] });
    await ui.load('1');
    await ui.load('2');
    assert.equal(ui.elements['event-result'].hidden, true);
    assert.equal(ui.elements['event-json'].textContent, '');
    assert.equal(ui.elements['event-summary'].innerHTML, '');
    assert.equal(ui.elements['download-json'].href, undefined);
    assert.deepEqual(ui.revoked, ['blob:1']);
    assert.match(ui.elements['event-status'].textContent, /Event inaccessible/);
    assert.equal(ui.elements['load-event'].disabled, false);
    assert.equal(ui.elements['event-id'].disabled, false);
});

test('network failures are reported and controls become available again', async () => {
    const ui = await inspector({ responses: [new Error('Network unavailable')] });
    await ui.load('1');
    assert.match(ui.elements['event-status'].textContent, /Network unavailable/);
    assert.equal(ui.elements['event-status'].dataset.error, 'true');
    assert.equal(ui.elements['event-result'].hidden, true);
    assert.equal(ui.elements['load-event'].disabled, false);
});
