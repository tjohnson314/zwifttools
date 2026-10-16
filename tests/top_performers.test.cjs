const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function element() {
    const listeners = {};
    const classes = new Set();
    return {
        children: [], attributes: {}, disabled: true, textContent: '',
        classList: {
            toggle(name, active) { if (active) classes.add(name); else classes.delete(name); },
            contains(name) { return classes.has(name); },
        },
        addEventListener(name, callback) { listeners[name] = callback; },
        click() { listeners.click(); },
        setAttribute(name, value) { this.attributes[name] = value; },
        set innerHTML(value) {
            this.children = [];
            this.input = element();
            this.input.checked = value.includes('checked');
        },
        querySelector() { return this.input; },
        appendChild(child) { this.children.push(child); },
    };
}

async function chartContext({ bike = 'road', failure = false } = {}) {
    const referenceFrame = { type: 'STANDARD', cda: 0.012, weight: 5050 };
    const referenceWheel = { cda: 0, cdaTT: -0.01, weight: 2350 };
    const wheel = { id: 'wheel', name: 'Wheel', cda: 0.001, cdaTT: -0.004, weight: 2000, default: true };
    const datasets = {
        '/api/top_performers': {
            referenceFrame, referenceWheel, wheels: [wheel],
            frames: [{ id: 'road', name: 'Road', type: 'STANDARD',
                cda: [0.02, 0.01], weight: [6000, 5500], default: true }],
        },
        '/api/top_performers_tt': {
            referenceFrame, referenceWheel, wheels: [wheel],
            frames: [
                { id: 'tt', name: 'TT', type: 'TT',
                    cda: [-0.015, -0.02], weight: [4000, 3500], default: true },
                { id: 'halo', name: 'Halo TT', type: 'TT', halo: true,
                    builtInWheel: { name: 'Built-in', cda: -0.006, weight: 1000 },
                    cda: [-0.02, -0.025], weight: [4000, 3500], default: true },
            ],
        },
    };
    const elements = {};
    const charts = [];
    const requests = [];
    const window = {
        location: { href: `http://localhost/top-performers${bike === 'tt' ? '?bike=tt' : ''}` },
        history: { replaceState(state, title, url) { window.location.href = String(url); } },
    };
    const context = vm.createContext({
        window, URL, console,
        document: {
            getElementById(id) { return elements[id] ||= element(); },
            createElement() { return element(); },
        },
        Chart: function(canvas, config) {
            this.config = config;
            this.destroy = () => { this.destroyed = true; };
            charts.push(this);
        },
        fetch: async url => {
            requests.push(url);
            return { ok: !failure, status: failure ? 503 : 200, json: async () => datasets[url] };
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'static', 'js', 'top_performers.js'), 'utf8'), context);
    await new Promise(resolve => setImmediate(resolve));
    return { elements, charts, requests, window,
        points: () => charts.at(-1).config.data.datasets[0].data };
}

test('road mode preserves raw negative values and switches to TT-specific wheel drag', async () => {
    const { elements, points, charts, requests, window } = await chartContext();
    assert.equal(points().length, 1);
    assert.ok(Math.abs(points()[0].x - (-0.009)) < 1e-10);
    assert.equal(points()[0].y, -600);
    assert.equal(elements.roadBtn.attributes['aria-pressed'], 'true');
    elements.ttBtn.click();
    assert.equal(points().length, 2);
    assert.ok(Math.abs(points()[0].x - 0.031) < 1e-10);
    assert.equal(points()[0].y, 1400);
    assert.ok(Math.abs(points()[1].x - 0.038) < 1e-10);
    assert.equal(points()[1].y, 2400);
    assert.equal(elements.ttBtn.attributes['aria-pressed'], 'true');
    assert.equal(elements.roadBtn.attributes['aria-pressed'], 'false');
    assert.equal(new URL(window.location.href).searchParams.get('bike'), 'tt');
    assert.equal(charts[0].destroyed, true);
    assert.deepEqual(requests.sort(), ['/api/top_performers', '/api/top_performers_tt']);
});

test('switching retains upgrade level and separate frame and wheel selections', async () => {
    const { elements, points, window } = await chartContext();
    elements.lvl5Btn.click();
    assert.equal(points()[0].y, -100);
    elements.frameNone.click();
    assert.equal(points().length, 0);
    elements.ttBtn.click();
    assert.equal(points().length, 2);
    assert.equal(points()[0].y, 1900);
    assert.equal(elements.lvl5Btn.classList.contains('active'), true);
    elements.wheelNone.click();
    assert.equal(points().length, 1);
    assert.match(points()[0].label, /built-in/);
    elements.roadBtn.click();
    assert.equal(points().length, 0);
    elements.frameAll.click();
    assert.equal(points().length, 1);
    assert.equal(points()[0].y, -100);
    assert.equal(new URL(window.location.href).searchParams.has('bike'), false);
    elements.ttBtn.click();
    assert.equal(points().length, 1);
    elements.wheelAll.click();
    assert.equal(points().length, 2);
});

test('shared TT links initialize the TT chart and active toggle', async () => {
    const { elements, points } = await chartContext({ bike: 'tt' });
    assert.match(points()[0].label, /^TT/);
    assert.equal(elements.ttBtn.attributes['aria-pressed'], 'true');
    assert.equal(elements.roadBtn.attributes['aria-pressed'], 'false');
    assert.match(elements.pointCount.textContent, /2 TT data points/);
});

test('HTTP failures produce a visible error instead of a misleading chart', async () => {
    const { elements, charts } = await chartContext({ failure: true });
    assert.equal(charts.length, 0);
    assert.match(elements.pointCount.textContent, /Failed to load data:.*503/);
});
