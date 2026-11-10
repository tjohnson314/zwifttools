const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function selectElement() {
    let value = '';
    return {
        options: [], disabled: false,
        addEventListener() {},
        closest() { return { before() {} }; },
        set innerHTML(html) {
            this.options = [...html.matchAll(/<option value="([^"]*)">/g)].map(match => ({ value: match[1] }));
            value = this.options[0]?.value || '';
        },
        get value() { return value; },
        set value(next) { value = this.options.some(option => option.value === next) ? next : ''; },
    };
}

function selectorContext({ failTopPerformers = false, framePlaceholder } = {}) {
    const frames = [
        { id: 'road', name: 'Road', frameType: 'Standard' },
        { id: 'tt', name: 'TT', frameType: 'TT' },
        { id: 'gravel', name: 'Gravel', frameType: 'Gravel' },
        { id: 'mtb', name: 'MTB', frameType: 'MTB' },
        { id: 'halo', name: 'Halo', frameType: 'Standard' },
        { id: 'tron', name: 'Tron', frameType: 'Standard', hasBuiltInWheels: true },
    ];
    const wheels = [
        { id: 'road1', name: 'Road 1', fitsFrame: 'Standard,TT' },
        { id: 'road2', name: 'Road 2', fitsFrame: 'Standard,TT' },
        { id: 'gravel1', name: 'Gravel 1', fitsFrame: 'Gravel' },
        { id: 'mtb1', name: 'MTB 1', fitsFrame: 'MTB' },
        { id: 'halo1', name: 'Halo 1', fitsFrame: 'Standard', exclusiveFrameId: 'halo' },
    ];
    const elements = { frame: selectElement(), wheels: selectElement(), level: selectElement(), weight: {}, cda: {} };
    const requests = [];
    const topData = {
        '/api/top_performers': {
            frames: [{ id: 'road', default: false }, { id: 'halo' }, { id: 'tron' }],
            wheels: [{ id: 'Road|2', default: false }],
        },
        '/api/top_performers_tt': { frames: [{ id: 'tt' }], wheels: [{ id: 'Road|1' }] },
    };
    const context = vm.createContext({
        document: {
            getElementById(id) { return elements[id]; },
            createElement() {
                return {
                    checked: false, disabled: false, textContent: '', children: [], listeners: {},
                    append(...children) { this.children.push(...children); },
                    setAttribute() {},
                    addEventListener(event, callback) { this.listeners[event] = callback; },
                };
            },
        },
        console: { error() {} },
        fetch: async url => {
            requests.push(url);
            if (topData[url]) {
                return { ok: !failTopPerformers, status: 503, json: async () => topData[url] };
            }
            if (url === '/api/frames') return { ok: true, json: async () => frames };
            if (url === '/api/wheels') return { ok: true, json: async () => wheels };
            return { ok: true, json: async () => ({ weight_kg: 8, cda_bias: -0.01 }) };
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/bike_selector.js'), 'utf8'), context);
    const BikeSelector = vm.runInContext('BikeSelector', context);
    const selector = new BikeSelector({ frameSelect: 'frame', wheelSelect: 'wheels',
        levelSelect: 'level', weightOut: 'weight', cdaOut: 'cda', framePlaceholder });
    selector.frames = frames;
    selector.wheels = wheels;
    selector._populateFrames();
    async function toggleFilter(checked) {
        selector.topPerformersCheckbox.checked = checked;
        await selector.topPerformersCheckbox.listeners.change();
    }
    return { selector, elements, requests, toggleFilter, BikeSelector };
}

test('shared selector limits wheels to the selected road, TT, gravel or MTB category', async () => {
    const { selector, elements } = selectorContext();
    for (const [frameId, expected] of [['road', ['road1', 'road2']], ['tt', ['road1', 'road2']],
        ['gravel', ['gravel1']], ['mtb', ['mtb1']]]) {
        await selector.setConfig({ frameId });
        assert.deepEqual(Array.from(elements.wheels.options, option => option.value), expected);
        assert.ok(expected.includes(selector.getConfig().wheelId));
        assert.equal(selector.isReady(), true);
    }
});

test('shared selector replaces incompatible saved wheels when switching frame categories', async () => {
    const { selector, requests } = selectorContext();
    await selector.setConfig({ frameId: 'road', wheelId: 'road2' });
    await selector.setConfig({ frameId: 'gravel', wheelId: 'road2' });
    assert.equal(selector.getConfig().wheelId, 'gravel1');
    await selector.setConfig({ frameId: 'road', wheelId: 'gravel1' });
    assert.equal(selector.getConfig().wheelId, 'road1');
    assert.ok(requests[1].includes('frame_id=gravel&wheel_id=gravel1'));
    assert.ok(requests[2].includes('frame_id=road&wheel_id=road1'));
});

test('shared selector keeps owner-only fixed wheels out of other frames', async () => {
    const { selector, elements } = selectorContext();
    await selector.setConfig({ frameId: 'halo', wheelId: 'road1' });
    assert.deepEqual(Array.from(elements.wheels.options, option => option.value), ['halo1']);
    assert.equal(selector.getConfig().wheelId, 'halo1');
    await selector.setConfig({ frameId: 'road', wheelId: 'halo1' });
    assert.equal(selector.getConfig().wheelId, 'road1');
});

test('filter uses the union of both full lists and restores the catalogue when unchecked', async () => {
    const { selector, elements, requests, toggleFilter } = selectorContext();
    assert.equal(selector.topPerformersCheckbox.checked, false);
    await selector.setConfig({ frameId: 'tt', wheelId: 'road2', level: 4 });
    await toggleFilter(true);
    assert.deepEqual(Array.from(elements.frame.options, o => o.value), ['road', 'tt', 'halo', 'tron']);
    assert.deepEqual(Array.from(elements.wheels.options, o => o.value), ['road1', 'road2']);
    assert.equal(selector.getConfig().frameId, 'tt');
    assert.equal(selector.getConfig().wheelId, 'road2');
    assert.equal(selector.getConfig().level, 4);
    await toggleFilter(false);
    assert.ok(elements.frame.options.some(o => o.value === 'gravel'));
    await toggleFilter(true);
    assert.equal(requests.filter(url => url === '/api/top_performers').length, 1);
    assert.equal(requests.filter(url => url === '/api/top_performers_tt').length, 1);
});

test('filter removes unlisted wheels as well as frames and updates the stats selection', async () => {
    const { selector, elements, requests, toggleFilter } = selectorContext();
    selector.wheels.push({ id: 'slow', name: 'Slow', fitsFrame: 'Standard,TT' });
    await selector.setConfig({ frameId: 'road', wheelId: 'slow' });
    await toggleFilter(true);
    assert.deepEqual(Array.from(elements.wheels.options, o => o.value), ['road1', 'road2']);
    assert.equal(selector.getConfig().wheelId, 'road1');
    assert.ok(requests.at(-1).includes('frame_id=road&wheel_id=road1'));
    await toggleFilter(false);
    assert.ok(elements.wheels.options.some(o => o.value === 'slow'));
    await selector.setConfig({ frameId: 'gravel', level: 3 });
    await toggleFilter(true);
    assert.equal(selector.getConfig().frameId, 'road');
    assert.equal(selector.getConfig().level, 3);
    assert.ok(requests.at(-1).includes('frame_id=road&wheel_id=road1'));
});

test('filter retains owner-only and built-in wheels for listed frames', async () => {
    const { selector, elements, toggleFilter } = selectorContext();
    await toggleFilter(true);
    await selector.setConfig({ frameId: 'halo', wheelId: 'halo1' });
    assert.equal(selector.topPerformersCheckbox.checked, true);
    assert.deepEqual(Array.from(elements.wheels.options, o => o.value), ['halo1']);
    await selector.setConfig({ frameId: 'tron' });
    assert.equal(selector.topPerformersCheckbox.checked, true);
    assert.equal(elements.wheels.disabled, true);
    assert.equal(selector.getConfig().wheelId, null);
});

test('explicit out-of-list restores disable the filter rather than discard the requested bike', async () => {
    const { selector, toggleFilter } = selectorContext();
    await toggleFilter(true);
    await selector.setConfig({ frameId: 'gravel', wheelId: 'gravel1', level: 5 });
    assert.equal(selector.topPerformersCheckbox.checked, false);
    assert.equal(selector.getConfig().frameId, 'gravel');
    assert.equal(selector.getConfig().wheelId, 'gravel1');
    assert.equal(selector.getConfig().level, 5);
});

test('filter preserves an empty frame placeholder', async () => {
    const { selector, elements, toggleFilter } = selectorContext({ framePlaceholder: 'Select a frame' });
    await toggleFilter(true);
    assert.equal(selector.getConfig().frameId, '');
    assert.equal(elements.wheels.disabled, true);
    assert.equal(selector.isReady(), false);
});

test('filter reports loading errors without changing the selected bike, and permits retry', async () => {
    const { selector, requests, toggleFilter } = selectorContext({ failTopPerformers: true });
    await selector.setConfig({ frameId: 'gravel', wheelId: 'gravel1' });
    await toggleFilter(true);
    assert.equal(selector.topPerformersCheckbox.checked, false);
    assert.equal(selector.topPerformersCheckbox.disabled, false);
    assert.match(selector.filterError.textContent, /Could not load top performers/);
    assert.equal(selector.getConfig().frameId, 'gravel');
    await toggleFilter(true);
    assert.equal(requests.filter(url => url === '/api/top_performers').length, 2);
});

test('instances share top-performer requests but keep independent filters', async () => {
    const { selector, elements, requests, toggleFilter, BikeSelector } = selectorContext();
    await toggleFilter(true);
    elements.frame2 = selectElement();
    elements.wheels2 = selectElement();
    elements.level2 = selectElement();
    const second = new BikeSelector({ frameSelect: 'frame2', wheelSelect: 'wheels2',
        levelSelect: 'level2', weightOut: 'weight', cdaOut: 'cda' });
    await second.load();
    assert.equal(second.topPerformersCheckbox.checked, false);
    assert.ok(elements.frame2.options.some(o => o.value === 'gravel'));
    assert.ok(!elements.frame.options.some(o => o.value === 'gravel'));
    second.topPerformersCheckbox.checked = true;
    await second._onFilterChange();
    assert.equal(selector.topPerformersCheckbox.checked, true);
    assert.equal(requests.filter(url => url === '/api/top_performers').length, 1);
});