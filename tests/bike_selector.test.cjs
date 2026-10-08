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
        set innerHTML(html) {
            this.options = [...html.matchAll(/<option value="([^"]*)">/g)].map(match => ({ value: match[1] }));
            value = this.options[0]?.value || '';
        },
        get value() { return value; },
        set value(next) { value = this.options.some(option => option.value === next) ? next : ''; },
    };
}

function selectorContext() {
    const frames = [
        { id: 'road', name: 'Road', frameType: 'Standard' },
        { id: 'tt', name: 'TT', frameType: 'TT' },
        { id: 'gravel', name: 'Gravel', frameType: 'Gravel' },
        { id: 'mtb', name: 'MTB', frameType: 'MTB' },
        { id: 'halo', name: 'Halo', frameType: 'Standard' },
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
    const context = vm.createContext({
        document: { getElementById(id) { return elements[id]; } },
        console,
        fetch: async url => {
            requests.push(url);
            return { ok: true, json: async () => ({ weight_kg: 8, cda_bias: -0.01 }) };
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/bike_selector.js'), 'utf8'), context);
    const BikeSelector = vm.runInContext('BikeSelector', context);
    const selector = new BikeSelector({ frameSelect: 'frame', wheelSelect: 'wheels',
        levelSelect: 'level', weightOut: 'weight', cdaOut: 'cda' });
    selector.frames = frames;
    selector.wheels = wheels;
    selector._populateFrames();
    return { selector, elements, requests };
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