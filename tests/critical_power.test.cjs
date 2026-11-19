const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/critical_power.js'), 'utf8');

function calculator(storage, riderId, blocked = false) {
    const elements = new Map();
    const listeners = {};
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            value: '', textContent: '', hidden: false, className: '',
            classList: { add() {}, remove() {} },
            events: {},
            addEventListener(name, fn) { this.events[name] = fn; },
            appendChild() {},
        });
        return elements.get(id);
    }
    const context = vm.createContext({
        window: { ZwiftRider: { get: () => riderId ? { id: riderId } : null, ready: Promise.resolve() } },
        document: {
            getElementById: element,
            addEventListener(name, callback) { listeners[name] = callback; },
            createElement() {
                return {
                    set innerHTML(value) {},
                    querySelector(selector) { return element(selector.slice(1)); },
                };
            },
        },
        localStorage: {
            getItem(key) { if (blocked) throw new Error('blocked'); return storage.get(key) ?? null; },
            setItem(key, value) { if (blocked) throw new Error('blocked'); storage.set(key, value); },
            removeItem(key) { if (blocked) throw new Error('blocked'); storage.delete(key); },
        },
        console: { warn() {} },
    });
    vm.runInContext(source, context);
    vm.runInContext('renderChart = () => {}; renderExpectedPowerChart = () => {}; renderTable = () => {};', context);
    return { context, element, init: () => listeners.DOMContentLoaded() };
}

test('calculator keeps inputs and models separate for each selected rider', async () => {
    const storage = new Map();
    const first = calculator(storage, '1');
    await first.init();
    assert.equal(storage.size, 0, 'example inputs are not automatically assigned to a rider');
    first.element('power-0').value = '400';
    first.context.calculate();
    const firstModel = storage.get('criticalPowerModelV2:1');
    assert.ok(firstModel);

    const second = calculator(storage, '2');
    await second.init();
    assert.equal(second.element('power-0').value, '370');
    assert.equal(storage.has('criticalPowerModelV2:2'), false);
    second.element('power-0').value = '390';
    second.context.calculate();
    assert.notEqual(storage.get('criticalPowerModelV2:2'), firstModel);

    const restored = calculator(storage, '1');
    await restored.init();
    assert.equal(restored.element('power-0').value, '400');
    assert.equal(JSON.parse(storage.get('criticalPowerModelV2:1')).cp, JSON.parse(firstModel).cp);
    second.context.resetForm();
    assert.equal(storage.has('criticalPowerModelV2:2'), false);
    assert.ok(storage.has('criticalPowerModelV2:1'));
});

test('unassigned calculations remain usable but are not saved', async () => {
    const storage = new Map();
    const page = calculator(storage, null);
    await page.init();
    page.context.calculate();
    assert.equal(storage.size, 0);
    assert.match(page.element('cpStorageStatus').textContent, /Select an active rider/);
    assert.match(page.element('cpValue').textContent, / W$/);
});

test('legacy inputs are only assigned after explicit import', async () => {
    const legacy = JSON.stringify({ rows: [
        { duration: '300', power: '410' },
        { duration: '600', power: '330' },
        { duration: '1200', power: '290' },
        { duration: '1800', power: '275' },
    ], whatIfDuration: '', whatIfPower: '' });
    const storage = new Map([
        ['criticalPowerCalculatorInputsV1', legacy],
        ['criticalPowerModelV1', JSON.stringify({ cp: 999, wPrime: 20000 })],
    ]);
    const page = calculator(storage, '42');
    await page.init();
    assert.equal(storage.has('criticalPowerModelV2:42'), false);
    assert.equal(page.element('importLegacyCp').hidden, false);
    page.element('importLegacyCp').events.click();
    assert.equal(page.element('power-0').value, '410');
    assert.ok(storage.has('criticalPowerModelV2:42'));
    assert.equal(storage.get('criticalPowerCalculatorInputsV1'), legacy);
});

test('storage failures are visible', async () => {
    const page = calculator(new Map(), '1', true);
    await page.init();
    page.context.calculate();
    assert.match(page.element('cpStorageStatus').textContent, /cannot be saved/);
});
