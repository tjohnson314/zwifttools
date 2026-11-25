const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

async function resolver() {
    const {MODEL} = await import('../mods/sauce_mod_power_breakdown/bike-model.mjs');
    const {createBikeResolver} = await import('../shared/sauce_bike_setup.mjs');
    return {MODEL, resolve: createBikeResolver(MODEL)};
}

function equipment() {
    return {
        frame: {name: 'Specialized Tarmac SL8', signature: 694663426},
        wheelFront: {name: 'Princeton Wake 6560 Lava', signature: 2015793973,
            imageName: 'Wheel_PrincetonCarbonWorksWake6560Red'},
        wheelRear: {name: 'Princeton Wake 6560 Lava', signature: 4293845976,
            imageName: 'Wheel_PrincetonCarbonWorksWake6560Red'},
    };
}

test('Sauce records resolve by exact frame aliases and wheel asset IDs, not guessed display names', async () => {
    const {MODEL, resolve} = await resolver();
    const setup = resolve(equipment(), 5);
    assert.equal(setup.error, undefined);
    assert.equal(setup.frame.id, 'SpecializedTarmacSL82024');
    assert.equal(setup.wheel.id, 'princetoncarbonworkswake6560red');
    assert.equal(setup.bikeWeightKg, (setup.frame.wt[5] + setup.wheel.wt) / 1000);
    assert.equal(setup.wheelBias, setup.wheel.cda);
    assert.match(setup.label, /assumed L5/);
    const tt = resolve({...equipment(), frame: {name: 'Zwift TT'}}, 0);
    assert.equal(tt.wheelBias, tt.wheel.cdaTt);
    assert.equal(tt.frameBias, MODEL.frames.find(frame => frame.id === 'Zwift_TT').cda[0]);
});

test('unknown, missing, mixed, and invalid equipment assumptions are explicitly unavailable', async () => {
    const {resolve} = await resolver();
    for (const data of [undefined, {}, {...equipment(), wheelRear: undefined}]) {
        assert.match(resolve(data, 5).error, /Equipment unavailable/);
    }
    const unknown = resolve({...equipment(), frame: {name: 'Unknown frame', signature: 0}}, 5);
    assert.match(unknown.error, /No physics stats for frame: Unknown frame/);
    assert.equal(unknown.frameBias, undefined);
    const unknownWheel = resolve({...equipment(), wheelFront: {signature: 123}}, 5);
    assert.match(unknownWheel.error, /No physics stats for wheels/);
    assert.match(unknownWheel.label, /123/);
    assert.match(resolve({...equipment(), wheelRear: {name: 'Enve SES 4.5 Pro'}}, 5).error, /Mixed wheelsets/);
    assert.match(resolve(equipment(), 6).error, /Upgrade assumption/);
});

test('standalone mods bundle the shared resolver and resolve integrated wheels rather than dropping wheel mass', async () => {
    const {MODEL} = await resolver();
    const source = await fs.readFile(path.join(__dirname, '..', 'shared', 'sauce_bike_setup.mjs'), 'utf8');
    for (const mod of ['sauce_mod_power_breakdown', 'sauce_mod_nearby_draft']) {
        const file = path.join(__dirname, '..', 'mods', mod, 'bike-setup.mjs');
        assert.equal(await fs.readFile(file, 'utf8'), source);
    }
    const {createBikeResolver} = await import('../mods/sauce_mod_power_breakdown/bike-setup.mjs');
    const setup = createBikeResolver(MODEL)({
        frame: {name: 'Zwift Concept Z1'},
        wheelFront: {name: 'Zwift Concept', imageName: 'Wheel_ZwiftConcept'},
        wheelRear: {name: 'Zwift Concept', imageName: 'Wheel_ZwiftConcept'},
    }, 5);
    assert.equal(setup.error, undefined);
    assert.ok(setup.wheel.wt > 0);
});

test('Equipment Peek uses the stock refresh RPC, preserves unknown IDs, and reports failures', async () => {
    const source = (await fs.readFile(path.join(__dirname,
        '..', 'mods', 'sauce_mod_equipment', 'equipment.mjs'), 'utf8'))
        .replace(/^import .*;\r?\n/m, '').replace('export function main()', 'function main()');
    const fields = new Map();
    const debug = {};
    let profile = {equipment: {cycling: equipment()}};
    let failure;
    let timer;
    const context = vm.createContext({
        Common: {enableSentry() {}, settingsStore: {setDefault() {}},
            rpc: {async getAthlete(id, options) {
                assert.equal(id, 'self');
                assert.deepEqual(Array.from(options.required), ['equipment']);
                assert.equal(options.refresh, true);
                if (failure) throw failure;
                return profile;
            }}},
        document: {
            querySelector(selector) { const field = {}; fields.set(selector, field); return field; },
            getElementById() { return debug; },
        },
        console: {error() {}},
        setInterval(callback) { timer = callback; },
        clearInterval() {},
        addEventListener() {},
    });
    vm.runInContext(source, context);
    vm.runInContext('main()', context);
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual([...fields.values()].map(field => field.textContent),
        ['Specialized Tarmac SL8', 'Princeton Wake 6560 Lava', 'Princeton Wake 6560 Lava']);
    profile = {equipment: {cycling: {frame: {signature: 0}}}};
    await timer();
    assert.equal([...fields.values()][0].textContent, '0');
    assert.match(debug.textContent, /wheelFront: unavailable/);
    failure = new Error('RPC disconnected');
    await timer();
    assert.ok([...fields.values()].every(field => field.textContent === 'Unavailable'));
    assert.match(debug.textContent, /RPC disconnected/);
});
