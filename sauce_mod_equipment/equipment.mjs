import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

// Map Zwift protobuf numeric IDs to human-readable names.
// Add entries here as you identify them.
const FRAME_NAMES = {
    3572756959: 'Zwift TT',
    3402382031: 'Cadex Tri',
    2629993294: 'Canyon Aeroad 2024',
    1029279076: 'Zwift Steel',
    1456463855: 'Zwift Concept Z1 (Tron)',
    3814159195: 'Handcycle',
    694663426: 'Specialized Tarmac SL8',
    142926447: 'Trek Emonda'
};
const WHEEL_NAMES = {
    1213183664: 'DT Swiss ARC 1100 DICUT 85/Disc (front)',
    590647095: 'DT Swiss ARC 1100 DICUT 85/Disc (rear)',
    2015793973: 'Princeton Wake 6560 Lava (front)',
    4293845976: 'Princeton Wake 6560 Lava (rear)',
    3114121871: 'ENVE SES 4.5 Pro (front)',
    635106022: 'ENVE SES 4.5 Pro (rear)',
    817265411: 'Princeton Mach TSV2/Blur Disc (front)',
    3710951039: 'Princeton Mach TSV2/Blur Disc (rear)',
    4029436085: 'Zwift 32mm Carbon (front)',
    4137014419: 'Zwift 32mm Carbon (rear)',
};

let fieldEls;
let debugEl;
let pollTimer;


export function main() {
    Common.settingsStore.setDefault({});
    fieldEls = {
        bikeFrame: document.querySelector('[data-field="bikeFrame"]'),
        bikeWheelFront: document.querySelector('[data-field="bikeWheelFront"]'),
        bikeWheelRear: document.querySelector('[data-field="bikeWheelRear"]'),
    };
    debugEl = document.getElementById('debug');

    refreshState();
    pollTimer = setInterval(refreshState, 5000);
}


function lookupName(id, map) {
    if (id == null) return '—';
    const name = map[id];
    return name != null ? name : String(id);
}


async function refreshState() {
    const log = [];
    try {
        let data = null;

        try {
            data = await Common.rpc.getPlayerProfile('self');
            log.push(`getPlayerProfile: ${data ? 'OK (' + Object.keys(data).length + ' keys)' : 'null'}`);
            if (data) {
                const bikeKeys = Object.keys(data).filter(k => /bike|frame|wheel|virtual/i.test(k));
                log.push(`  bike keys: ${bikeKeys.length ? bikeKeys.join(', ') : 'NONE'}`);
                bikeKeys.forEach(k => log.push(`    ${k}: ${data[k]}`));
            }
        } catch (e) {
            log.push(`getPlayerProfile ERROR: ${e?.message || e}`);
        }

        if (!data) {
            for (const el of Object.values(fieldEls)) el.textContent = '—';
            log.push('No data found');
            debugEl.textContent = log.join('\n');
            return;
        }

        const frameId = pickFirst(data, ['bikeFrame', 'bike_frame', 'frameId']);
        const frontId = pickFirst(data, ['bikeWheelFront', 'bike_wheel_front', 'wheelFront']);
        const rearId = pickFirst(data, ['bikeWheelRear', 'bike_wheel_rear', 'wheelRear']);

        fieldEls.bikeFrame.textContent = lookupName(frameId, FRAME_NAMES);
        fieldEls.bikeWheelFront.textContent = lookupName(frontId, WHEEL_NAMES);
        fieldEls.bikeWheelRear.textContent = lookupName(rearId, WHEEL_NAMES);

        log.push(`frame: ${frameId}, front: ${frontId}, rear: ${rearId}`);
    } catch (e) {
        for (const el of Object.values(fieldEls)) el.textContent = '—';
        log.push(`OUTER ERROR: ${e?.message || e}`);
    }
    debugEl.textContent = log.join('\n');
}


function pickFirst(obj, keys) {
    if (!obj) return undefined;
    for (const k of keys) {
        const v = obj[k];
        if (v != null && v !== '---' && v !== '') return v;
    }
    return undefined;
}