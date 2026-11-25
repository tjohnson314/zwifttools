import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

let fieldEls;
let debugEl;
let refreshing = false;

export function main() {
    Common.settingsStore.setDefault({});
    fieldEls = {
        frame: document.querySelector('[data-field="bikeFrame"]'),
        wheelFront: document.querySelector('[data-field="bikeWheelFront"]'),
        wheelRear: document.querySelector('[data-field="bikeWheelRear"]'),
    };
    debugEl = document.getElementById('debug');
    refreshState();
    const timer = setInterval(refreshState, 5000);
    globalThis.addEventListener('pagehide', () => clearInterval(timer), {once: true});
}

async function refreshState() {
    if (refreshing) return;
    refreshing = true;
    try {
        const athlete = await Common.rpc.getAthlete('self', {
            required: ['equipment'],
            refresh: true,
        });
        const equipment = athlete?.equipment?.cycling;
        const log = [];
        for (const [key, element] of Object.entries(fieldEls)) {
            const item = equipment?.[key];
            element.textContent = item?.name || (item?.signature != null ? String(item.signature) : 'Unavailable');
            log.push(`${key}: ${item?.signature ?? 'unavailable'}`);
        }
        debugEl.textContent = equipment ?
            log.join('\n') : 'Equipment unavailable. Requires Sauce 2.4 and an available rider profile.';
    } catch (error) {
        for (const element of Object.values(fieldEls)) element.textContent = 'Unavailable';
        debugEl.textContent = `Equipment lookup failed: ${error.message || error}`;
        console.error('Equipment lookup failed:', error);
    } finally {
        refreshing = false;
    }
}
