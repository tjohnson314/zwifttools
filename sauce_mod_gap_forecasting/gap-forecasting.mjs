import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

let forecastHorizon = 0;
const directionEls = new Map();

export function main() {
    Common.settingsStore.setDefault({forecastHorizon: 0});
    forecastHorizon = Common.settingsStore.get('forecastHorizon') ?? 0;

    Common.settingsStore.addEventListener('changed', () => {
        forecastHorizon = Common.settingsStore.get('forecastHorizon') ?? 0;
        updateForecastVisibility();
    });

    for (const panel of document.querySelectorAll('.gap-panel')) {
        directionEls.set(panel.dataset.direction, {
            gapValue: panel.querySelector('.gap-value'),
            forecastRow: panel.querySelector('.forecast-row'),
            forecastValue: panel.querySelector('.forecast-value'),
            forecastLabel: panel.querySelector('.forecast-label'),
        });
    }
    updateForecastVisibility();

    Common.subscribe('nearby', onNearby);
}

function onNearby(data) {
    if (!data || !data.length) {
        resetDisplay();
        return;
    }

    const watching = data.find(x => x.watching);
    if (!watching) {
        resetDisplay();
        return;
    }

    const mySpeed = watching?.state?.speed ?? 0; // km/h

    updateDirection('ahead', findClosestRider(data, rider => rider.gapDistance < 0), mySpeed);
    updateDirection('behind', findClosestRider(data, rider => rider.gapDistance > 0), mySpeed);
}

function updateForecastVisibility() {
    for (const {forecastRow} of directionEls.values()) {
        if (forecastRow) {
            forecastRow.style.display = forecastHorizon > 0 ? '' : 'none';
        }
    }
}

function resetDisplay() {
    updateDirection('ahead', null, 0);
    updateDirection('behind', null, 0);
}

function findClosestRider(data, predicate) {
    const riders = data.filter(x => !x.watching && x.gapDistance != null && predicate(x));
    if (!riders.length) {
        return null;
    }
    return riders.reduce((closest, rider) => Math.abs(rider.gapDistance) < Math.abs(closest.gapDistance) ? rider : closest);
}

function updateDirection(direction, rider, mySpeed) {
    const els = directionEls.get(direction);
    if (!els?.gapValue) {
        return;
    }

    if (!rider) {
        els.gapValue.textContent = '—';
        if (els.forecastValue) {
            els.forecastValue.textContent = '—';
        }
        if (els.forecastLabel) {
            els.forecastLabel.textContent = `in ${forecastHorizon}s`;
        }
        return;
    }

    const signedGap = rider.gapDistance;
    const currentGap = Math.abs(signedGap);
    els.gapValue.textContent = currentGap.toFixed(2);

    if (!els.forecastValue) {
        return;
    }

    if (forecastHorizon > 0) {
        const theirSpeed = rider.state?.speed ?? 0; // km/h
        const speedDiffMs = (mySpeed - theirSpeed) / 3.6;
        const directionSign = Math.sign(signedGap);
        const forecastedGap = Math.max(0, currentGap + directionSign * speedDiffMs * forecastHorizon);
        els.forecastValue.textContent = forecastedGap.toFixed(2);
        if (els.forecastLabel) {
            els.forecastLabel.textContent = `in ${forecastHorizon}s`;
        }
    } else {
        els.forecastValue.textContent = '—';
        if (els.forecastLabel) {
            els.forecastLabel.textContent = 'in 0s';
        }
    }
}

export function settingsMain() {
    Common.initInteractionListeners();
    Common.initSettingsForm('form#options')();
}
