import * as Common from '/pages/src/common.mjs';
import {MODEL} from './bike-model.mjs';
import * as Surface from './surface.mjs';

Common.enableSentry();

const {BASE_CDA, REF_HEIGHT_M, REF_WEIGHT_KG, HEIGHT_EXPONENT, WEIGHT_EXPONENT,
       AIR_DENSITY, GRAVITY} = MODEL.constants;

// Category definitions (order = display order); `cls` selects the bar colour.
const CATEGORIES = [
    {key: 'rider',    label: 'Rider power',      cls: 'in'},
    {key: 'draft',    label: 'Draft',            cls: 'in'},
    {key: 'aero',     label: 'Aero drag',        cls: 'out'},
    {key: 'rolling',  label: 'Rolling resist.',  cls: 'out'},
    {key: 'pe',       label: 'Δ Potential (climb)', cls: 'pe'},
    {key: 'ke',       label: 'Δ Kinetic (accel)',  cls: 'ke'},
    {key: 'residual', label: 'Residual',         cls: 'res'},
];

const DEFAULTS = {
    frameId: 'Zwift_TT',
    wheelId: '',
    upgradeLevel: 5,
    useZwiftWeight: true,
    weightKg: 75,
    heightCm: 183,
    autoSurface: true,
    manualSurface: 'Tarmac',
    smoothSec: 3,
    draftIsWatts: true,      // Sauce state.draft interpreted as watts
};

let settings;
let els = {};
let rowEls = {};            // key -> {value, barPos, barNeg}

// Smoothed running state.
let prev = null;            // {worldTime, altitude, speedMps}
let sm = {dAltDt: 0, dvDt: 0};     // smoothed derivatives
let smVals = null;          // smoothed category watts
let lastAthlete = {weightKg: null, heightCm: null};


function loadSettings() {
    settings = {...DEFAULTS};
    for (const k of Object.keys(DEFAULTS)) {
        const v = Common.settingsStore.get(k);
        if (v !== undefined) settings[k] = v;
    }
}


export function main() {
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();

    cacheEls();
    buildRows();
    updateConfigSummary();

    // Settings live in a separate child window; pick up changes as they happen.
    Common.settingsStore.addEventListener('changed', () => {
        loadSettings();
        updateConfigSummary();
    });
    Common.subscribe('athlete/self', onSelfUpdate);
}


// Entry point for the settings child window (shares the settings key).
export function settingsMain() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();

    cacheEls();
    populatePickers();
    bindSettings();
}


function cacheEls() {
    els.rows = document.querySelector('.breakdown-rows');
    els.bikeSummary = document.querySelector('.bike-summary');
    els.cdaSummary = document.querySelector('.cda-summary');
    els.massSummary = document.querySelector('.mass-summary');
    els.surfaceStyle = document.querySelector('.surface-style');
    els.surfaceType = document.querySelector('.surface-type');
    els.surfaceCrr = document.querySelector('.surface-crr');
    els.checkSum = document.querySelector('.check-sum');
    els.frameSel = document.querySelector('[data-set="frameId"]');
    els.wheelSel = document.querySelector('[data-set="wheelId"]');
    els.upgradeSel = document.querySelector('[data-set="upgradeLevel"]');
    els.useZwiftWeight = document.querySelector('[data-set="useZwiftWeight"]');
    els.weightKg = document.querySelector('[data-set="weightKg"]');
    els.heightCm = document.querySelector('[data-set="heightCm"]');
    els.smoothSec = document.querySelector('[data-set="smoothSec"]');
    els.autoSurface = document.querySelector('[data-set="autoSurface"]');
    els.manualSurface = document.querySelector('[data-set="manualSurface"]');
}


function buildRows() {
    els.rows.innerHTML = '';
    for (const cat of CATEGORIES) {
        const row = document.createElement('div');
        row.className = `brk-row ${cat.cls}`;
        row.innerHTML = `
            <div class="brk-label">${cat.label}</div>
            <div class="brk-bar"><div class="brk-neg"></div><div class="brk-pos"></div></div>
            <div class="brk-value">—</div>`;
        els.rows.appendChild(row);
        rowEls[cat.key] = {
            value: row.querySelector('.brk-value'),
            barPos: row.querySelector('.brk-pos'),
            barNeg: row.querySelector('.brk-neg'),
        };
    }
}


function populatePickers() {
    // Flat alphabetical frame list (already sorted by name in the data).
    els.frameSel.innerHTML = '';
    for (const f of MODEL.frames) {
        const o = document.createElement('option');
        o.value = f.id;
        o.textContent = f.name;
        els.frameSel.appendChild(o);
    }
    els.frameSel.value = settings.frameId;

    els.wheelSel.innerHTML = '<option value="">(Frame built-in wheels)</option>';
    for (const w of MODEL.wheels) {
        const o = document.createElement('option');
        o.value = w.id;
        o.textContent = `${w.make} ${w.model}`.trim();
        els.wheelSel.appendChild(o);
    }
    els.wheelSel.value = settings.wheelId;
    els.upgradeSel.value = String(settings.upgradeLevel);

    els.manualSurface.innerHTML = '';
    for (const type of ['Tarmac', 'Brick', 'Cobbles', 'Wood', 'Dirt', 'Gravel', 'Sand', 'Snow', 'Grass']) {
        const o = document.createElement('option');
        o.value = type; o.textContent = type;
        els.manualSurface.appendChild(o);
    }
    els.manualSurface.value = settings.manualSurface;

    // Reflect current settings into the other controls.
    els.useZwiftWeight.checked = settings.useZwiftWeight;
    els.weightKg.value = settings.weightKg;
    els.heightCm.value = settings.heightCm;
    els.smoothSec.value = settings.smoothSec;
    els.autoSurface.checked = settings.autoSurface;
    applyWeightEnable();
    applySurfaceEnable();
}


function bindSettings() {
    for (const el of document.querySelectorAll('[data-set]')) {
        const key = el.dataset.set;
        el.addEventListener('change', () => {
            let v;
            if (el.type === 'checkbox') v = el.checked;
            else if (el.type === 'number') v = parseFloat(el.value);
            else if (key === 'upgradeLevel') v = parseInt(el.value, 10);
            else v = el.value;
            settings[key] = v;
            Common.settingsStore.set(key, v);
            applyWeightEnable();
            applySurfaceEnable();
            updateConfigSummary();
        });
    }
}


function applyWeightEnable() {
    const manual = !els.useZwiftWeight.checked;
    els.weightKg.disabled = !manual;
    els.heightCm.disabled = !manual;
}

function applySurfaceEnable() {
    els.manualSurface.disabled = els.autoSurface.checked;
}


function getBikeSetup() {
    const frame = MODEL.frames.find(f => f.id === settings.frameId) || MODEL.frames[0];
    const wheel = settings.wheelId ? MODEL.wheels.find(w => w.id === settings.wheelId) : null;
    const lvl = Math.max(0, Math.min(5, settings.upgradeLevel | 0));
    const frameBias = frame.cda[lvl];
    const isTT = (frame.type || '').toUpperCase() === 'TT';
    const wheelBias = wheel ? (isTT ? wheel.cdaTt : wheel.cda) : 0;
    const bikeWeightKg = (frame.wt[lvl] + (wheel ? wheel.wt : 0)) / 1000;
    const bikeType = MODEL.frameTypeToBikeType[frame.type] || 'road_bike';
    return {frame, wheel, frameBias, wheelBias, bikeWeightKg, bikeType};
}


function computeCdA(setup, heightM, riderWeightKg) {
    // CdA = (BASE_CDA + frameBias + wheelBias) · (H/refH)^hExp · (M/refM)^wExp
    const scale = Math.pow(heightM / REF_HEIGHT_M, HEIGHT_EXPONENT) *
                  Math.pow(riderWeightKg / REF_WEIGHT_KG, WEIGHT_EXPONENT);
    return (BASE_CDA + setup.frameBias + setup.wheelBias) * scale;
}


function updateConfigSummary() {
    const setup = getBikeSetup();
    const {riderWeightKg, heightM} = getRider();
    const cda = computeCdA(setup, heightM, riderWeightKg);
    const mass = riderWeightKg + setup.bikeWeightKg;
    const wheelName = setup.wheel ? `${setup.wheel.make} ${setup.wheel.model}`.trim() : 'built-in';
    els.bikeSummary.textContent =
        `${setup.frame.name} · ${wheelName} · L${settings.upgradeLevel}`.trim();
    els.cdaSummary.textContent = `CdA ${cda.toFixed(4)} m²`;
    els.massSummary.textContent = `${mass.toFixed(1)} kg`;
}


function getRider() {
    let riderWeightKg, heightM;
    if (settings.useZwiftWeight && lastAthlete.weightKg) {
        riderWeightKg = lastAthlete.weightKg;
        heightM = lastAthlete.heightCm ? lastAthlete.heightCm / 100 : settings.heightCm / 100;
    } else {
        riderWeightKg = settings.weightKg;
        heightM = settings.heightCm / 100;
    }
    return {riderWeightKg, heightM};
}


function onSelfUpdate(data) {
    if (!data) return;
    const state = data.state;
    if (data.athlete) {
        if (data.athlete.weight) lastAthlete.weightKg = data.athlete.weight;
        if (data.athlete.height) lastAthlete.heightCm = data.athlete.height;
    }
    if (!state) return;

    if (state.courseId != null) Surface.preload(state.courseId);

    const setup = getBikeSetup();
    const {riderWeightKg, heightM} = getRider();
    const cda = computeCdA(setup, heightM, riderWeightKg);
    const mass = riderWeightKg + setup.bikeWeightKg;

    const power = Number.isFinite(state.power) ? state.power : 0;
    const v = (Number.isFinite(state.speed) ? state.speed : 0) / 3.6;   // km/h -> m/s
    const alt = Number.isFinite(state.altitude) ? state.altitude : (prev ? prev.altitude : 0);
    const wt = Number.isFinite(state.worldTime) ? state.worldTime : null;
    const draftRaw = Number.isFinite(state.draft) ? state.draft : 0;

    // --- Surface ---------------------------------------------------------
    let style = 'NORMAL';
    let surfaceType;
    if (settings.autoSurface && state.roadId != null) {
        const res = Surface.lookupStyle(state.courseId, state.roadId, state.roadTime);
        style = res.style;
        surfaceType = MODEL.styleMap[style] || 'Tarmac';
    } else {
        surfaceType = settings.manualSurface;
        style = `(manual) ${surfaceType}`;
    }
    const crrTable = MODEL.crr[setup.bikeType] || MODEL.crr.road_bike;
    const crr = crrTable[surfaceType] ?? crrTable.Tarmac;

    // --- Time derivatives (smoothed) ------------------------------------
    let dt = 0;
    if (prev && wt != null && prev.worldTime != null) dt = (wt - prev.worldTime) / 1000;
    let instAltDt = 0, instVdt = 0;
    if (dt > 0.05 && dt < 3) {
        instAltDt = (alt - prev.altitude) / dt;
        instVdt = (v - prev.speedMps) / dt;
        const alpha = 1 - Math.exp(-dt / Math.max(0.5, settings.smoothSec));
        sm.dAltDt += (instAltDt - sm.dAltDt) * alpha;
        sm.dvDt += (instVdt - sm.dvDt) * alpha;
    }
    prev = {worldTime: wt, altitude: alt, speedMps: v};

    const grad = v > 0.5 ? Math.max(-0.5, Math.min(0.5, sm.dAltDt / v)) : 0;
    const cosT = Math.cos(Math.atan(grad));

    // --- Power categories (W), signed so they sum to zero ---------------
    const riderToWheel = power;
    const draftW = settings.draftIsWatts ? draftRaw : draftRaw * (0.5 * AIR_DENSITY * cda * v * v * v) / 100;

    const cRider = riderToWheel;
    const cDraft = draftW;
    const cAero = -(0.5 * AIR_DENSITY * cda * v * v * v);
    const cRolling = -(crr * mass * GRAVITY * cosT * v);
    const cPE = -(mass * GRAVITY * sm.dAltDt);
    const cKE = -(mass * v * sm.dvDt);
    const cResidual = -(cRider + cDraft + cAero + cRolling + cPE + cKE);

    const raw = {rider: cRider, draft: cDraft, aero: cAero, rolling: cRolling,
                 pe: cPE, ke: cKE, residual: cResidual};

    // EMA display smoothing (linear -> preserves zero sum).
    if (!smVals) smVals = {...raw};
    const aDisp = dt > 0 ? 1 - Math.exp(-dt / Math.max(0.5, settings.smoothSec)) : 0.3;
    for (const k of Object.keys(raw)) smVals[k] += (raw[k] - smVals[k]) * aDisp;

    render(smVals, {style, surfaceType, crr, cda, mass, setup});
}


function render(vals, ctx) {
    let maxAbs = 1;
    for (const k of Object.keys(vals)) maxAbs = Math.max(maxAbs, Math.abs(vals[k]));

    for (const cat of CATEGORIES) {
        const w = vals[cat.key];
        const r = rowEls[cat.key];
        r.value.textContent = `${w >= 0 ? '+' : ''}${w.toFixed(0)} W`;
        const pct = Math.min(100, Math.abs(w) / maxAbs * 100);
        if (w >= 0) {
            r.barPos.style.width = `${pct}%`;
            r.barNeg.style.width = '0%';
        } else {
            r.barNeg.style.width = `${pct}%`;
            r.barPos.style.width = '0%';
        }
    }

    els.surfaceStyle.textContent = ctx.style;
    els.surfaceType.textContent = `→ ${ctx.surfaceType}`;
    els.surfaceCrr.textContent = `Crr ${ctx.crr.toFixed(4)}`;
    els.cdaSummary.textContent = `CdA ${ctx.cda.toFixed(4)} m²`;
    els.massSummary.textContent = `${ctx.mass.toFixed(1)} kg`;
    const wheelName = ctx.setup.wheel ? `${ctx.setup.wheel.make} ${ctx.setup.wheel.model}`.trim() : 'built-in';
    els.bikeSummary.textContent =
        `${ctx.setup.frame.name} · ${wheelName} · L${settings.upgradeLevel}`.trim();
}
