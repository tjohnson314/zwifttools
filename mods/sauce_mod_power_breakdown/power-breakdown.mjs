import * as Common from '/pages/src/common.mjs';
import {MODEL} from './bike-model.mjs';
import * as Surface from './surface.mjs';

Common.enableSentry();

const {AREA_COEFFICIENT, HEIGHT_EXPONENT, WEIGHT_EXPONENT, AREA_OFFSET,
    AIR_DENSITY, GRAVITY} = MODEL.constants;
const DRIVETRAIN_LOSS = MODEL.constants.DRIVETRAIN_LOSS ?? 0;

// Zwift's draft benefit saturates near half of the solo aero cost.
const MAX_DRAFT_FRACTION = 0.5;
const EARTH_RADIUS_M = 6371000;

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
    physicalSpeed: true,     // measure speed from world position, not route distance
};

let settings;
let els = {};
let rowEls = {};            // key -> {value, barPos, barNeg}

// Smoothed running state.
let prev = null;            // {worldTime, altitude, latlng, speedMps}
let sm = {dAltDt: 0, dvDt: 0, speed: 0};   // smoothed derivatives + speed
let smVals = null;          // smoothed category watts
let lastAthlete = {weightKg: null, heightCm: null};


// Horizontal ground distance (m) between two [lat, lng] pairs. Equirectangular
// is exact enough over the few metres covered between samples.
function groundDistance(a, b) {
    const rad = Math.PI / 180;
    const dLat = (b[0] - a[0]) * rad;
    const dLng = (b[1] - a[1]) * rad * Math.cos((a[0] + b[0]) * 0.5 * rad);
    return EARTH_RADIUS_M * Math.hypot(dLat, dLng);
}


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
    els.draftValue = document.querySelector('.draft-value');
    els.draftEff = document.querySelector('.draft-eff');
    els.draftRes = document.querySelector('.draft-res');
    els.draftMax = document.querySelector('.draft-max');
    els.checkSum = document.querySelector('.check-sum');
    els.frameSel = document.querySelector('[data-set="frameId"]');
    els.wheelSel = document.querySelector('[data-set="wheelId"]');
    els.upgradeSel = document.querySelector('[data-set="upgradeLevel"]');
    els.useZwiftWeight = document.querySelector('[data-set="useZwiftWeight"]');
    els.weightKg = document.querySelector('[data-set="weightKg"]');
    els.heightCm = document.querySelector('[data-set="heightCm"]');
    els.smoothSec = document.querySelector('[data-set="smoothSec"]');
    els.physicalSpeed = document.querySelector('[data-set="physicalSpeed"]');
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
    els.physicalSpeed.checked = settings.physicalSpeed;
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
    const riderArea = AREA_COEFFICIENT * Math.pow(heightM * 100, HEIGHT_EXPONENT) *
                      Math.pow(riderWeightKg, WEIGHT_EXPONENT) - AREA_OFFSET;
    return 0.5 * AIR_DENSITY * riderArea + setup.frameBias + setup.wheelBias;
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
    const vGame = (Number.isFinite(state.speed) ? state.speed : 0) / 3.6;   // km/h -> m/s
    const alt = Number.isFinite(state.altitude) ? state.altitude : (prev ? prev.altitude : 0);
    const wt = Number.isFinite(state.worldTime) ? state.worldTime : null;
    const draftRaw = Number.isFinite(state.draft) ? state.draft : 0;
    const latlng = Array.isArray(state.latlng) && Number.isFinite(state.latlng[0]) ?
        state.latlng : null;

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
    if (dt < 0 || dt > 30) dt = 0;      // session restart / world change
    const stepOk = dt > 0.05 && dt < 3;

    // Physical speed from the world position. Zwift's own `speed`/`distance`
    // track progress ALONG THE ROUTE, which under-reads whenever steering takes
    // a wider (or narrower) line than the road centreline.
    let vPhys = null;
    if (stepOk && latlng && prev.latlng) {
        const dGround = groundDistance(prev.latlng, latlng);
        const d3 = Math.hypot(dGround, alt - prev.altitude);
        if (d3 / dt < 40) vPhys = d3 / dt;      // ignore teleports / world changes
    }

    const usePhys = settings.physicalSpeed && vPhys != null;
    const vRaw = usePhys ? vPhys : vGame;

    if (stepOk) {
        const alpha = 1 - Math.exp(-dt / Math.max(0.5, settings.smoothSec));
        sm.dAltDt += ((alt - prev.altitude) / dt - sm.dAltDt) * alpha;
        // Position-derived speed is quantised, so smooth it before use; the game
        // speed is already clean and is taken as-is.
        if (usePhys && sm.speed === 0) sm.speed = vGame;
        sm.speed = usePhys ? sm.speed + (vRaw - sm.speed) * alpha : vRaw;
        sm.dvDt += ((sm.speed - prev.speedMps) / dt - sm.dvDt) * alpha;
    } else if (!usePhys) {
        sm.speed = vRaw;
    }
    const v = sm.speed;
    prev = {worldTime: wt, altitude: alt, speedMps: v, latlng};

    const grad = v > 0.5 ? Math.max(-0.5, Math.min(0.5, sm.dAltDt / v)) : 0;
    const cosT = Math.cos(Math.atan(grad));

    // --- Power categories (W), signed so they sum to zero ---------------
    const riderToWheel = power * (1 - DRIVETRAIN_LOSS);
    const cAero = -(0.5 * AIR_DENSITY * cda * v * v * v);
    const draftW = settings.draftIsWatts ? draftRaw : draftRaw * -cAero / 100;

    const cRider = riderToWheel;
    const cDraft = draftW;
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

    renderDraft(vals);

    els.surfaceStyle.textContent = ctx.style;
    els.surfaceType.textContent = `→ ${ctx.surfaceType}`;
    els.surfaceCrr.textContent = `Crr ${ctx.crr.toFixed(4)}`;
    els.cdaSummary.textContent = `CdA ${ctx.cda.toFixed(4)} m²`;
    els.massSummary.textContent = `${ctx.mass.toFixed(1)} kg`;
    const wheelName = ctx.setup.wheel ? `${ctx.setup.wheel.make} ${ctx.setup.wheel.model}`.trim() : 'built-in';
    els.bikeSummary.textContent =
        `${ctx.setup.frame.name} · ${wheelName} · L${settings.upgradeLevel}`.trim();
}


// Draft bar: full scale = the maximum draft Zwift gives (half the solo aero
// cost). A negative residual means the model over-credits the draft, so it is
// taken back out: green = effective draft, blue = the residual it swallowed.
function renderDraft(vals) {
    const maxDraft = Math.abs(vals.aero) * MAX_DRAFT_FRACTION;
    const draft = Math.max(0, vals.draft);
    const effective = vals.residual < 0 ? Math.max(0, draft + vals.residual) : draft;
    const resPart = draft - effective;
    const toPct = w => maxDraft > 1 ? Math.min(100, Math.max(0, w / maxDraft * 100)) : 0;

    els.draftEff.style.width = `${toPct(effective)}%`;
    els.draftRes.style.width = `${toPct(Math.min(resPart, Math.max(0, maxDraft - effective)))}%`;
    const pct = maxDraft > 1 ? (effective / maxDraft * 100) : 0;
    els.draftValue.textContent = `${effective.toFixed(0)} W (${pct.toFixed(0)}% of max)`;
    els.draftMax.textContent = `max ${maxDraft.toFixed(0)} W`;
}

