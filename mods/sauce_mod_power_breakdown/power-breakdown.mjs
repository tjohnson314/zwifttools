import * as Common from '/pages/src/common.mjs';
import {MODEL} from './bike-model.mjs';
import {createBikeResolver} from './bike-setup.mjs';
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
let equipment;
let setupKey;
let lastSupertuck = false;
let selfData;
let supertuckContext;
let contextRefreshing = false;
let contextError;
let cachedEvent;
let equipmentRefreshing = false;
let equipmentError;
const resolveBike = createBikeResolver(MODEL);


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
    refreshEquipment();
    refreshSupertuckContext();
    const equipmentTimer = setInterval(refreshEquipment, 5000);
    const contextTimer = setInterval(refreshSupertuckContext, 1000);
    globalThis.addEventListener('pagehide', () => {
        clearInterval(equipmentTimer);
        clearInterval(contextTimer);
    }, {once: true});
}

async function refreshEquipment() {
    if (equipmentRefreshing) return;
    equipmentRefreshing = true;
    try {
        const athlete = await Common.rpc.getAthlete('self', {
            required: ['equipment'],
            refresh: true,
        });
        if (!athlete?.equipment?.cycling) throw new Error('Rider equipment unavailable');
        equipmentError = null;
        els.equipmentStatus.textContent = '';
    } catch (error) {
        const message = String(error.message || error);
        els.equipmentStatus.textContent = `Equipment refresh failed; showing last available profile: ${message}`;
        if (equipmentError !== message) console.error('Equipment refresh failed:', error);
        equipmentError = message;
    } finally {
        equipmentRefreshing = false;
    }
}

async function refreshSupertuckContext() {
    if (contextRefreshing) return;
    contextRefreshing = true;
    const eventId = selfData?.eventSubgroupId ?? null;
    try {
        const connection = await Common.rpc.getGameConnectionStatus();
        if (connection?.state === 'disabled') {
            throw new Error('Enable Game Connection in Sauce Settings > General, then restart Sauce');
        }
        if (!connection?.connected) {
            throw new Error(`Game Connection not connected${connection?.error ? `: ${connection.error}` : ''}`);
        }
        const [workoutStatus, groups] = await Promise.all([
            Common.rpc.getWorkoutStatus(),
            Common.rpc.getGroupsData(),
        ]);
        if (!Array.isArray(groups)) throw new Error('Rider group data unavailable');
        if (!cachedEvent || cachedEvent.id !== eventId) {
            let banded = false;
            if (eventId != null) {
                const subgroup = await Common.rpc.getEventSubgroup(eventId);
                if (!subgroup) throw new Error('Event rules unavailable');
                banded = subgroup.rubberbanding || subgroup.rulesSet?.includes('RUBBERBANDING');
                if (!banded && subgroup.eventId != null) {
                    const event = await Common.rpc.getEvent(subgroup.eventId);
                    if (!event) throw new Error('Event rules unavailable');
                    banded = event.rubberbanding || event.rulesSet?.includes('RUBBERBANDING');
                }
            }
            cachedEvent = {id: eventId, banded: Boolean(banded)};
        }
        supertuckContext = {eventId, workoutStatus, groups, banded: cachedEvent.banded};
        contextError = null;
        els.supertuckStatus.textContent = '';
    } catch (error) {
        supertuckContext = null;
        const message = String(error.message || error);
        els.supertuckStatus.textContent = `Supertuck detection unavailable: ${message}`;
        if (contextError !== message) console.error('Supertuck context lookup failed:', error);
        contextError = message;
    } finally {
        contextRefreshing = false;
    }
}

function isSupertucking(data, setup, context) {
    const state = data.state;
    if (!state || !context || context.eventId !== (data.eventSubgroupId ?? null)) return false;
    if (!Number.isFinite(state.speed) || state.speed < 58 ||
        !Number.isFinite(state.grade) || state.grade > -0.03 ||
        !Number.isFinite(state.power) || state.power >= 11 || state.draft !== 0) return false;
    const frameType = (setup.frame.type || '').toUpperCase();
    if (frameType === 'TT' || frameType === 'MTB') return false;
    if (state.workoutZone ||
        (context.workoutStatus && !context.workoutStatus.state?.isFinished)) return false;
    const group = context.groups.find(item =>
        item.athletes.some(athlete => String(athlete.athleteId) === String(data.athleteId)));
    return context.banded || Boolean(group && group.athletes.length === 1);
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
    els.equipmentStatus = document.querySelector('.equipment-status');
    els.supertuckStatus = document.querySelector('.supertuck-status');
    els.surfaceStyle = document.querySelector('.surface-style');
    els.surfaceType = document.querySelector('.surface-type');
    els.surfaceCrr = document.querySelector('.surface-crr');
    els.draftValue = document.querySelector('.draft-value');
    els.draftEff = document.querySelector('.draft-eff');
    els.draftRes = document.querySelector('.draft-res');
    els.draftMax = document.querySelector('.draft-max');
    els.checkSum = document.querySelector('.check-sum');
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
            label: row.querySelector('.brk-label'),
            value: row.querySelector('.brk-value'),
            barPos: row.querySelector('.brk-pos'),
            barNeg: row.querySelector('.brk-neg'),
        };
    }
}


function populatePickers() {
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
    return resolveBike(equipment, settings.upgradeLevel);
}


function computeCdA(setup, heightM, riderWeightKg) {
    const riderArea = AREA_COEFFICIENT * Math.pow(heightM * 100, HEIGHT_EXPONENT) *
                      Math.pow(riderWeightKg, WEIGHT_EXPONENT) - AREA_OFFSET;
    return 0.5 * AIR_DENSITY * riderArea + setup.frameBias + setup.wheelBias;
}


function updateConfigSummary() {
    if (!els.bikeSummary) return;
    const setup = getBikeSetup();
    if (setup.error) {
        renderUnavailable(setup.error);
        return;
    }
    const {riderWeightKg, heightM} = getRider();
    const cda = computeCdA(setup, heightM, riderWeightKg);
    const mass = riderWeightKg + setup.bikeWeightKg;
    els.bikeSummary.textContent = setup.label;
    els.cdaSummary.textContent = `CdA ${cda.toFixed(4)} m²`;
    els.massSummary.textContent = `${mass.toFixed(1)} kg`;
}

function renderUnavailable(message) {
    lastSupertuck = false;
    rowEls.draft.label.textContent = 'Draft';
    els.bikeSummary.textContent = message;
    els.cdaSummary.textContent = 'CdA unavailable';
    els.massSummary.textContent = 'Mass unavailable';
    els.surfaceStyle.textContent = '—';
    els.surfaceType.textContent = '';
    els.surfaceCrr.textContent = '';
    els.draftValue.textContent = 'Unavailable';
    els.draftMax.textContent = 'max —';
    els.draftEff.style.width = els.draftRes.style.width = '0%';
    for (const row of Object.values(rowEls)) {
        row.value.textContent = '—';
        row.barPos.style.width = row.barNeg.style.width = '0%';
    }
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
    selfData = data;
    const state = data.state;
    if (data.athlete) {
        if (data.athlete.weight) lastAthlete.weightKg = data.athlete.weight;
        if (data.athlete.height) lastAthlete.heightCm = data.athlete.height;
    }
    equipment = data.athlete?.equipment?.cycling;
    const setup = getBikeSetup();
    if (setup.error || setup.key !== setupKey) {
        prev = null;
        sm = {dAltDt: 0, dvDt: 0, speed: 0};
        smVals = null;
        setupKey = setup.key;
    }
    if (setup.error) {
        renderUnavailable(setup.error);
        return;
    }
    if (!state) return;

    if (state.courseId != null) Surface.preload(state.courseId);
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
    const crr = (crrTable[surfaceType] ?? crrTable.Tarmac) - setup.crrReduction;

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
    const supertuck = isSupertucking(data, setup, supertuckContext);
    const draftW = supertuck ? -cAero / 3 :
        settings.draftIsWatts ? draftRaw : draftRaw * -cAero / 100;

    const cRider = riderToWheel;
    const cDraft = draftW;
    const cRolling = -(crr * mass * GRAVITY * cosT * v);
    const cPE = -(mass * GRAVITY * sm.dAltDt);
    const cKE = -(mass * v * sm.dvDt);
    const cResidual = -(cRider + cDraft + cAero + cRolling + cPE + cKE);

    const raw = {rider: cRider, draft: cDraft, aero: cAero, rolling: cRolling,
                 pe: cPE, ke: cKE, residual: cResidual};

    // EMA display smoothing (linear -> preserves zero sum).
    if (supertuck !== lastSupertuck) smVals = null;
    lastSupertuck = supertuck;
    if (!smVals) smVals = {...raw};
    const aDisp = dt > 0 ? 1 - Math.exp(-dt / Math.max(0.5, settings.smoothSec)) : 0.3;
    for (const k of Object.keys(raw)) smVals[k] += (raw[k] - smVals[k]) * aDisp;

    render(smVals, {style, surfaceType, crr, cda, mass, setup, supertuck});
}


function render(vals, ctx) {
    rowEls.draft.label.textContent = ctx.supertuck ? 'Draft (supertuck)' : 'Draft';
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
    els.bikeSummary.textContent = ctx.setup.label;
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

