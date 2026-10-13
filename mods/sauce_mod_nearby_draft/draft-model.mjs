import {MODEL} from './bike-model.mjs';

export const DEFAULTS = {
    frameId: 'Zwift_TT',
    wheelId: '',
    upgradeLevel: 5,
    weightKg: 75,
    heightCm: 183,
    smoothSec: 3,
    physicalSpeed: true,
};

export function selectNearby(entries, selfId, watchingId = selfId) {
    if (selfId == null || !Array.isArray(entries)) return [];
    const self = entries.find(entry => String(entry.athleteId) === String(selfId));
    const origin = Number.isFinite(self?.gap) ? self.gap :
        String(selfId) === String(watchingId) ? 0 : null;
    if (origin == null) return [];
    return entries.flatMap(entry => {
        if (entry.athleteId == null || !entry.state) return [];
        const isSelf = String(entry.athleteId) === String(selfId);
        const gap = isSelf ? 0 : Number.isFinite(entry.gap) ? entry.gap - origin : null;
        return gap != null && Math.abs(gap) <= 15 ? [{...entry, gap, isSelf}] : [];
    }).sort((left, right) => left.gap - right.gap);
}

export function summarizeDraft(values) {
    const maxDraft = Math.abs(values.aero) * 0.5;
    const effective = values.draft + Math.min(0, values.residual);
    const residual = values.draft - effective;
    return {effective, residual, maxDraft,
        percent: maxDraft > 0 ? effective / maxDraft * 100 : null};
}

export function draftGeometry(summary) {
    const greenStart = Math.min(0, summary.effective);
    const greenEnd = Math.max(0, summary.effective);
    const blueEnd = greenEnd + summary.residual;
    const minimum = greenStart;
    const maximum = Math.max(1, summary.maxDraft, greenEnd, blueEnd);
    const span = maximum - minimum;
    return {
        minimum,
        maximum,
        limit: (summary.maxDraft - minimum) / span * 100,
        zero: -minimum / span * 100,
        greenLeft: (greenStart - minimum) / span * 100,
        greenWidth: (greenEnd - greenStart) / span * 100,
        blueLeft: (greenEnd - minimum) / span * 100,
        blueWidth: summary.residual / span * 100,
    };
}

function groundDistance(left, right) {
    const radians = Math.PI / 180;
    const latitude = (right[0] - left[0]) * radians;
    const longitude = (right[1] - left[1]) * radians *
        Math.cos((left[0] + right[0]) * 0.5 * radians);
    return 6371000 * Math.hypot(latitude, longitude);
}

export class DraftTracker {
    constructor() {
        this.riders = new Map();
    }

    retain(ids) {
        const active = new Set(ids.map(String));
        for (const id of this.riders.keys()) {
            if (!active.has(id)) this.riders.delete(id);
        }
    }

    update(entry, settings = DEFAULTS, style = 'NORMAL') {
        const state = entry.state;
        const id = String(entry.athleteId);
        const worldTime = Number.isFinite(state.worldTime) ? state.worldTime : null;
        let history = this.riders.get(id);
        if (history && history.courseId !== state.courseId) history = null;
        if (history && worldTime != null && history.prev.worldTime != null) {
            const elapsed = (worldTime - history.prev.worldTime) / 1000;
            if (elapsed === 0) return history.summary;
            if (elapsed < 0 || elapsed >= 3) history = null;
        }
        const prev = history?.prev;
        const smooth = history?.smooth || {altitude: 0, acceleration: 0, speed: 0};
        const dt = prev && worldTime != null && prev.worldTime != null ?
            (worldTime - prev.worldTime) / 1000 : 0;
        const stepOk = dt > 0.05 && dt < 3;
        const alpha = dt > 0 ? 1 - Math.exp(-dt / Math.max(0.5, settings.smoothSec)) : 0.3;
        const altitude = Number.isFinite(state.altitude) ? state.altitude : prev?.altitude ?? 0;
        const latlng = Array.isArray(state.latlng) && state.latlng.length >= 2 &&
            state.latlng.every(Number.isFinite) ? state.latlng : null;
        const gameSpeed = Number.isFinite(state.speed) ? state.speed / 3.6 : 0;
        const physicalSpeed = settings.physicalSpeed && stepOk && latlng && prev.latlng ?
            Math.hypot(groundDistance(prev.latlng, latlng), altitude - prev.altitude) / dt : null;
        if (stepOk) {
            smooth.altitude += ((altitude - prev.altitude) / dt - smooth.altitude) * alpha;
            if (physicalSpeed != null && smooth.speed === 0) smooth.speed = gameSpeed;
            smooth.speed = physicalSpeed != null ?
                smooth.speed + (physicalSpeed - smooth.speed) * alpha : gameSpeed;
            smooth.acceleration += ((smooth.speed - prev.speed) / dt - smooth.acceleration) * alpha;
        } else {
            smooth.speed = gameSpeed;
        }

        const frame = MODEL.frames.find(candidate => candidate.id === settings.frameId) || MODEL.frames[0];
        const wheel = MODEL.wheels.find(candidate => candidate.id === settings.wheelId);
        const level = settings.upgradeLevel;
        const weight = Number.isFinite(entry.athlete?.weight) && entry.athlete.weight > 0 ?
            entry.athlete.weight : settings.weightKg;
        const height = Number.isFinite(entry.athlete?.height) && entry.athlete.height > 0 ?
            entry.athlete.height : settings.heightCm;
        const constants = MODEL.constants;
        const area = constants.AREA_COEFFICIENT * height ** constants.HEIGHT_EXPONENT *
            weight ** constants.WEIGHT_EXPONENT - constants.AREA_OFFSET;
        const wheelBias = wheel ? ((frame.type || '').toUpperCase() === 'TT' ? wheel.cdaTt : wheel.cda) : 0;
        const cda = 0.5 * constants.AIR_DENSITY * area + frame.cda[level] + wheelBias;
        const mass = weight + (frame.wt[level] + (wheel?.wt || 0)) / 1000;
        const bikeType = MODEL.frameTypeToBikeType[frame.type] || 'road_bike';
        const crrTable = MODEL.crr[bikeType] || MODEL.crr.road_bike;
        const crr = crrTable[MODEL.styleMap[style] || 'Tarmac'] ?? crrTable.Tarmac;
        const speed = smooth.speed;
        const gradient = speed !== 0 ? smooth.altitude / speed : 0;
        const aero = -(0.5 * constants.AIR_DENSITY * cda * speed ** 3);
        const draft = Number.isFinite(state.draft) ? state.draft : 0;
        const power = Number.isFinite(state.power) ? state.power : 0;
        const rolling = -(crr * mass * constants.GRAVITY * Math.cos(Math.atan(gradient)) * speed);
        const potential = -(mass * constants.GRAVITY * smooth.altitude);
        const kinetic = -(mass * speed * smooth.acceleration);
        const residual = -(power * (1 - (constants.DRIVETRAIN_LOSS ?? 0)) +
            draft + aero + rolling + potential + kinetic);
        const raw = {aero, draft, residual};
        const values = history?.values || {...raw};
        for (const key of Object.keys(raw)) values[key] += (raw[key] - values[key]) * alpha;
        const summary = summarizeDraft(values);
        this.riders.set(id, {courseId: state.courseId, smooth, values, summary,
            prev: {worldTime, altitude, latlng, speed}});
        return summary;
    }
}