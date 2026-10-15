import {MODEL} from './bike-model.mjs';

export const DEFAULTS = {
    radiusM: 50,
    maxRiders: 0,
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

export function positionOffset(origin, state) {
    if (!origin || !state || origin.courseId == null || origin.courseId !== state.courseId) return null;
    if ([origin.x, origin.y, state.x, state.y].every(Number.isFinite)) {
        return {x: (state.x - origin.x) / 100, y: (state.y - origin.y) / 100};
    }
    const valid = point => Array.isArray(point) && point.length >= 2 &&
        Number.isFinite(point[0]) && Number.isFinite(point[1]);
    if (!valid(origin.latlng) || !valid(state.latlng)) return null;
    const radians = Math.PI / 180;
    return {
        x: 6371000 * (state.latlng[1] - origin.latlng[1]) * radians * Math.cos(origin.latlng[0] * radians),
        y: -6371000 * (state.latlng[0] - origin.latlng[0]) * radians,
    };
}

export function selectSpatialNearby(entries, self, radiusM = DEFAULTS.radiusM) {
    if (!Array.isArray(entries) || self?.athleteId == null || !positionOffset(self.state, self.state)) return [];
    const unique = new Map(entries.filter(entry => entry.athleteId != null)
        .map(entry => [String(entry.athleteId), entry]));
    unique.set(String(self.athleteId), self);
    return [...unique.values()].flatMap(entry => {
        const position = positionOffset(self.state, entry.state);
        if (!position) return [];
        const distance = Math.hypot(position.x, position.y);
        return distance <= radiusM ? [{...entry, position, distance,
            isSelf: String(entry.athleteId) === String(self.athleteId)}] : [];
    }).sort((left, right) => left.distance - right.distance || Number(right.isSelf) - Number(left.isSelf));
}

export function selectDisplayedRiders(entries, self, settings = DEFAULTS) {
    const withinRadius = selectSpatialNearby(entries, self, settings.radiusM);
    return {
        riders: settings.maxRiders > 0 ? withinRadius.slice(0, settings.maxRiders) : withinRadius,
        actual: withinRadius.length,
    };
}

export function riderInitials(athlete, athleteId) {
    const name = (athlete?.sanitizedFullname || '').replace(/\[[^\]]*\]|\([^)]*\)/g, '').trim();
    const parts = name.split(/\s+/).filter(Boolean);
    if (!parts.length) return String(athleteId).slice(-2);
    return (Array.from(parts[0])[0] + (parts.length > 1 ? Array.from(parts[parts.length - 1])[0] : '')).toUpperCase();
}

export function donutSegments(summary) {
    const values = [summary.effective, summary.residual,
        summary.maxDraft - summary.effective - summary.residual];
    const total = values.reduce((sum, value) => sum + Math.abs(value), 0);
    return {
        values,
        fractions: total > 0 ? values.map(value => Math.abs(value) / total) : [0, 0, 1],
        signed: values.some(value => value < 0),
    };
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

export class PositionInterpolator {
    constructor() {
        this.riders = new Map();
        this.clock = null;
        this.timestamped = false;
        this.delayMs = 1000;
    }

    retain(ids) {
        const active = new Set(ids.map(String));
        for (const id of this.riders.keys()) {
            if (!active.has(id)) this.riders.delete(id);
        }
    }

    update(athleteId, state, now) {
        const id = String(athleteId);
        if (!positionOffset(state, state)) {
            this.riders.delete(id);
            return;
        }
        const timestamped = Number.isFinite(state.worldTime);
        if (timestamped && !this.timestamped) {
            this.riders.clear();
            this.clock = null;
            this.timestamped = true;
        }
        const time = timestamped ? state.worldTime : this.clock ?
            this.clock.time + now - this.clock.receivedAt : now;
        let previous = this.riders.get(id);
        const native = Number.isFinite(state.x) && Number.isFinite(state.y);
        const latest = previous?.samples[previous.samples.length - 1];
        if (latest && (latest.state.courseId !== state.courseId || time < latest.time - 3000)) {
            this.riders.clear();
            this.clock = null;
            previous = null;
        } else if (latest && time <= latest.time) {
            return;
        }
        if (previous && (previous.native !== native || now - previous.receivedAt >= 3000 || time - latest.time >= 3000)) previous = null;
        const target = {...state, latlng: state.latlng ? [...state.latlng] : state.latlng};
        const samples = previous?.samples || [];
        samples.push({time, state: target});
        while (samples.length > 2 && samples[1].time < time - 5000) samples.shift();
        this.riders.set(id, {samples, native, receivedAt: now});
        if (!this.clock || time > this.clock.time) this.clock = {time, receivedAt: now};
    }

    timeline(now, referenceId = this.riders.keys().next().value) {
        const reference = this.riders.get(String(referenceId));
        if (!reference || !this.clock) return null;
        const start = reference.samples[0].time;
        let end = reference.samples[reference.samples.length - 1].time;
        for (const {samples, receivedAt} of this.riders.values()) {
            const latest = samples[samples.length - 1].time;
            if (samples.length > 1 && now - receivedAt < 3000 && latest >= start) {
                end = Math.min(end, latest);
            }
        }
        return {time: Math.max(start, Math.min(end,
            this.clock.time + now - this.clock.receivedAt - this.delayMs)), end};
    }

    latencyMs(now, displayedTime = this.timeline(now)?.time) {
        if (!this.clock || !Number.isFinite(displayedTime)) return null;
        return this.clock.time + now - this.clock.receivedAt - displayedTime;
    }

    sample(athleteId, now, timeline = this.timeline(now)) {
        const motion = this.riders.get(String(athleteId));
        if (!motion || !timeline) return null;
        if (timeline.time < motion.samples[0].time ||
            timeline.time > motion.samples[motion.samples.length - 1].time) return null;
        let index = 0;
        while (index < motion.samples.length - 2 && motion.samples[index + 1].time < timeline.time) index++;
        const from = motion.samples[index];
        const target = motion.samples[index + 1] || from;
        const fraction = target.time > from.time ? (timeline.time - from.time) / (target.time - from.time) : 1;
        const state = {...target.state};
        if (Number.isFinite(state.worldTime)) state.worldTime = timeline.time;
        if (motion.native) {
            state.x = from.state.x + (target.state.x - from.state.x) * fraction;
            state.y = from.state.y + (target.state.y - from.state.y) * fraction;
        } else {
            state.latlng = target.state.latlng.map((value, coordinate) =>
                from.state.latlng[coordinate] + (value - from.state.latlng[coordinate]) * fraction);
        }
        return state;
    }

    isAnimating(now, referenceId) {
        const timeline = this.timeline(now, referenceId);
        return timeline != null && timeline.time < timeline.end;
    }
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