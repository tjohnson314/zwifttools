const MAX_SAMPLE_GAP_MS = 15000;

export function validCourseId(courseId) {
    return Number.isFinite(courseId) || (typeof courseId === 'string' && courseId.length > 0);
}

function crossing(previous, current, boundary, reverse) {
    const from = (previous.roadTime - 5000) / 1e6;
    const to = (current.roadTime - 5000) / 1e6;
    if (reverse ? !(from > boundary && to <= boundary) : !(from < boundary && to >= boundary)) {
        return null;
    }
    const fraction = (boundary - from) / (to - from);
    const distance = Number.isFinite(previous.distance) && Number.isFinite(current.distance) ?
        previous.distance + fraction * (current.distance - previous.distance) : null;
    return {time: previous.worldTime + fraction * (current.worldTime - previous.worldTime), distance};
}

export class SegmentTracker {
    constructor() {
        this.courses = new Map();
        this.riders = new Map();
    }

    setCourseSegments(courseId, segments) {
        const byRoad = new Map();
        let count = 0;
        for (const segment of segments) {
            if (segment.id == null || !segment.name || !Number.isFinite(segment.roadId) ||
                !Number.isFinite(segment.roadStart) || !Number.isFinite(segment.roadFinish) ||
                (!segment.loop && (segment.reverse ?
                    segment.roadStart <= segment.roadFinish : segment.roadStart >= segment.roadFinish))) {
                continue;
            }
            const key = `${segment.roadId}:${!!segment.reverse}`;
            if (!byRoad.has(key)) byRoad.set(key, []);
            byRoad.get(key).push(segment);
            count++;
        }
        this.courses.set(courseId, byRoad);
        return count;
    }

    update(entry) {
        const state = entry?.state;
        if (entry?.athleteId == null || !state || !Number.isFinite(state.worldTime) ||
            !validCourseId(state.courseId) || !Number.isFinite(state.roadId) ||
            !Number.isFinite(state.roadTime) || !this.courses.has(state.courseId)) {
            return [];
        }
        const id = String(entry.athleteId);
        let rider = this.riders.get(id);
        if (!rider) {
            rider = {previous: null, active: new Map()};
            this.riders.set(id, rider);
        }
        const previous = rider.previous;
        if (previous && state.worldTime <= previous.worldTime) return [];
        const current = {
            courseId: state.courseId, roadId: state.roadId, roadTime: state.roadTime,
            reverse: !!state.reverse, worldTime: state.worldTime, distance: state.distance,
        };
        rider.previous = current;
        if (!previous) return [];
        if (current.courseId !== previous.courseId ||
            current.worldTime - previous.worldTime > MAX_SAMPLE_GAP_MS) {
            rider.active.clear();
            return [];
        }
        if (current.roadId !== previous.roadId) {
            for (const [key, active] of rider.active) {
                if (!active.loop) rider.active.delete(key);
            }
            return [];
        }
        if (current.reverse !== previous.reverse) {
            rider.active.clear();
            return [];
        }

        const segments = this.courses.get(current.courseId).get(`${current.roadId}:${current.reverse}`) || [];
        const results = [];
        for (const segment of segments) {
            const key = String(segment.id);
            const loop = !!segment.loop;
            const start = crossing(previous, current, loop ? segment.roadFinish : segment.roadStart, current.reverse);
            const finish = loop ? start : crossing(previous, current, segment.roadFinish, current.reverse);
            const active = rider.active.get(key);
            if (loop && start) {
                if (active && Number.isFinite(segment.distance) && segment.distance > 0 &&
                    active.distance != null && start.distance != null) {
                    const travelled = start.distance - active.distance;
                    if (travelled >= segment.distance * 0.8 && travelled <= segment.distance * 2) {
                        results.push({name: segment.name, timeMs: start.time - active.time, finishTime: start.time});
                    }
                }
                rider.active.set(key, {...start, loop: true});
                continue;
            }
            if (start) rider.active.set(key, start);
            const attempt = rider.active.get(key);
            if (finish && attempt && finish.time > attempt.time) {
                results.push({name: segment.name, timeMs: finish.time - attempt.time, finishTime: finish.time});
                rider.active.delete(key);
            }
        }
        return results.sort((a, b) => a.finishTime - b.finishTime);
    }
}

export function formatTime(timeMs) {
    const hundredths = Math.round(timeMs / 10);
    const minutes = Math.floor(hundredths / 6000);
    const seconds = (hundredths % 6000) / 100;
    return minutes ? `${minutes}:${seconds.toFixed(2).padStart(5, '0')}` : `${seconds.toFixed(2)} s`;
}
