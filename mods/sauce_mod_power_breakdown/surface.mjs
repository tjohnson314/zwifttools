// Surface lookup via Sauce's own road data (coordinate-free, exact).
//
// Sauce serves per-course roads through Common.getRoads(courseId). Each road has:
//   - `id`           : matches the live rider state's `roadId`.
//   - `defaultStyle` : raw Zwift style name for the whole road ("" -> NORMAL).
//   - `styles`       : [{start, end, style}] sub-ranges in road-percent, where a
//                      mid-road sector overrides the base (may over-cover with
//                      start<0 / end>1). Last matching sector wins.
//
// The rider state carries `roadId` and the RAW `roadTime`; Sauce converts that
// to a road-percent via roadTimeToPercent(rt) = (rt - 5000) / 1e6. We resolve
// the surface style directly from (roadId, roadTime) with no coordinate maths,
// so it is immune to any world-origin / axis mismatch.

import * as Common from '/pages/src/common.mjs';

const roadsCache = new Map();   // courseId -> Map(roadId -> {default, styles}) | null
const pending = new Map();      // courseId -> Promise

function roadTimeToPercent(rt) {
    return (rt - 5000) / 1e6;
}

async function loadRoads(courseId) {
    if (roadsCache.has(courseId)) return roadsCache.get(courseId);
    if (pending.has(courseId)) return pending.get(courseId);

    const p = (async () => {
        try {
            const roads = await Common.getRoads(courseId);
            const map = new Map();
            for (const road of roads || []) {
                map.set(road.id, {
                    default: road.defaultStyle || 'NORMAL',
                    styles: (road.styles || []).map(s => ({
                        start: s.start, end: s.end, style: s.style,
                    })),
                });
            }
            roadsCache.set(courseId, map);
            return map;
        } catch (e) {
            roadsCache.set(courseId, null);   // no data for this course
            return null;
        } finally {
            pending.delete(courseId);
        }
    })();
    pending.set(courseId, p);
    return p;
}

// Kick off a background load so the first lookup is warm.
export function preload(courseId) {
    if (courseId != null) loadRoads(courseId);
}

// Returns {style, ready}. `style` is the raw Zwift style name for the rider's
// current road position; 'NORMAL' when the road/course is unknown or the road
// data has not loaded yet (ready === false while loading).
export function lookupStyle(courseId, roadId, roadTime) {
    const map = roadsCache.get(courseId);
    if (map === undefined) {          // not loaded yet: trigger + default
        loadRoads(courseId);
        return {style: 'NORMAL', ready: false};
    }
    if (map === null) {
        return {style: 'NORMAL', ready: true};
    }
    const road = roadId != null ? map.get(roadId) : null;
    if (!road) {
        return {style: 'NORMAL', ready: true};
    }
    let style = road.default || 'NORMAL';
    if (roadTime != null && Number.isFinite(roadTime) && road.styles.length) {
        const rp = roadTimeToPercent(roadTime);
        for (const s of road.styles) {
            if (rp >= s.start && rp <= s.end) {
                style = s.style;
            }
        }
    }
    return {style, ready: true};
}
