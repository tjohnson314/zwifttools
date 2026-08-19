// Surface lookup via authoritative road-style data (coordinate-free, exact).
//
// The primary source is `road-styles.mjs` (ROAD_STYLES), generated from the
// Zwift game client by tools/build_power_breakdown_data.py using the SAME,
// in-game-validated marker rules that build zwift_surfaces/world_*.json:
//   - a road marker only applies when it carries both road-times (an inactive
//     marker missing a road-time never renders);
//   - a styled marker (visible OR invisible) applies its style;
//   - a style-less INVISIBLE marker has no surface effect (base shows through);
//   - a style-less VISIBLE marker defaults to asphalt (NORMAL / style 0).
// Sauce's own `road.styles` projection carries none of the invisible/inactive
// metadata those rules need, so we cannot derive them from it — hence the
// bundled data. Sauce's roads remain a FALLBACK for courses we don't ship data
// for (portal roads, worlds without extractable road XML, etc.).
//
// ROAD_STYLES is keyed by Sauce courseId (the live `state.courseId`, which
// differs from the Zwift worldId, e.g. Makuri worldId 9 -> courseId 13):
//   { courseId: { roadId: { d: defaultStyleName, s: [[start, end, styleName], ...] } } }
// where start/end are normalised road-percent [0, 1] and the last covering
// sector wins over the default. The rider state carries `roadId` and the RAW
// `roadTime`, which Sauce maps to road-percent via (rt - 5000) / 1e6.

import * as Common from '/pages/src/common.mjs';
import {ROAD_STYLES} from './road-styles.mjs';

const roadsCache = new Map();   // courseId -> Map(roadId -> {default, styles}) | null
const pending = new Map();      // courseId -> Promise

function roadTimeToPercent(rt) {
    return (rt - 5000) / 1e6;
}

// Resolve a style from a bundled ROAD_STYLES road entry (default + sectors,
// last covering sector wins). Returns the raw Zwift style name.
function resolveBundled(road, roadTime) {
    let style = road.d || 'NORMAL';
    if (roadTime != null && Number.isFinite(roadTime) && road.s.length) {
        const rp = roadTimeToPercent(roadTime);
        for (const s of road.s) {
            if (rp >= s[0] && rp <= s[1]) {
                style = s[2];
            }
        }
    }
    return style;
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

// Kick off a background load of the Sauce fallback data for courses we don't
// ship authoritative road styles for, so the first lookup is warm.
export function preload(courseId) {
    if (courseId != null && !ROAD_STYLES[courseId]) loadRoads(courseId);
}

// Returns {style, ready}. `style` is the raw Zwift style name for the rider's
// current road position; 'NORMAL' when the road/course is unknown or the
// fallback road data has not loaded yet (ready === false while loading).
export function lookupStyle(courseId, roadId, roadTime) {
    // Primary: authoritative bundled road styles (matches the surface map).
    const courseRoads = courseId != null ? ROAD_STYLES[courseId] : null;
    if (courseRoads) {
        const road = roadId != null ? courseRoads[roadId] : null;
        // A road absent from our lean bundle is a plain, override-free tarmac
        // road (those are omitted at build time) -> NORMAL / Tarmac.
        return {style: road ? resolveBundled(road, roadTime) : 'NORMAL', ready: true};
    }

    // Fallback: Sauce's own road-style projection (portal roads, worlds without
    // extractable road XML, or a courseId we don't ship data for).
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
