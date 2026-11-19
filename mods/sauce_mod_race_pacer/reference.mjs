// Reference-ride CSV parsing and course matching for the Race Pacer overlay.
// No Sauce imports, so it can be unit-tested in Node.

/**
 * Zwift world ID (what ZwiftTools writes as `course_id`) → Sauce courseId
 * (the live `state.courseId`).  The two numberings differ, e.g. Watopia is
 * world 1 but course 6.  Same table as tools/build_power_breakdown_data.py,
 * sourced from Sauce's worldCourseDescs.
 */
export const WORLD_TO_COURSE = {
    1: 6, 2: 2, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16, 13: 17,
};

/** Sauce courseId → world name. */
export const COURSE_NAMES = {
    6: 'Watopia', 2: 'Richmond', 7: 'London', 8: 'New York', 9: 'Innsbruck', 10: 'Bologna',
    11: 'Yorkshire', 12: 'Crit City', 13: 'Makuri Islands', 14: 'France', 15: 'Paris',
    16: 'Gravel Mountain', 17: 'Scotland',
};

export function courseName(courseId) {
    return COURSE_NAMES[courseId] ?? `course ${courseId}`;
}

/** The Sauce courseId a CSV's `course_id` (a Zwift world ID) refers to, or null. */
export function csvCourseId(meta) {
    const worldId = parseInt(meta?.course_id, 10);
    return Number.isFinite(worldId) ? WORLD_TO_COURSE[worldId] ?? null : null;
}

/**
 * True when the live course is known and differs from the reference course.
 * Unknown on either side is not a mismatch, so the gap still shows.
 */
export function isWrongCourse(refCourseId, liveCourseId) {
    return refCourseId != null && liveCourseId != null && refCourseId !== liveCourseId;
}

/**
 * Text for the overlay's top line: Race Replay exports carry a route slug
 * ("tempus-fugit" → "Tempus Fugit"); TT plans carry a numeric route ID, so
 * fall back to the world name.
 */
export function routeLabel(meta) {
    const routeId = String(meta?.route_id ?? '').trim();
    if (routeId && !/^\d+$/.test(routeId)) {
        return routeId.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    const courseId = csvCourseId(meta);
    return courseId != null ? courseName(courseId) : '';
}

const parseNum = s => (s == null || s.trim() === '' ? null : Number(s));

/**
 * Parse a reference CSV:
 *   line 1 metadata keys, line 2 metadata values, line 3 data header, then data rows.
 * Rows with a blank `time_sec` or `distance_m` (telemetry gaps) are skipped;
 * a non-numeric value is an error.  Optional columns: power_watts, speed_kmh, bucket.
 */
export function parseReferenceCsv(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();

    if (lines.length < 5) {
        throw new Error('File too short — need metadata header, values, data header, and data rows.');
    }

    const metaKeys = lines[0].split(',').map(s => s.trim());
    const metaVals = lines[1].split(',').map(s => s.trim());
    const meta = {};
    for (let i = 0; i < metaKeys.length; i++) meta[metaKeys[i]] = metaVals[i] ?? '';

    const dataHeader = lines[2].split(',').map(s => s.trim().toLowerCase());
    const timeIdx = dataHeader.indexOf('time_sec');
    const distIdx = dataHeader.indexOf('distance_m');
    const pwrIdx  = dataHeader.indexOf('power_watts');
    const spdIdx  = dataHeader.indexOf('speed_kmh');
    const bktIdx  = dataHeader.indexOf('bucket');
    if (timeIdx === -1) throw new Error('Missing column: time_sec');
    if (distIdx === -1) throw new Error('Missing column: distance_m');

    const times = [];
    const distances = [];
    const powers = pwrIdx !== -1 ? [] : null;
    const speeds = spdIdx !== -1 ? [] : null;
    const buckets = bktIdx !== -1 ? [] : null;
    let skipped = 0;
    for (let i = 3; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const cols = line.split(',');
        const t = parseNum(cols[timeIdx]);
        const d = parseNum(cols[distIdx]);
        if (t === null || d === null) { skipped++; continue; }
        if (!Number.isFinite(t) || !Number.isFinite(d)) throw new Error(`Bad data on line ${i + 1}`);
        times.push(t);
        distances.push(d);
        if (powers !== null) {
            const p = parseNum(cols[pwrIdx]);
            powers.push(Number.isFinite(p) ? p : null);
        }
        if (speeds !== null) {
            const s = parseNum(cols[spdIdx]);
            speeds.push(Number.isFinite(s) ? s : null);
        }
        if (buckets !== null) {
            const b = parseInt(cols[bktIdx], 10);
            buckets.push(Number.isFinite(b) ? b : null);
        }
    }

    if (times.length < 2) throw new Error('Need at least 2 data rows.');
    return { meta, times, distances, powers, speeds, buckets, skipped };
}
