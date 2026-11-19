// Per-bucket power tracking for bucketed TT plans.  No Sauce imports, so it can
// be unit-tested in Node.

/** Gaps longer than this (seconds) between samples aren't counted (pauses, dropouts). */
export const MAX_SAMPLE_GAP_SEC = 5;

/**
 * Turn per-row bucket numbers into spans: [{bucket, start, end, target}].
 * Rows without a bucket are skipped.  Each span ends where the next one starts;
 * the last ends at its final row.  Returns [] when the file has no buckets.
 */
export function bucketSpans(distances, buckets, powers = null) {
    const spans = [];
    if (!buckets) return spans;
    for (let i = 0; i < distances.length; i++) {
        const b = buckets[i];
        if (b == null) continue;
        const last = spans[spans.length - 1];
        if (!last || last.bucket !== b) {
            spans.push({ bucket: b, start: distances[i], end: distances[i], target: powers?.[i] ?? null });
        } else {
            last.end = distances[i];
        }
    }
    for (let k = 0; k < spans.length - 1; k++) spans[k].end = spans[k + 1].start;
    return spans;
}

/** Index of the span containing distanceM, or -1 outside all spans. */
export function spanIndexAt(spans, distanceM) {
    if (!spans.length || distanceM < spans[0].start || distanceM > spans[spans.length - 1].end) return -1;
    let idx = 0;
    for (let k = 1; k < spans.length; k++) {
        if (spans[k].start <= distanceM) idx = k;
        else break;
    }
    return idx;
}

/**
 * Time-weighted average power within the current bucket.  Each sample's power
 * is held until the next sample.  Changing bucket, or time going backwards
 * (a reset), starts a new average.
 */
export class BucketAverage {
    constructor() { this.reset(); }

    reset() {
        this.bucket = null;
        this.weightedSum = 0;
        this.seconds = 0;
        this.lastTime = null;
        this.lastPower = null;
    }

    update(bucket, timeSec, power) {
        if (bucket !== this.bucket || (this.lastTime != null && timeSec < this.lastTime)) {
            this.reset();
            this.bucket = bucket;
        }
        if (this.lastTime != null && this.lastPower != null) {
            const dt = timeSec - this.lastTime;
            if (dt > 0 && dt <= MAX_SAMPLE_GAP_SEC) {
                this.weightedSum += this.lastPower * dt;
                this.seconds += dt;
            }
        }
        this.lastTime = timeSec;
        this.lastPower = Number.isFinite(power) ? power : null;
    }

    get average() {
        return this.seconds > 0 ? this.weightedSum / this.seconds : null;
    }
}

/**
 * Compare an average with the target: 'on' within ±max(2%, 3 W), else 'under' or 'over'.
 */
export function paceStatus(avgW, targetW) {
    if (avgW == null || targetW == null) return null;
    const margin = Math.max(3, targetW * 0.02);
    const diff = avgW - targetW;
    if (Math.abs(diff) <= margin) return 'on';
    return diff < 0 ? 'under' : 'over';
}
