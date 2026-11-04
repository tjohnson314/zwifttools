// Which detail lines the Race Pacer overlay shows.  Shared by the overlay and
// its settings window; no Sauce imports, so it can be unit-tested in Node.

export const DETAIL_DEFAULTS = {
    showRefPower: true,
    showRefSpeed: true,
    showBucketAvg: true,
    showBucketInfo: true,
    showNames: true,
};

/** Setting → class on <html> that hides that detail (see display.css). */
export const DETAIL_CLASSES = {
    showRefPower: 'hide-ref-power',
    showRefSpeed: 'hide-ref-speed',
    showBucketAvg: 'hide-bucket-avg',
    showBucketInfo: 'hide-bucket-info',
    showNames: 'hide-names',
};

/**
 * Map each hide-class to whether it should be on, given a settings getter.
 * Unset settings fall back to the defaults (shown).
 */
export function detailHideClasses(getSetting) {
    const out = {};
    for (const [key, cls] of Object.entries(DETAIL_CLASSES)) {
        out[cls] = !(getSetting(key) ?? DETAIL_DEFAULTS[key]);
    }
    return out;
}
