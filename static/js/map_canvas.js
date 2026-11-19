'use strict';

// Widths are CSS pixels; zoom grows them gently without obscuring nearby roads.
function mapStrokeWidth(baseWidth, zoom, dpr) {
    return baseWidth * Math.max(0.5, Math.min(2, Math.sqrt(zoom))) * dpr;
}

function mapScaleMeasurement(metersPerCssPixel, maxWidth = 120) {
    if (!Number.isFinite(metersPerCssPixel) || metersPerCssPixel <= 0 ||
        !Number.isFinite(maxWidth) || maxWidth <= 0) return null;
    const maximumMeters = metersPerCssPixel * maxWidth;
    const magnitude = 10 ** Math.floor(Math.log10(maximumMeters));
    const factor = [5, 2, 1].find(value => value * magnitude <= maximumMeters);
    const meters = factor * magnitude;
    const label = meters >= 1000 ? `${Number((meters / 1000).toPrecision(3))} km`
        : `${Number(meters.toPrecision(3))} m`;
    return { meters, width: meters / metersPerCssPixel, label };
}
