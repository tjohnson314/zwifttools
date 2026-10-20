'use strict';

// Widths are CSS pixels; zoom grows them gently without obscuring nearby roads.
function mapStrokeWidth(baseWidth, zoom, dpr) {
    return baseWidth * Math.max(0.5, Math.min(2, Math.sqrt(zoom))) * dpr;
}
