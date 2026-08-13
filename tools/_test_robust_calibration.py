"""Robust affine calibration test: align GPS to local geometry by trying both
directions and (for loops) all phase offsets, then fit local x/z -> latlng."""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race_replay.data_cleaner import load_route_data, haversine

ROOT = Path(__file__).resolve().parent.parent
SLUG = sys.argv[1] if len(sys.argv) > 1 else "shisa-shakedown"
MAP_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 9
N = 400


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9]+", " ", s.lower()).strip())


routes = json.load(open(ROOT / "zwift_routes" / f"world_{MAP_ID}.json", encoding="utf-8"))["routes"]
r = next(rr for rr in routes if norm(rr.get("name", "")) == SLUG)

# Concatenate leadin + route using the authoritative stored per-leg distances
# (anchored to Zwift header distances). Recomputing length from resampled x/z
# corner-cuts and underestimates arc length, so use the stored 'd'.
_xs, _zs, _ds = [], [], []
_offset = 0.0
for leg_key in ("leadin", "route"):
    leg = r.get(leg_key)
    if not (leg and leg.get("x")):
        continue
    _xs.extend(leg["x"]); _zs.extend(leg["z"])
    _ds.extend(d + _offset for d in leg["d"])
    _offset = _ds[-1]
rx = np.asarray(_xs, float); rz = np.asarray(_zs, float); rd = np.asarray(_ds, float)
route = load_route_data(SLUG)
gd = route.distance.astype(float)
glat = route.latlng[:, 0]; glng = route.latlng[:, 1]

is_loop = float(np.hypot(rx[-1] - rx[0], rz[-1] - rz[0])) < 50.0
print(f"{SLUG}: local {rd[-1]:.0f} m ({len(rx)} pts), GPS {gd[-1]:.0f} m ({len(gd)} pts), "
      f"{'LOOP' if is_loop else 'point-to-point'}")

# Resample local geometry to N equal-fraction points.
lf = np.linspace(0, 1, N)
lx = np.interp(lf * rd[-1], rd, rx)
lz = np.interp(lf * rd[-1], rd, rz)


def fit_at(shift, reverse):
    gf = lf.copy()
    if reverse:
        g = 1.0 - gf
    else:
        g = gf
    g = (g + shift) % 1.0 if is_loop else g
    plat = np.interp(g * gd[-1], gd, glat)
    plng = np.interp(g * gd[-1], gd, glng)
    A = np.column_stack([lx, lz, np.ones(N)])
    coef = np.linalg.lstsq(A, np.column_stack([plat, plng]), rcond=None)[0]
    f = A @ coef
    err = haversine(f[:, 0], f[:, 1], plat, plng)
    return float(np.percentile(err, 95)), coef


shifts = np.linspace(0, 1, 180, endpoint=False) if is_loop else [0.0]
best = None
for rev in (False, True):
    for sh in shifts:
        p95, coef = fit_at(sh, rev)
        if best is None or p95 < best[0]:
            best = (p95, coef, sh, rev)

p95, coef, sh, rev = best
print(f"BEST affine: p95={p95:.1f}m  (reverse={rev}, phase_shift={sh:.3f})")
print("coef:")
for row in coef.tolist():
    print("   ", row)
