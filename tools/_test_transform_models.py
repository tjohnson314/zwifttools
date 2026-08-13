"""Compare affine vs projective vs polynomial local->latlng fits for one route."""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race_replay.data_cleaner import load_route_data, haversine

ROOT = Path(__file__).resolve().parent.parent
SLUG = sys.argv[1] if len(sys.argv) > 1 else "red-zone-repeats"
MAP_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 9


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9]+", " ", s.lower()).strip())


routes = json.load(open(ROOT / "zwift_routes" / f"world_{MAP_ID}.json", encoding="utf-8"))["routes"]
r = next(rr for rr in routes if norm(rr.get("name", "")) == SLUG)
rt = r["route"]
rx = np.asarray(rt["x"], float); rz = np.asarray(rt["z"], float); rd = np.asarray(rt["d"], float)
route = load_route_data(SLUG)
frac = (route.distance - route.distance[0]) / (route.distance[-1] - route.distance[0])
tgt = frac * rd[-1]
mx = np.interp(tgt, rd, rx); mz = np.interp(tgt, rd, rz)
ll = route.latlng
lat = ll[:, 0]; lng = ll[:, 1]


def report(name, pred_lat, pred_lng):
    err = haversine(pred_lat, pred_lng, lat, lng)
    print(f"{name:16s} p50={np.percentile(err,50):6.1f}m  p95={np.percentile(err,95):6.1f}m  max={err.max():6.1f}m")


# Affine (current model)
A = np.column_stack([mx, mz, np.ones(len(mx))])
ca = np.linalg.lstsq(A, ll, rcond=None)[0]
fa = A @ ca
report("affine", fa[:, 0], fa[:, 1])

# Projective (homography): solve for each output separately with denom.
# Fit lat,lng = (a x + b z + c) / (g x + h z + 1)
def fit_proj(out):
    # linearize: out*(g x + h z + 1) = a x + b z + c
    M = np.column_stack([mx, mz, np.ones(len(mx)), -out * mx, -out * mz])
    p = np.linalg.lstsq(M, out, rcond=None)[0]
    a, b, c, g, h = p
    denom = g * mx + h * mz + 1
    return (a * mx + b * mz + c) / denom
report("projective", fit_proj(lat), fit_proj(lng))

# 2nd-order polynomial
P = np.column_stack([np.ones(len(mx)), mx, mz, mx * mx, mx * mz, mz * mz])
cp = np.linalg.lstsq(P, ll, rcond=None)[0]
fp = P @ cp
report("poly2", fp[:, 0], fp[:, 1])
