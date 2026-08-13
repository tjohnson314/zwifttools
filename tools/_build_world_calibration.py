"""Compute GPS calibration (local x/z -> latlng affine) for worlds that lack one.

Matches each cached ZwiftMap GPS route (zwiftmap_surfaces/*_route.json) to its
game route in zwift_routes/ by normalized name, fits the affine used by the
Surface Map projection, and keeps the lowest-p95 fit per world.
"""
import json
import glob
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race_replay.data_cleaner import load_route_data, haversine

ROOT = Path(__file__).resolve().parent.parent
P95_LIMIT = 40.0


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", "-", s)


# name-slug -> list of (map_id, route dict)
idx: dict[str, list] = {}
routes_by_world: dict[int, list] = {}
for f in glob.glob(str(ROOT / "zwift_routes" / "world_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    mid = d.get("mapID")
    for r in d.get("routes", []):
        idx.setdefault(norm(r.get("name", "")), []).append((mid, r))
        routes_by_world.setdefault(mid, []).append(r)

existing = json.load(open(ROOT / "zwift_surfaces" / "world_gps_calibration.json"))
print("already calibrated:", sorted(int(k) for k in existing))

strava = json.load(open(ROOT / "route_strava_segments.json"))
all_worlds = sorted(routes_by_world)
uncalibrated = [m for m in all_worlds if str(m) not in existing]


def endpoint_gap(r: dict) -> float:
    """Straight-line gap between a route's start and end (metres); 0 for loops."""
    rt = r.get("route") or {}
    x = rt.get("x") or []
    z = rt.get("z") or []
    if len(x) < 2:
        return 0.0
    return float(np.hypot(x[-1] - x[0], z[-1] - z[0]))


def fit(route, r) -> float | None:
    rt = r.get("route") or {}
    rx = np.asarray(rt.get("x", []), float)
    rz = np.asarray(rt.get("z", []), float)
    rd = np.asarray(rt.get("d", []), float)
    if len(rx) < 3 or len(rd) < 3:
        return None
    frac = (route.distance - route.distance[0]) / (route.distance[-1] - route.distance[0])
    tgt = frac * rd[-1]
    mx = np.interp(tgt, rd, rx)
    mz = np.interp(tgt, rd, rz)
    design = np.column_stack([mx, mz, np.ones(len(mx))])
    if np.linalg.matrix_rank(design) < 3:
        return None
    coef = np.linalg.lstsq(design, route.latlng, rcond=None)[0]
    f = design @ coef
    err = haversine(f[:, 0], f[:, 1], route.latlng[:, 0], route.latlng[:, 1])
    return coef, float(np.percentile(err, 95))


results: dict[int, dict] = {}


def consider(mid, r, slug):
    try:
        route = load_route_data(slug)
    except Exception as e:  # noqa: BLE001
        print(f"    {slug}: load error {e}")
        return
    if route is None:
        return
    out = fit(route, r)
    if out is None:
        return
    coef, p95 = out
    print(f"  cand world {mid:2d}: {r.get('name')!r:38s} slug={slug:26s} p95={p95:7.1f}m")
    cur = results.get(mid)
    if cur is None or p95 < cur["p95_m"]:
        results[mid] = {"coef": coef.tolist(), "route": r.get("name"),
                        "p95_m": round(p95, 1), "slug": slug}


# 1) Cached ZwiftMap GPS routes matched to game routes by name.
for path in glob.glob(str(ROOT / "zwiftmap_surfaces" / "*_route.json")):
    slug = os.path.basename(path).replace("_route.json", "").replace("_", "-")
    for mid, r in idx.get(slug, []):
        consider(mid, r, slug)

# 2) For worlds still lacking a good fit, fetch GPS for their point-to-point
#    routes (loops fit the affine poorly) that have a Strava segment mapping.
for mid in uncalibrated:
    if results.get(mid, {}).get("p95_m", 1e9) <= P95_LIMIT:
        continue
    candidates = sorted(routes_by_world[mid], key=endpoint_gap, reverse=True)
    for r in candidates[:12]:
        slug = norm(r.get("name", ""))
        if slug in strava:
            consider(mid, r, slug)
        if results.get(mid, {}).get("p95_m", 1e9) <= P95_LIMIT:
            break

print("\nCandidates:")
for mid in sorted(results):
    r = results[mid]
    status = "ACCEPT" if r["p95_m"] <= P95_LIMIT else "reject"
    print(f"  world {mid:2d}: {r['route']!r:40s} p95={r['p95_m']:6.1f}m  {status}")

# Merge accepted new worlds into the calibration file (keep existing entries).
merged = dict(existing)
added = []
for mid, r in results.items():
    key = str(mid)
    if key in merged:
        continue
    if r["p95_m"] <= P95_LIMIT:
        merged[key] = {"coef": r["coef"], "route": r["route"], "p95_m": r["p95_m"]}
        added.append(mid)

out = ROOT / "zwift_surfaces" / "world_gps_calibration.json"
json.dump({k: merged[k] for k in sorted(merged, key=int)}, open(out, "w"), indent=1)
print(f"\nAdded worlds: {sorted(added)}")
print(f"Now calibrated: {sorted(int(k) for k in merged)}")
