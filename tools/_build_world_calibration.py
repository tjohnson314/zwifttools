"""Compute GPS calibration (local x/z -> latlng affine) for worlds that lack one.

Matches each cached ZwiftMap GPS route (zwiftmap_surfaces/*_route.json) to its
game route in zwift_routes/ by normalized name and fits the affine used by the
Surface Map projection.

Per world it keeps the best fit, with two refinements:
- Lead-in auto-detect: ZwiftMap traces are inconsistent about whether they include
  the route lead-in, so both the main leg and lead-in+route are tried and the
  better-aligning one is kept.
- Coverage tie-break: when residuals are within COVERAGE_TIE_M, the route that
  covers more of the map wins (it extrapolates better to the corners).
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
# Accept a fit up to this p95. Makuri's best (wide-coverage) fit sits ~44 m, so
# the gate is above the ~40 m most worlds achieve to admit it.
P95_LIMIT = 45.0
# When two candidate routes fit within this p95 margin, prefer the one that
# covers more of the map (better corner extrapolation).
COVERAGE_TIE_M = 8.0


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", "-", s)


def endpoint_gap(r: dict) -> float:
    """Straight-line gap between a route's start and end (metres); 0 for loops."""
    rt = r.get("route") or {}
    x = rt.get("x") or []
    z = rt.get("z") or []
    if len(x) < 2:
        return 0.0
    return float(np.hypot(x[-1] - x[0], z[-1] - z[0]))


def _leg_geom(r, keys) -> tuple:
    """Concatenate the requested legs using authoritative stored per-leg 'd'.

    Recomputing length from resampled x/z would corner-cut and underestimate arc
    length, so the stored distances (anchored to Zwift header distances) are used.
    """
    xs, zs, ds = [], [], []
    offset = 0.0
    for key in keys:
        leg = r.get(key) or {}
        lx, lz, ld = leg.get("x"), leg.get("z"), leg.get("d")
        if not lx or not ld:
            continue
        xs.extend(lx)
        zs.extend(lz)
        ds.extend(d + offset for d in ld)
        offset = ds[-1]
    return np.asarray(xs, float), np.asarray(zs, float), np.asarray(ds, float)


def _fit_geom(route, rx, rz, rd):
    """Fit local x/z -> latlng affine for one geometry; return (coef, p95, coverage)."""
    if len(rx) < 3 or len(rd) < 3 or rd[-1] <= 0:
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
    coverage = float(np.hypot(mx.max() - mx.min(), mz.max() - mz.min()))
    return coef, float(np.percentile(err, 95)), coverage


def fit(route, r):
    """Fit affine, auto-detecting whether the GPS trace includes the lead-in.

    ZwiftMap traces are inconsistent: some include the lead-in, some don't. Try
    the main leg alone and lead-in+route, and keep whichever aligns better.
    Returns (coef, p95, coverage, keys) or None.
    """
    best = None
    for keys in (("route",), ("leadin", "route")):
        rx, rz, rd = _leg_geom(r, keys)
        out = _fit_geom(route, rx, rz, rd)
        if out is None:
            continue
        coef, p95, coverage = out
        if best is None or p95 < best[1]:
            best = (coef, p95, coverage, keys)
    return best


def _is_better(new: dict, cur: dict | None) -> bool:
    """Prefer lower p95; on a near-tie (within COVERAGE_TIE_M) prefer wider coverage.

    A route that visits more of the map yields an affine that extrapolates better
    to the corners, even if its own residual is marginally higher. An eligible
    fit (<= P95_LIMIT) always beats an ineligible one so an over-limit but wide
    route can never displace an acceptable fit.
    """
    if cur is None:
        return True
    new_ok = new["p95_m"] <= P95_LIMIT
    cur_ok = cur["p95_m"] <= P95_LIMIT
    if new_ok != cur_ok:
        return new_ok
    if new["p95_m"] <= cur["p95_m"] - COVERAGE_TIE_M:
        return True
    if cur["p95_m"] <= new["p95_m"] - COVERAGE_TIE_M:
        return False
    return new["coverage"] > cur["coverage"]


def consider(results, mid, r, slug):
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
    coef, p95, coverage, keys = out
    variant = "leadin+route" if len(keys) == 2 else "route"
    print(f"  cand world {mid:2d}: {r.get('name')!r:38s} slug={slug:26s} "
          f"p95={p95:7.1f}m cov={coverage:7.0f}m [{variant}]")
    new = {"coef": coef.tolist(), "route": r.get("name"), "p95_m": round(p95, 1),
           "slug": slug, "coverage": round(coverage, 1)}
    if _is_better(new, results.get(mid)):
        results[mid] = new


def main() -> None:
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

    results: dict[int, dict] = {}

    # 1) Cached ZwiftMap GPS routes matched to game routes by name.
    for path in glob.glob(str(ROOT / "zwiftmap_surfaces" / "*_route.json")):
        slug = os.path.basename(path).replace("_route.json", "").replace("_", "-")
        for mid, r in idx.get(slug, []):
            consider(results, mid, r, slug)

    # 2) For worlds still lacking a good fit, fetch GPS for their point-to-point
    #    routes (loops fit the affine poorly) that have a Strava segment mapping.
    for mid in uncalibrated:
        if results.get(mid, {}).get("p95_m", 1e9) <= P95_LIMIT:
            continue
        candidates = sorted(routes_by_world[mid], key=endpoint_gap, reverse=True)
        for r in candidates[:12]:
            slug = norm(r.get("name", ""))
            if slug in strava:
                consider(results, mid, r, slug)
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


if __name__ == "__main__":
    main()
