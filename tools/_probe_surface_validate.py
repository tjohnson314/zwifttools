"""Validate WAD surface extraction against zwift_routes geometry.

Loads a route's local-metre profile from ``zwift_routes/world_<mapID>.json`` and
assigns each point the surface of the nearest extracted road segment (same local
frame). Prints a distance-weighted surface breakdown so results can be sanity
checked against known routes.

Usage::

    python tools/_probe_surface_validate.py "Champs" [max_dist_m]
    python tools/_probe_surface_validate.py 3382019812
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = os.path.join(ROOT, "zwift_routes")
SURFACES = os.path.join(ROOT, "zwift_surfaces")


def load_route(query: str):
    index = json.load(open(os.path.join(ROUTES, "index.json"), encoding="utf-8"))
    q = query.lower()
    match = None
    for e in index:
        if query.isdigit() and int(query) == e["nameHash"]:
            match = e
            break
        if not query.isdigit() and q in e["name"].lower():
            match = e
            break
    if not match:
        raise SystemExit(f"no route matching {query!r}")
    world = json.load(open(os.path.join(ROUTES, match["file"]), encoding="utf-8"))
    route = next(r for r in world["routes"] if r["nameHash"] == match["nameHash"])
    return match, route


def load_surface_points(map_id: int):
    path = os.path.join(SURFACES, f"world_{map_id}.json")
    data = json.load(open(path, encoding="utf-8"))
    xs, zs, labels = [], [], []
    for seg in data["segments"]:
        xs.extend(seg["x"])
        zs.extend(seg["z"])
        labels.extend([seg["surface"]] * len(seg["x"]))
    return np.array(xs), np.array(zs), np.array(labels)


def breakdown(route: dict, map_id: int, max_dist: float):
    sx, sz, slab = load_surface_points(map_id)
    for section in ("leadin", "route"):
        prof = route.get(section)
        if not prof:
            continue
        rx = np.array(prof["x"])
        rz = np.array(prof["z"])
        rd = np.array(prof["d"])
        surf = np.empty(len(rx), dtype=object)
        dist = np.empty(len(rx))
        for i in range(len(rx)):
            d2 = (sx - rx[i]) ** 2 + (sz - rz[i]) ** 2
            j = int(np.argmin(d2))
            dist[i] = float(np.sqrt(d2[j]))
            surf[i] = slab[j] if dist[i] <= max_dist else "Tarmac"
        # distance-weighted breakdown using segment lengths between samples
        seg_len = np.diff(rd, prepend=rd[0])
        totals: dict[str, float] = {}
        for s, dl in zip(surf, seg_len):
            totals[s] = totals.get(s, 0.0) + dl
        total = sum(totals.values()) or 1.0
        print(f"\n[{section}] {rd[-1]/1000:.2f} km, {len(rx)} pts, "
              f"nearest-road dist median={np.median(dist):.1f}m p95={np.percentile(dist,95):.1f}m")
        for s, m in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"    {s:10s} {m/1000:6.2f} km  {100*m/total:5.1f}%")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "Figure 8"
    max_dist = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    match, route = load_route(query)
    print(f"Route: {match['name']}  (nameHash {match['nameHash']}, map {match['mapID']}, file {match['file']})")
    breakdown(route, match["mapID"], max_dist)


if __name__ == "__main__":
    main()
