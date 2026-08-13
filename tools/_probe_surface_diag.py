"""Diagnostic: for a route, how close is the nearest DIRT/COBBLE surface point,
and is it being out-competed by an overlapping TARMAC point?"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_surface_points(map_id):
    d = json.load(open(os.path.join(ROOT, "zwift_surfaces", f"world_{map_id}.json"), encoding="utf-8"))
    xs, zs, lab = [], [], []
    for s in d["segments"]:
        xs.extend(s["x"]); zs.extend(s["z"]); lab.extend([s["surface"]] * len(s["x"]))
    return np.array(xs), np.array(zs), np.array(lab)


def main():
    query = sys.argv[1]
    idx = json.load(open(os.path.join(ROOT, "zwift_routes", "index.json"), encoding="utf-8"))
    e = next(e for e in idx if query.lower() in e["name"].lower())
    world = json.load(open(os.path.join(ROOT, "zwift_routes", e["file"]), encoding="utf-8"))
    route = next(r for r in world["routes"] if r["nameHash"] == e["nameHash"])
    sx, sz, slab = load_surface_points(e["mapID"])

    prof = route["route"]
    rx, rz = np.array(prof["x"]), np.array(prof["z"])
    print(f"{e['name']} map {e['mapID']}, {len(rx)} pts")
    for target in ("Dirt", "Cobbles"):
        mask = slab == target
        if not mask.any():
            print(f"  no {target} points in world"); continue
        tx, tz = sx[mask], sz[mask]
        nearest_t = np.array([np.sqrt(np.min((tx - rx[i])**2 + (tz - rz[i])**2)) for i in range(len(rx))])
        nearest_any = np.array([np.sqrt(np.min((sx - rx[i])**2 + (sz - rz[i])**2)) for i in range(len(rx))])
        within10 = int((nearest_t <= 10).sum())
        # points where target is within 10m but a closer non-target point exists
        beaten = int(((nearest_t <= 10) & (nearest_any < nearest_t - 0.5)).sum())
        print(f"  {target}: pts with {target}<=10m: {within10}/{len(rx)}; "
              f"of those out-competed by nearer other surface: {beaten}; "
              f"min {target} dist {nearest_t.min():.1f}m")


if __name__ == "__main__":
    main()
