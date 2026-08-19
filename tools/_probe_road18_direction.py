"""Throwaway: map Sleepless City distance <-> road-18 t around the 2.5km marker."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared import surface_map_dev as smd

MAP_ID = 9
road_txt, styles = smd._read_road_style(MAP_ID)

# road 18 spline + cumulative -> t lookup
block = next(m.group(0) for m in smd._ROAD_BLOCK_RE.finditer(road_txt)
             if smd._ID_RE.search(m.group(0))
             and int(smd._ID_RE.search(m.group(0)).group(1)) == 18)
nodes = smd._exs._nodes_full(block)
looped = smd._LOOPED_RE.search(block) is not None
pts = np.asarray(smd._exs._spline_points(nodes, looped=looped))
cum = np.asarray(smd._exs._cumulative(pts))
total = cum[-1]

# Sleepless City geometry with cumulative ride distance
with open(os.path.join(ROOT, "zwift_routes", "world_9.json"), encoding="utf-8") as f:
    world = json.load(f)
route = next(r for r in world["routes"] if r["name"] == "Sleepless City")
xs, zs = [], []
for part in ("leadin", "route"):
    seg = route.get(part) or {}
    xs.extend(seg.get("x") or [])
    zs.extend(seg.get("z") or [])
xy = np.column_stack([xs, zs])
rd = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(zs)))])

# For every route point, nearest road-18 t (only where the route is on road 18)
print("route_km   road18_t   dist_to_road18(m)")
prev_on = False
for i in range(len(xy)):
    d2 = ((pts[:, 0] - xy[i, 0]) ** 2 + (pts[:, 1] - xy[i, 1]) ** 2)
    j = int(np.argmin(d2))
    dist = float(d2[j]) ** 0.5
    on = dist < 8.0
    t = cum[j] / total
    if on and rd[i] < 3500:
        print(f"{rd[i]/1000:7.3f}   {t:7.4f}   {dist:5.1f}")

print("\nmarker boundaries on road 18:")
print("  180000 visible   : t 0.8042 - 1.0000  (asphalt you can see)")
print("  180003 invisible : t 0.7538 - 1.0000  (asphalt only if invisible counts)")
print("  base ANCIENTBRICK (brick) below t 0.7538")
