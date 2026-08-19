"""Throwaway: identify the road/marker at ~2.5 km into Sleepless City (map 9)."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared import surface_map_dev as smd
from shared.surface_map import _seg_surface

MAP_ID = 9
ROUTE_NAME = "Sleepless City"
TARGET_D = 2500.0  # metres into the ride

road_txt, styles = smd._read_road_style(MAP_ID)

# --- route geometry (lead-in + route), with cumulative distance ---
with open(os.path.join(ROOT, "zwift_routes", "world_9.json"), encoding="utf-8") as f:
    world = json.load(f)
route = next(r for r in world["routes"] if r["name"] == ROUTE_NAME)

xs, zs = [], []
for part in ("leadin", "route"):
    seg = route.get(part) or {}
    xs.extend(seg.get("x") or [])
    zs.extend(seg.get("z") or [])
xy = np.column_stack([xs, zs])
d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(zs)))])
i = int(np.argmin(np.abs(d - TARGET_D)))
px, pz = xy[i]
print(f"{ROUTE_NAME}: total {d[-1]/1000:.2f} km; point @ {d[i]:.0f} m -> "
      f"x={px:.1f} z={pz:.1f}")

# --- current surface label at that point (from world_9.json surface segments) ---
with open(os.path.join(ROOT, "zwift_surfaces", "world_9.json"), encoding="utf-8") as f:
    surf = json.load(f)
best = (1e18, None, None)
for s in surf["segments"]:
    sx, sz = s.get("x", []), s.get("z", [])
    if not sx:
        continue
    dd = (np.asarray(sx) - px) ** 2 + (np.asarray(sz) - pz) ** 2
    j = int(np.argmin(dd))
    if dd[j] < best[0]:
        best = (float(dd[j]), s.get("style"), s.get("surface"))
print(f"current label near point: {best[1]} -> {best[2]}  "
      f"(dist {math.sqrt(best[0]):.1f} m)")


# --- which road + marker sits on that point ---
def road_blocks():
    for m in smd._ROAD_BLOCK_RE.finditer(road_txt):
        b = m.group(0)
        rid = smd._ID_RE.search(b)
        yield (int(rid.group(1)) if rid else -1), b


import re
STYLE_RE = re.compile(r'm_style="(\d+)"')
T1_RE = re.compile(r'm_roadTime1="([-\d.]+)"')
T2_RE = re.compile(r'm_roadTime2="([-\d.]+)"')
MID_RE = re.compile(r'm_markerId="(\d+)"')
INV_RE = re.compile(r'm_isInvisible="(\d+)"')

hits = []
for rid, b in road_blocks():
    nodes = smd._exs._nodes_full(b)
    if len(nodes) < 2:
        continue
    looped = smd._LOOPED_RE.search(b) is not None
    pts = smd._exs._spline_points(nodes, looped=looped)
    arr = np.asarray(pts)
    dd = (arr[:, 0] - px) ** 2 + (arr[:, 1] - pz) ** 2
    j = int(np.argmin(dd))
    dist = math.sqrt(dd[j])
    if dist > 12:
        continue
    cum = smd._exs._cumulative(pts)
    total = cum[-1]
    t_here = cum[j] / total if total else 0.0
    ds = smd._DEFSTYLE_RE.search(b)
    base = int(ds.group(1)) if ds else 31
    base_name = smd._style_name(styles, base)
    covering = []
    for tag in smd._MARKER_TAG_RE.findall(b):
        st = STYLE_RE.search(tag)
        t1 = T1_RE.search(tag)
        t2 = T2_RE.search(tag)
        a = float(t1.group(1)) if t1 else 0.0
        c = float(t2.group(1)) if t2 else 1.0
        if c < a:
            a, c = c, a
        if a <= t_here <= c:
            mid = MID_RE.search(tag)
            inv = INV_RE.search(tag)
            style_idx = int(st.group(1)) if st else 0
            covering.append({
                "markerId": mid.group(1) if mid else None,
                "style": style_idx, "style_name": smd._style_name(styles, style_idx),
                "surface": smd._surface_for(styles, style_idx),
                "invisible": bool(inv), "has_style": st is not None,
                "t0": round(a, 4), "t1": round(c, 4),
            })
    hits.append((dist, rid, t_here, base, base_name, covering))

hits.sort()
for dist, rid, t_here, base, base_name, covering in hits[:4]:
    print(f"\nroad {rid}: dist {dist:.1f} m, t={t_here:.4f}, "
          f"base defaultStyle={base} ({base_name} -> "
          f"{smd._surface_for(styles, base)})")
    if not covering:
        print("   no marker covers this point (uses base surface)")
    for c in covering:
        kind = "INVISIBLE" if c["invisible"] else "visible"
        src = "explicit" if c["has_style"] else "DEFAULT style0"
        print(f"   marker {c['markerId']} [{kind}] t {c['t0']}-{c['t1']}  "
              f"style {c['style']} ({c['style_name']} -> {c['surface']})  [{src}]")
