"""Scan all worlds; find routes whose exact road-id join traverses target styles."""
from __future__ import annotations

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, UNSET_STYLE  # noqa: E402
from _probe_route_surface_exact import parse_roads  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
TARGETS = {s.upper() for s in (sys.argv[1:] or ["PACKEDSAND", "INVISIBLE_DIRT"])}


def resolve_style(rid, t, roads, styles):
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    return styles[style]


def scan_world(world):
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    if not os.path.exists(wad):
        return
    entries = read_wad_entries(wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml"))
    try:
        road_xml = next(v for k, v in entries.items()
                        if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()).decode("utf-8", "replace")
        styles = parse_roadstyles(next(v for k, v in entries.items() if k.lower().endswith("roadstyle.xml")))
    except StopIteration:
        return
    if not (TARGETS & {s.upper() for s in styles}):
        return  # world has none of the target styles at all
    roads = parse_roads(road_xml)
    for nm, data in entries.items():
        if "/routes/" not in nm or not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        cp = root.find("highrescheckpoint")
        if r is None or cp is None:
            continue
        pts = cp.findall("entry")
        hit = {}
        prev = None
        for e in pts:
            x, z = float(e.get("x")), float(e.get("z"))
            sname = resolve_style(int(e.get("road")), float(e.get("time")), roads, styles).upper()
            if prev is not None and sname in TARGETS:
                dl = math.hypot(x - prev[0], z - prev[1]) / 100.0
                hit[sname] = hit.get(sname, 0.0) + dl
            prev = (x, z)
        if hit:
            frag = ", ".join(f"{k}={v/1000:.2f}km" for k, v in sorted(hit.items(), key=lambda kv: -kv[1]))
            print(f"  [{world}] {r.get('name')!r} (nameHash={r.get('nameHash')}): {frag}")


def main():
    print(f"Searching for routes traversing: {sorted(TARGETS)}\n")
    for i in range(15):
        scan_world(f"world{i}")


if __name__ == "__main__":
    main()
