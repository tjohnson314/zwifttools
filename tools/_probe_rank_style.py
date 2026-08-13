"""Rank routes in one world by distance on a given style (exact road-id join)."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, UNSET_STYLE  # noqa: E402
from _probe_route_surface_exact import parse_roads  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
WORLD = sys.argv[1] if len(sys.argv) > 1 else "world9"
TARGET = (sys.argv[2] if len(sys.argv) > 2 else "PACKEDSAND").upper()


def resolve_style(rid, t, roads, styles):
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    return styles[style]


def main():
    wad = os.path.join(ZWIFT, "assets", "Worlds", WORLD, "data_1.wad")
    entries = read_wad_entries(wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml"))
    road_xml = next(v for k, v in entries.items()
                    if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()).decode("utf-8", "replace")
    styles = parse_roadstyles(next(v for k, v in entries.items() if k.lower().endswith("roadstyle.xml")))
    roads = parse_roads(road_xml)

    rows = []
    for nm, data in entries.items():
        if "/routes/" not in nm or not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        cp = root.find("highrescheckpoint")
        if r is None or cp is None:
            continue
        pts = cp.findall("entry")
        km = 0.0
        total = 0.0
        prev = None
        for e in pts:
            x, z = float(e.get("x")), float(e.get("z"))
            sname = resolve_style(int(e.get("road")), float(e.get("time")), roads, styles).upper()
            if prev is not None:
                dl = math.hypot(x - prev[0], z - prev[1]) / 100.0
                total += dl
                if sname == TARGET:
                    km += dl
            prev = (x, z)
        if km > 0:
            rows.append((km, total, r.get("name")))

    print(f"{WORLD} routes ranked by {TARGET} distance:\n")
    for km, total, name in sorted(rows, reverse=True):
        pct = 100 * km / total if total else 0
        print(f"  {km/1000:6.2f} km  ({pct:4.1f}% of {total/1000:5.1f} km)  {name.strip()}")


if __name__ == "__main__":
    main()
