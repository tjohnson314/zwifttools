"""Rank one world's routes by WET coverage AND flatness (for a clean Crr test)."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, UNSET_STYLE  # noqa: E402
from _probe_route_surface_exact import parse_roads  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
WORLD = sys.argv[1] if len(sys.argv) > 1 else "world1"
TARGET = (sys.argv[2] if len(sys.argv) > 2 else "WET").upper()


def resolve_style(rid, t, roads, styles):
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    return styles[style].upper()


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
        total = 0.0
        wet = 0.0
        gain = 0.0            # sum of positive elevation deltas (m)
        wet_grade_abs = []    # |gradient| on WET segments
        prev = None
        for e in pts:
            x, z, y = float(e.get("x")), float(e.get("z")), float(e.get("y"))
            s = resolve_style(int(e.get("road")), float(e.get("time")), roads, styles)
            if prev is not None:
                dl = math.hypot(x - prev[0], z - prev[1]) / 100.0
                dy = y - prev[2]
                total += dl
                if dy > 0:
                    gain += dy
                if s == TARGET:
                    wet += dl
                    if dl > 0.5:
                        wet_grade_abs.append(abs(dy) / dl)
            prev = (x, z, y)
        if wet > 0 and total > 0:
            pct = 100 * wet / total
            mean_wet_grade = 100 * (sum(wet_grade_abs) / len(wet_grade_abs)) if wet_grade_abs else 0.0
            rows.append((wet / 1000, pct, total / 1000, gain, mean_wet_grade, r.get("name").strip()))

    # Flatness-first: routes with high WET km and low mean |grade| on WET.
    print(f"{WORLD} routes with {TARGET}, sorted by WET flatness (low mean |grade| first):\n")
    print(f"  {'WETkm':>6} {'WET%':>5} {'rtkm':>6} {'gain_m':>7} {'wet|grade|%':>11}  route")
    rows = [x for x in rows if x[0] >= 3.0]  # at least 3 km of WET to be useful
    for wetkm, pct, rtkm, gain, grade, name in sorted(rows, key=lambda x: x[4]):
        print(f"  {wetkm:6.2f} {pct:5.1f} {rtkm:6.1f} {gain:7.0f} {grade:11.2f}  {name}")


if __name__ == "__main__":
    main()
