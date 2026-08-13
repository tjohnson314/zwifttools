"""Per-segment surface breakdown for one route (exact road-id join).

Walks highrescheckpoints in order, resolves surface, and reports each continuous
run of a surface as [start_km, end_km] along the route (checkpoint distance).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, STYLE_TO_SURFACE, UNSET_STYLE  # noqa: E402
from _probe_route_surface_exact import parse_roads  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
WORLD = sys.argv[1] if len(sys.argv) > 1 else "world9"
NAME_HASH = sys.argv[2] if len(sys.argv) > 2 else "2833089100"


def resolve(rid, t, roads, styles):
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    name = styles[style]
    return STYLE_TO_SURFACE.get(name, "Tarmac"), name


def main():
    wad = os.path.join(ZWIFT, "assets", "Worlds", WORLD, "data_1.wad")
    entries = read_wad_entries(wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml"))
    road_xml = next(v for k, v in entries.items()
                    if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()).decode("utf-8", "replace")
    styles = parse_roadstyles(next(v for k, v in entries.items() if k.lower().endswith("roadstyle.xml")))
    roads = parse_roads(road_xml)

    for nm, data in entries.items():
        if "/routes/" not in nm or not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        cp = root.find("highrescheckpoint")
        if r is None or cp is None or r.get("nameHash") != NAME_HASH:
            continue
        pts = cp.findall("entry")

        # cumulative route distance + surface per checkpoint
        d = 0.0
        prev = None
        rows = []  # (dist_km, surface, style)
        for e in pts:
            x, z = float(e.get("x")), float(e.get("z"))
            if prev is not None:
                d += math.hypot(x - prev[0], z - prev[1]) / 100.0
            surf, style = resolve(int(e.get("road")), float(e.get("time")), roads, styles)
            rows.append((d, surf, style))
            prev = (x, z)

        total = rows[-1][0]
        print(f"{r.get('name').strip()}  ({total/1000:.2f} km, {len(pts)} checkpoints)\n")

        # group consecutive same-surface runs
        runs = []
        seg_start = rows[0][0]
        cur = rows[0][1]
        cur_style = rows[0][2]
        for i in range(1, len(rows)):
            if rows[i][1] != cur:
                runs.append((seg_start, rows[i][0], cur, cur_style))
                seg_start = rows[i][0]
                cur = rows[i][1]
                cur_style = rows[i][2]
        runs.append((seg_start, rows[-1][0], cur, cur_style))

        print("All surface segments along the route:")
        for a, b, surf, style in runs:
            length = b - a
            flag = "  <-- SAND" if surf == "Sand" else ""
            print(f"  {a/1000:6.2f} - {b/1000:6.2f} km  ({length:6.0f} m)  {surf:8s} [{style}]{flag}")

        print("\nContinuous SAND (PACKEDSAND) sections only:")
        n = 0
        for a, b, surf, style in runs:
            if surf == "Sand":
                n += 1
                print(f"  #{n}: {a/1000:6.2f} - {b/1000:6.2f} km  ({b - a:5.0f} m)")
        return
    print("route not found")


if __name__ == "__main__":
    main()
