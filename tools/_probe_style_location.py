"""Locate contiguous runs of a target style on one route (world, name substr, style)."""
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
NAME_SUB = (sys.argv[2] if len(sys.argv) > 2 else "Road to Sky").lower()
TARGET = (sys.argv[3] if len(sys.argv) > 3 else "INVISIBLE_DIRT").upper()


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

    match = None
    for nm, data in entries.items():
        if "/routes/" not in nm or not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        cp = root.find("highrescheckpoint")
        if r is None or cp is None:
            continue
        if NAME_SUB in (r.get("name") or "").lower():
            match = (r, cp)
            break
    if match is None:
        print(f"No route matching '{NAME_SUB}' in {WORLD}")
        return
    r, cp = match
    leadin = float(r.get("leadinDistanceInMeters") or 0.0)
    name = r.get("name").strip()

    runs = []
    cur = None
    d = 0.0
    prev = None
    total_target = 0.0
    for e in cp.findall("entry"):
        x, z = float(e.get("x")), float(e.get("z"))
        s = resolve_style(int(e.get("road")), float(e.get("time")), roads, styles)
        if prev is not None:
            d += math.hypot(x - prev[0], z - prev[1]) / 100.0
        prev = (x, z)
        if s == TARGET:
            total_target += 0.0 if cur is None else 0.0
            if cur is None:
                cur = [d, d]
            else:
                cur[1] = d
        else:
            if cur is not None:
                runs.append(tuple(cur))
                cur = None
    if cur is not None:
        runs.append(tuple(cur))

    print(f"{WORLD} | {name} | lead-in {leadin:.0f} m | total {d/1000:.2f} km")
    print(f"{TARGET} runs (distance from route start / from pen incl. lead-in):\n")
    if not runs:
        print("  (none)")
        return
    tot = 0.0
    for a, b in runs:
        length = b - a
        tot += length
        print(f"  route {a/1000:6.3f}-{b/1000:6.3f} km  "
              f"(pen {(a + leadin)/1000:6.3f}-{(b + leadin)/1000:6.3f} km)  "
              f"len {length:5.0f} m")
    print(f"\n  total {TARGET}: {tot:.0f} m across {len(runs)} run(s)")


if __name__ == "__main__":
    main()
