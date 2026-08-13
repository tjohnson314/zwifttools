"""Proof: resolve route surface EXACTLY via checkpoint road-id + time join.

For each highrescheckpoint entry: surface = road[road_id] base defaultStyle,
overridden by any ROADMARKER on that road whose [roadTime1, roadTime2] covers the
checkpoint's `time`. Distance-weighted breakdown printed for comparison with the
spatial nearest-point method.
"""
from __future__ import annotations

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, STYLE_TO_SURFACE, UNSET_STYLE  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"


def parse_roads(road_xml: str):
    """road_id -> (base_style_id, [(t0, t1, style_id), ...])."""
    roads = {}
    for m in re.finditer(r"<road>(.*?)</road>", road_xml, re.S):
        block = m.group(1)
        idm = re.search(r"<id>(\d+)</id>", block)
        if not idm:
            continue
        rid = int(idm.group(1))
        dm = re.search(r"<defaultStyle>(\d+)</defaultStyle>", block)
        base = int(dm.group(1)) if dm else UNSET_STYLE
        markers = []
        for em in re.finditer(r'<ent\b[^>]*type="ENTITY_TYPE_ROADMARKER"[^>]*>', block):
            tag = em.group(0)
            st = re.search(r'm_style="(\d+)"', tag)
            t1 = re.search(r'm_roadTime1="([-\d.]+)"', tag)
            t2 = re.search(r'm_roadTime2="([-\d.]+)"', tag)
            if st and t1 and t2:
                a, b = float(t1.group(1)), float(t2.group(1))
                if b < a:
                    a, b = b, a
                markers.append((a, b, int(st.group(1))))
        roads[rid] = (base, markers)
    return roads


def resolve(rid, t, roads, styles):
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    name = styles[style]
    return STYLE_TO_SURFACE.get(name, "Tarmac")


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else "world10"
    name_hash = sys.argv[2] if len(sys.argv) > 2 else "3919912289"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
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
        if r is None or r.get("nameHash") != name_hash:
            continue
        cp = root.find("highrescheckpoint")
        pts = cp.findall("entry")
        print(f"{r.get('name')}: {len(pts)} checkpoints, roads used: "
              f"{sorted(set(int(e.get('road')) for e in pts))}")
        totals = {}
        prev = None
        for e in pts:
            x, z = float(e.get("x")), float(e.get("z"))
            surf = resolve(int(e.get("road")), float(e.get("time")), roads, styles)
            if prev is not None:
                dl = math.hypot(x - prev[0], z - prev[1]) / 100.0
                totals[surf] = totals.get(surf, 0.0) + dl
            prev = (x, z)
        total = sum(totals.values()) or 1.0
        print(f"EXACT road-id join breakdown ({total/1000:.2f} km):")
        for s, mlen in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"    {s:10s} {mlen/1000:6.2f} km  {100*mlen/total:5.1f}%")
        # per-road default styles used
        for rid in sorted(set(int(e.get('road')) for e in pts)):
            base, markers = roads.get(rid, (UNSET_STYLE, []))
            bname = styles[base] if 0 <= base < len(styles) else f"#{base}"
            print(f"    road {rid}: default={bname}, markers={len(markers)}")
        return
    print("route not found")


if __name__ == "__main__":
    main()
