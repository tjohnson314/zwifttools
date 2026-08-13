"""Throwaway probe #3: m_style semantics + defaultStyle=31 meaning + style values."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_routes import read_wad_entries  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
STYLES = ["NORMAL","WOODEN","COBBLE","DIRT","ONELINE","WET","MOUNTAIN","LAVA",
          "INVISIBLE_DIRT","BEACHPATH","DESERT","TRAILDIRT","TRAILGRAVEL","TRACK",
          "DOTHINT","GRAVEL","INVISIBLE_GRAVEL","BOXHILL","PACKEDSAND","SNOW"]


def label(v: int) -> str:
    return STYLES[v] if 0 <= v < len(STYLES) else f"#{v}"


def main() -> None:
    world = sys.argv[1] if len(sys.argv) > 1 else "world1"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad)
    txt = next(v for k, v in entries.items() if k.lower().endswith("road.xml")).decode("utf-8", "replace")

    # all m_style values
    mstyles = [int(x) for x in re.findall(r'm_style="(\d+)"', txt)]
    print("== m_style value distribution ==")
    for v, c in Counter(mstyles).most_common():
        print(f"  {label(v)} ({v}): {c}")

    # split into roads; for each: defaultStyle, #nodes, #nodes with m_style
    roads = re.findall(r'<road>(.*?)</road>', txt, re.S)
    print(f"\n== {len(roads)} roads ==")
    ds31_all_styled = 0
    ds31_count = 0
    sample_31 = None
    sample_mixed = None
    for r in roads:
        dm = re.search(r'<defaultStyle>(\d+)</defaultStyle>', r)
        ds = int(dm.group(1)) if dm else -1
        nodes = re.findall(r'<ent\b[^>]*type="ENTITY_TYPE_ROADNODE"[^>]*>', r)
        styled = [n for n in nodes if 'm_style=' in n]
        if ds == 31:
            ds31_count += 1
            if styled and len(styled) == len(nodes):
                ds31_all_styled += 1
            if sample_31 is None and 0 < len(styled) < len(nodes):
                sample_31 = (ds, nodes)
        if sample_mixed is None and len({re.search(r'm_style="(\d+)"', n).group(1) for n in styled}) > 1:
            sample_mixed = (ds, nodes)

    print(f"defaultStyle=31 roads: {ds31_count}; of those, all-nodes-styled: {ds31_all_styled}")

    # Show a road that has MIXED m_style values (surface transition) to learn carry semantics
    if sample_mixed:
        ds, nodes = sample_mixed
        print(f"\n== sample road with mixed m_style (defaultStyle={label(ds)}) — node order ==")
        for i, n in enumerate(nodes[:60]):
            ms = re.search(r'm_style="(\d+)"', n)
            pos = re.search(r'm_pos="\{([^}]*)\}"', n)
            tag = label(int(ms.group(1))) if ms else "."
            xyz = pos.group(1) if pos else "?"
            print(f"  [{i:3d}] style={tag:16s} pos={xyz}")


if __name__ == "__main__":
    main()
