"""Throwaway probe #2: understand road/roadstyle surface structure.

- Full ordered list of roadstyle segment styles (index = defaultStyle).
- Whether <road> or <ent> road nodes carry per-node/per-segment style overrides.
- Distribution of defaultStyle across roads.
- All distinct attribute names appearing on <ent ENTITY_TYPE_ROADNODE>.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"


def main() -> None:
    world = sys.argv[1] if len(sys.argv) > 1 else "world1"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad)

    style_data = next(v for k, v in entries.items() if k.lower().endswith("roadstyle.xml"))
    road_data = next(v for k, v in entries.items() if k.lower().endswith("road.xml"))

    # 1. ordered styles
    styles = re.findall(r'<segment\s+style="([^"]+)"[^>]*?(?:sound="([^"]*)")?', style_data.decode("utf-8", "replace"))
    print("== roadstyle segments (index -> style / sound) ==")
    for i, (st, snd) in enumerate(styles):
        print(f"  [{i:2d}] {st:16s} sound={snd}")

    txt = road_data.decode("utf-8", "replace")

    # 2. distinct attribute names on road nodes
    node_attrs = Counter()
    for m in re.finditer(r'<ent\b([^>]*)>', txt):
        for a in re.findall(r'(\w+)=', m.group(1)):
            node_attrs[a] += 1
    print("\n== distinct <ent> attributes ==")
    for a, c in node_attrs.most_common():
        print(f"  {a}: {c}")

    # 3. any 'style' word anywhere on ent nodes or within <road> children (not defaultStyle)
    style_refs = re.findall(r'(\w*[Ss]tyle\w*)="?', txt)
    print("\n== style-ish tokens in road.xml ==")
    for a, c in Counter(style_refs).most_common():
        print(f"  {a}: {c}")

    # 4. defaultStyle distribution + per-road child tags
    ds = [int(x) for x in re.findall(r'<defaultStyle>(\d+)</defaultStyle>', txt)]
    print(f"\n== defaultStyle distribution across {len(ds)} roads ==")
    for v, c in Counter(ds).most_common():
        label = styles[v][0] if v < len(styles) else "?"
        print(f"  style {v} ({label}): {c} roads")

    # 5. per-road child element tags (to see if there's a per-segment style list)
    first_road = re.search(r'<road>(.*?)</road>', txt, re.S)
    if first_road:
        tags = Counter(re.findall(r'<(\w+)', first_road.group(1)))
        print("\n== child tags in first <road> ==")
        for t, c in tags.most_common():
            print(f"  {t}: {c}")

    # 6. search for any element that references a style index per segment (e.g. <style>, <segment>, styleOverride)
    for pat in ("styleOverride", "segmentStyle", "<style>", "roadStyle", "surface", "Surface", "styleIndex"):
        n = txt.count(pat)
        if n:
            print(f"  found '{pat}': {n}")


if __name__ == "__main__":
    main()
