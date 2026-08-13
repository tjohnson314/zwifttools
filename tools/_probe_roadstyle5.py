"""Throwaway probe #5: which ent type carries m_style (mid-road transition markers)."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_routes import read_wad_entries  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"


def main() -> None:
    world = sys.argv[1] if len(sys.argv) > 1 else "world1"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad)
    txt = next(v for k, v in entries.items() if k.lower().endswith("road.xml")).decode("utf-8", "replace")

    types_with_style = Counter()
    for m in re.finditer(r'<ent\b([^>]*)>', txt):
        attrs = m.group(1)
        if 'm_style=' in attrs:
            t = re.search(r'type="([^"]+)"', attrs)
            types_with_style[t.group(1) if t else "?"] += 1
    print("== ent types carrying m_style ==")
    for t, c in types_with_style.most_common():
        print(f"  {t}: {c}")

    # show a few full ent tags that carry m_style
    print("\n== sample m_style ents ==")
    shown = 0
    for m in re.finditer(r'<ent\b[^>]*m_style="[^"]*"[^>]*>', txt):
        print("  " + m.group(0)[:240])
        shown += 1
        if shown >= 8:
            break

    # all distinct ent types
    print("\n== all ent types ==")
    for t, c in Counter(re.findall(r'type="([^"]+)"', txt)).most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
