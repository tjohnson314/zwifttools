"""Throwaway probe: dump road.xml / roadstyle.xml structure from one world WAD."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_routes import read_wad_entries  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"


def main() -> None:
    world = sys.argv[1] if len(sys.argv) > 1 else "world1"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad)
    print(f"== {world} entries ({len(entries)}) ==")
    for name in sorted(entries):
        print(f"  {name}  ({len(entries[name])} bytes)")

    for key in ("road.xml", "roadstyle.xml"):
        match = [n for n in entries if n.lower().endswith(key)]
        if not match:
            print(f"\n!! no {key}")
            continue
        name = match[0]
        data = entries[name]
        print(f"\n===== {name} ({len(data)} bytes) first 4000 chars =====")
        print(data.decode("utf-8", "replace")[:4000])


if __name__ == "__main__":
    main()
