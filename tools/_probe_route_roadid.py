"""Probe: do raw route checkpoints carry a `road` id + `time` we can join to
road.xml road <id> + marker roadTime ranges?"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else "world10"
    name_hash = int(sys.argv[2]) if len(sys.argv) > 2 else 3919912289  # Casse-Pattes
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad, keep_substrings=("/routes/",))
    for nm, data in entries.items():
        if not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        if r is None or r.get("nameHash") != str(name_hash):
            continue
        print(f"FOUND {r.get('name')} in {nm}")
        cp = root.find("highrescheckpoint")
        entries_list = cp.findall("entry") if cp is not None else []
        print(f"highrescheckpoint entries: {len(entries_list)}")
        # distinct attributes
        attrs = set()
        for e in entries_list[:50]:
            attrs.update(e.attrib.keys())
        print("attrs:", sorted(attrs))
        for e in entries_list[:15]:
            print("  ", dict(e.attrib))
        # road id range + time range
        roads = [e.get("road") for e in entries_list if e.get("road") is not None]
        times = [float(e.get("time")) for e in entries_list if e.get("time") is not None]
        if times:
            print(f"time range: {min(times):.4f}..{max(times):.4f}")
        print(f"distinct road ids used: {sorted(set(roads))[:40]}")
        return
    print("route not found")


if __name__ == "__main__":
    main()
