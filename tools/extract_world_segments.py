"""Extract in-game leaderboard segment boundaries for one Zwift world WAD."""
from __future__ import annotations

import argparse
import json
import os
import re

from extract_zwift_routes import load_multiroot, parse_segment_data, read_wad_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift")
    parser.add_argument("--world", default="world11")
    parser.add_argument("--out", default="zwift_route_segments_wad.json")
    args = parser.parse_args()

    wad = os.path.join(args.zwift_dir, "assets", "Worlds", args.world, "data_1.wad")
    entries = read_wad_entries(wad, keep_substrings=("/routes/",))
    routes = {}
    for name, data in entries.items():
        if not name.endswith(".xml"):
            continue
        root = load_multiroot(data)
        route = root.find("route")
        if route is None:
            continue
        segments = parse_segment_data(root)
        if segments:
            routes[route.get("name", "").strip()] = {
                "name_hash": int(route.get("nameHash", "0")),
                "segments": segments,
            }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"world": args.world, "routes": routes}, handle, indent=2)
    print(f"Extracted segment data for {len(routes)} routes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())