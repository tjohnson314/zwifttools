"""Report routes newly added to the working tree, grouped by world."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("zwift_routes/index.json")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base(base: str) -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"{base}:{DATA_PATH.as_posix()}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    return json.loads(raw)


def key(route: dict) -> tuple[int, int | str]:
    name_hash = route.get("nameHash")
    return route.get("mapID", 0), name_hash if name_hash is not None else route.get("name", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git revision to compare against.")
    args = parser.parse_args()

    current = {key(route): route for route in load_json(ROOT / DATA_PATH)}
    previous = {key(route) for route in load_base(args.base)}
    rows = [route for route_key, route in current.items() if route_key not in previous]
    rows.sort(key=lambda route: (route.get("mapID", 0), route.get("name", "").lower()))

    print("| World | Route | Total distance (km) | Total elevation (m) |")
    print("|---|---|---:|---:|")
    last_world = None
    for route in rows:
        world = route.get("world", f"World {route.get('mapID', '?')}")
        if world != last_world:
            if last_world is not None:
                print()
            last_world = world
        distance_m = (route.get("distance_m") or 0) + (route.get("leadin_distance_m") or 0)
        ascent_m = (route.get("ascent_m") or 0) + (route.get("leadin_ascent_m") or 0)
        print(f"| {world} | {route.get('name', '')} | {distance_m / 1000:.2f} | {ascent_m:.2f} |")
    print(f"\n{len(rows)} new route(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())