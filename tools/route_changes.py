"""Report new and renamed routes in the working tree, grouped by world."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.world_config import WORLD_NAMES


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


def world_label(route: dict) -> str:
    world = route.get("world", f"world{route.get('mapID', '?')}")
    name = WORLD_NAMES.get(route.get("mapID"))
    return f"{name} ({world})" if name and name != world else world


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git revision to compare against.")
    args = parser.parse_args()

    current = {key(route): route for route in load_json(ROOT / DATA_PATH)}
    previous = {key(route): route for route in load_base(args.base)}
    rows = [route for route_key, route in current.items() if route_key not in previous]
    rows.sort(key=lambda route: (route.get("mapID", 0), route.get("name", "").lower()))
    renamed = [
        route for route_key, route in current.items()
        if route_key in previous
        and route.get("name", "") != previous[route_key].get("name", "")
    ]
    renamed.sort(key=lambda route: (route.get("mapID", 0), route.get("name", "").lower()))

    print("| World | Route | Total distance (km) | Total elevation (m) |")
    print("|---|---|---:|---:|")
    last_world = None
    for route in rows:
        world = world_label(route)
        if world != last_world:
            if last_world is not None:
                print()
            last_world = world
        distance_m = (route.get("distance_m") or 0) + (route.get("leadin_distance_m") or 0)
        ascent_m = (route.get("ascent_m") or 0) + (route.get("leadin_ascent_m") or 0)
        print(f"| {world} | {route.get('name', '')} | {distance_m / 1000:.2f} | {ascent_m:.2f} |")
    print(f"\n{len(rows)} new route(s)")
    print("\n| World | Previous name | New name |")
    print("|---|---|---|")
    for route in renamed:
        world = world_label(route)
        old_name = previous[key(route)].get("name", "")
        print(f"| {world} | {old_name} | {route.get('name', '')} |")
    print(f"\n{len(renamed)} renamed route(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())