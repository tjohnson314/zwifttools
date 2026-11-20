"""Publish catalogued Climb Portal ascents for the surface map.

Usage: python tools\\extract_climb_portal_routes.py [--zwift-dir PATH]
       [--out-routes zwift_routes] [--out-surfaces zwift_surfaces]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.extract_zwift_routes import _cumulative_planar, _resample, read_wad_entries
from tools.extract_zwift_surfaces import _bezier, _markers, _nodes_full, UNITS_PER_METRE

CLIMB_PORTAL_MAP_ID = 1001


def _text(element: ET.Element, tag: str) -> str:
    value = element.findtext(tag)
    if value is None or not value.strip():
        raise ValueError(f"Missing portal field: {tag}")
    return value.strip()


def _climb_points(road: ET.Element, start: float, finish: float) -> list[tuple[float, float, float]]:
    block = ET.tostring(road, encoding="unicode")
    horizontal = _nodes_full(block, auto_tangents=True)
    vertical = _nodes_full(block, auto_tangents=True, axes=(1, 2))
    count = len(road.findall("ent[@type='ENTITY_TYPE_ROADNODE']"))
    if count < 2 or len(horizontal) != count or len(vertical) != count:
        raise ValueError("Portal road has missing or invalid spline nodes")
    points = []
    for index in range(count - 1):
        low = max(0.0, start * (count - 1) - index)
        high = min(1.0, finish * (count - 1) - index)
        if high <= low:
            continue
        a, _, ao = horizontal[index]
        b, bi, _ = horizontal[index + 1]
        ay, _, ayo = vertical[index]
        by, byi, _ = vertical[index + 1]
        controls = (
            (a[0], ay[0], a[1]),
            (a[0] + ao[0], ay[0] + ayo[0], a[1] + ao[1]),
            (b[0] + bi[0], by[0] + byi[0], b[1] + bi[1]),
            (b[0], by[0], b[1]),
        )
        control_length = sum(math.dist(p, q) for p, q in zip(controls, controls[1:]))
        steps = max(1, math.ceil(control_length * (high - low) / 2))
        for step in range(steps + 1):
            if points and step == 0:
                continue
            u = low + (high - low) * step / steps
            x, z = _bezier(*[(p[0], p[2]) for p in controls], u)
            altitude, _ = _bezier(*[(p[1], p[2]) for p in controls], u)
            points.append((x * UNITS_PER_METRE, altitude * UNITS_PER_METRE,
                           z * UNITS_PER_METRE))
    return points


def parse_portal_climb(data: bytes, source: str) -> dict | None:
    road = ET.fromstring(data).find("./roads/road")
    if road is None:
        raise ValueError(f"No portal road in {source}")
    metadata = road.find("metadata")
    if metadata is None:
        raise ValueError(f"No portal metadata in {source}")
    if metadata.findtext("m_PortalRoadExcludeFromDictionary") in ("1", "true"):
        return None
    if (_text(road, "IsPortalRoad").lower() != "true"
            or _text(road, "splineType") != "Bezier" or _text(road, "looped") != "0"):
        raise ValueError(f"Unsupported portal spline in {source}")
    partitions = road.findall("portalRoadPartitions/partition")
    if not partitions:
        raise ValueError(f"No portal climb partitions in {source}")
    start = float(partitions[0].attrib["startTime"])
    finish = float(_text(road, "portalRoadEndGateRoadTime"))
    if not (0 <= start < finish <= 1):
        raise ValueError(f"Invalid portal climb boundaries in {source}: {start}, {finish}")
    block = ET.tostring(road, encoding="unicode")
    active_styles = {int(_text(road, "defaultStyle"))}
    active_styles.update(style for low, high, style in _markers(block)
                         if low < finish and high > start)
    if active_styles - {0, 31}:
        raise ValueError(f"Unresolved portal surface styles in {source}: {active_styles}")
    points = _climb_points(road, start, finish)
    distances = [distance / UNITS_PER_METRE for distance in _cumulative_planar(points)]
    if len(points) < 2 or distances[-1] <= 0:
        raise ValueError(f"Empty portal ascent in {source}")
    profile = _resample(distances, points, 600, ["Tarmac"] * len(points))
    road_id = int(_text(road, "id"))
    return {
        "mapID": CLIMB_PORTAL_MAP_ID,
        "nameHash": int(_text(metadata, "m_PortalRoadHash")),
        "name": _text(metadata, "m_PortalRoadUserFacingName"),
        "source_name": _text(metadata, "m_PortalRoadInternalName"),
        "source_file": source,
        "portal_road_id": road_id,
        "portal_start_time": start,
        "portal_finish_time": finish,
        "elevation_scale": 1.0,
        "distance_m": float(_text(metadata, "m_PortalRoadCourseLength")) / UNITS_PER_METRE,
        "ascent_m": float(_text(metadata, "m_PortalRoadCourseAscentF")) / UNITS_PER_METRE,
        "leadin_distance_m": 0.0, "leadin_ascent_m": 0.0,
        "sport_type": 1, "event_only": False,
        "leadin": None, "route": profile, "segments": [],
        "road_nodes": [{"road_id": road_id}],
        "start_locations": {"free_ride": {"leadin_distance_m": 0.0, "leadin_ascent_m": 0.0}},
        "route_versions": {
            mode: {"available": True, "geometry_status": "portal_spline",
                   "leadin_distance_m": 0.0, "leadin_ascent_m": 0.0,
                   "road_ids": [road_id]}
            for mode in ("race", "free_ride")
        },
    }


def extract_climb_portal_routes(zwift_dir: str, out_routes: str, out_surfaces: str) -> int:
    archive = Path(zwift_dir) / "assets" / "Worlds" / "roads.wad"
    entries = read_wad_entries(str(archive), keep_substrings=("Worlds/portal/road_",))
    routes = []
    for source, data in sorted(entries.items()):
        if not source.lower().endswith(".xml"):
            continue
        route = parse_portal_climb(data, source)
        if route is not None:
            routes.append(route)
    if not routes:
        raise ValueError("No catalogued Climb Portal ascents found")
    if len({route["nameHash"] for route in routes}) != len(routes):
        raise ValueError("Duplicate Climb Portal road hashes")
    routes.sort(key=lambda route: route["name"].casefold())
    segments = [{
        "road_id": route["portal_road_id"], "style": "NORMAL", "surface": "Tarmac",
        "x": route["route"]["x"], "z": route["route"]["z"],
    } for route in routes]
    metadata = {"mapID": CLIMB_PORTAL_MAP_ID, "world": "Climb Portal",
                "source": "zwift_wad_portal", "coordinate_system": "zwift_local_m"}
    exports = [
        (out_routes, json.dumps({**metadata, "routes": routes}, ensure_ascii=True,
                               separators=(",", ":"), allow_nan=False)),
        (out_surfaces, json.dumps({**metadata, "segments": segments}, ensure_ascii=True,
                                 separators=(",", ":"), allow_nan=False)),
    ]
    for directory, payload in exports:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "climb_portal.json").write_text(payload, encoding="utf-8")
    return len(routes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift")
    parser.add_argument("--out-routes", default="zwift_routes")
    parser.add_argument("--out-surfaces", default="zwift_surfaces")
    args = parser.parse_args()
    count = extract_climb_portal_routes(args.zwift_dir, args.out_routes, args.out_surfaces)
    print(f"Exported {count} Climb Portal ascents")


if __name__ == "__main__":
    main()
