"""Complete, directional segment traversals from recorded activity telemetry."""

import hashlib
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from bike_comparison.physics import estimate_draft_efficiency, frontal_area_from_rider
from shared import surface_map
from shared.utils import calculate_normalized_power
from shared.zrl_leaderboard import _crossing_time_at_route_position

SEGMENT_CROSSING_CORRIDOR_M = 20


def catalog_version():
    paths = [surface_map.CALIBRATION_FILE, surface_map.ZRL_SEGMENTS_FILE,
             surface_map.WAD_SEGMENTS_FILE, *sorted(surface_map.ROUTE_DIR.glob("world_*.json"))]
    video_path = surface_map._route_path(surface_map.VIDEO_ROUTES_MAP_ID)
    if video_path.exists():
        paths.append(video_path)
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    return _catalog_version(signature)


@lru_cache(maxsize=1)
def _catalog_version(signature):
    digest = hashlib.sha256()
    for filename, _, _ in signature:
        with open(filename, "rb") as source:
            digest.update(source.read())
    for loader in (surface_map._load_route_world, surface_map._load_wad_segments,
                   surface_map._load_segment_names, surface_map._load_calibration,
                   surface_map._projection):
        loader.cache_clear()
    return "segmentPRsV6:" + digest.hexdigest()[:16]


def segment_map_id(activity):
    """Resolve API world IDs and video route hashes to the stored catalog."""
    world_id = activity.get("worldId")
    world_id = int(world_id) if world_id is not None else None
    if world_id == surface_map.VIDEO_ROUTES_MAP_ID:
        return world_id
    video_path = surface_map._route_path(surface_map.VIDEO_ROUTES_MAP_ID)
    if video_path.exists():
        routes = surface_map._load_route_world(surface_map.VIDEO_ROUTES_MAP_ID).get("routes", [])
        if any(
            (world_id is not None and world_id == route.get("source_mapID"))
            or (activity.get("routeId") is not None
                and str(activity["routeId"]) == str(route.get("nameHash")))
            for route in routes
        ):
            return surface_map.VIDEO_ROUTES_MAP_ID
    if world_id is None:
        raise ValueError("Activity has no world ID for segment matching")
    return world_id


@lru_cache(maxsize=16)
def world_segments(map_id, version):
    """Deduplicate route occurrences of the same directional timing segment."""
    candidates = {}
    for route in surface_map._load_route_world(map_id).get("routes", []):
        leadin = surface_map._pack_leg(map_id, route.get("leadin"))
        main = surface_map._pack_leg(map_id, route.get("route"))
        for marker in surface_map._route_segment_markers(map_id, route, leadin, main):
            path = marker.get("latlng")
            if not path or len(path) < 2:
                continue
            points = np.asarray(path, dtype=float)
            scale = np.cos(np.radians(points[:, 0].mean()))
            xy = (points - points[0]) * [111_320, 111_320 * scale]
            distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            key = f"{map_id}:{marker['hash']}"
            # Short lead-in occurrences must not replace a complete segment.
            if key not in candidates or distance > candidates[key]["path_distance_m"]:
                candidates[key] = {
                    "segment_id": key, "name": marker["name"], "latlng": path,
                    "path_distance_m": distance, "direction": marker["direction"],
                }
    by_name = defaultdict(list)
    for candidate in candidates.values():
        by_name[candidate["name"]].append(candidate)
    for name, variants in by_name.items():
        if len(variants) > 1 and {variant["direction"] for variant in variants} == {"F", "R"}:
            base_name = re.sub(r"\s+Reverse$", "", name, flags=re.IGNORECASE)
            for variant in variants:
                direction = "Forward" if variant["direction"] == "F" else "Reverse"
                variant["name"] = f"{base_name} ({direction})"
    return list(candidates.values())


def _crossings(frame, path, position):
    remaining = frame
    while len(remaining) >= 2:
        crossing = _crossing_time_at_route_position(
            remaining, path, position, max_lateral_m=SEGMENT_CROSSING_CORRIDOR_M,
        )
        if crossing is None:
            break
        yield crossing
        # Keep the sample after the crossing, not the edge just consumed.
        remaining = remaining[remaining["time_sec"] > crossing]


def _slice(frame, start, end):
    times = frame["time_sec"].to_numpy(dtype=float)
    interior = frame[(frame["time_sec"] > start) & (frame["time_sec"] < end)]
    boundaries = []
    for time in (start, end):
        boundaries.append({
            column: time if column == "time_sec" else float(np.interp(time, times, frame[column]))
            for column in frame.columns
        })
    return pd.concat([pd.DataFrame([boundaries[0]]), interior,
                      pd.DataFrame([boundaries[1]])], ignore_index=True)


def _mean(frame, column):
    if column not in frame or not np.isfinite(frame[column]).all():
        return None
    values = frame[column].to_numpy(dtype=float)
    dt = np.diff(frame["time_sec"].to_numpy(dtype=float))
    return float(np.sum((values[:-1] + values[1:]) * 0.5 * dt) / dt.sum())


def traversal_metrics(frame, start, end, weight_kg, height_cm, setup):
    portion = _slice(frame, start, end)
    times = portion["time_sec"].to_numpy(dtype=float)
    power = portion.get("power_watts")
    normalized = None
    if power is not None and np.isfinite(power).all():
        if end - start < 30:
            normalized = _mean(portion, "power_watts")
        else:
            # Regular midpoint samples avoid biasing toward the entry sample.
            edges = np.linspace(start, end, int(np.ceil(end - start)) + 1)
            regular_times = (edges[:-1] + edges[1:]) * 0.5
            normalized = calculate_normalized_power(
                np.interp(regular_times, times, power), time_sec=regular_times,
            )
    draft = None
    if (weight_kg is not None and height_cm is not None and setup is not None
            and {"power_watts", "speed_kmh", "altitude_m"}.issubset(portion.columns)
            and np.isfinite(portion[["power_watts", "speed_kmh", "altitude_m"]]).all().all()):
        estimate = estimate_draft_efficiency(
            portion, weight_kg, setup, frontal_area_from_rider(height_cm / 100, weight_kg),
        )
        draft = estimate["avg_draft_watts"] if estimate else None
    return {
        "start_time_sec": start, "end_time_sec": end, "total_time_sec": end - start,
        "distance_m": float((portion["distance_km"].iloc[-1] - portion["distance_km"].iloc[0]) * 1000),
        "elevation_change_m": (
            float(portion["altitude_m"].iloc[-1] - portion["altitude_m"].iloc[0])
            if "altitude_m" in portion and np.isfinite(portion["altitude_m"]).all() else None
        ),
        "avg_power": _mean(portion, "power_watts"),
        "normalized_power": normalized, "avg_draft_watts": draft,
        "initial_speed_kmh": (
            float(portion["speed_kmh"].iloc[0])
            if "speed_kmh" in portion and np.isfinite(portion["speed_kmh"].iloc[0]) else None
        ),
    }


def find_traversals(frame, segments, weight_kg=None, height_cm=None, setup=None):
    required = {"time_sec", "lat", "lng", "distance_km"}
    if not required.issubset(frame.columns):
        raise ValueError("GPS, time and distance telemetry are required to identify segments.")
    if len(frame) < 2 or not np.isfinite(frame[list(required)]).all().all():
        raise ValueError("Segment matching requires finite GPS, time and distance samples.")
    if not (np.diff(frame["time_sec"]) > 0).all():
        raise ValueError("Activity timestamps must be strictly increasing.")
    results = []
    for segment in segments:
        path = np.asarray(segment["latlng"], dtype=float)
        scale = np.cos(np.radians(path[:, 0].mean()))
        xy = (path - path[0]) * [111_320, 111_320 * scale]
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        length = distances[-1]
        if length <= 0:
            continue
        # Interior directional gates verify the route, not merely its endpoints.
        gates = np.interp(np.arange(100, length, 100), distances, np.arange(len(path)))
        starts = list(_crossings(frame, path, 0))
        ends = list(_crossings(frame, path, len(path) - 1))
        pass_number = 0
        for index, start in enumerate(starts):
            limit = starts[index + 1] + 1e-6 if index + 1 < len(starts) else np.inf
            for end in (value for value in ends if start < value <= limit):
                portion = _slice(frame, start, end)
                travelled = (portion["distance_km"].iloc[-1] - portion["distance_km"].iloc[0]) * 1000
                if abs(travelled - length) > max(60, length * 0.1):
                    continue
                gate_time = start
                for gate in gates:
                    crossing = _crossing_time_at_route_position(
                        portion[portion["time_sec"] >= gate_time], path, gate,
                        max_lateral_m=SEGMENT_CROSSING_CORRIDOR_M,
                    )
                    if crossing is None or crossing <= gate_time:
                        break
                    gate_time = crossing
                else:
                    pass_number += 1
                    results.append({
                        "segment_id": segment["segment_id"], "segment_name": segment["name"],
                        "pass": pass_number,
                        **traversal_metrics(frame, start, end, weight_kg, height_cm, setup),
                    })
                    break
    return sorted(results, key=lambda row: (row["start_time_sec"], row["segment_id"]))
