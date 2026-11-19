"""Power Play leaderboards from directional arch starts and recorded telemetry."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from shared.event_route import _arch_catalog
from shared.data_fetcher import convert_telemetry_to_dataframe
from shared.segment_prs import _crossings, _mean, _slice
from shared.surface_map import local_to_latlng

logger = logging.getLogger(__name__)
DURATIONS = (5, 10, 20)
RACE_ID_PATTERN = r"race_(?:data|event)_[0-9]+"


def is_power_play(title):
    title = (title or "").casefold()
    return "zwift power play" in title and re.search(r"\bstage\s+1\b", title) is not None


def power_arches(map_id, route_id):
    if map_id is None:
        worlds = [world for world, routes in _arch_catalog().items() if str(route_id) in routes]
        if len(worlds) != 1:
            raise ValueError("The race world and route could not be identified.")
        map_id = worlds[0]
    arches = _arch_catalog().get(str(map_id), {}).get(str(route_id), [])
    result = []
    seen = set()
    for arch in arches:
        if arch["type"] != "power" or not arch["enabled"]:
            continue
        if arch.get("duration_sec") not in DURATIONS or not arch.get("start_line"):
            raise ValueError("Power arch timing data is missing from the route catalog.")
        key = (arch["road_id"], arch["direction"], arch["duration_sec"])
        if key in seen:
            continue
        seen.add(key)
        path = np.asarray(arch["start_line"], dtype=float)
        latlng = local_to_latlng(int(map_id), path[:, 0], path[:, 1])
        if latlng is None:
            raise ValueError("GPS calibration is unavailable for this Power Play route.")
        result.append({**arch, "latlng": latlng})
    if not result:
        raise ValueError("No Power Play arches are available for this race route.")
    return result


def best_efforts(frame, arches, start_time=0, end_time=None):
    required = ["time_sec", "lat", "lng", "power_watts"]
    if not set(required).issubset(frame.columns) or len(frame) < 2:
        raise ValueError("GPS, power and time telemetry are required.")
    times = frame["time_sec"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Telemetry timestamps must be finite and strictly increasing.")
    cadence = float(np.median(np.diff(times)))
    limit = min(times[-1], end_time) if end_time is not None else times[-1]
    best = {}
    incomplete = 0
    for arch in arches:
        duration = int(arch["duration_sec"])
        for start in _crossings(frame, arch["latlng"], 0):
            if start < start_time or start > limit:
                continue
            end = start + duration
            if end > limit:
                incomplete += 1
                continue
            left = max(0, np.searchsorted(times, start, side="right") - 1)
            right = np.searchsorted(times, end, side="left") + 1
            samples = frame.iloc[left:right]
            if (not np.isfinite(samples[required]).all().all()
                    or (np.diff(samples["time_sec"]) > cadence * 1.5).any()):
                incomplete += 1
                continue
            power = _mean(_slice(frame[["time_sec", "power_watts"]], start, end), "power_watts")
            if duration not in best or power > best[duration]["avg_power"]:
                best[duration] = {
                    "duration_sec": duration, "avg_power": power,
                    "start_time_sec": start, "end_time_sec": end,
                    "arch_name": arch["name"], "road_id": arch["road_id"],
                }
    return best, incomplete


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_leaderboard(race_path):
    """Read the current roster and original samples, never resampled replay power."""
    race_path = Path(race_path)
    manifest_path = race_path / "event_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        title = manifest.get("race_name", "")
        groups = manifest["subgroups"]
        source_activity_id = None
        event_id = manifest.get("event_id")
    else:
        race_meta = _read_json(race_path / "race_meta.json")
        title = race_meta.get("race_name", "")
        source_activity_id = race_meta.get("source_activity_id")
        event_id = race_meta.get("event_id")
        groups = [{"race_id": race_path.name, "label": None}]
    if not is_power_play(title):
        raise ValueError('This leaderboard is only available for Stage 1 - Zwift Power Play.')

    warnings = []
    best_by_rider = {}
    rider_count = 0
    for group in groups:
        group_id = group["race_id"]
        if not re.fullmatch(RACE_ID_PATTERN, group_id):
            raise ValueError("Invalid cached race ID.")
        directory = race_path.parent / group_id
        meta = _read_json(directory / "race_meta.json")
        source_activity_id = source_activity_id or meta.get("source_activity_id")
        summary = pd.read_csv(
            directory / "complete_race_summary.csv",
            dtype={"activity_id": str, "player_id": str}, keep_default_na=False,
        )
        for rider in summary.to_dict("records"):
            rider_count += 1
            name = rider.get("name") or f"Rider {rider['rank']}"
            if rider.get("status", "SUCCESS") != "SUCCESS":
                warnings.append(f"{name}: telemetry unavailable ({rider['status']}).")
                continue
            activity_id = str(rider["activity_id"])
            if not activity_id.isascii() or not activity_id.isdigit():
                raise ValueError("Invalid rider activity ID in cached roster.")
            stem = f"rank{int(rider['rank'])}_{activity_id}"
            try:
                samples = pd.read_csv(directory / f"{stem}.csv")
                raw = _read_json(directory / f"{stem}_raw.json")
                if not {"lat", "lng"}.issubset(samples.columns):
                    gps = convert_telemetry_to_dataframe(raw)
                    if not np.array_equal(samples["time_sec"], gps["time_sec"]):
                        raise ValueError("GPS samples do not match the recorded power timestamps.")
                    samples[["lat", "lng"]] = gps[["lat", "lng"]]
                route_id = meta.get("route_id") or raw.get("routeId")
                arches = power_arches(meta.get("world_id"), route_id)
                start = 0
                activity_start = rider.get("activity_start_time")
                if not activity_start and activity_id == str(meta.get("source_activity_id")):
                    activity_start = meta.get("activity_start_time")
                if meta.get("race_start_time") and activity_start:
                    race_start_offset = (pd.Timestamp(meta["race_start_time"])
                                         - pd.Timestamp(activity_start)).total_seconds()
                    start = max(0, race_start_offset)
                else:
                    raise ValueError("Race/activity start time unavailable; re-fetch the race.")
                elapsed = rider.get("elapsed_ms")
                if elapsed not in (None, ""):
                    elapsed = float(elapsed) / 1000
                    if not np.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError("Race result duration must be finite and positive.")
                    end = race_start_offset + elapsed
                else:
                    end = None
                    warnings.append(f"{name}: official finish time unavailable; using the recording end.")
                efforts, incomplete = best_efforts(samples, arches, start, end)
            except (OSError, ValueError, KeyError, TypeError) as error:
                logger.warning("Power Play telemetry unavailable for %s: %s", activity_id, error)
                warnings.append(f"{name}: {error}")
                continue
            if incomplete:
                warnings.append(f"{name}: {incomplete} incomplete effort(s) excluded.")
            if not efforts:
                warnings.append(f"{name}: no complete power-arch efforts found.")
                continue
            weight = rider.get("weight_kg")
            recorded = str(rider.get("weight_is_event_recorded", "")).casefold() == "true"
            weight = float(weight) if recorded and weight not in (None, "") else None
            if weight is None or not np.isfinite(weight) or weight <= 0:
                weight = None
                warnings.append(f"{name}: event weight unavailable; W/kg omitted. Re-fetch to refresh weights.")
            rider_key = rider.get("player_id") or activity_id
            for duration, effort in efforts.items():
                key = (rider_key, duration)
                previous = best_by_rider.get(key)
                if previous is None or effort["avg_power"] > previous["avg_power"]:
                    best_by_rider[key] = {
                        **effort, "name": name, "activity_id": activity_id,
                        "player_id": rider.get("player_id") or None,
                        "category": group.get("label"), "weight_kg": weight,
                        "watts_per_kg": effort["avg_power"] / weight if weight else None,
                    }
    rows = list(best_by_rider.values())
    return {
        "race_id": race_path.name, "race_name": title, "rider_count": rider_count,
        "event_id": str(event_id) if event_id else None,
        "source_activity_id": str(source_activity_id) if source_activity_id else None,
        "durations": list(DURATIONS), "rows": rows, "warnings": warnings,
    }
