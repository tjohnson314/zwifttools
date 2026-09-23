"""Build a ZRL points-race leaderboard from rider telemetry."""

import json
from pathlib import Path

import numpy as np

from shared.zrl_scoring import score_points_race


SEGMENTS_FILE = Path(__file__).parent.parent / "zrl_route_segments.json"
ROUTE_FILE = Path(__file__).parent.parent / "zwift_routes" / "world_11.json"
CALIBRATION_FILE = Path(__file__).parent.parent / "zwift_surfaces" / "world_gps_calibration.json"


def load_route_segments(route_slug="montmartre-mixer"):
    with SEGMENTS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)[route_slug]["segments"]


def _load_calibrated_route_boundaries():
    """Project persisted Paris WAD segment paths directly into GPS space."""
    if not ROUTE_FILE.exists() or not CALIBRATION_FILE.exists():
        return None
    with ROUTE_FILE.open(encoding="utf-8") as handle:
        routes = json.load(handle).get("routes", [])
    route = next((r for r in routes if r.get("name") == "Montmartre Mixer"), None)
    if not route or not route.get("segment_path"):
        return None
    with CALIBRATION_FILE.open(encoding="utf-8") as handle:
        calibration = json.load(handle).get("11", {}).get("coef")
    if not calibration:
        return None
    coefficient = np.asarray(calibration, dtype=float)
    path = route["segment_path"]
    positions = np.arange(len(path["x"]), dtype=float)
    projected = np.column_stack([path["x"], path["z"], np.ones(len(positions))]) @ coefficient
    return projected, positions


def interpolate_time_at_distance(df, distance_km):
    """Interpolate telemetry time at a distance, returning seconds or None."""
    if df is None or len(df) < 2 or not {"time_sec", "distance_km"}.issubset(df.columns):
        return None
    times = np.asarray(df["time_sec"], dtype=float)
    distances = np.asarray(df["distance_km"], dtype=float)
    valid = np.isfinite(times) & np.isfinite(distances)
    times = times[valid]
    distances = distances[valid]
    if len(times) < 2:
        return None
    order = np.argsort(distances, kind="stable")
    distances = distances[order]
    times = times[order]
    unique_distances, unique_indices = np.unique(distances, return_index=True)
    unique_times = times[unique_indices]
    if distance_km < unique_distances[0] or distance_km > unique_distances[-1]:
        return None
    return float(np.interp(distance_km, unique_distances, unique_times))


def _crossing_time_at_point(df, latitude, longitude, start_index=0, max_distance_m=100):
    """Find the first ordered GPS passage near a segment boundary."""
    if df is None or not {"time_sec", "lat", "lng"}.issubset(df.columns):
        return None, start_index
    lat = np.asarray(df["lat"], dtype=float)
    lng = np.asarray(df["lng"], dtype=float)
    times = np.asarray(df["time_sec"], dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(times)
    if valid.sum() < 2:
        return None, start_index
    lat = lat[valid]
    lng = lng[valid]
    times = times[valid]
    start_index = min(max(start_index, 0), len(times) - 2)
    lat_scale = np.cos(np.radians(latitude))
    target = np.array([latitude, longitude * lat_scale])
    candidates = []
    for index in range(start_index, len(times) - 1):
        first = np.array([lat[index], lng[index] * lat_scale])
        second = np.array([lat[index + 1], lng[index + 1] * lat_scale])
        vector = second - first
        length_sq = float(np.dot(vector, vector))
        fraction = 0.0 if length_sq == 0 else float(np.clip(np.dot(target - first, vector) / length_sq, 0, 1))
        projection = first + fraction * vector
        distance_m = float(np.linalg.norm(projection - target) * 111_320)
        if distance_m <= max_distance_m:
            candidates.append((index, fraction, distance_m))
    if not candidates:
        return None, start_index
    # Keep only the first contiguous near-boundary passage. Within that passage
    # choose the closest projected telemetry segment, which preserves route
    # order while avoiding a nearby whole-second sample masking interpolation.
    passage = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate[0] <= passage[-1][0] + 2:
            passage.append(candidate)
        else:
            break
    index, fraction, _ = min(passage, key=lambda candidate: candidate[2])
    return float(times[index] + fraction * (times[index + 1] - times[index])), index + 1


def _add_reference_gps_boundaries(segments, telemetry_by_activity):
    """Project WAD route percentages onto a rider GPS track.

    WAD segment boundaries are normalized checkpoint positions, while the API
    gives us rider latitude/longitude. A complete reference track supplies the
    spatial coordinate for each WAD position; every rider is then matched
    against that coordinate spatially, never against their own odometer.
    """
    calibrated_route = _load_calibrated_route_boundaries()
    if calibrated_route is not None:
        projected, positions = calibrated_route
        enriched = []
        for segment in segments:
            segment = dict(segment)
            if "wad_percent_start" in segment and "wad_percent_end" in segment:
                start = float(segment["wad_percent_start"]) * positions[-1]
                end = float(segment["wad_percent_end"]) * positions[-1]
                segment.update({
                    "start_lat": float(np.interp(start, positions, projected[:, 0])),
                    "start_lng": float(np.interp(start, positions, projected[:, 1])),
                    "end_lat": float(np.interp(end, positions, projected[:, 0])),
                    "end_lng": float(np.interp(end, positions, projected[:, 1])),
                })
            enriched.append(segment)
        return enriched

    candidates = [
        frame for frame in telemetry_by_activity.values()
        if {"time_sec", "lat", "lng"}.issubset(frame.columns) and len(frame) >= 2
    ]
    if not candidates:
        return segments
    reference = max(candidates, key=len)
    lat = np.asarray(reference["lat"], dtype=float)
    lng = np.asarray(reference["lng"], dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lng)
    lat = lat[valid]
    lng = lng[valid]
    if len(lat) < 2:
        return segments
    positions = np.arange(len(lat), dtype=float)
    enriched = []
    for segment in segments:
        segment = dict(segment)
        if "wad_percent_start" in segment and "wad_percent_end" in segment:
            start = float(segment["wad_percent_start"]) * positions[-1]
            end = float(segment["wad_percent_end"]) * positions[-1]
            segment.update({
                "start_lat": float(np.interp(start, positions, lat)),
                "start_lng": float(np.interp(start, positions, lng)),
                "end_lat": float(np.interp(end, positions, lat)),
                "end_lng": float(np.interp(end, positions, lng)),
            })
        enriched.append(segment)
    return enriched


def _format_seconds(seconds):
    if seconds is None:
        return None
    total_ms = int(round(float(seconds) * 1000))
    minutes, remainder_ms = divmod(total_ms, 60_000)
    return f"{minutes}:{remainder_ms // 1000:02d}.{remainder_ms % 1000:03d}"


def build_leaderboard(participants, telemetry_by_activity, segments=None):
    """Return summary, FAL rows, and FTS rows for one points race.

    Participant metadata must contain ``activity_id``, ``name``, ``rank`` and
    optional ``elapsed_ms``. Telemetry is keyed by activity ID and contains a
    DataFrame with time, latitude, and longitude columns.
    """
    segments = _add_reference_gps_boundaries(
        segments or load_route_segments(), telemetry_by_activity
    )
    spatial_segments = [
        segment for segment in segments
        if all(key in segment for key in ("start_lat", "start_lng", "end_lat", "end_lng"))
    ]
    rider_inputs = []
    fal_rows = []
    fts_rows = []
    for participant in participants:
        activity_id = str(participant["activity_id"])
        df = telemetry_by_activity.get(activity_id)
        elapsed_ms = participant.get("elapsed_ms")
        finish_time = float(elapsed_ms) / 1000 if elapsed_ms is not None else None
        finished = finish_time is not None and participant.get("rank") is not None
        rider_inputs.append({
            "id": activity_id,
            "name": participant.get("name", ""),
            "finished": finished,
            "finish_order": participant.get("rank") if finished else None,
            "disqualified": bool(participant.get("disqualified", False)),
        })

        path_index = 0
        for segment in segments:
            if not all(key in segment for key in ("start_lat", "start_lng", "end_lat", "end_lng")):
                continue
            start_time, path_index = _crossing_time_at_point(
                df, segment["start_lat"], segment["start_lng"], path_index
            )
            end_time, path_index = _crossing_time_at_point(
                df, segment["end_lat"], segment["end_lng"], path_index
            )
            if end_time is None:
                continue
            segment_id = str(segment["wad_hash"])
            occurrence_id = f"{segment_id}:{segment['pass']}"
            fal_rows.append({
                "segment_id": occurrence_id,
                "pass": segment["pass"],
                "rider_id": activity_id,
                "crossing_seconds": end_time,
                "elapsed_seconds": end_time - start_time if start_time is not None else None,
                "include_fts": False,
            })
            segment_duration = end_time - start_time if start_time is not None else None
            if start_time is not None and 0 <= segment_duration <= 300:
                fts_rows.append({
                    "segment_id": segment_id,
                    "pass": segment["pass"],
                    "rider_id": activity_id,
                    "crossing_seconds": end_time,
                    "elapsed_seconds": segment_duration,
                    "include_fal": False,
                })

    scores = {
        score["rider_id"]: score
        for score in score_points_race(rider_inputs, fal_rows + fts_rows)
    }
    participant_by_id = {str(p["activity_id"]): p for p in participants}
    summary = []
    for rider_id, score in scores.items():
        participant = participant_by_id[rider_id]
        summary.append({
            **score,
            "name": participant.get("name", ""),
            "activity_id": rider_id,
            "finish_time": _format_seconds(float(participant["elapsed_ms"]) / 1000)
            if participant.get("elapsed_ms") is not None else None,
        })
    summary.sort(key=lambda row: (-row["total"], -row["fin"], -row["fal"], -row["fts"], row["name"]))

    def detail_rows(source_rows, points_key):
        detail = []
        ranked_points = {}
        grouped = {}
        for row in source_rows:
            grouped.setdefault(row["segment_id"], []).append(row)
        eligible_ids = {
            rider["id"] for rider in rider_inputs
            if rider["finished"] and not rider["disqualified"]
        }
        for segment_id, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: row["crossing_seconds"] if points_key == "fal" else row["elapsed_seconds"])
            for place, row in enumerate(ordered, start=1):
                points = max(len(participants) - place + 1, 0) if points_key == "fal" else 0
                if points_key == "fts" and place <= 10:
                    points = (15, 12, 10, 8, 6, 5, 4, 3, 2, 1)[place - 1]
                if row["rider_id"] not in eligible_ids:
                    points = 0
                ranked_points[(segment_id, row["pass"], row["rider_id"])] = points
        for row in source_rows:
            participant = participant_by_id[row["rider_id"]]
            segment = next(
                segment for segment in segments
                if str(segment["wad_hash"]) == row["segment_id"].split(":")[0]
                and segment["pass"] == row["pass"]
            )
            detail.append({
                "segment_key": f"{segment['wad_hash']}:{segment['pass']}",
                "segment_name": segment["name"],
                "pass": segment["pass"],
                "name": participant.get("name", ""),
                "activity_id": row["rider_id"],
                "time": _format_seconds(row["crossing_seconds"] if points_key == "fal" else row["elapsed_seconds"]),
                "time_ms": int(round((row["crossing_seconds"] if points_key == "fal" else row["elapsed_seconds"]) * 1000)),
                "points": ranked_points[(row["segment_id"], row["pass"], row["rider_id"])],
            })
        detail.sort(key=lambda row: -row["points"])
        return detail

    return {
        "summary": summary,
        "fal": detail_rows(fal_rows, "fal"),
        "fts": detail_rows(fts_rows, "fts"),
        "segments": segments,
        "segment_geometry_available": len(spatial_segments) == len(segments),
    }