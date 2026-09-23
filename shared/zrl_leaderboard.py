"""Build a ZRL points-race leaderboard from rider telemetry."""

import json
from pathlib import Path

import numpy as np

from shared.zrl_scoring import score_points_race


SEGMENTS_FILE = Path(__file__).parent.parent / "zrl_route_segments.json"
ROUTE_FILE = Path(__file__).parent.parent / "zwift_routes" / "world_11.json"
CALIBRATION_FILE = Path(__file__).parent.parent / "zwift_surfaces" / "world_gps_calibration.json"
ROUTE_MATCH_LOOKAHEAD_POINTS = 10


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


def _crossing_time_at_point(
    df, latitude, longitude, start_index=0, max_distance_m=100,
    expected_time_sec=None,
):
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
    if expected_time_sec is not None:
        index, fraction, _ = min(
            candidates,
            key=lambda candidate: abs(
                (times[candidate[0]] + candidate[1] * (times[candidate[0] + 1] - times[candidate[0]]))
                - expected_time_sec
            ),
        )
        return float(times[index] + fraction * (times[index + 1] - times[index])), index + 1

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


def _crossing_time_at_route_position(
    df, route_points, position, expected_time_sec=None, max_lateral_m=100,
):
    """Interpolate a rider's forward crossing of a route-normal line."""
    if df is None or not {"time_sec", "lat", "lng"}.issubset(df.columns):
        return None
    lat = np.asarray(df["lat"], dtype=float)
    lng = np.asarray(df["lng"], dtype=float)
    times = np.asarray(df["time_sec"], dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(times)
    lat = lat[valid]
    lng = lng[valid]
    times = times[valid]
    if len(times) < 2:
        return None

    route = np.asarray(route_points, dtype=float)
    position = float(np.clip(position, 0, len(route) - 1))
    indices = np.arange(len(route), dtype=float)
    target = np.array([
        np.interp(position, indices, route[:, 0]),
        np.interp(position, indices, route[:, 1]),
    ])
    before = route[max(0, int(np.floor(position)) - 1)]
    after = route[min(len(route) - 1, int(np.ceil(position)) + 1)]
    scale = np.cos(np.radians(target[0]))
    target_xy = np.array([target[0], target[1] * scale])
    direction = np.array([
        after[0] - before[0],
        (after[1] - before[1]) * scale,
    ])
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0:
        return None
    direction /= direction_norm

    rider_xy = np.column_stack([lat, lng * scale])
    along = (rider_xy - target_xy) @ direction
    candidates = []
    for index in np.where((along[:-1] <= 0) & (along[1:] >= 0))[0]:
        span = along[index + 1] - along[index]
        if span <= 0:
            continue
        fraction = float(-along[index] / span)
        crossing_xy = rider_xy[index] + fraction * (
            rider_xy[index + 1] - rider_xy[index]
        )
        lateral_m = float(np.linalg.norm(crossing_xy - target_xy) * 111_320)
        if lateral_m <= max_lateral_m:
            crossing_time = float(
                times[index] + fraction * (times[index + 1] - times[index])
            )
            candidates.append((crossing_time, lateral_m))
    if not candidates:
        return None
    if expected_time_sec is not None:
        return min(candidates, key=lambda item: abs(item[0] - expected_time_sec))[0]
    return candidates[0][0]


def _finish_clock_time(df, finish_boundary, expected_time_sec):
    """Estimate telemetry-clock finish time for the per-rider clock offset."""
    if finish_boundary is None:
        return None
    crossing, _ = _crossing_time_at_point(
        df, finish_boundary[0], finish_boundary[1],
        max_distance_m=100, expected_time_sec=expected_time_sec,
    )
    if crossing is not None:
        return crossing
    if df is None or "time_sec" not in df.columns:
        return None
    finite_times = np.asarray(df["time_sec"], dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    if not {"lat", "lng"}.issubset(df.columns):
        return float(finite_times[-1]) if len(finite_times) else None
    lat = np.asarray(df["lat"], dtype=float)
    lng = np.asarray(df["lng"], dtype=float)
    times = np.asarray(df["time_sec"], dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(times)
    if valid.sum() < 1:
        return None
    lat = lat[valid]
    lng = lng[valid]
    times = times[valid]
    scale = np.cos(np.radians(finish_boundary[0]))
    error_m = np.sqrt(
        ((lat - finish_boundary[0]) * 111_320) ** 2
        + ((lng - finish_boundary[1]) * scale * 111_320) ** 2
    )
    near = np.where(np.abs(times - expected_time_sec) <= 30)[0]
    if len(near) == 0:
        near = np.arange(len(times))
    return float(times[near[np.argmin(error_m[near])]])


def _match_rider_to_route(df, route_points):
    """Project rider GPS samples onto the ordered route axis."""
    if df is None or not {"time_sec", "lat", "lng"}.issubset(df.columns):
        return None
    lat = np.asarray(df["lat"], dtype=float)
    lng = np.asarray(df["lng"], dtype=float)
    times = np.asarray(df["time_sec"], dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(times)
    if valid.sum() < 2:
        return None
    lat = lat[valid]
    lng = lng[valid]
    times = times[valid]
    route = np.asarray(route_points, dtype=float)
    scale = np.cos(np.radians(np.mean(route[:, 0])))
    rider_xy = np.column_stack([lat, lng * scale])
    route_xy = np.column_stack([route[:, 0], route[:, 1] * scale])
    progress = np.empty(len(rider_xy), dtype=float)
    first_stop = min(len(route), ROUTE_MATCH_LOOKAHEAD_POINTS + 1)
    first_squared = ((route_xy[:first_stop] - rider_xy[0]) ** 2).sum(axis=1)
    current = int(np.argmin(first_squared))
    progress[0] = current
    for sample_index in range(1, len(rider_xy)):
        stop = min(len(route), current + ROUTE_MATCH_LOOKAHEAD_POINTS + 1)
        candidate_squared = (
            (route_xy[current:stop] - rider_xy[sample_index]) ** 2
        ).sum(axis=1)
        current += int(np.argmin(candidate_squared))
        progress[sample_index] = current
    return times, progress


def _route_crossing_time(matched, target, start_index=0, expected_time_sec=None):
    """Binary-search route progress and interpolate the crossing timestamp."""
    if matched is None:
        return None, start_index
    times, progress = matched
    search_start = min(max(start_index, 0), len(progress) - 1)
    index = int(np.searchsorted(progress[search_start:], target, side="left")) + search_start
    if index <= search_start or index >= len(progress):
        return None, start_index
    previous = index - 1
    span = progress[index] - progress[previous]
    fraction = 0.0 if span <= 0 else (target - progress[previous]) / span
    crossing = times[previous] + fraction * (times[index] - times[previous])
    if expected_time_sec is not None and abs(crossing - expected_time_sec) > 30:
        return None, start_index
    return float(crossing), index + 1


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
    calibrated_route = _load_calibrated_route_boundaries()
    segments = _add_reference_gps_boundaries(
        segments or load_route_segments(), telemetry_by_activity
    )
    route_points = None if calibrated_route is None else calibrated_route[0]
    spatial_segments = [
        segment for segment in segments
        if all(key in segment for key in ("start_lat", "start_lng", "end_lat", "end_lng"))
    ]
    rider_inputs = []
    fal_rows = []
    fts_rows = []
    clock_offsets = {}
    route_matched_riders = 0
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

        matched_route = _match_rider_to_route(df, route_points) if route_points is not None else None
        telemetry_finish = None
        if finish_time is not None and matched_route is not None:
            telemetry_finish, _ = _route_crossing_time(
                matched_route, len(route_points) - 1
            )
            if telemetry_finish is None:
                telemetry_finish, _ = _route_crossing_time(
                    matched_route, max(0, len(route_points) - 2)
                )
            refined_finish = _crossing_time_at_route_position(
                df,
                route_points,
                len(route_points) - 1,
                expected_time_sec=telemetry_finish,
            )
            if refined_finish is None:
                refined_finish = _finish_clock_time(
                    df, route_points[-1], telemetry_finish
                )
            if refined_finish is not None:
                telemetry_finish = refined_finish
        clock_offset = (
            telemetry_finish - finish_time
            if telemetry_finish is not None and finish_time is not None
            else 0.0
        )
        clock_offsets[activity_id] = round(float(clock_offset), 3)

        if matched_route is not None:
            route_matched_riders += 1

        path_index = 0
        for segment in segments:
            start_time = None
            end_time = None
            if matched_route is not None and "wad_percent_start" in segment and "wad_percent_end" in segment:
                start_position = (
                    float(segment["wad_percent_start"]) * (len(route_points) - 1)
                )
                end_position = (
                    float(segment["wad_percent_end"]) * (len(route_points) - 1)
                )
                start_time, path_index = _route_crossing_time(
                    matched_route,
                    start_position,
                    path_index,
                )
                end_time, path_index = _route_crossing_time(
                    matched_route,
                    end_position,
                    path_index,
                )
                refined_start = _crossing_time_at_route_position(
                    df, route_points, start_position, expected_time_sec=start_time
                )
                if start_time is not None and all(
                    key in segment for key in ("start_lat", "start_lng")
                ) and refined_start is None:
                    refined_start, _ = _crossing_time_at_point(
                        df,
                        segment["start_lat"],
                        segment["start_lng"],
                        max_distance_m=100,
                        expected_time_sec=start_time,
                    )
                if refined_start is not None:
                    start_time = refined_start
                refined_end = _crossing_time_at_route_position(
                    df, route_points, end_position, expected_time_sec=end_time
                )
                if end_time is not None and all(
                    key in segment for key in ("end_lat", "end_lng")
                ) and refined_end is None:
                    refined_end, _ = _crossing_time_at_point(
                        df,
                        segment["end_lat"],
                        segment["end_lng"],
                        max_distance_m=100,
                        expected_time_sec=end_time,
                    )
                if refined_end is not None:
                    end_time = refined_end
            if end_time is None:
                continue
            segment_id = str(segment["wad_hash"])
            occurrence_id = f"{segment_id}:{segment['pass']}"
            fal_rows.append({
                "segment_id": occurrence_id,
                "pass": segment["pass"],
                "rider_id": activity_id,
                "crossing_seconds": end_time - clock_offset,
                "elapsed_seconds": end_time - start_time if start_time is not None else None,
                "include_fts": False,
            })
            segment_duration = end_time - start_time if start_time is not None else None
            max_segment_duration = 90 if segment.get("type") == "sprint" else 300
            if start_time is not None and 0 <= segment_duration <= max_segment_duration:
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
            "telemetry_available": rider_id in telemetry_by_activity,
            "clock_offset_sec": clock_offsets.get(rider_id, 0.0),
            "finish_time_ms": int(round(float(participant["elapsed_ms"])))
            if participant.get("elapsed_ms") is not None else None,
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
                "segment_key": (
                    str(segment["wad_hash"])
                    if points_key == "fts"
                    else f"{segment['wad_hash']}:{segment['pass']}"
                ),
                "segment_name": segment["name"],
                "pass": segment["pass"],
                "name": participant.get("name", ""),
                "activity_id": row["rider_id"],
                "time": _format_seconds(row["crossing_seconds"] if points_key == "fal" else row["elapsed_seconds"]),
                "time_ms": int(round((row["crossing_seconds"] if points_key == "fal" else row["elapsed_seconds"]) * 1000)),
                "points": ranked_points[(row["segment_id"], row["pass"], row["rider_id"])],
            })
        detail.sort(key=lambda row: row["time_ms"])
        return detail

    return {
        "summary": summary,
        "fal": detail_rows(fal_rows, "fal"),
        "fts": detail_rows(fts_rows, "fts"),
        "segments": segments,
        "segment_geometry_available": len(spatial_segments) == len(segments),
        "route_matched_riders": route_matched_riders,
        "segment_rows": {"fal": len(fal_rows), "fts": len(fts_rows)},
    }