import numpy as np
import pandas as pd
import json

from shared.zrl_leaderboard import (
    _load_calibrated_route_boundaries,
    _match_rider_to_route,
    _route_crossing_time,
    build_leaderboard,
    load_route_segments,
)


def test_segment_hashes_survive_json_as_exact_strings():
    segments = json.loads(json.dumps(load_route_segments()))
    tchou = next(
        segment for segment in segments
        if segment["name"] == "Tchou Tchou Sprint"
    )

    assert tchou["wad_hash"] == "-9223372035804541048"


def test_route_matcher_keeps_early_pass_at_overlapping_geometry():
    first_pass = np.column_stack([
        48.0 + np.arange(30) * 0.0001,
        2.0 + np.arange(30) * 0.0001,
    ])
    later_pass = first_pass + np.array([0.000001, 0.000001])
    route = np.vstack([first_pass, later_pass])
    telemetry = pd.DataFrame({
        "time_sec": np.arange(len(first_pass), dtype=float),
        "lat": later_pass[:, 0],
        "lng": later_pass[:, 1],
    })

    _, progress = _match_rider_to_route(telemetry, route)

    assert progress[0] == 0
    assert progress[-1] == len(first_pass) - 1
    assert np.all(np.diff(progress) >= 0)


def test_finish_offset_shifts_fal_but_not_fts_duration():
    route, _ = _load_calibrated_route_boundaries()
    telemetry = pd.DataFrame({
        "time_sec": np.arange(len(route), dtype=float),
        "lat": route[:, 0],
        "lng": route[:, 1],
    })
    participant = {
        "activity_id": "rider",
        "name": "Test Rider",
        "rank": 1,
        "elapsed_ms": (len(route) - 1 + 10) * 1000,
    }
    matched = _match_rider_to_route(telemetry, route)
    first_segment = load_route_segments()[0]
    start_time, next_index = _route_crossing_time(
        matched,
        first_segment["wad_percent_start"] * (len(route) - 1),
    )
    end_time, _ = _route_crossing_time(
        matched,
        first_segment["wad_percent_end"] * (len(route) - 1),
        next_index,
    )

    result = build_leaderboard([participant], {"rider": telemetry})

    assert result["summary"][0]["clock_offset_sec"] == -10.0
    assert result["fal"][0]["time_ms"] == round((end_time + 10) * 1000)
    assert result["fts"][0]["time_ms"] == round((end_time - start_time) * 1000)