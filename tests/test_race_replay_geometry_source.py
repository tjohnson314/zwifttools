from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from race_replay.data_cleaner import (
    CalibratedGameRoute,
    align_riders_to_game_route,
    load_preferred_route_geometry,
)


def test_montmartre_prefers_zwift_wad_without_loading_strava():
    reference = pd.DataFrame({
        "distance_km": [0.0, 1.0],
        "altitude_m": [35.0, 40.0],
    })

    with patch(
        "race_replay.data_cleaner.load_route_data",
        side_effect=AssertionError("Strava fallback must not run"),
    ):
        calibrated, route_data, source = load_preferred_route_geometry(
            1247427185,
            "Montmartre Mixer",
            "montmartre-mixer",
            reference,
        )

    assert source == "zwift_wad"
    assert route_data is None
    assert calibrated.horizontal_p95_m == 3.1
    assert len(calibrated.leadin_latlng) == 259
    assert len(calibrated.route_latlng) == 600


def test_strava_is_used_only_when_zwift_geometry_is_unavailable():
    strava_route = Mock()
    with (
        patch(
            "race_replay.data_cleaner.load_game_route_profile",
            return_value=None,
        ),
        patch(
            "race_replay.data_cleaner.load_route_data",
            return_value=strava_route,
        ) as load_strava,
    ):
        calibrated, route_data, source = load_preferred_route_geometry(
            999,
            "Missing Route",
            "missing-route",
            None,
        )

    load_strava.assert_called_once_with("missing-route")
    assert calibrated is None
    assert route_data is strava_route
    assert source == "zwiftmap_strava"


def test_on_time_starters_use_first_pass_of_overlapping_route():
    early_gps = np.column_stack([
        np.zeros(40), np.arange(40) * 0.0001,
    ])
    repeated_gps = early_gps + [0.000001, 0.0]
    route = CalibratedGameRoute(
        route_name="Overlapping Route",
        leadin_distance_m=0.0,
        lap_distance_m=7390.0,
        leadin_distance=np.array([]),
        leadin_altitude=np.array([]),
        leadin_latlng=np.empty((0, 2)),
        route_distance=np.concatenate([np.arange(40) * 10.0,
                                       7000.0 + np.arange(40) * 10.0]),
        route_altitude=np.zeros(80),
        route_latlng=np.vstack([early_gps, repeated_gps]),
        horizontal_p95_m=3.0,
    )
    riders = []
    for gps in (early_gps, repeated_gps):
        riders.append({
            "activity_start_time": "2026-10-02T20:59:00Z",
            "data": pd.DataFrame({
                "distance_km": np.arange(40) * 0.01,
                "lat": gps[:, 0],
                "lng": gps[:, 1],
                "speed_kmh": np.full(40, 36.0),
            }),
        })
    riders.append({
        "activity_start_time": "2026-10-02T21:01:00Z",
        "data": riders[1]["data"].copy(),
    })

    aligned, _ = align_riders_to_game_route(
        riders, route, race_start_time="2026-10-02T21:00:00Z",
    )

    for rider in aligned[:2]:
        np.testing.assert_allclose(
            rider["data"]["distance_km"], np.arange(40) * 0.01,
            atol=0.001,
        )
    np.testing.assert_allclose(
        aligned[2]["data"]["distance_km"], 7.0 + np.arange(40) * 0.01,
        atol=0.001,
    )