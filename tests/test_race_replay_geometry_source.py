from unittest.mock import Mock, patch

import pandas as pd

from race_replay.data_cleaner import load_preferred_route_geometry


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