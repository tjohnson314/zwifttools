from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from race_replay.data_cleaner import (
    CalibratedGameRoute,
    CleanedRaceData,
    RiderData,
    align_riders_to_game_route,
    build_replay_route_segments,
    load_preferred_route_geometry,
)
from tools.extract_zwift_routes import parse_segment_definitions


def test_timing_arch_without_road_id_uses_road_zero():
    definitions = parse_segment_definitions(b'''
        <world><entities>
            <ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.4924368"
                 m_startLineSplineTimeF="0.483" m_overrideNetworkHashF="13941612717"
                 m_ArchFriendlyName="Country Sprint" m_ArchSegmentDistanceF="0.146304" />
        </entities></world>
    ''')
    assert len(definitions) == 1
    assert definitions[0]['road_id'] == 0
    assert definitions[0]['name'] == 'Country Sprint'
    assert definitions[0]['start_time'] == 0.483
    assert definitions[0]['end_time'] == 0.4924368


def _segment_race(finish_km=27.5, riders=None):
    return CleanedRaceData(
        race_id='segment-test', route_name='Montmartre Mixer',
        finish_line_km=finish_km, riders=riders or [],
        elevation_profile=pd.DataFrame(), min_time=0, max_time=100,
        world='PARIS',
    )


def test_replay_route_segments_preserve_known_occurrences():
    segments = build_replay_route_segments(_segment_race(), 'PARIS')
    assert len(segments) == 7
    koms = [segment for segment in segments if segment['type'] == 'kom']
    assert [segment['pass'] for segment in koms] == [1, 2]
    assert koms[0]['start_distance_km'] == 15.05675
    assert len(koms[0]['latlng']) > 2


def test_makuri_40_replay_uses_embedded_sprint_segments():
    race = _segment_race(40.256)
    race.route_name = 'Makuri 40'
    race.world = 'MAKURI'
    segments = build_replay_route_segments(race, race.world)
    assert segments == build_replay_route_segments(race, 'MAKURIISLANDS')
    assert [segment['name'] for segment in segments] == [
        'Village Sprint', 'Country Sprint',
        'ALLEY SPRINT', 'CASTLE PARK SPRINT', 'SHISA SPRINT',
    ]
    assert all(segment['type'] == 'sprint' for segment in segments)
    assert all(len(segment['latlng']) > 2 for segment in segments)
    assert np.isclose(segments[0]['start_distance_km'], 1.70447)
    assert np.isclose(segments[1]['start_distance_km'], 7.42658)
    assert np.isclose(segments[-1]['end_distance_km'], 27.73835)


def test_replay_route_segments_repeat_laps_and_respect_shortened_races():
    segments = build_replay_route_segments(_segment_race(53), 'PARIS')
    first = segments[0]
    repeated = next(segment for segment in segments if segment['lap'] == 2)
    assert repeated['name'] == first['name']
    assert abs(repeated['start_distance_km'] - first['start_distance_km'] - 25.1062) < 1e-6
    shortened = build_replay_route_segments(_segment_race(3.6), 'PARIS')
    assert len(shortened) == 1
    assert shortened[0]['end_distance_km'] == 3.70982
    assert build_replay_route_segments(_segment_race(), None) == []


def test_replay_segment_boundaries_follow_rider_gps_distance():
    race = _segment_race()
    segment = build_replay_route_segments(race, 'PARIS')[0]
    path = np.asarray(segment['latlng'])
    distances = np.linspace(segment['start_distance_km'], segment['end_distance_km'], len(path))
    race.riders = [RiderData(
        rank=1, activity_id='test', name='Reference', team='', finish_time_sec=100,
        data=pd.DataFrame({
            'distance_km': distances - 0.095,
            'lat': path[:, 0], 'lng': path[:, 1],
        }),
    )]
    aligned = build_replay_route_segments(race, 'PARIS')[0]
    assert abs(aligned['start_distance_km'] - segment['start_distance_km'] + 0.095) < 1e-6
    assert abs(aligned['end_distance_km'] - segment['end_distance_km'] + 0.095) < 1e-6


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