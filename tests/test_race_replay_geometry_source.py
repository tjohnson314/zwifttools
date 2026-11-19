from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from race_replay.data_cleaner import (
    CalibratedGameRoute,
    CleanedRaceData,
    RiderData,
    align_riders_to_game_route,
    build_game_elevation_profile,
    build_replay_route_segments,
    load_game_route_profile,
    load_preferred_route_geometry,
)
from tools.extract_zwift_routes import parse_segment_definitions, parse_segment_names


@pytest.mark.parametrize("reverse_name, expected", [
    ("Epic KOM", "Epic KOM Reverse"),
    ("Epic KOM Reverse", "Epic KOM Reverse"),
    ("Reverse Epic KOM", "Reverse Epic KOM"),
])
def test_timing_arch_names_follow_source_direction_not_hash_sign(reverse_name, expected):
    source = f'''<world><entities>
        <ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.5"
             m_startLineSplineTimeF="0.1" m_overrideNetworkHashF="-123"
             m_startLineSplineTimeR="0.9" m_overrideNetworkHashR="456"
             m_ArchFriendlyName="Epic KOM" m_ArchFriendlyNameR="{reverse_name}"
             m_ArchSegmentDistanceF="1" m_ArchSegmentDistanceR="1" />
    </entities></world>'''.encode()

    names = parse_segment_names(source)

    assert names == {-123: "Epic KOM", 456: expected}
    assert {definition["hash"]: definition["name"]
            for definition in parse_segment_definitions(source)} == names


def test_power_to_the_tower_labels_reverse_and_forward_epic_kom_separately():
    from shared.surface_map import _load_route_world, get_route

    route = next(route for route in _load_route_world(1)["routes"] if route["nameHash"] == 2064290391)
    climbs = sorted(
        (segment for segment in route["segments"] if segment["name"].startswith("Epic KOM")),
        key=lambda segment: segment["start_distance_m"],
    )

    assert [segment["name"] for segment in climbs] == ["Epic KOM Reverse", "Epic KOM"]
    assert all(segment["boundary_source"] == "timing_arch" for segment in climbs)
    assert all(0 <= segment["max_distance_m"] - segment["distance_m"] <= 25 for segment in climbs)
    displayed = sorted(
        (segment for segment in get_route(1, route["nameHash"])["segments"]
         if segment["name"].startswith("Epic KOM")),
        key=lambda segment: segment["start_distance_m"],
    )
    assert [segment["name"] for segment in displayed] == ["Epic KOM Reverse", "Epic KOM"]


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


def test_timing_arch_catalog_excludes_internal_props_without_filtering_declared_lengths():
    definitions = parse_segment_definitions(b'''
        <world><entities>
            <ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.9"
                 m_startLineSplineTimeF="0.1" m_overrideNetworkHashF="7"
                 m_Name="Internal prop" m_ArchSegmentDistanceF="1" />
            <ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.9"
                 m_startLineSplineTimeF="0.1" m_overrideNetworkHashF="8"
                 m_ArchFriendlyName="Incomplete definition" />
            <ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.9"
                 m_startLineSplineTimeF="0.1" m_overrideNetworkHashF="9"
                 m_ArchFriendlyName="Sprint" m_ArchSegmentDistanceF="1000" />
        </entities></world>
    ''')

    assert len(definitions) == 1
    assert definitions[0]['hash'] == 9
    assert definitions[0]['distance_m'] == 1000000


def _segment_race(finish_km=27.5, riders=None):
    return CleanedRaceData(
        race_id='segment-test', route_name='Montmartre Mixer',
        finish_line_km=finish_km, riders=riders or [],
        elevation_profile=pd.DataFrame(), min_time=0, max_time=100,
        world='PARIS',
    )


def test_replay_route_segments_preserve_occurrences_within_length_tolerance():
    from shared.surface_map import _load_route_world

    source = next(route for route in _load_route_world(11)['routes'] if route['name'] == 'Montmartre Mixer')
    assert sum(segment['hash'] == 1055881124 for segment in source['native_segments']) == 2
    segments = build_replay_route_segments(_segment_race(), 'PARIS')
    assert len(segments) == 6
    koms = [segment for segment in segments if segment['type'] == 'kom']
    assert [segment['pass'] for segment in koms] == [1]
    assert koms[0]['start_distance_km'] == 18.32678
    assert len(koms[0]['latlng']) > 2
    sprints = [segment for segment in segments if segment['name'] == 'Monceau Sprint']
    assert [segment['pass'] for segment in sprints] == [1, 2]


def test_makuri_40_replay_uses_embedded_sprint_segments():
    from shared.surface_map import _load_route_world

    race = _segment_race(40.256)
    race.route_name = 'Makuri 40'
    race.world = 'MAKURI'
    segments = build_replay_route_segments(race, race.world)
    assert segments == build_replay_route_segments(race, 'MAKURIISLANDS')
    source = next(route for route in _load_route_world(9)['routes'] if route['name'] == race.route_name)
    source_segments = sorted(source['segments'], key=lambda segment: segment['start_distance_m'])
    assert [segment['name'] for segment in source_segments] == [
        'Village Sprint', 'Country Sprint',
        'ALLEY SPRINT', 'CASTLE PARK SPRINT', 'SHISA SPRINT',
    ]
    assert all(segment['boundary_source'] == 'timing_arch' for segment in source['segments'])
    assert all(segment['distance_m'] <= segment['max_distance_m'] for segment in source['segments'])
    expected = [segment for segment in source_segments
                if round(segment['max_distance_m'] - segment['distance_m'], 2) <= 25]
    assert len(expected) == 5
    assert [segment['name'] for segment in segments] == [segment['name'] for segment in expected]
    assert all(segment['type'] == 'sprint' for segment in segments)
    assert all(len(segment['latlng']) > 2 for segment in segments)


def test_replay_route_segments_repeat_laps_and_respect_shortened_races():
    segments = build_replay_route_segments(_segment_race(53), 'PARIS')
    first = segments[0]
    repeated = next(segment for segment in segments if segment['lap'] == 2)
    assert repeated['name'] == first['name']
    assert abs(repeated['start_distance_km'] - first['start_distance_km'] - 25.1062) < 1e-6
    shortened = build_replay_route_segments(_segment_race(3.6), 'PARIS')
    assert len(shortened) == 1
    assert shortened[0]['end_distance_km'] == pytest.approx(first['end_distance_km'])
    shortened = build_replay_route_segments(_segment_race(first['start_distance_km'] + 0.01), 'PARIS')
    assert len(shortened) == 1
    assert shortened[0]['name'] == first['name']
    assert shortened[0]['end_distance_km'] == first['end_distance_km']
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


def test_volcano_leadin_preserves_checkpoint_spacing_and_joins_main_lap():
    profile = load_game_route_profile(686828068, "Volcano Circuit")

    assert profile is not None
    assert profile.leadin_distance[-1] == pytest.approx(profile.leadin_distance_m)
    assert profile.leadin_distance[0] == pytest.approx(211.72)
    assert np.ptp(profile.leadin_distance) == pytest.approx(2546.38)

    route, _, _ = load_preferred_route_geometry(
        686828068, "Volcano Circuit", "volcano-circuit", None,
    )
    elevation = build_game_elevation_profile(
        route, route.leadin_distance_m + route.lap_distance_m,
    )
    np.testing.assert_allclose(
        elevation["distance_km"].to_numpy()[:len(route.leadin_distance) - 1],
        route.leadin_distance[:-1] / 1000.0,
    )
    assert elevation["distance_km"].is_monotonic_increasing


def test_volcano_projection_does_not_add_missing_pen_approach_at_lap_join():
    route, _, _ = load_preferred_route_geometry(
        686828068, "Volcano Circuit", "volcano-circuit", None,
    )
    # Start at the first available checkpoint, not the unrecorded pen approach.
    gps = np.vstack([route.leadin_latlng[:-1], route.route_latlng])
    traveled_m = np.concatenate([
        route.leadin_distance[:-1] - route.leadin_distance[0],
        np.ptp(route.leadin_distance) + route.route_distance,
    ])
    rider = {"data": pd.DataFrame({
        "distance_km": traveled_m / 1000.0,
        "lat": gps[:, 0], "lng": gps[:, 1],
        "speed_kmh": np.full(len(gps), 36.0),
    })}

    aligned, _ = align_riders_to_game_route([rider], route)

    np.testing.assert_allclose(
        aligned[0]["data"]["distance_km"], traveled_m / 1000.0, atol=0.015,
    )


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


@pytest.mark.parametrize("detect_late_joiners", [True, False])
def test_missing_leadin_geometry_does_not_match_a_later_route_pass(detect_late_joiners):
    main_gps = np.column_stack([
        np.zeros(100), 0.005 + np.arange(100) * 0.0001,
    ])
    pen_gps = np.column_stack([
        np.zeros(40), np.arange(40) * 0.0001,
    ])
    route = CalibratedGameRoute(
        route_name="Missing Lead-in",
        leadin_distance_m=500.0,
        lap_distance_m=19390.0,
        leadin_distance=np.array([]),
        leadin_altitude=np.array([]),
        leadin_latlng=np.empty((0, 2)),
        route_distance=np.concatenate([
            np.arange(100) * 10.0, 19000.0 + np.arange(40) * 10.0,
        ]),
        route_altitude=np.zeros(140),
        route_latlng=np.vstack([main_gps, pen_gps]),
        horizontal_p95_m=3.0,
    )
    gps = np.vstack([pen_gps, main_gps])
    distances = np.concatenate([
        np.arange(40) * 0.01, 0.5 + np.arange(100) * 0.01,
    ])
    rider = {
        "activity_start_time": "2026-10-13T04:06:34Z",
        "data": pd.DataFrame({
            "distance_km": distances,
            "lat": gps[:, 0],
            "lng": gps[:, 1],
            "speed_kmh": np.full(140, 36.0),
        }),
    }
    late_rider = {
        "activity_start_time": "2026-10-13T04:11:00Z",
        "data": rider["data"].iloc[:40].copy(),
    }

    aligned, _ = align_riders_to_game_route(
        [rider, late_rider] if detect_late_joiners else [rider],
        route,
        race_start_time="2026-10-13T04:10:00Z",
        detect_late_joiners=detect_late_joiners,
    )

    np.testing.assert_allclose(
        aligned[0]["data"]["distance_km"], distances, atol=0.001,
    )
    if detect_late_joiners:
        np.testing.assert_allclose(
            aligned[1]["data"]["distance_km"],
            19.5 + np.arange(40) * 0.01,
            atol=0.001,
        )