import json
import math
from bisect import bisect_right

import pytest

from tools import extract_zwift_routes
from tools.extract_zwift_routes import build_leadin_view, parse_paddock_roads, parse_route


def test_longest_paddock_road_uses_sampled_spline_length_and_deterministic_ties():
    from tools.analyze_paddock_groups import make_groups

    source = b'''<road><id>8</id><roadIsPaddock>true</roadIsPaddock>
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}" m_tangentOut="{0,0,10000}" />
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{10000,0,0}" m_tangentIn="{0,0,10000}" />
        </road><road><id>9</id><roadIsPaddock>true</roadIsPaddock>
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}" m_bStraight="1" />
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{15000,0,0}" />
        </road>'''
    roads = parse_paddock_roads(source)
    group = make_groups(1, roads, road_data=source)[0]
    assert group["representative_road_id"] == 8
    assert group["representative_road_length_m"] > 150
    tie = make_groups(1, [{**road, "road_id": road_id} for road_id, road in [(10, roads[1]), (9, roads[1])]])[0]
    assert tie["representative_road_id"] == 9


def test_route_start_modes_preserve_spawn_ranges_and_missing_distances():
    source = b'''<route name="Test" leadinDistanceInMeters="100"
        freeRideLeadinDistanceInMeters="0" meetupLeadinDistanceInMeters="25" />
        <spawn_area road="4" forward="0" starttime="0.9" endtime="0.8" />
        <private_spawn_area road="4" forward="0" starttime="0.7" endtime="0.6" />
        <highrescheckpoint><entry road="4" dir="0" time="0.5" x="0" y="0" z="0" />
        <entry road="4" dir="0" time="0.4" x="10000" y="0" z="0" /></highrescheckpoint>'''
    route = parse_route(source, 1, 10)
    assert route["start_locations"]["free_ride"] == {
        "spawn_area": {"road": "4", "forward": "0", "starttime": "0.9", "endtime": "0.8"},
        "leadin_distance_m": 0.0, "leadin_ascent_m": None,
    }
    assert route["start_locations"]["meetup"]["leadin_distance_m"] == 25
    assert route["leadin_distance_m"] == 100
    assert route["route"]["d"][-1] == 100


@pytest.mark.parametrize("direction,times", [("1", [0, 0.5, 1, 1.1]), ("0", [1, 0.5, 0, -0.1])])
def test_free_ride_approach_uses_native_spawn_midpoint_without_rescaling(direction, times):
    source = f'''<route name="Test" leadinDistanceInMeters="100" freeRideLeadinDistanceInMeters="70" />
        <spawn_area road="4" forward="{direction}" starttime="0.4" endtime="0.6" />
        <leadinhighrescheckpoint>
        <entry road="4" dir="{direction}" time="{times[0]}" x="0" y="0" z="0" />
        <entry road="4" dir="{direction}" time="{times[1]}" x="5000" y="0" z="0" />
        <entry road="4" dir="{direction}" time="{times[2]}" x="10000" y="0" z="0" />
        </leadinhighrescheckpoint><highrescheckpoint>
        <entry road="4" dir="{direction}" time="{times[2]}" x="10000" y="0" z="0" />
        <entry road="4" dir="{direction}" time="{times[3]}" x="11000" y="0" z="0" />
        </highrescheckpoint>'''.encode()
    route = parse_route(source, 1, 10)
    version = route["route_versions"]["free_ride"]
    assert version["leadin_distance_m"] == 70
    assert version["leadin"]["d"][-1] == 50
    assert version["leadin"]["x"][0] == 50
    assert version["geometry_status"] == "native"
    assert route["leadin"]["d"][-1] == 100


def test_free_ride_approach_uses_reference_path_when_own_leadin_is_absent():
    from tools.extract_zwift_routes import native_start_paths

    source = b'''<route name="Test" freeRideLeadinDistanceInMeters="50" />
        <spawn_area road="4" forward="1" starttime="0.4" endtime="0.6" />
        <highrescheckpoint><entry road="4" dir="1" time="1" x="10000" y="0" z="0" />
        <entry road="4" dir="1" time="1.1" x="11000" y="0" z="0" /></highrescheckpoint>'''
    reference = extract_zwift_routes.load_multiroot(b'''<highrescheckpoint>
        <entry road="4" dir="1" time="0" x="0" y="0" z="0" />
        <entry road="4" dir="1" time="1" x="10000" y="0" z="0" /></highrescheckpoint>''')
    missing = parse_route(source, 1, 10)
    assert missing["route_versions"]["free_ride"]["geometry_status"] == "missing_leadin"
    route = parse_route(source, 1, 10, reference_paths=native_start_paths(reference, "Reference"))
    version = route["route_versions"]["free_ride"]
    assert version["leadin"]["d"][-1] == 50
    assert version["source_route"] == "Reference"


@pytest.mark.parametrize("direction, start_time, end_time, coordinates, start_distance, end_distance", [
    (1, 0.25, 0.75, [0, 100, 400], 50, 250),
    (0, 0.75, 0.25, [400, 100, 0], 150, 350),
])
def test_timing_arch_crossings_override_native_percentages_without_length_guessing(
    monkeypatch, direction, start_time, end_time, coordinates, start_distance, end_distance):
    times = [0, 0.5, 1] if direction else [1, 0.5, 0]
    checkpoints = ''.join(
        f'<entry road="4" time="{time}" dir="{direction}" x="{coordinate * 100}" y="0" z="0" />'
        for time, coordinate in zip(times, coordinates))
    source = f'''<route name="Test" distanceInMeters="400" />
        <highrescheckpoint>{checkpoints}</highrescheckpoint>
        <segment_data><entry hash="7" percentStart="0" percentEnd="0.25" /></segment_data>'''.encode()
    definition = {"hash": 7, "name": "Sprint", "road_id": 4, "start_time": start_time,
                  "end_time": end_time, "distance_m": 1000}

    route = parse_route(source, 9, 2, segment_definitions=[definition])

    assert len(route["segments"]) == 1
    segment = route["segments"][0]
    assert segment["boundary_source"] == "timing_arch"
    assert segment["start_distance_m"] == start_distance
    assert segment["end_distance_m"] == end_distance
    assert segment["start_checkpoint"] == pytest.approx(0.5)
    assert segment["end_checkpoint"] == pytest.approx(1.5)
    assert start_distance in route["route"]["d"]
    assert end_distance in route["route"]["d"]
    assert 0.5 in route["segment_path"]["d"]
    assert 1.5 in route["segment_path"]["d"]
    assert route["native_segments"] == [{"hash": 7, "percent_start": 0, "percent_end": 0.25}]

    from shared import surface_map

    monkeypatch.setattr(surface_map, "_projection", lambda _: {"project": lambda xs, zs: (xs, zs)})
    monkeypatch.setattr(surface_map, "_load_segment_names", lambda: {})
    monkeypatch.setattr(surface_map, "local_to_latlng", lambda *args: [])
    segment["percent_start"], segment["percent_end"] = 0, 1
    marker = surface_map._route_segment_markers(9, route, None, route["route"])[0]
    assert marker["path"][0]["x"] == (50 if direction else 250)
    assert marker["path"][-1]["x"] == (250 if direction else 50)


@pytest.mark.parametrize("times, roads, checkpoint_direction, start_time, end_time, expected", [
    ([0, 1], [4, 4], 1, 0.25, 0.75, 1),
    ([0, 0.25, 0.75, 1], [4] * 4, 1, 0.25, 0.75, 1),
    ([0, 0.5, 1, 0, 0.5, 1], [4] * 6, 1, 0.25, 0.75, 2),
    ([0.5, 0.75, 1], [4] * 3, 1, 0.25, 0.75, 0),
    ([0, 0.25, 0.5], [4] * 3, 1, 0.25, 0.75, 0),
    ([0, 0.5, 1], [4, 9, 4], 1, 0.25, 0.75, 0),
    ([0, 0.5, 1], [4] * 3, 0, 0.25, 0.75, 0),
    ([0.7, 0.9, 1, 0, 0.1, 0.3], [4] * 6, 1, 0.8, 0.2, 1),
    ([0.7, 0.9, 1, 0], [4] * 4, 1, 0.8, 0, 1),
    ([0.9, 0.1], [4, 4], 1, 0.5, 0.75, 0),
])
def test_timing_arch_passages_require_both_crossings_and_do_not_interpolate_road_wraps(
        times, roads, checkpoint_direction, start_time, end_time, expected):
    entries = ''.join(
        f'<entry road="{road}" time="{time}" dir="{checkpoint_direction}" x="{index * 10000}" y="0" z="0" />'
        for index, (road, time) in enumerate(zip(roads, times)))
    source = f'<route name="Test" /><highrescheckpoint>{entries}</highrescheckpoint>'.encode()
    definitions = [{"hash": 7, "name": "Sprint", "road_id": 4, "direction": "F",
                    "start_time": start_time, "end_time": end_time, "distance_m": None}]

    route = parse_route(source, 9, 2, segment_definitions=definitions)

    assert len(route["segments"]) == expected
    assert all(segment["end_distance_m"] > segment["start_distance_m"] for segment in route["segments"])


@pytest.mark.parametrize("direction,times,start_time,end_time,begin_timer", [
    ("F", [0.4, 0.6, 0.9, 1, 0, 0.25, 0.4, 0.6], 0.500001, 0.5, 0.25),
    ("R", [0.6, 0.4, 0.1, 0, 1, 0.75, 0.6, 0.4], 0.500001, 0.5, 0.75),
    ("F", [0.4, 0.5, 0.5, 0.9, 1, 0, 0.25, 0.5], 0.5, 0.5, 0.25),
    ("F", [0.5, 0.6, 0.9, 1, 0, 0.25, 0.4, 0.5], 0.499999, 0.5, 0.25),
])
def test_close_timing_boundaries_require_next_lap_arming_crossing(
        direction, times, start_time, end_time, begin_timer):
    checkpoints = ''.join(
        f'<entry road="4" time="{time}" dir="{1 if direction == "F" else 0}" '
        f'x="{index * 10000}" y="0" z="0" />'
        for index, time in enumerate(times))
    source = f'<route name="Lap" /><highrescheckpoint>{checkpoints}</highrescheckpoint>'.encode()
    definition = {"hash": 7, "name": "Lap", "road_id": 4, "direction": direction,
                  "start_time": start_time, "end_time": end_time, "distance_m": 1000,
                  "is_loop": True, "begin_timer_time": begin_timer}

    route = parse_route(source, 1, 2, segment_definitions=[definition])

    assert len(route["segments"]) == 1
    passage = route["segments"][0]
    assert passage["start_checkpoint"] < 2
    assert passage["end_checkpoint"] > 6
    assert passage["end_distance_m"] - passage["start_distance_m"] > 500
    assert passage["start_time"] == start_time
    assert passage["end_time"] == end_time
    assert passage["declared_distance_m"] == 1000


@pytest.mark.parametrize("attributes, expected", [
    ('m_telemetryArchType="4"', True),
    ('m_bStopwatchRequiresAllCheckpoints="1"', True),
    ('m_archGdePath="data/ZwiftLapArch.gde"', True),
    ('m_telemetryArchType="1"', False),
])
def test_nearby_timing_fields_require_source_lap_classification(attributes, expected):
    source = f'''<ent type="ENTITY_TYPE_TIMINGARCH" m_roadTime="0.5"
        m_startLineSplineTimeF="0.499999" m_overrideNetworkHashF="7"
        m_ArchFriendlyName="Marker" m_ArchSegmentDistanceF="0.00999"
        m_beginTimerSplineTimeF="0.25" {attributes} />'''.encode()
    definition = extract_zwift_routes.parse_segment_definitions(source)[0]

    assert definition["is_loop"] is expected
    assert definition["start_time"] == 0.499999
    assert definition["end_time"] == 0.5
    assert definition["distance_m"] == pytest.approx(9.99)


def test_lap_traversal_does_not_bridge_a_different_road():
    checkpoints = ''.join(
        f'<entry road="{road}" time="{time}" dir="1" x="{index * 10000}" y="0" z="0" />'
        for index, (road, time) in enumerate([(4, 0.4), (4, 0.6), (9, 0.8), (4, 0.25), (4, 0.6)]))
    source = f'<route name="Detour" /><highrescheckpoint>{checkpoints}</highrescheckpoint>'.encode()
    definition = {"hash": 7, "name": "Lap", "road_id": 4, "direction": "F",
                  "start_time": 0.5, "end_time": 0.5, "distance_m": 1000,
                  "is_loop": True, "begin_timer_time": 0.25}

    assert parse_route(source, 1, 2, segment_definitions=[definition])["segments"] == []


@pytest.mark.parametrize("direction,times", [
    ("F", [0, 0.5, 1, 0]),
    ("R", [1, 0.5, 0, 1]),
])
def test_lap_wrap_without_timer_cannot_finish_before_arming(direction, times):
    checkpoints = ''.join(
        f'<entry road="4" time="{time}" dir="{1 if direction == "F" else 0}" '
        f'x="{index * 10000}" y="0" z="0" />'
        for index, time in enumerate(times))
    source = f'<route name="Lap" /><highrescheckpoint>{checkpoints}</highrescheckpoint>'.encode()
    definition = {"hash": 7, "name": "Lap", "road_id": 4, "direction": direction,
                  "start_time": times[0], "end_time": times[-2], "distance_m": 1000,
                  "is_loop": True, "begin_timer_time": None}

    route = parse_route(source, 1, 2, segment_definitions=[definition])

    assert len(route["segments"]) == 1
    passage = route["segments"][0]
    assert passage["start_checkpoint"] == 0
    assert passage["lap_armed_checkpoint"] == passage["end_checkpoint"] == 3


@pytest.mark.parametrize("direction,times,begin_timer", [
    ("F", [0.49, 0.51, 0.9, 1, 0, 0.25, 0.49], 0.25),
    ("R", [0.51, 0.49, 0.1, 0, 1, 0.75, 0.51], 0.75),
])
def test_closed_native_lap_preserves_geometry_when_arch_is_inside_first_edge(
    monkeypatch, direction, times, begin_timer):
    coordinates = [(0, 0), (100, 0), (100, 100), (0, 100), (-100, 100), (-100, 0), (0, 0)]
    checkpoints = ''.join(
        f'<entry road="4" time="{time}" dir="{1 if direction == "F" else 0}" '
        f'x="{point[0] * 100}" y="0" z="{point[1] * 100}" />'
        for time, point in zip(times, coordinates))
    source = f'<route name="Lap" supportedLaps="1" /><highrescheckpoint>{checkpoints}</highrescheckpoint>'.encode()
    definition = {"hash": 7, "name": "Lap", "road_id": 4, "direction": direction,
                  "start_time": 0.5, "end_time": 0.5, "distance_m": 1000,
                  "is_loop": True, "begin_timer_time": begin_timer}
    original = parse_route(source, 1, 100)

    route = parse_route(source, 1, 100, segment_definitions=[definition])

    assert len(route["segments"]) == 1
    passage = route["segments"][0]
    assert passage["boundary_source"] == "closed_route_lap"
    assert passage["start_checkpoint"] == 0
    assert passage["end_checkpoint"] == 6
    assert passage["timing_arch_checkpoint"] == pytest.approx(0.5)
    assert passage["timing_arch_distance_m"] == 50
    assert passage["end_distance_m"] - passage["start_distance_m"] == 600
    assert passage["declared_distance_m"] == 1000
    assert route["route"] == original["route"]

    from shared import surface_map

    monkeypatch.setattr(surface_map, "_projection", lambda _: {"project": lambda xs, zs: (xs, zs)})
    monkeypatch.setattr(surface_map, "_load_segment_names", lambda: {})
    monkeypatch.setattr(surface_map, "local_to_latlng", lambda *args: [])
    marker = surface_map._route_segment_markers(1, route, None, route["route"])[0]
    assert marker["path"][0] == marker["path"][-1] == {"x": 50, "y": 0}
    assert marker["start_distance_m"] == 0
    assert marker["end_distance_m"] == 600


@pytest.mark.parametrize("change", [
    "not_lap_enabled", "open_path", "other_road", "wrong_direction",
    "missing_arm", "not_lap_arch", "arch_after_first_edge",
])
def test_closed_lap_fallback_rejects_incomplete_or_unrelated_paths(change):
    times = [0.49, 0.51, 0.9, 1, 0, 0.25, 0.49]
    if change == "missing_arm":
        times = [0.49, 0.51, 0.9, 1, 0.3, 0.4, 0.49]
    if change == "arch_after_first_edge":
        times = [0.4, 0.49, 0.9, 1, 0, 0.25, 0.4]
    checkpoints = ''.join(
        f'<entry road="{9 if change == "other_road" and index == 3 else 4}" '
        f'time="{time}" dir="{0 if change == "wrong_direction" else 1}" '
        f'x="{(0 if index == 6 and change != "open_path" else index) * 10000}" y="0" z="0" />'
        for index, time in enumerate(times))
    source = (f'<route name="Lap" supportedLaps="{0 if change == "not_lap_enabled" else 1}" />'
              f'<highrescheckpoint>{checkpoints}</highrescheckpoint>').encode()
    definition = {"hash": 7, "name": "Lap", "road_id": 4, "direction": "F",
                  "start_time": 0.5, "end_time": 0.5, "distance_m": 1000,
                  "is_loop": change != "not_lap_arch", "begin_timer_time": 0.25}

    assert parse_route(source, 1, 100, segment_definitions=[definition])["segments"] == []


def test_cached_laps_require_a_complete_armed_passage():
    from pathlib import Path

    from shared.surface_map import _load_route_world, get_route

    routes = {route["name"]: route for route in _load_route_world(1)["routes"]}
    hilly_reverse_hash = -9223372035789709482
    volcano = routes["Volcano Circuit Run"]
    assert hilly_reverse_hash not in {segment["hash"] for segment in volcano["segments"]}
    assert hilly_reverse_hash not in {
        segment["hash"] for segment in get_route(1, volcano["nameHash"])["segments"]}
    for name in ("Hilly Route Reverse", "Hilly Route Reverse Run"):
        passage = next(segment for segment in routes[name]["segments"]
                       if segment["hash"] == hilly_reverse_hash)
        assert passage["distance_m"] == pytest.approx(9165.8, abs=0.1)
        assert passage["declared_distance_m"] == pytest.approx(9010.0002289)

    for cycling, running, segment_hash, distance in [
        ("Volcano Circuit", "Volcano Circuit Run", 18241635062, 4095.31),
        ("Volcano Circuit CCW", "Volcano Circuit CCW Run", -9223372018613140746, 4095.29),
    ]:
        route = routes[cycling]
        passage = next(segment for segment in route["segments"] if segment["hash"] == segment_hash)
        assert passage["boundary_source"] == "closed_route_lap"
        assert passage["distance_m"] == distance
        assert passage["start_checkpoint"] == route["leadin_checkpoint_count"]
        assert passage["end_checkpoint"] == route["leadin_checkpoint_count"] + route["route_checkpoint_count"] - 1
        cycle_marker = next(segment for segment in get_route(1, route["nameHash"])["segments"]
                            if segment["hash"] == segment_hash)
        run_marker = next(segment for segment in get_route(1, routes[running]["nameHash"])["segments"]
                          if segment["hash"] == segment_hash)
        assert cycle_marker["path"][0] == cycle_marker["path"][-1] == run_marker["path"][0] == run_marker["path"][-1]

    for path in (Path(__file__).resolve().parents[1] / "zwift_routes").glob("world_[0-9]*.json"):
        for route in json.loads(path.read_text(encoding="utf-8"))["routes"]:
            for passage in route.get("segments", []):
                if not passage.get("is_loop"):
                    continue
                assert passage["start_checkpoint"] < passage["lap_armed_checkpoint"] <= passage["end_checkpoint"]
                assert passage["effective_start_time"] in (passage["start_time"], passage["end_time"])
                assert passage["effective_end_time"] in (passage["start_time"], passage["end_time"])
                assert passage["distance_m"] == round(
                    passage["end_distance_m"] - passage["start_distance_m"], 2)


def test_cached_alley_passages_use_timing_crossings_and_remain_visible():
    from shared.surface_map import _load_route_world, _route_path_slice, get_route

    routes = {route["name"]: route for route in _load_route_world(9)["routes"]}
    cases = [
        ("Neon Flats", 44010445334, 386.50),
        ("Neokyo All-Nighter", 44010445334, 386.49),
        ("Electric Loop", -9223371992844330474, 414.56),
    ]
    endpoints = {}
    for name, segment_hash, length in cases:
        source = routes[name]
        passage = next(segment for segment in source["segments"] if segment["hash"] == segment_hash)
        assert passage["boundary_source"] == "timing_arch"
        assert passage["distance_m"] == pytest.approx(length, abs=0.03)
        assert passage["declared_distance_m"] == pytest.approx(482.8020036)
        assert segment_hash in {segment["hash"] for segment in get_route(9, source["nameHash"])["segments"]}
        path = _route_path_slice(source, None, source["route"], passage["percent_start"], passage["percent_end"])
        endpoints[name] = (path[0], path[-1])
    for boundary in (0, 1):
        first = endpoints["Neon Flats"][boundary]
        second = endpoints["Neokyo All-Nighter"][boundary]
        assert math.dist(first, second) < 0.5
    assert math.dist(endpoints["Neon Flats"][1], endpoints["Electric Loop"][1]) < 0.5


def test_segment_distances_record_global_hash_maxima_without_replacing_raw_data(tmp_path):
    from copy import deepcopy

    segment_hash = 9223372036854775807
    payloads = {
        "world_1.json": {"routes": [{"segments": [
            {"hash": segment_hash, "name": "Climb", "percent_start": 0.1,
             "percent_end": 0.9, "start_distance_m": 100, "end_distance_m": 500},
            {"hash": -segment_hash, "name": "Climb", "start_distance_m": 10,
             "end_distance_m": 310},
        ]}]},
        "world_10.json": {"routes": [{"segments": [
            {"hash": segment_hash, "name": "Other label", "start_distance_m": 20.01,
             "end_distance_m": 520.02},
            {"hash": segment_hash, "name": "Climb", "start_distance_m": 1000,
             "end_distance_m": 1500.01},
        ]}]},
        "world_altitude_scale.json": {"1": 0.5},
        "index.json": [{"name": "Route"}],
    }
    before = deepcopy(payloads)
    for filename, payload in payloads.items():
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    extract_zwift_routes.update_segment_distances(tmp_path)

    for filename, original in before.items():
        updated = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        if filename in ("index.json", "world_altitude_scale.json"):
            assert updated == original
            continue
        for route in updated["routes"]:
            for segment in route["segments"]:
                assert segment.pop("distance_m") == round(
                    segment["end_distance_m"] - segment["start_distance_m"], 2)
                assert segment.pop("max_distance_m") == (
                    500.01 if segment["hash"] == segment_hash else 300)
        assert updated == original

    bytes_before = {filename: (tmp_path / filename).read_bytes() for filename in before}
    extract_zwift_routes.update_segment_distances(tmp_path)
    assert {filename: (tmp_path / filename).read_bytes() for filename in before} == bytes_before


@pytest.mark.parametrize("distance, expected_passes", [
    (500, 3), (499.99, 3), (475.01, 3), (475, 3), (474.99, 2),
])
def test_segment_markers_allow_25_metre_shortfalls_without_mutating_source(monkeypatch, distance, expected_passes):
    from copy import deepcopy
    from shared import surface_map

    route = {"segment_path": {"d": [0, 1], "x": [0, 1000], "z": [0, 0]},
             "segments": [
                 {"hash": 7, "name": "Climb", "percent_start": 0, "percent_end": distance / 1000,
                  "start_distance_m": 0, "end_distance_m": distance,
                  "distance_m": distance, "max_distance_m": 500},
                 {"hash": 7, "name": "Climb", "percent_start": 0, "percent_end": 0.475,
                  "start_distance_m": 0, "end_distance_m": 475,
                  "distance_m": 475, "max_distance_m": 500},
                 {"hash": 7, "name": "Climb", "percent_start": 0.5, "percent_end": 1,
                  "start_distance_m": 500, "end_distance_m": 1000,
                  "distance_m": 500, "max_distance_m": 500},
                 {"hash": -7, "name": "Climb", "percent_start": 0, "percent_end": 0.3,
                  "start_distance_m": 0, "end_distance_m": 300,
                  "distance_m": 300, "max_distance_m": 300},
             ]}
    before = deepcopy(route)
    monkeypatch.setattr(surface_map, "_projection", lambda _: {"project": lambda xs, zs: (xs, zs)})
    monkeypatch.setattr(surface_map, "_load_segment_names", lambda: {})
    monkeypatch.setattr(surface_map, "local_to_latlng", lambda *args: [])

    markers = surface_map._route_segment_markers(13, route, None, route["segment_path"])

    assert [(marker["hash"], marker["pass"]) for marker in markers] == [
        *((7, index) for index in range(1, expected_passes + 1)), (-7, 1),
    ]
    assert route == before


def test_la_reine_keeps_native_segment_data_but_only_ven_10_displays_the_full_climb():
    from copy import deepcopy
    from shared.surface_map import _load_route_world, get_route

    world = _load_route_world(10)
    before = deepcopy(world)
    la_reine = next(route for route in world["routes"] if route["name"] == "La Reine")
    ven_10 = next(route for route in world["routes"] if route["name"] == "Ven-10")
    partial = next(segment for segment in la_reine["native_segments"] if segment["hash"] == 1034851390)
    complete = next(segment for segment in ven_10["segments"] if segment["hash"] == 1034851390)

    assert partial["percent_start"] == 0.48545861
    assert partial["percent_end"] == 0.84608501
    assert 1034851390 not in {segment["hash"] for segment in la_reine["segments"]}
    assert complete["boundary_source"] == "timing_arch"
    assert complete["distance_m"] == complete["max_distance_m"]
    assert 1034851390 not in {segment["hash"] for segment in get_route(10, la_reine["nameHash"])["segments"]}
    assert 1034851390 in {segment["hash"] for segment in get_route(10, ven_10["nameHash"])["segments"]}
    assert world == before


def test_route_sampling_retains_bends_beyond_the_initial_point_budget():
    profile = extract_zwift_routes.build_profile(
        [(0, 0, 0), (10000, 5000, 10000), (20000, 0, 0)], 2,
    )

    assert profile["x"] == [0, 100, 200]
    assert profile["z"] == [0, 100, 0]
    assert profile["alt"] == [0, 50, 0]
    assert profile["d"][-1] == pytest.approx(282.84)


def test_route_sampling_preserves_exact_fractional_boundary_distances():
    profile = extract_zwift_routes.build_profile(
        [(0, 0, 0), (10000, 0, 0), (20000, 0, 0)], 2,
        preserve_distances=[75.25],
    )

    assert profile["d"] == [0, 75.25, 200]
    assert profile["x"] == [0, 75.25, 200]


@pytest.mark.parametrize("closed, laps, resolves", [
    (True, 1, True), (False, 1, False), (True, 0, False),
])
def test_zero_segment_endpoint_resolves_only_for_closed_lap_routes(closed, laps, resolves):
    from shared.surface_map import _route_segment_markers

    last_x, last_z = (10000, 0) if closed else (30000, 10000)
    source = f'''<route name="Loop" supportedLaps="{laps}" leadinDistanceInMeters="200" />
        <leadinhighrescheckpoint>
            <entry x="-10000" y="0" z="0" />
            <entry x="0" y="0" z="0" />
        </leadinhighrescheckpoint>
        <highrescheckpoint>
            <entry x="10000" y="0" z="0" />
            <entry x="20000" y="0" z="0" />
            <entry x="20000" y="0" z="10000" />
            <entry x="{last_x}" y="0" z="{last_z}" />
        </highrescheckpoint>
        <segment_data><entry hash="7" percentStart="0.4" percentEnd="0" /></segment_data>'''.encode()
    route = parse_route(source, 13, 600, segment_names={7: "Loop"})
    segment = route["segments"][0]

    assert segment["percent_start"] == 0.4
    assert segment["percent_end"] == 0
    if resolves:
        assert segment["resolved_percent_end"] == 1
        assert segment["end_distance_m"] == round(200 + route["route"]["d"][-1], 2)
        marker = _route_segment_markers(13, route, None, route["route"])[0]
        assert len(marker["path"]) >= 4
        assert marker["path"][0] == marker["path"][-1]
        assert marker["start_distance_m"] < marker["end_distance_m"]
    else:
        assert "resolved_percent_end" not in segment
        assert segment["end_distance_m"] == 0


def test_route_sampling_retains_segment_boundaries_in_profiles_and_map_paths():
        from shared.surface_map import _route_path_slice

        source = b'''<route name="Bends" distanceInMeters="400" />
            <highrescheckpoint>
                <entry x="0" y="0" z="0" />
                <entry x="10000" y="0" z="0" />
                <entry x="10000" y="0" z="10000" />
                <entry x="20000" y="0" z="10000" />
                <entry x="30000" y="0" z="10000" />
            </highrescheckpoint>
            <segment_data><entry hash="7" percentStart="0.375" percentEnd="0.875" /></segment_data>'''
        route = parse_route(source, 13, 2)

        assert 150 in route["route"]["d"]
        assert 350 in route["route"]["d"]
        assert 1.5 in route["segment_path"]["d"]
        assert 3.5 in route["segment_path"]["d"]
        segment = route["segments"][0]
        assert segment["start_distance_m"] == 150
        assert segment["end_distance_m"] == 350
        segment_path = _route_path_slice(route, None, None, 0.375, 0.875)
        assert segment_path[0] == [100, 50]
        assert segment_path[-1] == [250, 100]


def test_route_sampling_bounds_position_and_altitude_error_without_clamping():
        points = [(index * 1000, math.sin(index / 5) * 3000, math.cos(index / 10) * 4000)
                            for index in range(100)]
        profile = extract_zwift_routes.build_profile(points, 4)
        distances = [distance / 100 for distance in extract_zwift_routes._cumulative_planar(points)]

        assert min(profile["alt"]) < 0
        assert profile["d"][-1] == round(distances[-1], 2)
        for distance, source in zip(distances, points):
                index = max(0, min(bisect_right(profile["d"], distance) - 1, len(profile["d"]) - 2))
                weight = (distance - profile["d"][index]) / (profile["d"][index + 1] - profile["d"][index])
                sampled = [profile[key][index] + weight * (profile[key][index + 1] - profile[key][index])
                                     for key in ("x", "alt", "z")]
                assert math.dist(sampled, [value / 100 for value in source]) <= 1.02


def test_route_sampling_preserves_boundaries_in_trimmed_paddock_profiles():
        source = b'''<route name="Pen" leadinDistanceInMeters="105" />
            <leadinhighrescheckpoint>
                <entry x="0" y="0" z="0" road="11" />
                <entry x="10000" y="0" z="0" road="9" />
                <entry x="20000" y="0" z="0" road="9" />
            </leadinhighrescheckpoint>
            <highrescheckpoint>
                <entry x="20500" y="0" z="0" road="9" />
                <entry x="30500" y="0" z="0" road="9" />
            </highrescheckpoint>
            <segment_data><entry hash="7" percentStart="0.375" percentEnd="0.75" /></segment_data>'''
        route = parse_route(source, 13, 2, paddock_road_ids={11})

        assert route["leadin_view"]["paddock_distance_m"] == 100
        assert 50 in route["leadin_view"]["profile"]["d"]
        assert route["segments"][0]["start_distance_m"] == 150


@pytest.mark.parametrize("published, expected", [
    (105, True), (95.001, True), (114.999, True),
    (95, False), (115, False), (155, False), (205, False),
])
def test_leadin_view_uses_strict_error_thresholds_and_includes_endpoint_gap(published, expected):
    points = [(0, 100, 0), (10000, 200, 0), (20000, 300, 0)]
    view = build_leadin_view(points, [(11, 0), (9, 0), (9, 1)],
                             [(20500, 400, 0)], published, {11}, 600,
                             ["Tarmac", "Dirt", "Wood"])

    assert (view is not None) is expected
    if expected:
        assert view["full_distance_m"] == 205
        assert view["distance_m"] == 105
        assert view["paddock_distance_m"] == 100
        assert view["error_before_m"] == abs(205 - published)
        assert view["error_after_m"] == abs(105 - published)
        assert view["excluded_checkpoint_count"] == 1
        assert view["paddock_road_ids"] == [11]
        assert view["profile"] == {
            "d": [0, 100], "x": [100, 200], "z": [0, 0],
            "alt": [2, 3], "surface": ["Dirt", "Wood"],
        }


@pytest.mark.parametrize("paddock_distance, expected", [(49.999, False), (50, False), (50.001, True)])
def test_leadin_view_requires_initial_error_strictly_over_fifty_metres(paddock_distance, expected):
    prefix = paddock_distance * 100
    view = build_leadin_view(
        [(0, 0, 0), (prefix, 0, 0), (prefix + 10000, 0, 0)],
        [(11, 0), (9, 0), (9, 1)], [(prefix + 10500, 0, 0)], 105, {11}, 600,
    )
    assert (view is not None) is expected


@pytest.mark.parametrize("road_ids, main_points, paddocks", [
    ([9, 9, 9], [(20500, 0, 0)], {11}),
    ([11, 11, 11], [(20500, 0, 0)], {11}),
    ([11, 9, 11], [(20500, 0, 0)], {11}),
    ([11, 9, 9], [], {11}),
    ([11, 9, 9], [(20500, 0, 0)], set()),
])
def test_leadin_view_keeps_unmeasurable_and_non_prefix_geometry(road_ids, main_points, paddocks):
    assert build_leadin_view(
        [(0, 0, 0), (10000, 0, 0), (20000, 0, 0)],
        [(road_id, 0) for road_id in road_ids], main_points, 105, paddocks, 600,
    ) is None


def test_route_adds_display_profile_without_mutating_raw_geometry_or_metadata():
    source = b'''<route name="Test" leadinDistanceInMeters="105" />
      <leadinhighrescheckpoint>
        <entry x="0" y="100" z="0" road="11" />
        <entry x="10000" y="200" z="0" road="9" />
        <entry x="20000" y="300" z="0" road="9" />
      </leadinhighrescheckpoint>
      <highrescheckpoint>
        <entry x="20500" y="400" z="0" road="9" />
        <entry x="30500" y="500" z="0" road="9" />
      </highrescheckpoint>'''
    original = parse_route(source, 13, 600)
    updated = parse_route(source, 13, 600, paddock_road_ids={11})

    view = updated.pop("leadin_view")
    assert updated == original
    assert view["profile"]["x"] == [100, 200]
    assert view["error_before_m"] == 100
    assert view["error_after_m"] == 0


def test_surface_map_route_modes_preserve_cache_and_rebase_course_segments(monkeypatch):
    from copy import deepcopy
    from shared import surface_map

    raw = {"d": [0, 100, 200], "x": [0, 100, 200], "z": [0, 0, 0],
           "alt": [10, 20, 30], "surface": ["Tarmac"] * 3}
    free = {key: values[1:] for key, values in raw.items()}
    free["d"] = [0, 100]
    cached = {"nameHash": 123, "leadin_distance_m": 205, "leadin": raw,
              "route": {"d": [0, 100], "x": [205, 305], "z": [0, 0],
                        "alt": [40, 50], "surface": ["Tarmac"] * 2},
              "leadin_view": {"paddock_distance_m": 100, "profile": free},
              "route_versions": {"race": {"available": True},
                                 "free_ride": {"available": True, "leadin": free, "leadin_distance_m": 90}}}
    before = deepcopy(cached)
    monkeypatch.setattr(surface_map, "_load_route_world", lambda _: {"routes": [cached]})
    monkeypatch.setattr(surface_map, "_projection", lambda _: {"project": lambda xs, zs: (xs, zs)})
    monkeypatch.setattr(surface_map, "get_world_altitude_scale", lambda _: 2)
    monkeypatch.setattr(surface_map, "_route_segment_markers", lambda *args: [
        {"start_distance_m": 50, "end_distance_m": 70},
        {"start_distance_m": 215, "end_distance_m": 225}])
    race = surface_map.get_route(13, 123, mode="race")
    free_ride = surface_map.get_route(13, 123, mode="free_ride")
    assert race["leadin"]["local_x"] == [0, 100, 200]
    assert race["route_offset_m"] == 205
    assert free_ride["leadin"]["local_x"] == [100, 200]
    assert free_ride["route_offset_m"] == 105
    assert free_ride["leadin_distance_m"] == 90
    assert free_ride["segments"] == [{"start_distance_m": 115, "end_distance_m": 125}]
    assert cached == before


@pytest.mark.parametrize("adjusted", [False, True])
def test_surface_map_uses_display_profile_and_offsets_without_mutating_cache(monkeypatch, adjusted):
    from copy import deepcopy
    from shared import surface_map

    raw_profile = {
        "d": [0, 100, 200], "x": [0, 100, 200], "z": [0, 0, 0],
        "alt": [10, 20, 30], "surface": ["Tarmac"] * 3,
    }
    cached = {
        "nameHash": 123, "leadin_distance_m": 105,
        "leadin": raw_profile,
        "route": {"d": [0, 100], "x": [205, 305], "z": [0, 0],
                  "alt": [40, 50], "surface": ["Tarmac"] * 2},
        "escalators": [
            {"leg": "leadin", "start_distance_m": 150, "end_distance_m": 160},
            {"leg": "route", "start_distance_m": 115, "end_distance_m": 125},
        ],
    }
    if adjusted:
        cached["leadin_view"] = {
            "paddock_distance_m": 100,
            "profile": {key: values[1:] for key, values in raw_profile.items()},
        }
        cached["leadin_view"]["profile"]["d"] = [0, 100]
    before = deepcopy(cached)
    monkeypatch.setattr(surface_map, "_load_route_world", lambda _: {"routes": [cached]})
    monkeypatch.setattr(surface_map, "_projection", lambda _: {"project": lambda xs, zs: (xs, zs)})
    monkeypatch.setattr(surface_map, "get_world_altitude_scale", lambda _: 2)

    def markers(map_id, route, leadin, main):
        assert leadin["local_x"] == [0, 100, 200]
        return [{"start_distance_m": 215, "end_distance_m": 225}]

    monkeypatch.setattr(surface_map, "_route_segment_markers", markers)
    displayed = surface_map.get_route(13, 123)

    assert cached == before
    assert displayed["leadin_distance_m"] == 105
    assert displayed["route_offset_m"] == (105 if adjusted else 205)
    assert displayed["route"]["alt"] == [70, 90]
    assert displayed["leadin"]["local_x"] == ([100, 200] if adjusted else [0, 100, 200])
    assert displayed["leadin"]["d"] == ([0, 100] if adjusted else [0, 100, 200])
    assert displayed["segments"][0]["start_distance_m"] == (115 if adjusted else 215)
    assert displayed["escalators"][0]["start_distance_m"] == (50 if adjusted else 150)
    assert displayed["escalators"][1]["start_distance_m"] == 115
    assert displayed["breakdown"][0]["distance_m"] == (200 if adjusted else 300)


@pytest.mark.parametrize("leadin", [None, {}, {"d": []}])
def test_surface_map_profile_offset_ignores_metadata_without_leadin_geometry(leadin):
    from shared.surface_map import _route_profile_offset_m

    route = {"leadin_distance_m": 248.3, "leadin": leadin}
    assert _route_profile_offset_m(route) == 0
    assert route["leadin_distance_m"] == 248.3


@pytest.mark.parametrize("attributes, paddocks, ramp", [
    ('eventPaddocks="21,22,23,24,25" useAlternateEventRamp="2"',
     [21, 22, 23, 24, 25], 2),
    ('eventPaddocks="25, 21,25," useAlternateEventRamp="0"', [25, 21, 25], 0),
    ('eventPaddocks=""', [], None),
    ('', [], None),
])
def test_route_preserves_event_paddock_references_and_ramp_mode(attributes, paddocks, ramp):
    route = parse_route(f'<route name="Test" {attributes} />'.encode(), 1, 600)

    assert route["event_paddocks"] == paddocks
    assert route["alternate_event_ramp"] == ramp


def test_paddock_roads_preserve_local_nodes_handles_and_spawn_references():
    roads = b'''<world><roads>
        <road><id>164</id><roadName>Pen</roadName><roadIsPaddock>true</roadIsPaddock>
          <paddockExitRoadTime>0</paddockExitRoadTime>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{-1000,-2000,3000}"
               m_tangentIn="{-100,200,-300}" m_tangentOut="{400,500,600}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,-1000,4000}" />
          <ent type="ENTITY_TYPE_ROADMARKER" m_pos="{999,999,999}" />
        </road>
        <road><id>166</id><roadIsPaddock>1</roadIsPaddock></road>
        <road><id>149</id><roadIsPaddock>false</roadIsPaddock></road>
        <road><id>150</id></road>
    </roads></world>'''
    entities = b'''<world><entities>
        <ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="Climb Mtn"
             m_roadId="164" m_roadTime="0.1747803277" />
        <ent type="ENTITY_TYPE_SPAWNMARKER" m_roadId="149" m_roadTime="0.5" />
        <ent type="ENTITY_TYPE_SPAWNMARKER" />
        <ent type="ENTITY_TYPE_PROP" m_roadId="164" m_roadTime="0.8" />
    </entities></world>'''

    pen, empty_pen = parse_paddock_roads(roads, entities)

    assert pen == {
        "road_id": 164, "name": "Pen", "paddock_exit_road_time": 0,
        "nodes": [
            {"x": -10, "alt": -20, "z": 30,
             "tangent_in": [-1, 2, -3], "tangent_out": [4, 5, 6]},
            {"x": 0, "alt": -10, "z": 40, "tangent_in": None, "tangent_out": None},
        ],
        "spawn_markers": [{"name": "Climb Mtn", "road_time": 0.1747803277}],
    }
    assert empty_pen == {
        "road_id": 166, "name": "", "paddock_exit_road_time": None,
        "nodes": [], "spawn_markers": [],
    }
    assert parse_paddock_roads(roads)[0]["spawn_markers"] == []
    assert parse_paddock_roads(b'<world><roads /></world>') == []


def test_extraction_publishes_paddock_references_and_world_locations(tmp_path, monkeypatch):
    world = tmp_path / "client" / "assets" / "Worlds" / "world1"
    world.mkdir(parents=True)
    (world / "data_1.wad").touch()
    output = tmp_path / "routes"
    monkeypatch.setattr(extract_zwift_routes, "read_wad_entries", lambda *args, **kwargs: {
        "Worlds/world1/routes/routes0.xml": b'<route name="Test" nameHash="123" '
        b'eventPaddocks="21,22" useAlternateEventRamp="2" leadinDistanceInMeters="105" />'
        b'<leadinhighrescheckpoint>'
        b'<entry x="0" y="0" z="0" road="164" />'
        b'<entry x="10000" y="0" z="0" road="149" />'
        b'<entry x="20000" y="0" z="0" road="149" />'
        b'</leadinhighrescheckpoint><highrescheckpoint>'
        b'<entry x="20500" y="0" z="0" road="149" />'
        b'<entry x="30500" y="0" z="0" road="149" />'
        b'</highrescheckpoint>',
        "Worlds/world1/road.xml": b'<world><roads><road><id>164</id>'
        b'<roadIsPaddock>true</roadIsPaddock>'
        b'<ent type="ENTITY_TYPE_ROADNODE" m_pos="{-1000,0,2000}" />'
        b'</road></roads></world>',
    })
    monkeypatch.setattr("sys.argv", [
        "extract_zwift_routes.py", "--zwift-dir", str(tmp_path / "client"),
        "--out", str(output),
    ])

    assert extract_zwift_routes.main() == 0

    index = json.loads((output / "index.json").read_text())
    world_data = json.loads((output / "world_1.json").read_text())
    for metadata in (index[0], world_data["routes"][0]):
        assert metadata["event_paddocks"] == [21, 22]
        assert metadata["alternate_event_ramp"] == 2
    assert world_data["coordinate_system"] == "zwift_local_m"
    assert world_data["paddock_roads"][0]["road_id"] == 164
    assert world_data["paddock_roads"][0]["nodes"][0]["x"] == -10
    assert world_data["routes"][0]["leadin_view"]["error_before_m"] == 100
    assert world_data["routes"][0]["leadin_view"]["error_after_m"] == 0