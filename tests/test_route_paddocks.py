import json
import math
from bisect import bisect_right

import pytest

from tools import extract_zwift_routes
from tools.extract_zwift_routes import build_leadin_view, parse_paddock_roads, parse_route


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
    assert displayed["route"]["alt"] == [70, 90]
    assert displayed["leadin"]["local_x"] == ([100, 200] if adjusted else [0, 100, 200])
    assert displayed["leadin"]["d"] == ([0, 100] if adjusted else [0, 100, 200])
    assert displayed["segments"][0]["start_distance_m"] == (115 if adjusted else 215)
    assert displayed["escalators"][0]["start_distance_m"] == (50 if adjusted else 150)
    assert displayed["escalators"][1]["start_distance_m"] == 115
    assert displayed["breakdown"][0]["distance_m"] == (200 if adjusted else 300)


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