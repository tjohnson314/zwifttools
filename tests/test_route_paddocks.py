import json

import pytest

from tools import extract_zwift_routes
from tools.extract_zwift_routes import build_leadin_view, parse_paddock_roads, parse_route


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