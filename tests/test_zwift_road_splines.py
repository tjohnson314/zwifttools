import xml.etree.ElementTree as ET
import json

import pytest

from shared.surface_map_dev import _node_segments
from tools.extract_zwift_surfaces import _nodes_full, _spline_points, resolve_road


def test_route_road_membership_distinguishes_course_and_mode_specific_approaches():
     from tools.extract_zwift_routes import build_route_versions

     root = ET.fromstring('''<root>
          <leadinhighrescheckpoint>
               <entry road="12" dir="1" time="0" x="-100" y="0" z="0" />
               <entry road="0" dir="1" time="0.2" x="0" y="0" z="0" />
          </leadinhighrescheckpoint>
          <highrescheckpoint>
               <entry road="0" dir="1" time="0.5" x="1000" y="0" z="0" />
               <entry road="2" dir="1" time="0.1" x="2000" y="0" z="0" />
          </highrescheckpoint>
          </root>''')
     route = {"leadin_distance_m": 1, "leadin_ascent_m": 0, "leadin": {},
              "start_locations": {"free_ride": {"spawn_area": {
                   "road": "0", "forward": "1", "starttime": "0.3", "endtime": "0.4"}}}}
     versions = build_route_versions(root, route)
     assert versions["race"]["road_ids"] == [0, 2, 12]
     assert versions["free_ride"]["road_ids"] == [0, 2]
     assert versions["free_ride"]["geometry_status"] == "native"
     assert versions["free_ride"]["road_paths"]["0"][0]["x"][0] == pytest.approx(5)
     assert "12" not in versions["free_ride"]["road_paths"]
     assert "road_ids" not in route


def test_world_api_exposes_projected_point_paths_without_mutating_the_cache(monkeypatch):
     from shared import surface_map

     source = {"routes": [{"nameHash": 1, "name": "Partial", "route_versions": {
          "race": {"available": True, "road_ids": [0], "road_paths": {
               "0": [{"x": [0, 40], "z": [0, 10]}]}}}}]}
     before = json.dumps(source)
     monkeypatch.setattr(surface_map, "_load_surface_world", lambda _: {"segments": []})
     monkeypatch.setattr(surface_map, "_load_route_world", lambda _: source)
     monkeypatch.setattr(surface_map, "_projection", lambda _: {
          "project": lambda xs, zs: (xs, [-value for value in zs]), "mode": "local",
          "background": None, "bounds": {"min_x": 0, "max_x": 40, "min_y": -10, "max_y": 0}})
     world = surface_map.get_world_surfaces(1)
     assert world["road_match_tolerance"] > 0
     assert world["routes"][0]["road_paths"]["race"]["0"] == [{"x": [0, 40], "y": [0, -10]}]
     assert json.dumps(source) == before


def test_timing_road_position_interpolates_bracketing_checkpoints_without_clamping():
     from shared.surface_map_dev import _road_time_point

     intervals = [(0, 0.5, (0, 0), (100, 0), "Route"),
                     (1, 0.5, (400, 0), (100, 0), "Reverse"),
                     (0.1, 0.2, (20, 0), (40, 0), "Dense")]

     point = _road_time_point(intervals, 0.125)
     assert point == {"local_x": 25, "local_z": 0, "source_route": "Dense"}
     assert _road_time_point(intervals, 0.5)["local_x"] == 100
     assert _road_time_point(intervals, 1)["local_x"] == 400
     assert _road_time_point(intervals, -0.01) is None
     assert _road_time_point(intervals, 1.01) is None
     assert _road_time_point([], 0.5) is None


def test_dev_timing_arches_keep_exact_hashes_raw_xml_and_unlocated_boundaries(monkeypatch):
     from shared import surface_map_dev as dev

     entries = {
          "Worlds/world9/entities.xml": b'''<world><ent type="ENTITY_TYPE_TIMINGARCH"
               m_Name="Alley" m_roadId="4" m_roadTime="0.75"
               m_startLineSplineTimeF="0.25" m_overrideNetworkHashF="9223372036854775807"
               m_startLineSplineTimeR="1.1" m_overrideNetworkHashR="-9223372036854775807"
               m_ArchFriendlyName="Alley Sprint" m_ArchSegmentDistanceF="0.4828020036" />
               <ent type="ENTITY_TYPE_TIMINGARCH" m_Name="Missing road" m_roadId="99"
                     m_roadTime="0.5" m_startLineSplineTimeF="0.1" />
                  <ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="Alley spawn" m_roadId="4" m_roadTime="0.25" />
               <ent type="ENTITY_TYPE_ROADMARKER" /></world>''',
          "Worlds/world9/routes/routes1.xml": b'''<route name="Reference" />
               <highrescheckpoint>
                    <entry road="4" dir="1" time="0" x="0" z="0" />
                    <entry road="4" dir="1" time="0.5" x="10000" z="0" />
                    <entry road="4" dir="1" time="1" x="40000" z="0" />
               </highrescheckpoint>''',
     }
     monkeypatch.setattr(dev, "_world_wads", lambda: {9: "test.wad"})
     monkeypatch.setattr(dev, "read_wad_entries", lambda *args, **kwargs: entries)
     monkeypatch.setattr(dev, "_projection", lambda _: {"project": lambda xs, zs: (xs, [-value for value in zs])})
     dev.get_world_timing_arches.cache_clear()
     try:
          payload = dev.get_world_timing_arches(9)
          arches = payload["arches"]
          assert len(arches) == 3
          forward = next(arch for arch in arches if arch["hash"] == "9223372036854775807")
          reverse = next(arch for arch in arches if arch["hash"] == "-9223372036854775807")
          assert forward["start"]["x"] == 50
          assert forward["finish"]["x"] == 250
          assert forward["start"]["source_route"] == "Reference"
          assert forward["declared_distance_m"] == pytest.approx(482.8020036)
          assert 'm_startLineSplineTimeF="0.25"' in forward["xml"]
          assert reverse["start_time"] == 1.1
          assert reverse["start"] is None and reverse["finish"] is not None
          assert reverse["name"] == "Alley Sprint Reverse"
          missing = next(arch for arch in arches if arch["road_id"] == 99)
          assert missing["start"] is None and missing["finish"] is None
     finally:
          dev.get_world_timing_arches.cache_clear()


def test_dev_spawn_markers_preserve_raw_times_and_distinguish_road_anchors(monkeypatch):
     from shared import surface_map_dev as dev

     entities = b'''<ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="AdZ_Paddock"
          m_label="Alpe spawn" m_roadId="130" m_roadTime="0.0303317708" />
          <ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="Native" m_roadId="2" m_roadTime="0.25" />
          <ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="Outside" m_roadId="130" m_roadTime="-2" />
          <ent type="ENTITY_TYPE_SPAWNMARKER" m_Name="Missing" m_roadId="999" m_roadTime="0.5" />
          <ent type="ENTITY_TYPE_SPAWNMARKER" m_label="Invalid" m_roadId="unknown" m_roadTime="nan" />'''
     road = '''<road><id>130</id><looped>0</looped>
          <ent type="ENTITY_TYPE_ROADNODE" m_bStraight="1" m_pos="{0,0,0}" m_tangentOut="{0,0,10000}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{10000,0,0}" m_tangentIn="{0,0,10000}" />
          </road>'''
     monkeypatch.setattr(dev, "_read_road_style", lambda _: (road, []))
     monkeypatch.setattr(dev, "_projection", lambda _: {"project": lambda xs, zs: (xs, [-value for value in zs])})
     intervals = {2: [(0, 1, (0, 0), (40, 80), "Native route")]}
     markers = {marker["name"]: marker for marker in dev._spawn_markers(entities, intervals, 1)}
     spawn = markers["AdZ_Paddock"]
     assert spawn["label"] == "Alpe spawn"
     assert spawn["road_time"] == 0.0303317708
     assert spawn["position"] == {"local_x": 50, "local_z": 0, "source_route": None, "x": 50, "y": 0}
     assert spawn["position_source"] == "road_geometry_midpoint"
     assert 'm_Name="AdZ_Paddock"' in spawn["xml"]
     assert markers["Outside"]["road_time"] == -2
     assert markers["Outside"]["position_source"] == "road_geometry_midpoint"
     assert markers["Missing"]["position"] is None
     assert markers["Invalid"]["road_id"] is None
     assert markers["Invalid"]["road_time"] is None
     assert markers["Invalid"]["position"] is None
     assert markers["Native"]["position"]["x"] == 10
     assert markers["Native"]["position"]["y"] == -20
     assert markers["Native"]["position_source"] == "native_route_checkpoint_interpolation"


def test_scotland_road_9_straight_flags_override_both_sideways_handles():
    block = '''
        <ent type="ENTITY_TYPE_ROADNODE" m_bStraight="1"
             m_pos="{-247367.5625,15415.63671875,54064.2890625}"
             m_tangentIn="{0,34.8306770325,640.4641723633}"
             m_tangentOut="{1862.8078613281,-85.8150253296,365.9859924316}" />
        <ent type="ENTITY_TYPE_ROADNODE" m_bStraight="1"
             m_pos="{-247367.5625,15121.6064453125,48657.6796875}"
             m_tangentIn="{-1896.8389892578,109.5636978149,36.8779067993}"
             m_tangentOut="{4600.94921875,-238.1001892090,52.8761787415}" />
        <ent type="ENTITY_TYPE_ROADNODE"
             m_pos="{-247367.5625,14938.64453125,44883.171875}"
             m_tangentIn="{-4599.09375,241.0039215088,135.9091186523}"
             m_tangentOut="{0,-73.5879287720,-1518.1204833984}" />
    '''

    nodes = _nodes_full(block)
    points = _spline_points(nodes)

    assert all(point[0] == pytest.approx(-2473.675625) for point in points)
    assert points[0] == nodes[0][0]
    assert points[-1] == nodes[-1][0]
    assert all(current[1] >= following[1] for current, following in zip(points, points[1:]))
    for _, _, segment in _node_segments(nodes, False):
        assert all(point[0] == pytest.approx(-2473.675625) for point in segment)


@pytest.mark.parametrize("flag", ["1", "true"])
def test_straight_closing_segment_preserves_neighboring_curves(flag):
    block = f'''
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{0,0,0}}"
             m_tangentIn="{{-500,0,500}}" m_tangentOut="{{0,0,500}}" />
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{1000,0,0}}"
             m_tangentIn="{{0,0,500}}" m_tangentOut="{{500,0,0}}" />
        <ent type="ENTITY_TYPE_ROADNODE" m_bStraight="{flag}" m_pos="{{1000,0,1000}}"
             m_tangentIn="{{500,0,0}}" m_tangentOut="{{-500,0,500}}" />
    '''

    nodes = _nodes_full(block)
    segments = _node_segments(nodes, True)

    assert nodes[0][1] == nodes[2][2] == (0, 0)
    assert max(point[1] for point in segments[0][2]) > 1
    assert max(point[0] for point in segments[1][2]) > 11
    assert all(point[0] == pytest.approx(point[1]) for point in segments[2][2])
    assert _spline_points(nodes, looped=True)[-1] == nodes[0][0]


def test_false_or_missing_straight_flags_keep_source_handles():
    nodes = _nodes_full('''
        <ent type="ENTITY_TYPE_ROADNODE" m_bStraight="0" m_pos="{0,0,0}"
             m_tangentOut="{0,0,500}" />
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,0}"
             m_tangentIn="{0,0,500}" />
    ''')

    assert nodes[0][2] == nodes[1][1] == (0, 5)
    assert max(point[1] for point in _spline_points(nodes)) > 1
    assert _nodes_full("") == []


def test_missing_handles_can_use_uniform_catmull_rom_without_changing_default():
     block = '''<road><looped>0</looped>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,1000}" />
          </road>'''
     original = _nodes_full(block)
     nodes = _nodes_full(block, auto_tangents=True)

     assert all(node[1] == node[2] == (0, 0) for node in original)
     assert nodes[0][2] == pytest.approx((10 / 6, 0))
     assert nodes[1][1] == pytest.approx((-10 / 6, -10 / 6))
     assert nodes[1][2] == pytest.approx((10 / 6, 10 / 6))
     assert nodes[-1][1] == pytest.approx((0, -10 / 6))
     points = _spline_points(nodes)
     assert points[0] == nodes[0][0] and points[-1] == nodes[-1][0]
     assert min(point[1] for point in points) < 0
     assert max(point[0] for point in points) > 10


def test_automatic_handles_wrap_at_loop_seam():
     block = '''<road><looped>1</looped>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,1000}" />
          </road>'''
     nodes = _nodes_full(block, auto_tangents=True)

     assert nodes[0][2] == pytest.approx((0, -10 / 6))
     assert nodes[-1][2] == pytest.approx((-10 / 6, 0))
     assert _spline_points(nodes, looped=True)[-1] == nodes[0][0]


@pytest.mark.parametrize("looped", [0, 1])
def test_main_surface_segments_use_the_dev_maps_automatic_handles(looped):
     block = f'''<road><id>0</id><looped>{looped}</looped><defaultStyle>0</defaultStyle>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{0,0,0}}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{1000,0,0}}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{1000,0,1000}}" />
          </road>'''
     segment, = resolve_road(block, 0.5, ["NORMAL"])

     assert segment["surface"] == "Tarmac"
     assert segment["road_id"] == 0
     assert min(segment["z"]) < 0 and max(segment["x"]) > 10
     assert (segment["x"][0], segment["z"][0]) == (0, 0)
     assert (segment["x"][-1], segment["z"][-1]) == ((0, 0) if looped else (10, 10))


def test_automatic_handles_preserve_explicit_zero_handles_and_straight_flags():
     block = '''<road><looped>0</looped>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}"
                m_tangentOut="{0,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,0}"
                m_bStraight="1" m_tangentIn="{0,0,500}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,1000}" />
          </road>'''
     nodes = _nodes_full(block, auto_tangents=True)

     assert nodes[0][2] == (0, 0)
     assert nodes[1][1] == (0, 5)
     assert nodes[1][2] == nodes[2][1] == (0, 0)
     segments = _node_segments(nodes, False)
     assert max(point[1] for point in segments[0][2]) > 1
     assert all(point[0] == pytest.approx(10) for point in segments[1][2])


def test_dev_road_overview_detail_and_node_segments_use_automatic_handles(monkeypatch):
     from shared import surface_map_dev as dev

     block = '''<road><id>0</id><looped>0</looped><defaultStyle>0</defaultStyle>
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,0}" />
          <ent type="ENTITY_TYPE_ROADNODE" m_pos="{1000,0,1000}" />
          </road>'''
     monkeypatch.setattr(dev, "_read_road_style", lambda _: (block, ["NORMAL"]))
     monkeypatch.setattr(dev, "_projection", lambda _: {
          "project": lambda xs, zs: (xs, zs), "mode": "local",
          "background": None, "bounds": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10},
     })
     overview = dev.get_world_roads(1)["roads"][0]
     detail = dev.get_road_detail(1, 0)

     assert min(overview["y"]) < 0 and max(overview["x"]) > 10
     assert min(detail["full"]["y"]) < 0 and max(detail["full"]["x"]) > 10
     assert min(detail["segments"][0]["y"]) < 0
     assert max(detail["segments"][1]["x"]) > 10
     assert overview["length_m"] == detail["length_m"]