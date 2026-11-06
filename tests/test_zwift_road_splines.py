import pytest

from shared.surface_map_dev import _node_segments
from tools.extract_zwift_surfaces import _nodes_full, _spline_points


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