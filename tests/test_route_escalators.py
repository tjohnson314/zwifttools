from unittest.mock import patch

import numpy as np
import pytest

from shared.surface_map import get_route
from tools.extract_zwift_routes import (
    derive_escalator_passages,
    load_multiroot,
    parse_escalator_definitions,
    parse_route,
)


def _definitions():
    return parse_escalator_definitions(b'''
        <world><entities>
          <ent type="ENTITY_TYPE_VOLUME_EVENTTRIGGER"
               m_triggerTagName="SubwayB_Forward" m_roadId="221"
               m_backRoadTime="0.3" m_frontRoadTime="0.7"
               m_bDirectionalTrigger="1" m_triggerDir="1"
               m_selectedResponsesForSave="Set Bonus Watts;Clear Bonus Watts"
               m_selectedResponseParametersForSave="15.000000;"
               m_trainerSlopeScale="0.2" />
          <ent type="ENTITY_TYPE_VOLUME_EVENTTRIGGER"
               m_triggerTagName="SubwayB_Reverse" m_roadId="240"
               m_backRoadTime="0.3" m_frontRoadTime="0.7"
               m_bDirectionalTrigger="1"
               m_selectedResponsesForSave="Set Bonus Watts;Clear Bonus Watts"
               m_selectedResponseParametersForSave="15.000000;"
               m_trainerSlopeScale="0.2" />
          <ent type="ENTITY_TYPE_VOLUME_EVENTTRIGGER"
               m_triggerTagName="Unrelated"
               m_selectedResponsesForSave="Set Bonus Watts" />
          <ent type="ENTITY_TYPE_PROP" m_triggerTagName="SubwayProp" />
        </entities></world>
    ''')


def _entry(road, time, direction, distance):
    return (f'<entry road="{road}" time="{time}" dir="{direction}" '
            f'x="{distance * 100}" y="-1000" z="0" />')


def test_escalator_definitions_use_trigger_bounds_and_direction_defaults():
    forward, reverse = _definitions()
    assert forward == {
        "name": "SubwayB_Forward", "road_id": 221,
        "start_time": 0.3, "end_time": 0.7, "direction": 0,
        "bonus_watts": 15.0, "trainer_slope_scale": 0.2,
    }
    assert reverse["direction"] == 1


def test_escalator_edges_interpolate_even_without_samples_inside_zone():
    xml = ('<route name="Test" leadinDistanceInMeters="50" />'
           '<highrescheckpoint>'
           + _entry(221, 1, 0, 0) + _entry(221, 0, 0, 100)
           + '</highrescheckpoint>')
    route = parse_route(xml.encode(), 4, 2, escalator_definitions=_definitions())
    assert len(route["escalators"]) == 1
    passage = route["escalators"][0]
    assert passage["start_distance_m"] == 80
    assert passage["end_distance_m"] == 120
    assert passage["leg"] == "route"
    assert route["route"]["alt"] == [-10.0, -10.0]


def test_escalators_filter_wrong_roads_and_directions_and_keep_repeated_passes():
    entries = [
        _entry(221, 1, 0, 0), _entry(221, 0.5, 0, 50), _entry(221, 0, 0, 100),
        _entry(221, 0, 1, 100), _entry(221, 1, 1, 200),
        _entry(999, 0, 1, 200), _entry(999, 1, 1, 300),
        _entry(240, 0, 1, 300), _entry(240, 1, 1, 400),
        _entry(221, 1, 0, 400), _entry(221, 0, 0, 500),
    ]
    root = load_multiroot(('<highrescheckpoint>' + ''.join(entries)
                          + '</highrescheckpoint>').encode())
    passages = derive_escalator_passages(root, _definitions(), 0)
    assert [(p["start_distance_m"], p["end_distance_m"]) for p in passages] == [
        (30, 70), (330, 370), (430, 470),
    ]
    assert [p["direction"] for p in passages] == [0, 1, 0]


def test_escalator_leadin_offsets_and_absent_metadata():
    checkpoints = _entry(221, 1, 0, 0) + _entry(221, 0, 0, 100)
    xml = ('<route name="Test" leadinDistanceInMeters="150" />'
           f'<leadinhighrescheckpoint>{checkpoints}</leadinhighrescheckpoint>'
           f'<highrescheckpoint>{checkpoints}</highrescheckpoint>').encode()
    route = parse_route(xml, 4, 2, escalator_definitions=_definitions())
    assert [(p["leg"], p["start_distance_m"], p["end_distance_m"])
            for p in route["escalators"]] == [("leadin", 30, 70), ("route", 180, 220)]
    assert parse_route(xml, 4, 2)["escalators"] == []


def test_green_to_screen_cached_escalators_reach_route_api():
    route = get_route(4, 1178708985)
    assert len(route["escalators"]) == 4
    assert [p["name"] for p in route["escalators"]] == [
        "SubwayC_Forward", "SubwayB_Forward", "SubwayB_Reverse", "SubwayC_Reverse",
    ]
    assert all(p["end_distance_m"] > p["start_distance_m"] for p in route["escalators"])
    assert all(p["bonus_watts"] == 15 for p in route["escalators"])
    assert all(p["trainer_slope_scale"] == pytest.approx(0.2)
               for p in route["escalators"])
    main = route["route"]
    for passage in route["escalators"]:
        start, end = (
            passage[key] - route["leadin_distance_m"]
            for key in ("start_distance_m", "end_distance_m")
        )
        assert np.interp(end, main["d"], main["alt"]) > np.interp(
            start, main["d"], main["alt"]
        )
    with patch("shared.surface_map._load_route_world", return_value={
        "routes": [{"nameHash": 1, "name": "Older cache"}],
    }):
        assert get_route(4, 1)["escalators"] == []
