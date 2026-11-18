import pytest

from shared import event_route
from tools.extract_zwift_routes import (
    derive_arch_passages,
    load_multiroot,
    parse_arch_definitions,
)


@pytest.fixture(autouse=True)
def clear_preview_cache():
    event_route._geometry.cache_clear()
    event_route._arch_catalog.cache_clear()
    yield
    event_route._geometry.cache_clear()
    event_route._arch_catalog.cache_clear()


def _entry(road, time, direction, distance):
    return (f'<entry road="{road}" time="{time}" dir="{direction}" '
            f'x="{distance * 100}" y="-1000" z="0" />')


def test_arch_crossings_preserve_missing_ids_flags_directions_and_repeated_passes():
    definitions = parse_arch_definitions(b'''
      <ent type="ENTITY_TYPE_TIMINGARCH" m_Name="Checkpoint" m_roadId="9"
           m_roadTime="0.5" m_bGivesPowerUp="0" />
      <ent type="ENTITY_TYPE_POWERARCH" m_Name="Power" m_roadId="9"
           m_ArchFriendlyName="Outbound" m_ArchFriendlyNameR="Inbound"
           m_roadTime="0.75" m_ArchId="11" m_bEnableBackward="0" />
    ''')
    checkpoints = ''.join([
        _entry(9, 0, 1, 0), _entry(9, 0.5, 1, 50), _entry(9, 1, 1, 100),
        _entry(9, 1, 0, 100), _entry(9, 0, 0, 200),
        _entry(999, 0, 1, 200), _entry(999, 1, 1, 300),
    ])
    root = load_multiroot(f'<highrescheckpoint>{checkpoints}</highrescheckpoint>'.encode())
    passages = derive_arch_passages(root, definitions)
    assert [(p["arch_id"], p["distance_m"]) for p in passages] == [
        (None, 50), (11, 75), (11, 125), (None, 150),
    ]
    assert [p["name"] for p in passages] == ["Checkpoint", "Outbound", "Inbound", "Checkpoint Reverse"]
    assert passages[0]["gives_powerup"] is False
    assert passages[2]["enabled"] is False
    assert passages[1]["type"] == "power"


def test_arch_crossings_include_leadin_connector_offset_and_route_finish():
    definition = parse_arch_definitions(
        b'<ent type="ENTITY_TYPE_TIMINGARCH" m_roadId="9" m_roadTime="1" />')
    xml = ('<leadinhighrescheckpoint>'
           + _entry(9, 0, 1, 0) + _entry(9, 1, 1, 10)
           + '</leadinhighrescheckpoint><highrescheckpoint>'
           + _entry(9, 0, 1, 20) + _entry(9, 1, 1, 120)
           + '</highrescheckpoint>')
    passages = derive_arch_passages(load_multiroot(xml.encode()), definition)
    assert [(p["leg"], p["distance_m"]) for p in passages] == [("leadin", 10), ("route", 120)]


def test_arch_crossings_handle_road_time_wrap_without_crossing_unrelated_arches():
    definitions = parse_arch_definitions(b'''
      <ent type="ENTITY_TYPE_TIMINGARCH" m_roadId="9" m_roadTime="0" m_ArchId="1" />
      <ent type="ENTITY_TYPE_TIMINGARCH" m_roadId="9" m_roadTime="0.5" m_ArchId="2" />
    ''')
    xml = ('<highrescheckpoint>'
           + _entry(9, 0.99, 1, 0) + _entry(9, 0.01, 1, 100)
           + '</highrescheckpoint>')
    passages = derive_arch_passages(load_multiroot(xml.encode()), definitions)
    assert len(passages) == 1
    assert passages[0]["arch_id"] == 1
    assert passages[0]["distance_m"] == 50


def test_arch_at_shared_leadin_and_route_boundary_is_not_counted_twice():
    definitions = parse_arch_definitions(
        b'<ent type="ENTITY_TYPE_TIMINGARCH" m_roadId="9" m_roadTime="0.5" />')
    xml = ('<leadinhighrescheckpoint>'
           + _entry(9, 0, 1, 0) + _entry(9, 0.5, 1, 50)
           + '</leadinhighrescheckpoint><highrescheckpoint>'
           + _entry(9, 0.5, 1, 50) + _entry(9, 1, 1, 100)
           + '</highrescheckpoint>')
    passages = derive_arch_passages(load_multiroot(xml.encode()), definitions)
    assert len(passages) == 1
    assert passages[0]["leg"] == "leadin"
    assert passages[0]["distance_m"] == 50


def test_power_play_cache_has_all_six_power_arches_and_both_bridge_crossings():
    preview = event_route.build_event_route(4, 2898229208, [
        ("Event tags", ['arch_powerup="0,2,4,1,5,1,6,1,7,1,8,1,9,1,11,1"']),
    ], [])
    van_arches = [arch for arch in preview["arches"] if arch["effective_id"] in (4, 5, 6, 7, 8, 9, 11)]
    assert [arch["effective_id"] for arch in van_arches] == [4, 5, 11, 6, 7, 11, 8, 9]
    assert all(arch["awards"] == [("Event tags", "Arch override: Draft Boost (Van) (ID 1)")]
               for arch in van_arches)
    sprints = [arch for arch in preview["arches"] if arch["arch_id"] == 1]
    assert len(sprints) == 2
    assert all("Default random:" in arch["awards"][0][1] for arch in sprints)
    assert sprints[-1]["distance_km"] == pytest.approx(preview["total_km"], abs=0.0001)
    assert len(preview["segments"]) == 4


def test_grade_checkpoints_show_feathers_and_uncertainty_without_relabeling_id_74():
    preview = event_route.build_event_route(1, 3536020075, [
        ("Event tags", ['arch_powerup="0,0,74,0"', 'powerup_percent="2,95,3,5"']),
    ], [])
    checkpoints = [arch for arch in preview["arches"] if arch["entity_name"].startswith("Grade")]
    assert len(checkpoints) == 10
    assert all(arch["arch_id"] is None and arch["effective_id"] == 0 for arch in checkpoints)
    assert all("Feather / Lightweight" in arch["awards"][0][1] and arch["warning"] for arch in checkpoints)
    woodland = next(arch for arch in preview["arches"] if arch["arch_id"] == 74)
    assert woodland["name"] == "Woodland Sprint Reverse"
    assert "Feather / Lightweight" in woodland["awards"][0][1]
    finish = next(arch for arch in preview["arches"] if arch["arch_id"] == 77)
    assert "95%" in finish["awards"][0][1]
    assert "overlap" in finish["warning"]


@pytest.mark.parametrize("tags,expected", [
    (['arch_powerup="11,8"', 'powerup_percent="1,100"'], "Arch override: Anvil (ID 8)"),
    (['arch_powerup="74,0"', 'powerup_percent="1,150,5,-50"'], "150%; Aero Boost (Helmet) (ID 5): -50%"),
    (['arch_powerup="11,123"'], "Unknown powerup (ID 123)"),
    (['arch_powerup="11"'], "Unknown award: unrecognized arch override"),
    (['powerup_percent="1,nope"'], "Unrecognized distribution"),
])
def test_expected_awards_preserve_overrides_raw_values_and_invalid_configuration(tags, expected):
    assert expected in event_route._award(tags, 11)


def test_disabled_and_conflicting_sources_are_not_silently_merged():
    sources = [("Event tags", ['arch_powerup="11,8"']), ("Pen tags", ['arch_powerup="11,0"'])]
    preview = event_route.build_event_route(4, 2898229208, sources, [])
    arch = next(arch for arch in preview["arches"] if arch["arch_id"] == 11)
    assert arch["awards"] == [("Event tags", "Arch override: Anvil (ID 8)"),
                              ("Pen tags", "Arch override: Feather / Lightweight (ID 0)")]
    disabled = event_route.build_event_route(4, 2898229208, sources, ["NO_POWERUPS"])
    assert all(arch["awards"] == [("Event/pen rules", "Powerups disabled (NO_POWERUPS)")]
               and arch["warning"] is None for arch in disabled["arches"])


def test_missing_geometry_and_arch_metadata_are_explicit(monkeypatch):
    assert "valid mapId and routeId" in event_route.build_event_route(None, 1, [], [])["error"]
    monkeypatch.setattr(event_route, "get_route", lambda *args, **kwargs: None)
    assert "not available" in event_route.build_event_route(999, 1, [], [])["error"]
    event_route._geometry.cache_clear()
    monkeypatch.setattr(event_route, "get_route", lambda *args, **kwargs: {
        "name": "Below sea level", "route_offset_m": 0, "leadin": None,
        "route": {"d": [0, 100], "alt": [-50, -20], "x": [0, 100], "y": [0, 0]},
        "segments": [],
    })
    monkeypatch.setattr(event_route, "get_world_background", lambda *args: {"background": None})
    monkeypatch.setattr(event_route, "_arch_catalog", lambda: {})
    preview = event_route.build_event_route(999, 1, [], [])
    assert preview["min_altitude"] == -50
    assert preview["max_altitude"] == -20
    assert preview["paths"]["route"]["profile"] == "60.00,230.00 880.00,30.00"
    assert "not available" in preview["arch_error"]
