import json
from unittest.mock import Mock

import pytest
import requests

import app as app_module
from shared.zwift_auth import ZwiftAPIError


@pytest.fixture
def upstream(monkeypatch):
    headers = {"Authorization": "Bearer server-only"}
    monkeypatch.setattr(app_module, "get_headers", lambda: headers)
    fetch = Mock()
    monkeypatch.setattr(app_module, "_request_with_retry", fetch)
    return fetch


def event_response(payload, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


def test_inspector_page_does_not_require_rider_or_fetch(upstream):
    response = app_module.app.test_client().get("/event-inspector?event_id=5795670")
    assert response.status_code == 200
    assert b'id="event-form"' in response.data
    assert b'id="event-json"' in response.data
    upstream.assert_not_called()


def test_all_event_settings_are_preserved_with_server_only_auth(upstream):
    event = {
        "id": 5795670, "name": "<script>alert(1)</script>",
        "tags": ["starting_powerup=8"], "customUrl": "", "workoutHash": None,
        "overrideMapPreferences": False,
        "eventSubgroups": [{
            "id": 123, "rulesSet": ["NO_TT_BIKES"],
            "tags": ['powerup_percent="1,50,5,50"'],
            "unknownSetting": {"large": 9223372036854775807, "raw": -99999.5},
        }],
    }
    upstream.return_value = event_response(event)
    response = app_module.app.test_client().get("/api/events/005795670")
    assert response.status_code == 200
    assert response.get_json() == event
    assert b'9223372036854775807' in response.data
    assert b'Bearer' not in response.data
    assert response.headers["Cache-Control"] == "no-store"
    assert response.mimetype == "application/json"
    upstream.assert_called_once_with(
        "GET", f"{app_module.BASE_URL}/events/5795670",
        headers={"Authorization": "Bearer server-only"}, timeout=30)


@pytest.mark.parametrize("event_id", ["0", "-1", "abc", "1.5", "١", "1" * 21])
def test_invalid_event_ids_do_not_fetch(upstream, event_id):
    response = app_module.app.test_client().get(f"/api/events/{event_id}")
    assert response.status_code == 400
    assert "positive numeric ID" in response.get_json()["error"]
    upstream.assert_not_called()


@pytest.mark.parametrize("status,expected", [(403, 403), (404, 404), (429, 429), (401, 503), (500, 503)])
def test_event_failures_are_explicit_and_preserve_retry_after(upstream, status, expected):
    upstream.return_value = event_response({"upstream": "not exposed"}, status)
    upstream.return_value.headers["Retry-After"] = "9"
    response = app_module.app.test_client().get("/api/events/5795670")
    assert response.status_code == expected
    assert "Could not fetch event settings" in response.get_json()["error"]
    assert response.headers["Retry-After"] == "9"
    assert b"upstream" not in response.data


@pytest.mark.parametrize("payload", [None, [], {}, {"id": 1}])
def test_invalid_event_responses_are_reported(upstream, payload):
    upstream.return_value = event_response(payload)
    response = app_module.app.test_client().get("/api/events/5795670")
    assert response.status_code == 502
    assert response.get_json()["error"] == "Invalid event response from Zwift."


def test_non_json_event_response_is_reported(upstream):
    upstream.return_value = event_response(None)
    upstream.return_value._content = b"<html>not JSON</html>"
    response = app_module.app.test_client().get("/api/events/5795670")
    assert response.status_code == 502


def test_service_auth_failure_is_reported(upstream, monkeypatch):
    def unavailable():
        raise ZwiftAPIError("Zwift service account is not configured.")
    monkeypatch.setattr(app_module, "get_headers", unavailable)
    response = app_module.app.test_client().get("/api/events/5795670")
    assert response.status_code == 503
    assert "not configured" in response.get_json()["error"]
    upstream.assert_not_called()


def test_readable_pens_resolve_course_decode_settings_and_keep_original_json(upstream):
    tags = ["showplacements", "starting_powerup=8", "steering_disabled", "nobraking"]
    event = {
        "id": 5795670, "name": "DIRT Racing Series", "eventType": "RACE",
        "sport": "CYCLING", "worldId": 1, "mapId": 7, "routeId": 2905381067,
        "laps": 1, "distanceInMeters": 0.0, "durationInSeconds": 0,
        "eventStart": "2026-11-18T18:15:00.000+0000",
        "rulesSet": ["ENFORCE_NO_ZPOWER", "NO_TT_BIKES", "SHOW_RACE_RESULTS"],
        "tags": tags, "unlisted": True, "categoryEnforcement": False,
        "eventSubgroups": [
            {"id": 7481192, "subgroupLabel": "C", "name": "DIRT (C)",
             "eventSubgroupStart": "2026-11-18T18:17:00.000+0000",
             "totalEntrantCount": 52, "totalJoinedCount": 49, "signedUp": True,
             "tags": tags, "rulesSet": ["NO_TT_BIKES"]},
            {"id": 7481193, "subgroupLabel": "A", "name": "DIRT (A)",
             "eventSubgroupStart": "2026-11-18T18:15:00.000+0000",
             "totalEntrantCount": 29, "totalJoinedCount": 24, "signedUp": False},
        ],
        "unknownField": 9223372036854775807,
    }
    upstream.return_value = event_response(event)
    response = app_module.app.test_client().get("/api/events/5795670?view=readable")
    assert response.status_code == 200
    data = response.get_json()
    html = data["summary_html"]
    assert json.loads(data["raw_json"]) == event
    assert "9223372036854775807" in data["raw_json"]
    assert "Yorkshire (mapId 7)" in html
    assert "Watopia" not in html
    assert "Royal Pump Room 8 (ID 2905381067)" in html
    assert "27712.1 m" in html
    assert "0.0 m" in html
    assert "2026-11-18 18:17:00 UTC" in html
    assert html.index("Pen A</h2>") < html.index("Pen C</h2>")
    assert "Anvil (ID 8)" in html
    assert "Default for arches without a matching override: random Feather" in html
    assert "Fallback arch award distribution</dt><dd>Not specified in these tags" in html
    assert "Steering disabled" in html and "Braking disabled" in html
    assert "zPower / estimated power not allowed" in html
    assert "Service account signed up" in html
    assert "not the active rider" in html
    assert ">52</dd>" in html and ">49</dd>" in html
    assert response.headers["Cache-Control"] == "no-store"
    assert upstream.call_count == 1


def test_readable_pen_overrides_stay_distinct_and_unknown_values_are_safe(upstream):
    event = {
        "id": 1, "name": "<script>unsafe()</script>", "mapId": 7,
        "tags": ["starting_powerup=8"], "rulesSet": ["NO_POWERUPS"],
        "eventSubgroups": [{
            "id": 2, "subgroupLabel": "X", "mapId": 999, "routeId": 999999999999,
            "tags": ["starting_powerup=123", 'powerup_percent="1,150,5,-50"',
                     "unknown_tag=<script>unsafe()</script>"],
            "rulesSet": ["UNKNOWN_RULE"], "distanceInMeters": -200,
            "bikeHash": 9223372036854775807,
        }],
    }
    upstream.return_value = event_response(event)
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    html = response.get_json()["summary_html"]
    assert "Anvil (ID 8)" in html
    assert "Unknown powerup (ID 123)" in html
    assert "Draft Boost (Van) (ID 1): 150%" in html
    assert "Aero Boost (Helmet) (ID 5): -50%" in html
    assert "Unknown world (mapId 999)" in html
    assert "Unknown route (ID 999999999999)" in html
    assert "Unknown rule (UNKNOWN_RULE)" in html
    assert "-200 m" in html
    assert "9223372036854775807" in html
    assert "NO_POWERUPS is present" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "unknown_tag=" in html


@pytest.mark.parametrize("tag", ['powerup_percent="1,50,5"', 'powerup_percent="1,nope"',
                                  'powerup_percent="1,NaN"'])
def test_unrecognized_powerup_distribution_is_explicit(upstream, tag):
    upstream.return_value = event_response({"id": 1, "tags": [tag]})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    assert "Unrecognized distribution:" in html
    assert "Fallback arch award distribution</dt><dd>Not specified in these tags" not in html


def test_missing_settings_are_not_zeroes_and_disabled_powerups_are_explicit(upstream):
    upstream.return_value = event_response({"id": 1, "rulesSet": ["NO_POWERUPS"]})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    html = response.get_json()["summary_html"]
    assert "Duration setting</dt><dd>Not specified</dd>" in html
    assert "Distance setting</dt><dd>Not specified</dd>" in html
    assert "NO_POWERUPS is present" in html
    assert "Default for arches without a matching override" not in html
    assert "powerup tags are retained for inspection but do not apply" in html


@pytest.mark.parametrize("mode,expected", [
    ("CULLING_EVENT_ONLY", "Only riders in the same event, across all pens (CULLING_EVENT_ONLY)"),
    ("CULLING_SUBGROUP_ONLY", "Only riders in the same pen (CULLING_SUBGROUP_ONLY)"),
    ("UNKNOWN_MODE", "UNKNOWN_MODE"),
    (None, "Not specified"),
])
def test_readable_visibility_distinguishes_event_and_pen_scope(upstream, mode, expected):
    upstream.return_value = event_response({"id": 1, "cullingType": mode})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    assert f"Rider visibility mode</dt><dd>{expected}</dd>" in response.get_json()["summary_html"]


@pytest.mark.parametrize("event_settings,pen_settings,expected", [
    ({}, {}, True),
    ({"tags": ["starting_powerup=8"]}, {}, True),
    ({}, {"tags": ["starting_powerup=8"]}, True),
    ({"rulesSet": ["NO_POWERUPS"]}, {}, False),
    ({}, {"rulesSet": ["NO_POWERUPS"]}, False),
    ({"tags": ['powerup_percent="1,50,5,50"']}, {}, False),
    ({}, {"tags": ['powerup_percent="1,50,5,50"']}, False),
    ({}, {"tags": ['powerup_percent="unrecognized"']}, False),
    ({"tags": ['arch_powerup="0,2,4,1,11,1"']}, {}, True),
    ({}, {"tags": ['arch_powerup="74,0"']}, True),
])
def test_default_arch_powerup_note_only_when_neither_source_disables_or_sets_distribution(
    upstream, event_settings, pen_settings, expected
):
    upstream.return_value = event_response({
        "id": 1, **event_settings,
        "eventSubgroups": [{"id": 2, "subgroupLabel": "A", **pen_settings}],
    })
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    assert ("Default for arches without a matching override" in html) is expected
    if expected:
        assert "Feather, Draft Boost (Van), or Aero Boost (Helmet)" in html
        assert "Default probabilities are not specified." in html


def test_arch_overrides_decode_separately_from_starting_and_fallback_powerups(upstream):
    event = {
        "id": 1,
        "tags": ["starting_powerup=0", 'arch_powerup="0,0,74,0"',
                 'powerup_percent="2,95,3,5"'],
    }
    upstream.return_value = event_response(event)
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    data = response.get_json()
    html = data["summary_html"]
    assert "Starting powerup</dt><dd>Feather / Lightweight (ID 0)" in html
    assert "Arch 0 (including arches without an ID)</dt><dd>Feather / Lightweight (ID 0)" in html
    assert "Arch 74</dt><dd>Feather / Lightweight (ID 0)" in html
    assert "Fallback arch award distribution</dt><dd>Small XP Bonus (ID 2): 95%; Large XP Bonus (ID 3): 5%" in html
    assert "arch_powerup takes priority over powerup_percent" in html
    assert "powerup_percent applies only to arches without a matching override" in html
    assert "Default for arches without a matching override" not in html
    assert json.loads(data["raw_json"]) == event


@pytest.mark.parametrize("value", ['"0,0,74"', '""', '"<script>,0"'])
def test_malformed_arch_overrides_are_explicit_and_escaped(upstream, value):
    upstream.return_value = event_response({"id": 1, "tags": [f"arch_powerup={value}"]})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    assert "Unrecognized arch override:" in html
    assert "<script>" not in html


def test_disabled_powerups_override_starting_arch_and_distribution_tags(upstream):
    upstream.return_value = event_response({
        "id": 1, "tags": ["starting_powerup=8", 'arch_powerup="11,1"'],
        "eventSubgroups": [{
            "id": 2, "rulesSet": ["NO_POWERUPS"],
            "tags": ['arch_powerup="74,123"', 'powerup_percent="2,100"'],
        }],
    })
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    html = response.get_json()["summary_html"]
    assert "Powerups disabled: NO_POWERUPS is present" in html
    assert "powerup tags are retained for inspection but do not apply" in html
    assert "Arch 11</dt><dd>Draft Boost (Van) (ID 1)" in html
    assert "Arch 74</dt><dd>Unknown powerup (ID 123)" in html
    assert "Event tags</h4>" in html and "Pen tags</h4>" in html
    assert "Default for arches without a matching override" not in html


@pytest.mark.parametrize("identical", [True, False])
def test_arch_sources_are_not_repeated_when_event_and_pen_tags_match(upstream, identical):
    tags = ['arch_powerup="11,1"']
    upstream.return_value = event_response({
        "id": 1, "mapId": 4, "routeId": 2898229208, "tags": tags,
        "eventSubgroups": [{
            "id": 2, "subgroupLabel": "A",
            "tags": tags if identical else ['arch_powerup="11,0"'],
        }],
    })
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    route_html = html.split('<section class="event-route"', 1)[1]
    assert ("Event and pen powerup tags are identical;" in route_html) is identical
    assert "Event and pen tags (identical):" not in route_html
    assert "Arch override: Draft" in route_html
    if identical:
        assert "<strong>" not in route_html
    else:
        assert "<strong>Event tags:</strong>" in route_html
        assert "<strong>Pen tags:</strong>" in route_html
        assert "Arch override: Feather" in route_html


@pytest.mark.parametrize("map_id,route_id,expected_count", [(4, 1178708985, 4), (7, 2905381067, 0)])
def test_event_route_escalator_overlays_and_details(upstream, map_id, route_id, expected_count):
    upstream.return_value = event_response({"id": 1, "mapId": map_id, "routeId": route_id})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    assert html.count('class="escalator-zone"') == expected_count
    if expected_count:
        assert f"Escalators ({expected_count})" in html
        assert "trainer slope: 20%" in html
        assert "SubwayC_Forward" in html
        assert '<polyline class="escalator-line"' in html
    else:
        assert "Escalators (" not in html
        assert 'class="escalator-key"' not in html


@pytest.mark.parametrize("identical", [True, False])
def test_powerup_section_merges_matching_event_and_pen_tags(upstream, identical):
    tags = ["starting_powerup=0", 'arch_powerup="74,0"']
    upstream.return_value = event_response({
        "id": 1, "tags": tags, "eventSubgroups": [{
            "id": 2, "subgroupLabel": "A",
            "tags": tags if identical else ["starting_powerup=8"],
        }],
    })
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    html = response.get_json()["summary_html"]
    section = html.split('<section class="powerup-settings">', 1)[1].split("</section>", 1)[0]
    assert ("Event and pen powerup tags are identical; these powerup settings apply to both." in section) is identical
    assert section.count("Starting powerup</dt>") == (1 if identical else 2)
    if identical:
        assert "<h4>" not in section
    else:
        assert "<h4>Event tags</h4>" in section
        assert "<h4>Pen tags</h4>" in section
        assert "Anvil (ID 8)" in section


@pytest.mark.parametrize("pen_powerups,merged", [
    (['arch_powerup="0,2,4,1,11,1"', "starting_powerup=0"], True),
    (['arch_powerup="0,2,4,1,11,1"', "starting_powerup=8"], False),
    (['arch_powerup="0,2,4,1,11,1"'], False),
])
def test_powerup_merging_ignores_unrelated_tags_and_tag_order(upstream, pen_powerups, merged):
    upstream.return_value = event_response({
        "id": 5781526, "mapId": 4, "routeId": 2898229208,
        "tags": ["starting_powerup=0", 'arch_powerup="0,2,4,1,11,1"', "timestamp=1794850009577"],
        "eventSubgroups": [{
            "id": 7463519, "subgroupLabel": "A",
            "tags": ["timestamp=1794850018998", "enable_power_segments", *pen_powerups],
        }],
    })
    response = app_module.app.test_client().get("/api/events/5781526?view=readable")
    assert response.status_code == 200
    data = response.get_json()
    html = data["summary_html"]
    section = html.split('<section class="powerup-settings">', 1)[1].split("</section>", 1)[0]
    route_html = html.split('<section class="event-route"', 1)[1].split("</table>", 1)[0]
    assert ("Event and pen powerup tags are identical;" in section) is merged
    assert ("Event and pen powerup tags are identical;" in route_html) is merged
    assert section.count("Starting powerup</dt>") == (1 if merged else 2)
    if merged:
        assert "<strong>Event tags:</strong>" not in route_html
        assert "<strong>Pen tags:</strong>" not in route_html
    else:
        assert "<strong>Event tags:</strong>" in route_html
        assert "<strong>Pen tags:</strong>" in route_html
    assert "timestamp=1794850009577" in html
    assert "timestamp=1794850018998" in html
    assert "<h3>Event tags</h3>" in html and "<h3>Pen tags</h3>" in html
    assert json.loads(data["raw_json"])["tags"][-1] == "timestamp=1794850009577"


def test_readable_pens_render_tabs_and_individual_routes_with_arch_awards(upstream):
    event = {
        "id": 1, "mapId": 1, "routeId": 3536020075,
        "eventSubgroups": [
            {"id": 2, "subgroupLabel": "A", "tags": ['arch_powerup="0,0,74,0"']},
            {"id": 3, "subgroupLabel": "B", "mapId": 4, "routeId": 2898229208,
             "tags": ['arch_powerup="11,1"']},
        ],
    }
    upstream.return_value = event_response(event)
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    assert response.status_code == 200
    data = response.get_json()
    html = data["summary_html"]
    assert 'role="tablist" aria-label="Event pens"' in html
    assert 'role="tabpanel" aria-labelledby="pen-tab-1" tabindex="0">' in html
    assert 'role="tabpanel" aria-labelledby="pen-tab-2" tabindex="0" hidden>' in html
    assert "Route preview: Three Step Sisters" in html
    assert "Route preview: Watts the Limit" in html
    assert "Woodland Sprint Reverse" in html and "Grade01" in html
    assert "Whether the event configuration enables an award is uncertain" in html
    assert "Arch override: Feather / Lightweight" in html
    assert "Arch override: Draft Boost (Van)" in html
    assert 'id="pen-1-arch-1"' in html and 'id="pen-2-arch-1"' in html
    assert 'class="route-map"' in html and 'class="route-profile"' in html
    assert "not expanded or shortened for event laps, distance, or time limits" in html
    assert json.loads(data["raw_json"]) == event
