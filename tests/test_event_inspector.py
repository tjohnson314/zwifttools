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
    assert "random arch awards are not ruled out" in html
    assert "Arch award distribution</dt><dd>Not specified in these tags" in html
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
    assert "Arch award distribution</dt><dd>Not specified in these tags" not in html


def test_missing_settings_are_not_zeroes_and_disabled_powerups_are_explicit(upstream):
    upstream.return_value = event_response({"id": 1, "rulesSet": ["NO_POWERUPS"]})
    response = app_module.app.test_client().get("/api/events/1?view=readable")
    html = response.get_json()["summary_html"]
    assert "Duration setting</dt><dd>Not specified</dd>" in html
    assert "Distance setting</dt><dd>Not specified</dd>" in html
    assert "NO_POWERUPS is present" in html
    assert "random arch awards are not ruled out" not in html
