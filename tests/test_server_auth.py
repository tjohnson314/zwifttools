import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import numpy as np
import pandas as pd

import app as app_module
from shared import data_fetcher, zwift_auth


def response(status=200, payload=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result.headers.update(headers or {})
    return result


def tokens(name="server"):
    return zwift_auth.ZwiftTokens(name, "refresh", time.time() + 3600)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Unexpected network request")
    monkeypatch.setattr(requests, "request", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests, "get", blocked)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("ZWIFT_USERNAME", "server@example.invalid")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-only")
    manager = zwift_auth.ServerTokenManager()
    monkeypatch.setattr(manager, "wait", Mock())
    monkeypatch.setattr(app_module, "server_token_manager", manager)
    login = Mock(return_value=tokens())
    monkeypatch.setattr(zwift_auth, "get_token_with_password", login)
    return manager, login


def test_server_headers_outside_request_and_legacy_cookie_cleared(service, monkeypatch):
    manager, login = service
    assert app_module.get_headers()["Authorization"] == "Bearer server"
    monkeypatch.setattr(requests, "request", Mock(return_value=response(200, {
        "id": 7, "firstName": "Public", "lastName": "Rider",
        "height": 1800, "weight": 75000, "ftp": 275,
    })))
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["tokens"] = {"access_token": "visitor-secret"}
        session["oauth_state"] = "legacy"
    result = client.get("/api/profiles/7")
    assert result.status_code == 200
    assert result.get_json() == {
        "id": "7", "name": "Public Rider", "height_cm": 180, "weight_kg": 75, "ftp_w": 275,
    }
    assert b"server" not in result.data and b"visitor-secret" not in result.data
    with client.session_transaction() as session:
        assert "tokens" not in session and "oauth_state" not in session
    login.assert_called_once_with("server@example.invalid", "test-only")
    assert manager._tokens.access_token == "server"


@pytest.mark.parametrize("path", ["/api/profiles/7", "/api/my_profile?zwift_id=7"])
@pytest.mark.parametrize("measurements,expected", [
    ({"heightInCentimeters": 171, "weightInGrams": 67890}, (171, 67.9)),
    ({"heightInCentimeters": "171.5", "weightInGrams": "67890"}, (171.5, 67.9)),
    ({"heightInMillimeters": "1800", "weight": "75000"}, (180, 75)),
    ({"height": 1800, "heightInCentimeters": 171}, (180, None)),
    ({"height": None, "heightInCentimeters": 0, "weight": 0}, (0, 0)),
    ({"heightInCentimeters": -10, "weightInGrams": 9000000}, (-10, 9000)),
])
def test_profile_converts_available_height_and_weight_fields(service, monkeypatch, path, measurements, expected):
    profile = {"id": 7, "firstName": "Public", **measurements}
    monkeypatch.setattr(requests, "request", Mock(return_value=response(200, profile)))
    result = app_module.app.test_client().get(path)
    assert result.status_code == 200
    data = result.get_json()
    assert (data["height_cm"], data["weight_kg"]) == expected


@pytest.mark.parametrize("measurements", [
    {"heightInCentimeters": "invalid"},
    {"heightInCentimeters": "NaN"},
    {"weightInGrams": "Infinity"},
])
def test_malformed_profile_measurements_are_reported(service, monkeypatch, measurements):
    profile = {"id": 7, "firstName": "Public", **measurements}
    monkeypatch.setattr(requests, "request", Mock(return_value=response(200, profile)))
    result = app_module.app.test_client().get("/api/profiles/7")
    assert result.status_code == 502
    assert result.get_json()["error"] == "Invalid profile measurements from Zwift."


@pytest.mark.parametrize("path", [
    "/api/my_profile", "/api/my_activities", "/api/power_dashboard",
    "/api/power_dashboard/activities", "/api/segment_prs/activities",
    "/api/segment_prs/activity/11", "/api/power_dashboard/activity/11",
])
@pytest.mark.parametrize("suffix", ["", "?zwift_id=0", "?zwift_id=-1", "?zwift_id=abc",
                                   "?zwift_id=1.5", "?zwift_id=%D9%A1",
                                   "?zwift_id=111111111111111111111"])
def test_rider_scoped_requests_require_valid_explicit_id(path, suffix):
    assert app_module.app.test_client().get(path + suffix).status_code == 400


@pytest.mark.parametrize("value", ["0", "-1", "me", "1.5", "1e3", "١"])
def test_profile_path_rejects_non_numeric_positive_ids(value):
    assert app_module.app.test_client().get(f"/api/profiles/{value}").status_code == 400


@pytest.mark.parametrize("status", [403, 404, 429])
def test_profile_and_activity_errors_preserve_upstream_status(service, monkeypatch, status):
    monkeypatch.setattr(requests, "request", Mock(return_value=response(
        status, {}, {"Retry-After": "9"})))
    client = app_module.app.test_client()
    for path in ("/api/profiles/7", "/api/my_profile?zwift_id=7",
                 "/api/power_dashboard/activity/11?zwift_id=7",
                 "/api/workout_activity/11"):
        result = client.get(path)
        assert result.status_code == status
        if status == 429:
            assert result.headers["Retry-After"] == "9"


@pytest.mark.parametrize("payload", [None, [], {}, {"id": 8}])
def test_missing_or_malformed_profile_is_not_a_valid_selection(service, monkeypatch, payload):
    monkeypatch.setattr(requests, "request", Mock(return_value=response(200, payload)))
    assert app_module.app.test_client().get("/api/profiles/7").status_code == 502


def test_selected_identity_drives_all_lists_and_aggregate(service, monkeypatch):
    urls = []
    def upstream(method, url, **kwargs):
        urls.append(url)
        assert kwargs["headers"]["Authorization"] == "Bearer server"
        return response(200, [] if url.endswith("/activities") else {"id": 7})
    monkeypatch.setattr(requests, "request", upstream)
    client = app_module.app.test_client()
    for path in ("/api/my_activities", "/api/power_dashboard/activities",
                 "/api/segment_prs/activities", "/api/power_dashboard"):
        assert client.get(path + "?zwift_id=7").status_code == 200
    assert all("/profiles/7" in url for url in urls)
    assert sum(url.endswith("/activities") for url in urls) == 4


def test_list_errors_are_not_empty_success(service, monkeypatch):
    def upstream(method, url, **kwargs):
        return response(403, {}) if url.endswith("/activities") else response(200, {"id": 7})
    monkeypatch.setattr(requests, "request", upstream)
    client = app_module.app.test_client()
    for path in ("/api/my_activities", "/api/power_dashboard/activities",
                 "/api/segment_prs/activities", "/api/power_dashboard"):
        assert client.get(path + "?zwift_id=7").status_code == 403


def test_invalid_activity_list_is_not_empty_success(service, monkeypatch):
    def upstream(method, url, **kwargs):
        return response(200, {} if url.endswith("/activities") else {"id": 7})
    monkeypatch.setattr(requests, "request", upstream)
    client = app_module.app.test_client()
    for path in ("/api/my_activities", "/api/power_dashboard/activities",
                 "/api/segment_prs/activities", "/api/power_dashboard"):
        assert client.get(path + "?zwift_id=7").status_code == 502


@pytest.mark.parametrize("status", [401, 503])
def test_upstream_auth_and_service_failures_are_explicit_503(service, monkeypatch, status):
    monkeypatch.setattr(zwift_auth, "refresh_access_token", Mock(return_value=tokens("new")))
    monkeypatch.setattr(data_fetcher._time, "sleep", Mock())
    upstream = Mock(return_value=response(status))
    monkeypatch.setattr(requests, "request", upstream)
    result = app_module.app.test_client().get("/api/profiles/7")
    assert result.status_code == 503
    assert upstream.call_count == (2 if status == 401 else 3)


@pytest.mark.parametrize("path", ["/api/power_dashboard/activity/11", "/api/segment_prs/activity/11"])
def test_selected_rider_must_own_activity(service, monkeypatch, path):
    monkeypatch.setattr(requests, "request", Mock(return_value=response(200, {"id": 7})))
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", Mock(return_value=(
        {"powerInWatts": [100]}, {"profileId": 8}, None)))
    result = app_module.app.test_client().get(path + "?zwift_id=7")
    assert result.status_code == 403


def test_workout_activity_returns_actual_owner_not_selected_rider(service, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", Mock(return_value=(
        {"timeInSec": [0, 1], "powerInWatts": [-100, 9000]},
        {"profileId": 8, "name": "Ride"}, None)))
    result = app_module.app.test_client().get("/api/workout_activity/11?zwift_id=7")
    assert result.status_code == 200
    assert result.get_json()["profile_id"] == "8"
    assert result.get_json()["power_watts"] == [-100, 9000]


def test_power_no_power_is_distinct_from_failed_fetch(service, monkeypatch):
    fetch = Mock(return_value=({}, {"profileId": 7}, None))
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", fetch)
    client = app_module.app.test_client()
    path = "/api/power_dashboard/activity/11?zwift_id=7"
    assert client.get(path).get_json()["has_power"] is False
    fetch.return_value = (None, None, "Telemetry error: upstream unavailable")
    result = client.get(path)
    assert result.status_code == 502
    assert "has_power" not in result.get_json()


@pytest.mark.parametrize("telemetry,error,available", [
    ({"samples": True}, None, True),
    (None, "Telemetry error: upstream unavailable", False),
])
def test_activity_enrichment_marks_telemetry_availability(service, monkeypatch, telemetry, error, available):
    monkeypatch.setattr(app_module, "get_activity_details", lambda *args: (
        {"eventInfo": {"eventSubGroupId": 2, "id": 3}, "routeId": 1}, None))
    monkeypatch.setattr(app_module, "get_route_info", lambda *args: None)
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (telemetry, {}, error))
    monkeypatch.setattr(app_module, "convert_telemetry_to_dataframe", lambda data: pd.DataFrame({
        "power_watts": [100, 200], "hr_bpm": [0, 150],
    }))
    result = app_module.app.test_client().get("/api/activity_enrichment/11")
    assert result.status_code == 200
    data = result.get_json()
    assert data["telemetry_available"] is available
    assert data["is_race"] is True
    assert data["event_id"] == 3
    assert data["avg_hr"] == (150 if available else None)


def test_captured_headers_renew_expired_tokens_and_retry_401_once(service, monkeypatch):
    manager, _ = service
    captured = app_module.get_headers()
    manager._tokens.expires_at = 0
    refresh = Mock(side_effect=[tokens("renewed"), tokens("after401")])
    monkeypatch.setattr(zwift_auth, "refresh_access_token", refresh)
    upstream = Mock(side_effect=[response(401), response(200, {"id": 7})])
    monkeypatch.setattr(requests, "request", upstream)
    result = data_fetcher._request_with_retry("GET", "https://example.invalid", headers=captured)
    assert result.status_code == 200
    assert refresh.call_count == 2
    assert manager.wait.call_count == 2
    assert captured["Authorization"] == "Bearer server"
    upstream.side_effect = None
    upstream.return_value = response(401)
    refresh.side_effect = None
    refresh.return_value = tokens("rejected")
    with pytest.raises(zwift_auth.ZwiftAPIError, match="rejected"):
        data_fetcher._request_with_retry("GET", "https://example.invalid", headers=captured)
    assert upstream.call_count == 4


def test_concurrent_initial_auth_and_rejected_token_refresh_are_single_flight(service, monkeypatch):
    manager, login = service
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: manager.access_token(), range(16))) == ["server"] * 16
    assert login.call_count == 1
    refresh = Mock(return_value=tokens("new"))
    monkeypatch.setattr(zwift_auth, "refresh_access_token", refresh)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: manager.access_token(rejected_token="server"), range(16)))
    assert values == ["new"] * 16
    assert refresh.call_count == 1


def test_failed_auth_has_shared_cooldown_without_exposing_credentials(service):
    manager, login = service
    login.side_effect = requests.ConnectionError("sensitive upstream message")
    for _ in range(3):
        with pytest.raises(zwift_auth.ZwiftAPIError, match="service authentication failed"):
            manager.access_token()
    assert login.call_count == 1


def test_token_rate_limit_is_honored(service):
    manager, login = service
    login.side_effect = requests.HTTPError(response=response(429, {}, {"Retry-After": "120"}))
    with pytest.raises(zwift_auth.ZwiftAPIError) as error:
        manager.access_token()
    assert error.value.status_code == 429
    assert error.value.retry_after == "120"
    assert manager._retry_at > time.monotonic() + 100


def test_plain_test_headers_do_not_use_server_auth(monkeypatch):
    upstream = Mock(return_value=response(401))
    monkeypatch.setattr(requests, "request", upstream)
    result = data_fetcher._request_with_retry(
        "GET", "https://example.invalid", headers={"Authorization": "test"})
    assert result.status_code == 401
    assert upstream.call_count == 1


def test_unconfigured_service_is_explicit_but_manual_pages_stay_public(monkeypatch):
    monkeypatch.delenv("ZWIFT_USERNAME", raising=False)
    monkeypatch.delenv("ZWIFT_PASSWORD", raising=False)
    client = app_module.app.test_client()
    assert client.get("/api/profiles/7").status_code == 503
    assert app_module.get_headers(required=False) is None
    for path in ("/power", "/segment-prs", "/my-activities", "/workout-analysis", "/workout-route"):
        assert client.get(path).status_code == 200
    assert client.post("/api/workouts/upload").status_code == 400
    assert client.post("/api/workout_route_plan", json={}).status_code == 400


@pytest.mark.parametrize("path", [
    "/", "/?tab=archived", "/bike-comparison", "/critical-power", "/race-replay",
    "/dirt", "/zrl/leaderboard", "/surface-map", "/surface-map-dev", "/top-performers",
    "/my-activities", "/workout-analysis", "/workout-route", "/power", "/segment-prs",
    "/tt-pacing", "/ttt-pacing", "/ride-compare", "/ttt-analysis",
])
def test_all_tool_pages_share_one_public_rider_selector(monkeypatch, path):
    monkeypatch.delenv("ZWIFT_USERNAME", raising=False)
    monkeypatch.delenv("ZWIFT_PASSWORD", raising=False)
    monkeypatch.setattr(app_module, "_surface_map_dev_available", lambda: True)
    monkeypatch.setattr(app_module, "get_route_info", lambda route_id: None)
    monkeypatch.setattr(app_module.server_token_manager, "headers",
                        Mock(side_effect=AssertionError("Page attempted service authentication")))
    client = app_module.app.test_client()
    result = client.get(path, base_url="http://localhost")
    assert result.status_code == 200
    html = result.get_data(as_text=True)
    assert html.count('id="activeRiderForm"') == 1
    assert not re.search(r'type\s*=\s*["\']?password', html, re.IGNORECASE)
    assert not re.search(r'href\s*=\s*["\']/auth/login', html, re.IGNORECASE)
    assert client.get_cookie(app_module.app.config["SESSION_COOKIE_NAME"]) is None


@pytest.mark.parametrize("subgroup_id", ["", "123"])
def test_manual_csv_works_without_server_auth(monkeypatch, subgroup_id):
    monkeypatch.delenv("ZWIFT_USERNAME", raising=False)
    monkeypatch.delenv("ZWIFT_PASSWORD", raising=False)
    header = "athlete_id,world_time,power,speed,altitude,distance,draft,lat,lng\n"
    rows = [f"42,{second * 1000},250,36,10,{second * 10},30,0.1,0.2"
            for second in range(10)]
    result = app_module.app.test_client().post("/api/ttt/fetch_csv", data={
        "csv_file": (io.BytesIO((header + "\n".join(rows)).encode()), "telemetry.csv"),
        "subgroup_id": subgroup_id,
    })
    assert result.status_code == 200
    app_module._ttt_cache.pop(result.get_json()["cache_key"], None)


@pytest.mark.parametrize("failure", [
    "entries", "entries_error", "rate_limit", "activity", "event", "profile", "unconfigured",
])
def test_csv_analysis_survives_unavailable_optional_metadata(monkeypatch, failure):
    headers = {"Authorization": "test"} if failure != "unconfigured" else None
    monkeypatch.setattr(app_module, "get_headers", lambda required=True: headers)
    participant = {"activity_id": "11", "player_id": 42, "name": "Public",
                   "weight_kg": 68, "height_cm": 175, "weight_is_event_recorded": True}
    error = zwift_auth.ZwiftAPIError("Metadata unavailable", 429 if failure == "rate_limit" else 403)
    entries = Mock(
        side_effect=error if failure in ("entries", "rate_limit") else None,
        return_value=(None, "Unavailable") if failure == "entries_error" else ([participant], None))
    monkeypatch.setattr(app_module, "get_race_entries", entries)
    monkeypatch.setattr(app_module, "get_activity_details", Mock(
        side_effect=error if failure == "activity" else None,
        return_value=({"eventInfo": {"id": 8}}, None)))
    def request(method, url, **kwargs):
        if failure == "event":
            raise error
        if url.endswith("/events/8"):
            return response(200, {"name": "Public TTT", "eventSubgroups": [{"id": 123}]})
        if failure == "profile":
            raise error
        return response(200, {"id": 42, "firstName": "Public",
                              "weightInGrams": 68000, "heightInCentimeters": 175})
    request_mock = Mock(side_effect=request)
    monkeypatch.setattr(app_module, "_request_with_retry", request_mock)
    if failure == "profile":
        participant["player_id"] = 99
    monkeypatch.setattr(app_module, "_ttt_cache", {})
    header = "athlete_id,world_time,power,speed,altitude,distance,draft,lat,lng\n"
    rows = [f"42,{second * 1000},250,36,10,{second * 10},30,0.1,0.2"
            for second in range(10)]
    client = app_module.app.test_client()
    result = client.post("/api/ttt/fetch_csv", data={
        "subgroup_id": "123",
        "csv_file": (io.BytesIO((header + "\n".join(rows)).encode()), "telemetry.csv"),
    })
    assert result.status_code == 200
    data = result.get_json()
    assert data["warnings"]
    if failure in ("entries", "entries_error", "rate_limit", "activity", "event", "unconfigured"):
        assert any("Event metadata unavailable" in warning for warning in data["warnings"])
    if failure in ("rate_limit", "profile", "unconfigured"):
        assert any("default weight" in warning for warning in data["warnings"])
    assert entries.call_count <= 1
    if failure in ("rate_limit", "unconfigured"):
        request_mock.assert_not_called()
    rider = app_module._ttt_cache[data["cache_key"]]["all_processed"]["42"]
    assert rider["power"].tolist() == [250] * 10
    reassigned = client.post("/api/ttt/reassign", json={
        "subgroup_id": data["cache_key"], "assignments": {"42": "Team"},
    })
    assert reassigned.get_json()["warnings"] == data["warnings"]


def test_csv_analysis_uses_optional_metadata_when_available(monkeypatch):
    from shared import route_lookup
    from race_replay import data_cleaner

    monkeypatch.setattr(app_module, "get_headers", lambda required=True: {"Authorization": "test"})
    participant = {"activity_id": "11", "player_id": 42, "name": "Public Rider",
                   "weight_kg": "68", "height_cm": "175", "weight_is_event_recorded": True}
    monkeypatch.setattr(app_module, "get_race_entries", lambda *args: ([participant], None))
    monkeypatch.setattr(app_module, "get_activity_details", lambda *args: ({"eventInfo": {"id": 8}}, None))
    monkeypatch.setattr(app_module, "_request_with_retry", lambda *args, **kwargs: response(
        200, {"name": "Public TTT", "eventSubgroups": [{"id": 123, "routeId": 10}]}))
    monkeypatch.setattr(route_lookup, "get_route_info", lambda route_id: {
        "name": "Test Route", "distanceInMeters": 1000,
    })
    monkeypatch.setattr(data_cleaner, "load_route_data", lambda slug: None)
    monkeypatch.setattr(app_module, "_ttt_cache", {})
    header = "athlete_id,world_time,power,speed,altitude,distance,draft,lat,lng\n"
    rows = [f"42,{second * 1000},250,36,10,{second * 10},30,0.1,0.2"
            for second in range(10)]
    result = app_module.app.test_client().post("/api/ttt/fetch_csv", data={
        "subgroup_id": "123",
        "csv_file": (io.BytesIO((header + "\n".join(rows)).encode()), "telemetry.csv"),
    })
    assert result.status_code == 200
    data = result.get_json()
    assert data["warnings"] == []
    assert data["event_name"] == "Public TTT"
    rider = app_module._ttt_cache[data["cache_key"]]["all_processed"]["42"]
    assert (rider["name"], rider["weight_kg"], rider["height_cm"]) == ("Public Rider", 68, 175)


def test_retired_auth_never_reads_visitor_password(monkeypatch):
    def forbidden(*args):
        pytest.fail("Visitor form was accessed")
    monkeypatch.setattr(app_module.app.request_class, "form", property(forbidden))
    client = app_module.app.test_client()
    assert client.post("/auth/login", data="password=visitor").status_code == 410
    assert client.get("/auth/login?next=https://evil.invalid").location == "/"
    assert client.get("/auth/callback?code=anything").status_code == 410
    with client.session_transaction() as session:
        session["tokens"] = {"access_token": "old"}
        session["oauth_state"] = "old-state"
    assert client.get("/auth/logout").location == "/"
    with client.session_transaction() as session:
        assert not session


def test_fit_upload_and_derived_outputs_are_never_cacheable(monkeypatch, caplog):
    records = [{"timestamp": index, "power": 9000, "speed": 10, "distance": index * 10,
                "altitude": 0} for index in range(8)]
    monkeypatch.setattr(app_module, "HAS_FIT_SDK", True)
    monkeypatch.setattr(app_module, "Stream", SimpleNamespace(from_byte_array=lambda value: value),
                        raising=False)
    monkeypatch.setattr(app_module, "Decoder", lambda value: SimpleNamespace(
        read=lambda **kwargs: ({"record_mesgs": records}, [])), raising=False)
    client = app_module.app.test_client()
    result = client.post("/api/upload_fit", data={
        "file": (io.BytesIO(b"FIT-private-test-payload"), "ride.fit")})
    # The invalid header must not be logged, persisted, or cached.
    assert result.status_code == 400
    assert result.headers["Cache-Control"] == "no-store"
    result = client.post("/api/upload_fit", data={
        "file": (io.BytesIO(b"\0" * 8 + b".FIT"), "ride.fit")})
    assert result.status_code == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert result.get_json()["telemetry"]["power_watts"] == [9000] * 8
    monkeypatch.setattr(app_module, "get_bike_stats", lambda *args: None)
    result = client.post("/api/compare", json={
        "actual": {"frame_id": "missing"}, "alternative": {"frame_id": "missing"},
        "telemetry": {"power_watts": [9000]},
    })
    assert result.status_code == 400
    assert result.headers["Cache-Control"] == "no-store"
    result = client.post("/api/best_bikes", json={})
    assert result.headers["Cache-Control"] == "no-store"
    assert "FIT-private-test-payload" not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [403, 404])
def test_zrl_skips_inaccessible_rider_and_keeps_other_results(monkeypatch, stream, status):
    from shared import zrl_leaderboard, wtrl_teams

    participants = [{"activity_id": "1", "name": "Public"}, {"activity_id": "2", "name": "Private"}]
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "get_race_entries", lambda *args: (participants, None))
    def fetch(activity_id, headers):
        if activity_id == "2":
            raise zwift_auth.ZwiftAPIError("Unavailable", status)
        return {"powerInWatts": [200]}, {}, None
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", fetch)
    monkeypatch.setattr(app_module, "convert_telemetry_to_dataframe", lambda data: data)
    build = Mock(return_value={"summary": [], "segment_geometry_available": True})
    monkeypatch.setattr(zrl_leaderboard, "build_leaderboard", build)
    monkeypatch.setattr(wtrl_teams, "add_team_scores", lambda *args: None)
    suffix = "_stream" if stream else ""
    result = app_module.app.test_client().get(f"/api/zrl/leaderboard{suffix}?subgroup_id=1")
    assert result.status_code == 200
    payload = (json.loads(result.get_data(as_text=True).strip().split("data: ")[-1])
               if stream else result.get_json())
    assert payload["participants_total"] == 2
    assert payload["telemetry_riders"] == 1
    assert list(build.call_args.args[1]) == ["1"]


@pytest.mark.parametrize("stream", [False, True])
def test_zrl_reports_service_failures_instead_of_skipping(monkeypatch, stream):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "get_race_entries", lambda *args: ([{"activity_id": "1"}], None))
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", Mock(
        side_effect=zwift_auth.ZwiftAPIError("Rate limited", 429)))
    suffix = "_stream" if stream else ""
    result = app_module.app.test_client().get(f"/api/zrl/leaderboard{suffix}?subgroup_id=1")
    if stream:
        assert json.loads(result.get_data(as_text=True).strip().split("data: ")[-1])["error"] == "Rate limited"
    else:
        assert result.status_code == 429


def test_ride_compare_skips_private_and_retries_partial_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    telemetry = {"powerInWatts": [100, 200]}
    activity = {"fitnessData": {}, "profile": {"firstName": "Public"}}
    fetch = Mock(side_effect=[
        zwift_auth.ZwiftAPIError("Private", 403), (telemetry, activity, None),
        (telemetry, activity, None), (telemetry, activity, None),
    ])
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", fetch)
    frame = pd.DataFrame({"time_sec": [0, 1], "power_watts": [100, 200],
                          "hr_bpm": [100, 101]})
    monkeypatch.setattr(app_module, "convert_telemetry_to_dataframe", lambda data: frame)
    monkeypatch.setattr(app_module, "save_rider_data", lambda *args: None)
    client = app_module.app.test_client()
    def fetch_result():
        response = client.get("/api/ride_compare/fetch_stream?activity_ids=1,2")
        return json.loads(response.get_data(as_text=True).strip().split("data: ")[-1])
    first = fetch_result()
    assert first["success"] and "1 unavailable" in first["message"]
    second = fetch_result()
    assert second["success"] and "2 rides" in second["message"]
    assert fetch.call_count == 4


@pytest.mark.parametrize("status", [403, 429])
def test_ride_compare_reports_all_unavailable_or_service_failure(monkeypatch, tmp_path, status):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", Mock(
        side_effect=zwift_auth.ZwiftAPIError("Rate limited" if status == 429 else "Private", status)))
    response = app_module.app.test_client().get("/api/ride_compare/fetch_stream?activity_ids=1,2")
    result = json.loads(response.get_data(as_text=True).strip().split("data: ")[-1])
    assert result["error"] == ("Rate limited" if status == 429 else "No accessible ride telemetry could be fetched.")
    assert not list(tmp_path.glob("race_data/*/complete_race_summary.csv"))


def test_ttt_skips_private_riders_including_first_and_manual(monkeypatch):
    participants = [
        {"activity_id": "1", "name": "Private [A]", "weight_kg": 75},
        {"activity_id": "2", "name": "Public [A]", "weight_kg": 75},
    ]
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "get_race_entries", lambda *args: (participants, None))
    def details(activity_id, headers):
        if activity_id in ("1", "3"):
            raise zwift_auth.ZwiftAPIError("Private", 403)
        return {"eventInfo": {}}, None
    monkeypatch.setattr(app_module, "get_activity_details", details)
    def fetch(activity_id, headers):
        if activity_id == "1":
            raise zwift_auth.ZwiftAPIError("Private", 403)
        return {"timeInSec": list(range(12))}, {}, None
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", fetch)
    frame = pd.DataFrame({
        "time_sec": np.arange(12), "speed_kmh": np.full(12, 36),
        "distance_km": np.arange(12) / 100, "altitude_m": np.zeros(12),
        "power_watts": np.full(12, 200),
    })
    monkeypatch.setattr(app_module, "convert_telemetry_to_dataframe", lambda data: frame)
    monkeypatch.setattr(app_module, "get_bike_stats", lambda *args: SimpleNamespace(
        cda_bias=0, weight_kg=10))
    monkeypatch.setattr(app_module, "_build_ttt_team_results", lambda processed, assignments, tags: [])
    monkeypatch.setattr(app_module, "_ttt_cache", {})
    response = app_module.app.test_client().post("/api/ttt/fetch", json={
        "subgroup_id": 1, "team_tags": "A", "manual_activities": "3",
    })
    assert response.status_code == 200
    assert "2 of 3 riders" in response.get_json()["warnings"][0]
    assert list(app_module._ttt_cache[1]["all_processed"]) == ["2"]
    reassigned = app_module.app.test_client().post("/api/ttt/reassign", json={
        "subgroup_id": 1, "assignments": {},
    })
    assert reassigned.status_code == 200
    assert "2 of 3 riders" in reassigned.get_json()["warnings"][0]
