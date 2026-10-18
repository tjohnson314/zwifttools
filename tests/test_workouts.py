import json
import math

import pytest
import requests

import app as app_module
from shared import workouts


def definition(steps, sport="bike"):
    return f"<workout_file><name>Test</name><sportType>{sport}</sportType><workout>{steps}</workout></workout_file>".encode()


def test_parser_expands_repeats_ranges_ramps_and_free_ride():
    result = workouts.parse_workout("Workouts/test.xml", definition("""
        <Warmup Duration="10" PowerLow=".3" PowerHigh=".8"/>
        <IntervalsT Repeat="2" OnDuration="5" OffDuration="3"
            PowerOnLow="1.1" PowerOnHigh="1.2" PowerOffLow=".5" PowerOffHigh=".6"/>
        <Cooldown Duration="4" PowerLow=".7" PowerHigh=".3"/>
        <FreeRide Duration="2" Target="1.4"/>
        <MaxEffort Duration="1"/>
    """))
    assert result["duration_sec"] == 33
    assert [segment["start_sec"] for segment in result["segments"]] == [0, 10, 15, 18, 23, 26, 30, 32]
    assert result["segments"][0]["ramp"]
    assert result["segments"][1]["power_low"] == 1.1
    assert result["segments"][1]["power_high"] == 1.2
    assert not result["segments"][1]["ramp"]
    assert result["segments"][5]["power_high"] == .3
    assert result["segments"][6]["power_low"] is None
    assert result["segments"][7]["power_low"] is None


def test_parser_handles_aliases_and_does_not_guess_zone_power():
    result = workouts.parse_workout("test.xml", definition("""
        <SolidState Duration="5" Power="-.1"/>
        <cooldown Duration="5" Power=".2"/>
        <SteadyState Duration="5" Zone="2"/>
        <textevent message="Do not export coaching text"/>
    """, "Bike"))
    assert result["segments"][0]["power_low"] == -.1
    assert result["segments"][1]["ramp"]
    assert result["segments"][2]["power_low"] is None
    assert result["warnings"]
    assert "coaching text" not in json.dumps(result)
    assert workouts.parse_workout("run.xml", definition('<SteadyState Duration="5" Power="1"/>', "run")) is None
    assert workouts.parse_workout("rest.xml", definition("<RestDay/>")) is None


def test_fixed_reference_tests_preserve_absolute_power_and_distance_plans_are_excluded():
    data = definition('<SteadyState Duration="60" Power="1.2"/>').replace(
        b"<workout_file>", b"<workout_file><ftpOverride>100</ftpOverride>")
    result = workouts.parse_workout("ramp.xml", data)
    assert result["ftp_override"] == 100
    assert any("fixed 100 W" in warning for warning in result["warnings"])
    data = data.replace(b"<ftpOverride>100</ftpOverride>", b"<durationType>distance</durationType>")
    assert workouts.parse_workout("distance.xml", data) is None


@pytest.mark.parametrize("step", [
    '<SteadyState Duration="-1" Power=".5"/>',
    '<SteadyState Duration="1" Power="NaN"/>',
    '<IntervalsT Repeat="1.5" OnPower="1" OffPower=".5" OnDuration="1" OffDuration="1"/>',
    '<Unknown Duration="10"/>',
])
def test_parser_rejects_invalid_or_unknown_prescriptions(step):
    with pytest.raises(ValueError):
        workouts.parse_workout("invalid.xml", definition(step))


def test_bundled_catalog_contains_gorby_and_all_numeric_blocks_are_contiguous():
    catalog = workouts.load_workouts()
    assert len(catalog) > 1700
    gorby = next(workout for workout in catalog.values() if workout["name"] == "The Gorby")
    assert gorby["duration_sec"] == 3600
    assert len(gorby["segments"]) == 11
    assert sum(segment["kind"] == "Effort" for segment in gorby["segments"]) == 5
    for workout in catalog.values():
        end = 0
        for segment in workout["segments"]:
            assert segment["start_sec"] == end
            assert segment["end_sec"] > end
            end = segment["end_sec"]
            for key in ("power_low", "power_high"):
                assert segment[key] is None or math.isfinite(segment[key])
        assert end == workout["duration_sec"]


def test_power_samples_preserve_raw_outliers_zeros_and_null():
    data = {"timeInSec": [0, 1, 4, 5], "powerInWatts": [-20, 0, 9000, None]}
    assert workouts.power_telemetry(data) == {
        "time_sec": [0, 1, 4, 5], "power_watts": [-20, 0, 9000, None],
    }


@pytest.mark.parametrize("data", [
    {}, {"timeInSec": [], "powerInWatts": []},
    {"timeInSec": [0, 1], "powerInWatts": [10]},
    {"timeInSec": [0, 0], "powerInWatts": [10, 20]},
    {"timeInSec": [1, 0], "powerInWatts": [10, 20]},
    {"timeInSec": [0], "powerInWatts": [float("inf")]},
])
def test_power_samples_reject_invalid_data(data):
    with pytest.raises(ValueError):
        workouts.power_telemetry(data)


def test_workout_page_preserves_activity_on_login_and_renders_controls():
    client = app_module.app.test_client()
    response = client.get("/workout-analysis?activity_id=123")
    assert response.status_code == 302
    assert "activity_id%3D123" in response.location
    with client.session_transaction() as session:
        session["tokens"] = {}
    html = client.get("/workout-analysis?activity_id=123").get_data(as_text=True)
    for field in ("ftp", "intensity", "offset", "workoutSearch", "workoutSelect", "powerChart"):
        assert f'id="{field}"' in html
    assert 'min="0.01"' in html
    assert b"/workout-analysis?activity_id=" in client.get("/my-activities").data


def test_catalog_api_returns_lightweight_index_and_full_detail():
    client = app_module.app.test_client()
    index = client.get("/api/workouts").get_json()["workouts"]
    assert index
    assert "segments" not in index[0]
    detail = client.get(f"/api/workouts/{index[0]['id']}").get_json()
    assert detail["segments"]
    assert client.get("/api/workouts/does-not-exist").status_code == 404


def test_missing_catalog_is_an_explicit_error(monkeypatch):
    def unavailable():
        raise FileNotFoundError("Missing catalog")
    monkeypatch.setattr(app_module, "workout_index", unavailable)
    response = app_module.app.test_client().get("/api/workouts")
    assert response.status_code == 503
    assert "catalog unavailable" in response.get_json()["error"]


def test_activity_api_auth_validation_and_raw_samples(monkeypatch):
    client = app_module.app.test_client()
    monkeypatch.setattr(app_module, "get_headers", lambda: None)
    assert client.get("/api/workout_activity/123").status_code == 401
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    assert client.get("/api/workout_activity/nope").status_code == 400
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (
        {"timeInSec": [0, 1, 2], "powerInWatts": [-100, 0, 9000]}, {"name": "Ride"}, None))
    response = client.get("/api/workout_activity/123")
    assert response.status_code == 200
    assert response.get_json()["power_watts"] == [-100, 0, 9000]
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: ({}, {}, None))
    assert client.get("/api/workout_activity/123").status_code == 422


def test_activity_upstream_failures_are_not_success(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (None, None, "No fitness data available"))
    client = app_module.app.test_client()
    response = client.get("/api/workout_activity/123")
    assert response.status_code == 502
    assert response.get_json()["error"] == "No fitness data available"
    def timeout(*args):
        raise requests.Timeout()
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", timeout)
    assert client.get("/api/workout_activity/123").status_code == 502


@pytest.mark.parametrize("ftp,expected", [(280, 280), ("285.5", 285.5), (None, None),
                                         (0, None), ("unknown", None), ("NaN", None)])
def test_profile_exposes_current_ftp_without_breaking_measurements(monkeypatch, ftp, expected):
    class ProfileResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"firstName": "Test", "height": 1800, "weight": 75000, "ftp": ftp}
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module.requests, "get", lambda *args, **kwargs: ProfileResponse())
    response = app_module.app.test_client().get("/api/my_profile")
    assert response.status_code == 200
    assert response.get_json() == {"name": "Test", "height_cm": 180, "weight_kg": 75, "ftp_w": expected}
