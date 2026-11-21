from types import SimpleNamespace

import pytest

import app as app_module


@pytest.fixture
def matching_client(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    monkeypatch.setattr(app_module, "_fetch_profile", lambda profile_id, _: {
        "id": profile_id, "weightInGrams": 83000, "heightInCentimeters": 191,
    })
    return app_module.app.test_client()


def test_matching_segments_simulates_all_filters_range_and_sorts_percentage(matching_client, monkeypatch):
    times = [140, 99, 80, 105, 102, 101, 160]
    segments = [{"id": str(index), "name": f"Segment {index}", "world_name": "Watopia"}
                for index in range(len(times))]
    monkeypatch.setattr(app_module, "list_segments", lambda: segments)
    monkeypatch.setattr(app_module, "load_segment_profile", lambda identifier: identifier)
    setup = SimpleNamespace(weight_kg=7, cda_bias=-.01, crr_reduction=.0003)
    monkeypatch.setattr(app_module, "_get_efficiency_setup", lambda: setup)
    calls = []
    def simulate(identifier, **options):
        calls.append(options)
        return times[int(identifier)]
    monkeypatch.setattr(app_module, "constant_power_time", simulate)
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": 253.7, "duration_sec": 100, "critical_power_w": 300})
    assert response.status_code == 200
    data = response.get_json()
    assert [row["segment_id"] for row in data["matches"]] == ["3", "4", "5", "1"]
    assert data["matches"][0]["completion_percent"] == pytest.approx(100 / 105 * 100)
    assert data["matches"][-1]["completion_percent"] == pytest.approx(100 / 99 * 100)
    assert data["matches"][-1]["difference_sec"] == 1
    assert len(calls) == 7
    assert all(call == {
        "power_w": 254, "rider_weight_kg": 83, "bike_weight_kg": 7,
        "cda": app_module.rider_cda(1.91, 83) - .01, "crr_reduction": .0003,
        "initial_speed_mps": data["initial_speed_mps"],
    } for call in calls)
    assert data["weight_kg"] == 83
    assert data["height_cm"] == 191
    assert data["power_w"] == 254
    assert app_module.calculate_power_for_speed(
        data["initial_speed_mps"], 0, 83, 7, app_module.rider_cda(1.91, 83) - .01, .0037,
    ) == pytest.approx(300)


@pytest.mark.parametrize("power,cp,expected_power,expected_cp", [
    (253.5, 212.5, 254, 213), (253.4, 212.4, 253, 212),
])
def test_matching_simulation_rounds_power_and_cp_to_displayed_watts(
    matching_client, monkeypatch, power, cp, expected_power, expected_cp,
):
    monkeypatch.setattr(app_module, "list_segments", lambda: [])
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": power, "duration_sec": 100, "critical_power_w": cp})
    assert response.status_code == 200
    data = response.get_json()
    assert data["power_w"] == expected_power
    assert data["critical_power_w"] == expected_cp
    setup = app_module._get_efficiency_setup()
    assert app_module.calculate_power_for_speed(
        data["initial_speed_mps"], 0, 83, setup.weight_kg,
        app_module.rider_cda(1.91, 83) + setup.cda_bias, .004 - setup.crr_reduction,
    ) == pytest.approx(expected_cp)


@pytest.mark.parametrize("times,expected", [
    ([1101, 1100, 900, 899], [90, 110]),
    ([1101, 899], []),
    ([900, 950, 1000, 1050, 1100, 1020], [90, 100 * 990 / 1050, 100 * 990 / 1020,
                                                       99, 100 * 990 / 950]),
])
def test_matching_range_includes_boundaries_and_limits_sorted_results_to_five(
    matching_client, monkeypatch, times, expected,
):
    segments = [{"id": str(index), "name": f"Segment {index}", "world_name": "Watopia"}
                for index in range(len(times))]
    monkeypatch.setattr(app_module, "list_segments", lambda: segments)
    monkeypatch.setattr(app_module, "load_segment_profile", lambda identifier: identifier)
    monkeypatch.setattr(app_module, "constant_power_time", lambda identifier, **kwargs: times[int(identifier)])
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": 300, "duration_sec": 990, "critical_power_w": 250})
    assert response.status_code == 200
    assert [row["completion_percent"] for row in response.get_json()["matches"]] == pytest.approx(expected)


@pytest.mark.parametrize("payload", [None, [], {}, {"power_w": -20, "duration_sec": 300},
                                    {"power_w": True, "duration_sec": 300},
                                    {"power_w": 300, "duration_sec": 0}])
def test_matching_segments_rejects_invalid_inputs(matching_client, payload):
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


@pytest.mark.parametrize("cp", [None, 0, -10, True, "300", float("nan"), float("inf")])
def test_matching_segments_requires_valid_cp(matching_client, cp):
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": 300, "duration_sec": 100, "critical_power_w": cp})
    assert response.status_code == 400
    assert "critical power" in response.get_json()["error"]


def test_rolling_start_is_not_capped_for_large_raw_cp(matching_client, monkeypatch):
    monkeypatch.setattr(app_module, "list_segments", lambda: [])
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": 300, "duration_sec": 100, "critical_power_w": 50000})
    assert response.status_code == 200
    data = response.get_json()
    setup = app_module._get_efficiency_setup()
    assert data["initial_speed_mps"] > 30
    assert app_module.calculate_power_for_speed(
        data["initial_speed_mps"], 0, 83, setup.weight_kg,
        app_module.rider_cda(1.91, 83) + setup.cda_bias, .004 - setup.crr_reduction,
    ) == pytest.approx(50000)


def test_matching_segments_requires_current_measurements_and_service(matching_client, monkeypatch):
    url = "/api/power_dashboard/matching_segments?zwift_id=7"
    payload = {"power_w": 300, "duration_sec": 100, "critical_power_w": 250}
    monkeypatch.setattr(app_module, "_fetch_profile", lambda *args: {"id": 7})
    response = matching_client.post(url, json=payload)
    assert response.status_code == 422
    assert "height and weight" in response.get_json()["error"]
    monkeypatch.setattr(app_module, "get_headers", lambda: None)
    assert matching_client.post(url, json=payload).status_code == 503


def test_matching_segments_reports_simulation_failure(matching_client, monkeypatch):
    monkeypatch.setattr(app_module, "list_segments", lambda: [{"id": "1"}])
    monkeypatch.setattr(app_module, "load_segment_profile", lambda _: None)
    def fail(*args, **kwargs):
        raise ValueError("Invalid segment geometry")
    monkeypatch.setattr(app_module, "constant_power_time", fail)
    response = matching_client.post("/api/power_dashboard/matching_segments?zwift_id=7",
                                    json={"power_w": 300, "duration_sec": 100, "critical_power_w": 250})
    assert response.status_code == 422
    assert response.get_json()["error"] == "Invalid segment geometry"
