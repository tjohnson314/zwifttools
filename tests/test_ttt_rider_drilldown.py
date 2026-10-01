import numpy as np

import app as app_module
from race_replay.data_cleaner import _unwrap_loop_route_distances


def _rider_with_distance(distance_m):
    sample_count = len(distance_m)
    return {
        "name": "Rider",
        "activity_id": "activity",
        "weight_kg": 75.0,
        "time_sec": np.arange(sample_count, dtype=float),
        "speed_mps": np.full(sample_count, 10.0),
        "distance_m": np.asarray(distance_m, dtype=float),
        "altitude_m": np.arange(sample_count, dtype=float),
        "power": np.full(sample_count, 250.0),
        "draft_watts": np.full(sample_count, 25.0),
    }


def test_rider_distance_sampling_keeps_full_trace_after_projection_reset():
    rider = _rider_with_distance([0, 50, 100, 150, 200, 250, 300, 50])
    rider["route_distance_m"] = rider["distance_m"]

    distance_axis, indices = app_module._sample_ttt_rider_distance(rider, step_m=50)

    np.testing.assert_array_equal(distance_axis, [0, 50, 100, 150, 200, 250])
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4, 5])


def test_rider_distance_sampling_ignores_nonfinite_projection_samples():
    rider = _rider_with_distance([0, 50, 100, 150])
    rider["route_distance_m"] = np.array([0, np.nan, 100, 150], dtype=float)

    distance_axis, indices = app_module._sample_ttt_rider_distance(rider, step_m=50)

    np.testing.assert_array_equal(distance_axis, [0, 50, 100])
    np.testing.assert_array_equal(indices, [0, 2, 2])


def test_rider_endpoint_returns_full_trace_after_projection_reset(monkeypatch):
    rider = _rider_with_distance([0, 50, 100, 150, 200, 250, 300, 50])
    rider.update({
        "name": "Finished Rider",
        "activity_id": "123",
        "weight_kg": 75.0,
        "route_distance_m": rider["distance_m"].copy(),
    })
    cache_key = 654321
    monkeypatch.setitem(
        app_module._ttt_cache,
        cache_key,
        {"all_processed": {"123": rider}},
    )

    response = app_module.app.test_client().post(
        "/api/ttt/rider",
        json={"subgroup_id": cache_key, "activity_id": "123"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["distance_km"] == [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    assert data["power_watts"] == [250.0] * 6


def test_team_chart_reaches_finisher_maximum_when_manual_rider_drops_out():
    finisher = _rider_with_distance([0, 50, 100, 150, 200, 250, 300, 50])
    finisher.update({
        "name": "Finisher",
        "activity_id": "finished",
        "route_distance_m": finisher["distance_m"].copy(),
    })
    manual_dnf = _rider_with_distance([0, 50, 100, 150, 200])
    manual_dnf.update({
        "name": "Manual DNF",
        "activity_id": "dnf",
        "route_distance_m": manual_dnf["distance_m"].copy(),
    })

    teams = app_module._build_ttt_team_results(
        {"finished": finisher, "dnf": manual_dnf},
        {"finished": "RTB", "dnf": "RTB"},
        ["RTB"],
    )

    assert teams[0]["distance_km"] == [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]


def test_loop_distance_unwrap_supports_more_than_two_laps():
    projected_on_lap = np.array(
        [0, 500, 0, 500, 0, 500, 0, 500, 0, 500], dtype=float
    )
    odometer_progress = np.arange(0, 5000, 500, dtype=float)

    unwrapped = _unwrap_loop_route_distances(
        projected_on_lap, odometer_progress, lap_distance_m=1000
    )

    np.testing.assert_array_equal(unwrapped, odometer_progress)


def test_team_aggregation_preserves_millisecond_offsets():
    def build_with_offset(offset_sec):
        rider = _rider_with_distance([0, 100, 200])
        rider['time_sec'] = rider['time_sec'] + offset_sec
        return app_module._build_ttt_team_results(
            {'activity': rider}, {'activity': 'RTB'}, ['RTB']
        )[0]

    earlier = build_with_offset(-0.001)
    later = build_with_offset(0.001)

    assert earlier['lead_time_sec'][0] == 0.0
    assert earlier['lead_time_sec'][2] == 1.0
    assert later['lead_time_sec'][1] == 1.0
    assert later['lead_time_sec'][3] == 2.0


def test_position_deviation_is_measured_from_team_leader():
    ahead = _rider_with_distance([0, 100, 200])
    ahead.update({"name": "Ahead", "activity_id": "ahead"})
    behind = _rider_with_distance([0, 80, 160])
    behind.update({"name": "Behind", "activity_id": "behind"})

    team = app_module._build_ttt_team_results(
        {"ahead": ahead, "behind": behind},
        {"ahead": "RTB", "behind": "RTB"},
        ["RTB"],
    )[0]

    deviations = {
        rider["activity_id"]: rider
        for rider in team["position_deviation"]["riders"]
    }
    assert deviations["ahead"]["distance_km"] == [0.0, 0.1, 0.2]
    assert deviations["ahead"]["deviation_m"] == [0.0, 0.0, 0.0]
    assert deviations["behind"]["deviation_m"] == [0.0, -20.0, -40.0]


def test_position_deviation_responds_to_rider_time_offset():
    reference = _rider_with_distance([0, 100, 200])
    reference.update({"name": "Reference", "activity_id": "reference"})
    shifted = _rider_with_distance([0, 100, 200])
    shifted.update({"name": "Shifted", "activity_id": "shifted"})
    shifted["time_sec"] = shifted["time_sec"] + 0.5

    team = app_module._build_ttt_team_results(
        {"reference": reference, "shifted": shifted},
        {"reference": "RTB", "shifted": "RTB"},
        ["RTB"],
    )[0]

    deviations = {
        rider["activity_id"]: rider["deviation_m"]
        for rider in team["position_deviation"]["riders"]
    }
    assert deviations["reference"] == [0.0, 0.0, 0.0]
    assert deviations["shifted"] == [-50.0, -50.0]


def test_position_deviation_leader_axis_never_moves_backwards():
    early_leader = _rider_with_distance([0, 100, 200])
    early_leader.update({"name": "Early", "activity_id": "early"})
    continuing_rider = _rider_with_distance([0, 50, 100, 150])
    continuing_rider.update({"name": "Continuing", "activity_id": "continuing"})

    team = app_module._build_ttt_team_results(
        {"early": early_leader, "continuing": continuing_rider},
        {"early": "RTB", "continuing": "RTB"},
        ["RTB"],
    )[0]

    for rider in team["position_deviation"]["riders"]:
        assert np.all(np.diff(rider["distance_km"]) > 0)