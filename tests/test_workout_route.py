from io import BytesIO

import numpy as np
import pytest

import app as app_module
from bike_comparison.pacing_planner import RouteProfile
from bike_comparison.physics import speed_from_power
from bike_comparison.workout_route import simulate_workout_route, target_power
from shared.workouts import parse_workout


def workout(steps='<SteadyState Duration="60" Power="1"/>', extra=""):
    return parse_workout("custom.zwo", (
        f"<workout_file><name>Test workout</name>{extra}<workout>{steps}</workout></workout_file>"
    ).encode())


def route(length=10000, grade=0, surface="Tarmac"):
    return RouteProfile("Test route", np.array([0., length]), np.array([-20., -20 + length * grade]),
                        surfaces=np.array([surface, surface]))


def simulate(plan=None, course=None, **options):
    return simulate_workout_route(
        course if course is not None else route(), plan if plan is not None else workout(),
        rider_weight_kg=75., bike_weight_kg=8., cda=.3, ftp_w=options.pop("ftp_w", 250.),
        **options,
    )


def test_time_targets_use_rider_ftp_ranges_ramps_and_explicit_untargeted_effort():
    plan = workout('<Warmup Duration="10" PowerLow=".4" PowerHigh="1"/>'
                   '<SteadyState Duration="10" PowerLow=".8" PowerHigh="1"/>'
                   '<FreeRide Duration="10"/>')
    assert target_power(plan["segments"][0], 5, 250, 300, .75) == pytest.approx(175)
    assert target_power(plan["segments"][1], 15, 250, 300, .75) == 225
    assert target_power(plan["segments"][2], 25, 100, 300, .75) == 225
    result = simulate(plan, ftp_w=300., untargeted_percent=80.)
    assert result["profile"]["power_w"][-1] == 240
    assert any("80%" in warning for warning in result["warnings"])


def test_fixed_reference_does_not_scale_with_rider_ftp():
    plan = workout(extra="<ftpOverride>100</ftpOverride>")
    low, high = simulate(plan, ftp_w=100.), simulate(plan, ftp_w=500.)
    assert low["distance_covered_km"] == high["distance_covered_km"]
    assert high["reference_ftp_w"] == 100
    assert high["profile"]["power_w"] == [100.] * len(high["profile"]["power_w"])


@pytest.mark.parametrize(("ftp", "expected"), [
    (251., 250.), (253., 255.), (252.5, 255.), (262.5, 265.), (2.5, 5.), (0., 0.),
])
def test_targets_round_to_nearest_five_with_halfway_values_up(ftp, expected):
    segment = workout()["segments"][0]
    assert target_power(segment, 0, ftp, ftp, .75) == expected


def test_rounding_follows_scaling_and_range_midpoint_without_changing_source():
    plan = workout('<SteadyState Duration="60" PowerLow=".8" PowerHigh=".9"/>')
    segment = plan["segments"][0]
    assert target_power(segment, 0, 253., 253., .75) == 215.
    result = simulate(plan, ftp_w=253.)
    rounded = simulate(workout(), ftp_w=215.)
    assert result["distance_covered_km"] == pytest.approx(rounded["distance_covered_km"], abs=1e-10)
    assert set(result["profile"]["power_w"]) == {215.}
    assert segment["power_low"] == .8
    assert segment["power_high"] == .9


def test_fixed_reference_and_untargeted_power_are_rounded_after_scaling():
    fixed = simulate(workout(extra="<ftpOverride>252.5</ftpOverride>"), ftp_w=500.)
    assert set(fixed["profile"]["power_w"]) == {255.}
    assumed = simulate(workout('<FreeRide Duration="60"/>'), ftp_w=253., untargeted_percent=75.)
    assert set(assumed["profile"]["power_w"]) == {190.}


@pytest.mark.parametrize("kind", ["Warmup", "Cooldown"])
def test_rounded_ramps_match_exact_five_watt_plateaus(kind):
    low, high = (0., .1) if kind == "Warmup" else (.1, 0.)
    ramp = workout(f'<{kind} Duration="10" PowerLow="{low}" PowerHigh="{high}"/>')
    powers = [0, .05, .1] if kind == "Warmup" else [.1, .05, 0]
    plateaus = workout(
        f'<SteadyState Duration="2.5" Power="{powers[0]}"/>'
        f'<SteadyState Duration="5" Power="{powers[1]}"/>'
        f'<SteadyState Duration="2.5" Power="{powers[2]}"/>'
    )
    actual = simulate(ramp, route(10000, -.05), ftp_w=100.)
    expected = simulate(plateaus, route(10000, -.05), ftp_w=100.)
    assert actual["distance_covered_km"] == pytest.approx(expected["distance_covered_km"], abs=1e-10)
    assert {2.5, 7.5}.issubset(actual["profile"]["time_sec"])
    assert set(actual["profile"]["power_w"]) == {0., 5., 10.}


def test_excessive_rounded_ramp_steps_fail_explicitly_instead_of_clipping():
    plan = workout('<Warmup Duration="10" PowerLow="0" PowerHigh="1000"/>')
    with pytest.raises(ValueError, match="10,000-step simulation limit"):
        simulate(plan)
    assert plan["segments"][0]["power_high"] == 1000


def test_standing_start_and_flat_equilibrium_match_existing_physics():
    result = simulate(workout('<SteadyState Duration="600" Power="1"/>'))
    assert result["profile"]["speed_kph"][0] == 0
    expected = speed_from_power(250., 0., 75., 8., .3)
    assert result["profile"]["speed_kph"][-1] / 3.6 == pytest.approx(expected, rel=1e-4)
    assert result["profile"]["altitude_m"][0] == -20
    assert result["distance_remaining_km"] > 0
    assert not result["completed"]


def test_completion_is_at_finish_not_workout_end_and_has_remaining_time():
    result = simulate(course=route(200))
    assert result["completed"]
    assert 0 < result["finish_time_sec"] < 60
    assert result["workout_time_remaining_sec"] == 60 - result["finish_time_sec"]
    assert result["distance_covered_km"] == .2
    assert result["distance_remaining_km"] == 0
    assert result["profile"]["distance_km"][-1] == pytest.approx(.2, abs=1e-8)
    assert result["profile"]["time_sec"][-1] == result["finish_time_sec"]


def test_completion_threshold_uses_actual_workout_distance():
    baseline = simulate()
    distance = baseline["distance_covered_km"] * 1000
    assert simulate(course=route(distance - .01))["completed"]
    assert not simulate(course=route(distance + .01))["completed"]


def test_zero_power_is_stationary_on_flat_but_rolls_downhill_from_rest():
    zero = workout('<FreeRide Duration="60"/>')
    stopped = simulate(zero, untargeted_percent=0.)
    assert stopped["distance_covered_km"] == 0
    assert stopped["profile"]["speed_kph"] == [0.] * len(stopped["profile"]["speed_kph"])
    downhill = simulate(zero, route(10000, -.05), untargeted_percent=0.)
    assert downhill["distance_covered_km"] > .5


def test_momentum_survives_power_transition_and_uphill_coasting_stops():
    plan = workout('<SteadyState Duration="20" Power="1"/>'
                   '<FreeRide Duration="180"/>')
    result = simulate(plan, route(10000, .1), untargeted_percent=0.)
    times, speeds = result["profile"]["time_sec"], result["profile"]["speed_kph"]
    transition = [speed for time, speed in zip(times, speeds) if time == 20]
    assert len(transition) == 2
    assert transition[0] == transition[1] > 0
    assert speeds[-1] == 0


def test_more_ftp_and_faster_surfaces_increase_distance():
    assert simulate(ftp_w=300.)["distance_covered_km"] > simulate(ftp_w=200.)["distance_covered_km"]
    assert simulate(course=route(surface="Tarmac"))["distance_covered_km"] > simulate(course=route(surface="Dirt"))["distance_covered_km"]


def test_zero_length_geometry_joins_do_not_create_infinite_gradients():
    geometry = RouteProfile(
        "Joined route", np.array([0., 100., 100., 200.]),
        np.array([-20., -20., -19.99, -19.99]),
        surfaces=np.array(["Tarmac"] * 4),
    )
    result = simulate(course=geometry)
    assert result["completed"]
    assert result["route_distance_km"] == .2


def test_real_wad_route_includes_leadin_and_supports_geometry_joins():
    info = next(route for route in app_module.list_routes() if route["name"] == "Volcano Circuit")
    geometry = app_module.load_route_profile(str(info["id"]), info["name"], world=info["world"], include_leadin=True)
    without_leadin = app_module.load_route_profile(str(info["id"]), info["name"], world=info["world"], include_leadin=False)
    assert geometry.distance_m[-1] > without_leadin.distance_m[-1]
    result = simulate(course=geometry)
    assert result["distance_covered_km"] > 0
    assert not result["completed"]


def test_unusual_power_and_gradients_are_not_clipped():
    plan = workout('<SteadyState Duration="10" Power="20"/>')
    result = simulate(plan)
    assert result["profile"]["power_w"][0] == 5000
    steep = simulate(course=route(10000, .6))
    assert steep["distance_covered_km"] < simulate(course=route(10000, .4))["distance_covered_km"]
    with pytest.raises(ValueError, match="not clipped"):
        simulate(workout('<SteadyState Duration="10" Power="-.1"/>'))


@pytest.mark.parametrize("change", [
    {"duration_sec": float("nan")}, {"duration_sec": 90000},
    {"segments": []}, {"ftp_override": -1},
    {"segments": [{"start_sec": 0, "end_sec": 60, "power_low": None, "power_high": 1, "ramp": False}]},
    {"segments": [{"start_sec": 1, "end_sec": 60, "power_low": 1, "power_high": 1, "ramp": False}]},
])
def test_invalid_custom_numeric_plans_are_rejected(change):
    plan = workout()
    plan.update(change)
    with pytest.raises(ValueError):
        simulate(plan)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "list_routes", lambda: [
        {"id": "test", "name": "Test route", "world": "WATOPIA"},
    ])
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: route(200))
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["tokens"] = {}
    return client


def request_body():
    db = app_module.get_db()
    frame_id = next(frame_id for frame_id in db.frames if db.get_bike_stats(frame_id) is not None)
    return {
        "route_id": "test", "custom_workout": workout(), "rider_weight_kg": 75.,
        "rider_height_cm": 175., "ftp_w": 250., "frame_id": frame_id,
    }


def test_page_requires_login_and_has_all_inputs(client):
    anonymous = app_module.app.test_client()
    assert anonymous.get("/workout-route").status_code == 302
    assert anonymous.post("/api/workout_route_plan", json={}).status_code == 401
    response = client.get("/workout-route")
    assert response.status_code == 200
    for field in ("ftp", "riderWeight", "riderHeight", "frameSelect", "routeSelect", "workoutSelect", "workoutFile"):
        assert f'id="{field}"' in response.text
    assert b'/workout-route' in client.get("/").data


def test_api_accepts_uploaded_and_catalog_workouts(client):
    uploaded = client.post("/api/workouts/upload", data={
        "file": (BytesIO(b'<workout_file><workout><SteadyState Duration="60" Power="1"/></workout></workout_file>'), "test.zwo"),
    })
    assert uploaded.status_code == 200
    body = request_body()
    body["custom_workout"] = uploaded.get_json()
    result = client.post("/api/workout_route_plan", json=body)
    assert result.status_code == 200
    assert result.get_json()["completed"]
    body.pop("custom_workout")
    body["workout_id"] = next(iter(app_module.load_workouts()))
    assert client.post("/api/workout_route_plan", json=body).status_code == 200


@pytest.mark.parametrize("change", [
    {"ftp_w": 0}, {"ftp_w": float("nan")}, {"rider_weight_kg": True},
    {"rider_height_cm": -1}, {"upgrade_level": 1.5}, {"route_id": "missing"},
    {"frame_id": "missing"}, {"include_leadin": "false"},
    {"custom_workout": {}}, {"workout_id": "missing", "custom_workout": None},
])
def test_api_rejects_bad_inputs(client, change):
    body = request_body()
    body.update(change)
    response = client.post("/api/workout_route_plan", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_missing_leadin_is_not_silently_excluded(client, monkeypatch):
    geometry = route()
    geometry.surfaces = None
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: geometry)
    body = request_body()
    response = client.post("/api/workout_route_plan", json=body)
    assert response.status_code == 400
    assert "Lead-in geometry unavailable" in response.get_json()["error"]
    body["include_leadin"] = False
    response = client.post("/api/workout_route_plan", json=body)
    assert response.status_code == 200
    assert any("CRR" in warning for warning in response.get_json()["warnings"])
