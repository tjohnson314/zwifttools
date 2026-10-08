import numpy as np
import pytest

from bike_comparison.pacing_planner import PacingPlanResult, RouteProfile, plan_tt_pacing, truncate_route_profile


def make_route_profile():
    return RouteProfile(
        name="Test Route",
        distance_m=np.array([0.0, 100.0, 200.0]),
        altitude_m=np.array([10.0, 20.0, 15.0]),
        surfaces=np.array(["road", "gravel", "road"], dtype=object),
        source_distance_m=200.0,
        source_ascent_m=10.0,
    )


def test_truncate_route_profile_interpolates_endpoint_and_drops_full_route_totals():
    route = make_route_profile()

    truncated = truncate_route_profile(route, 150.0)

    assert np.allclose(truncated.distance_m, [0.0, 100.0, 150.0])
    assert np.allclose(truncated.altitude_m, [10.0, 20.0, 17.5])
    assert truncated.surfaces.tolist() == ["road", "gravel", "gravel"]
    assert truncated.total_ascent_m == 10.0
    assert truncated.source_distance_m is None
    assert truncated.source_ascent_m is None


def test_truncate_route_profile_uses_surface_at_exact_endpoint():
    truncated = truncate_route_profile(make_route_profile(), 100.0)

    assert truncated.surfaces.tolist() == ["road", "gravel"]


@pytest.mark.parametrize("known_surfaces", [True, False])
def test_tt_chart_surfaces_align_with_profile_and_preserve_transitions(known_surfaces):
    route = RouteProfile(
        name="Mixed", distance_m=np.array([0.0, 100.0, 200.0, 300.0]),
        altitude_m=np.array([-5.0, 5.0, 0.0, 10.0]),
        surfaces=np.array(["Tarmac", "Dirt", "Tarmac", "Tarmac"]) if known_surfaces else None)

    result = plan_tt_pacing(
        route, rider_weight_kg=75.0, rider_height_m=1.8, bike_weight_kg=8.0,
        cda=0.3, power_target_w=250.0, max_chunk_m=50.0,
        downsample_points=2, bisection_iterations=12)

    expected = ["Tarmac", "Tarmac", "Dirt", "Dirt", "Tarmac", "Tarmac"] if known_surfaces else ["Unknown"] * 2
    assert result.surface_type == expected
    assert len(result.distance_km) == len(result.altitude_m) == len(result.surface_type)
    assert len(result.power_w) == len(result.speed_kph) == len(result.surface_type)
    assert result.altitude_m[0] < 0.0
    assert result.crr == ([0.004, 0.004, 0.016, 0.016, 0.004, 0.004] if known_surfaces else [0.004, 0.004])


@pytest.mark.parametrize("surface, road_crr, gravel_crr", [("Dirt", 0.016, 0.009), ("Tarmac", 0.004, 0.008)])
@pytest.mark.parametrize("num_buckets", [None, 2])
def test_tt_bike_surface_resistance_changes_speed_at_the_same_power(surface, road_crr, gravel_crr, num_buckets):
    route = RouteProfile(
        name=surface, distance_m=np.array([0.0, 300.0]), altitude_m=np.zeros(2),
        surfaces=np.array([surface, surface]))
    params = dict(rider_weight_kg=75.0, rider_height_m=1.8, bike_weight_kg=8.0,
                  cda=0.3, power_target_w=250.0, max_chunk_m=50.0,
                  bisection_iterations=12, num_buckets=num_buckets)

    road = plan_tt_pacing(route, **params, bike_type="road_bike")
    gravel = plan_tt_pacing(route, **params, bike_type="gravel_bike")

    assert road.crr == [road_crr] * 6
    assert gravel.crr == [gravel_crr] * 6
    assert (gravel.total_time_seconds < road.total_time_seconds) == (gravel_crr < road_crr)
    assert road.normalized_power_w == pytest.approx(250.0, rel=0.02)
    assert gravel.normalized_power_w == pytest.approx(250.0, rel=0.02)


def test_tt_missing_surfaces_preserves_custom_crr_fallback():
    route = RouteProfile(name="Unknown", distance_m=np.array([0.0, 300.0]), altitude_m=np.zeros(2))
    result = plan_tt_pacing(
        route, rider_weight_kg=75.0, rider_height_m=1.8, bike_weight_kg=8.0,
        cda=0.3, power_target_w=250.0, crr=0.011, bike_type="gravel_bike",
        max_chunk_m=50.0, bisection_iterations=12)

    assert result.surface_type == ["Unknown"] * 6
    assert result.crr == [0.011] * 6


@pytest.mark.parametrize("frame_type, bike_type", [("Standard", "road_bike"), ("Gravel", "gravel_bike"), ("MTB", "mtb")])
def test_tt_api_passes_selected_bike_category_and_returns_crr(monkeypatch, frame_type, bike_type):
    from types import SimpleNamespace
    import app as app_module

    setup = SimpleNamespace(weight_kg=8.0, cda_bias=0.0, frame_type=frame_type)
    monkeypatch.setattr(app_module, "get_db", lambda: SimpleNamespace(get_bike_stats=lambda *args: setup))
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: make_route_profile())
    result = PacingPlanResult(
        route_name="Test Route", total_time_seconds=20.0, total_distance_km=0.2,
        total_ascent_m=10.0, avg_speed_kph=36.0, avg_power_w=250.0,
        normalized_power_w=250.0, max_power_w=300.0, min_power_w=200.0,
        surface_type=["Dirt"], crr=[0.009])

    def optimizer(**kwargs):
        assert kwargs["bike_type"] == bike_type
        return result

    monkeypatch.setattr(app_module, "plan_tt_pacing", optimizer)
    response = app_module.app.test_client().post("/api/tt_pacing_plan", json={
        "route_name": "Test Route", "frame_id": "test", "rider_weight_kg": 75.0,
        "rider_height_cm": 180.0, "avg_power_watts": 250.0})

    assert response.status_code == 200
    assert response.get_json()["profile"]["crr"] == [0.009]
    assert response.get_json()["profile"]["surface_type"] == ["Dirt"]


@pytest.mark.parametrize("distance_m", [0.0, 200.0, float("nan")])
def test_truncate_route_profile_rejects_invalid_distance(distance_m):
    with pytest.raises(ValueError, match="Custom distance"):
        truncate_route_profile(make_route_profile(), distance_m)