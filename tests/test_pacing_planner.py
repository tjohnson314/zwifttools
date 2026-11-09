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


def test_route_portion_interpolates_both_boundaries_and_rebases_distance():
    route = make_route_profile()
    route.lats = np.array([1.0, 2.0, 3.0])
    route.lngs = np.array([4.0, 5.0, 6.0])
    portion = truncate_route_profile(route, 150.0, start_distance_m=50.0)

    assert np.allclose(portion.distance_m, [0.0, 50.0, 100.0])
    assert np.allclose(portion.altitude_m, [15.0, 20.0, 17.5])
    assert np.allclose(portion.lats, [1.5, 2.0, 2.5])
    assert np.allclose(portion.lngs, [4.5, 5.0, 5.5])
    assert portion.surfaces.tolist() == ["road", "gravel", "gravel"]
    assert portion.total_ascent_m == 5.0
    assert portion.source_distance_m is None
    assert np.allclose(route.distance_m, [0.0, 100.0, 200.0])


def test_route_portion_defaults_to_full_route_end():
    portion = truncate_route_profile(make_route_profile(), start_distance_m=100.0)
    assert portion.distance_m.tolist() == [0.0, 100.0]
    assert portion.altitude_m.tolist() == [20.0, 15.0]
    assert portion.surfaces.tolist() == ["gravel", "road"]


def test_segment_catalog_uses_complete_directional_checkpoint_geometry(monkeypatch):
    from bike_comparison import pacing_planner as planner

    path = {"d": [0, 1, 2, 3], "x": [0, 100, 200, 300], "z": [0, 0, 0, 0],
            "alt": [0, 20, 10, 30]}
    segment = {"hash": 123, "name": "Test KOM", "direction": "F", "distance_m": 150,
               "max_distance_m": 150, "start_checkpoint": 0.5, "end_checkpoint": 2,
               "percent_start": 1 / 6, "percent_end": 2 / 3}
    monkeypatch.setattr(planner.surface_map, "_WORLD_NAMES", {1: "Watopia"})
    monkeypatch.setattr(planner.surface_map, "_load_route_world", lambda _: {"routes": [
        {"segment_path": path, "segments": [segment, dict(segment, distance_m=50)]}]})
    monkeypatch.setattr(planner.surface_map, "get_world_altitude_scale", lambda _: 0.5)
    planner._segment_catalog.cache_clear()
    try:
        entries = planner.list_segments()
        assert len(entries) == 1
        assert entries[0]["id"] == "1:123"
        assert entries[0]["distance_km"] == 0.15
        assert entries[0]["ascent_m"] == 5
        leg = planner._segment_catalog()["1:123"]["leg"]
        assert leg["d"] == [0, 50, 150]
        assert leg["alt"] == [10, 15, 10]
    finally:
        planner._segment_catalog.cache_clear()


@pytest.mark.parametrize("start,end", [(-1, None), (100, 100), (150, 100), (200, None),
                                            (float("nan"), None), (0, 201)])
def test_route_portion_rejects_invalid_ranges(start, end):
    with pytest.raises(ValueError, match="Custom distance"):
        truncate_route_profile(make_route_profile(), end, start_distance_m=start)


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


@pytest.mark.parametrize("distance_m", [0.0, 201.0, float("nan")])
def test_truncate_route_profile_rejects_invalid_distance(distance_m):
    with pytest.raises(ValueError, match="Custom distance"):
        truncate_route_profile(make_route_profile(), distance_m)


@pytest.fixture
def tt_api(monkeypatch):
    from types import SimpleNamespace
    import app as app_module

    setup = SimpleNamespace(weight_kg=8.0, cda_bias=0.0, frame_type="Standard")
    monkeypatch.setattr(app_module, "get_db", lambda: SimpleNamespace(get_bike_stats=lambda *args: setup))
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: make_route_profile())
    seen = []

    def optimizer(**kwargs):
        route = kwargs["route"]
        seen.append(route)
        return PacingPlanResult(
            route_name=route.name, total_time_seconds=20, total_distance_km=route.total_distance_km,
            total_ascent_m=route.total_ascent_m, avg_speed_kph=36, avg_power_w=250,
            normalized_power_w=250, max_power_w=300, min_power_w=200)

    monkeypatch.setattr(app_module, "plan_tt_pacing", optimizer)
    body = {"route_name": "Test Route", "frame_id": "test", "rider_weight_kg": 75,
            "rider_height_cm": 180, "avg_power_watts": 250}
    return app_module, body, seen


@pytest.mark.parametrize("end,expected", [(0.15, [0, 50, 100]), (None, [0, 50, 150])])
def test_tt_api_selects_start_and_end(tt_api, end, expected):
    module, body, seen = tt_api
    body.update(start_distance_km=0.05, custom_distance_km=end)
    response = module.app.test_client().post("/api/tt_pacing_plan", json=body)
    assert response.status_code == 200
    assert seen[0].distance_m.tolist() == expected
    assert response.get_json()["start_distance_km"] == 0.05


@pytest.mark.parametrize("start,end", [(-1, None), (0.15, 0.1), (0.2, None), ("bad", None),
                                           (float("inf"), None), (0, float("nan"))])
def test_tt_api_rejects_invalid_portion(tt_api, start, end):
    module, body, seen = tt_api
    body.update(start_distance_km=start, custom_distance_km=end)
    response = module.app.test_client().post("/api/tt_pacing_plan", json=body)
    assert response.status_code == 400
    assert not seen


def test_tt_api_uses_full_segment_ignoring_route_only_options(tt_api, monkeypatch):
    module, body, seen = tt_api
    segment = make_route_profile()
    segment.name = "Test KOM"
    monkeypatch.setattr(module, "load_segment_profile", lambda segment_id: segment if segment_id == "1:123" else None)
    body.pop("route_name")
    body.update(segment_id="1:123", include_leadin=True, laps="invalid",
                start_distance_km=999, custom_distance_km=-1)
    response = module.app.test_client().post("/api/tt_pacing_plan", json=body)
    assert response.status_code == 200
    assert seen == [segment]
    assert response.get_json()["route_name"] == "Test KOM"
    assert response.get_json()["start_distance_km"] == 0


def test_tt_api_unknown_segment_is_bad_request(tt_api):
    module, body, seen = tt_api
    body["segment_id"] = "missing"
    response = module.app.test_client().post("/api/tt_pacing_plan", json=body)
    assert response.status_code == 400
    assert not seen