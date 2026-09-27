import numpy as np
import pytest

from bike_comparison.pacing_planner import RouteProfile, truncate_route_profile


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


@pytest.mark.parametrize("distance_m", [0.0, 200.0, float("nan")])
def test_truncate_route_profile_rejects_invalid_distance(distance_m):
    with pytest.raises(ValueError, match="Custom distance"):
        truncate_route_profile(make_route_profile(), distance_m)