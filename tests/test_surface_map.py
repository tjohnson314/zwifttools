import pytest

import app as app_module
from bike_comparison.pacing_planner import list_segments
from bike_comparison import pacing_planner
from shared import surface_map


def test_world_lists_complete_directional_segments_without_changing_routes(monkeypatch):
    monkeypatch.setattr(surface_map, "get_world_surfaces", lambda _: {"routes": [{"name": "Route"}]})
    monkeypatch.setattr(app_module, "list_segments", lambda: [
        {"id": "1:-123", "course_id": 1}, {"id": "2:456", "course_id": 2},
    ])
    data = app_module.app.test_client().get("/api/surface_map/world/1").get_json()
    assert data["timing_segments"] == [{"id": "1:-123", "course_id": 1}]
    assert data["routes"] == [{"name": "Route"}]


def test_segment_detail_uses_the_simulated_geometry_with_no_leadin():
    metadata = next(segment for segment in list_segments() if segment["id"] == "1:-9223372027220687690")
    response = app_module.app.test_client().get("/api/surface_map/segment/1:-9223372027220687690")
    assert response.status_code == 200
    data = response.get_json()
    assert data["segment_id"] == metadata["id"]
    assert data["mapID"] == 1
    assert data["name"] == metadata["name"]
    assert data["leadin"] is None
    assert data["leadin_distance_m"] == 0
    assert data["route_offset_m"] == 0
    assert data["route"]["d"][0] == 0
    assert data["route"]["d"][-1] == pytest.approx(metadata["distance_km"] * 1000, abs=.1)
    assert data["distance_m"] == pytest.approx(metadata["distance_km"] * 1000)
    assert len(data["route"]["surface"]) == len(data["route"]["alt"]) == len(data["route"]["d"])
    assert sum(row["distance_m"] for row in data["breakdown"]) == pytest.approx(data["route"]["d"][-1])
    assert data["bounds"]["min_x"] == min(data["route"]["x"])
    assert data["bounds"]["max_y"] == max(data["route"]["y"])


@pytest.mark.parametrize("identifier", ["1:missing", "999:123", "invalid"])
def test_unknown_segment_reports_not_found(identifier):
    response = app_module.app.test_client().get(f"/api/surface_map/segment/{identifier}")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_segment_read_failure_is_reported(monkeypatch):
    def fail(_):
        raise OSError("Unavailable geometry")
    monkeypatch.setattr(app_module, "load_segment_geometry", fail)
    response = app_module.app.test_client().get("/api/surface_map/segment/1:123")
    assert response.status_code == 502
    assert response.get_json()["error"]


def test_segment_routes_include_all_complete_occurrences_once_and_preserve_direction(monkeypatch):
    path = {"d": [0, 1, 2, 3], "x": [0, 100, 200, 300], "z": [0, 0, 0, 0], "alt": [0, 0, 0, 0]}
    segment = {"hash": -123, "name": "Sprint", "distance_m": 200, "max_distance_m": 200,
               "percent_start": 0, "percent_end": 2 / 3, "direction": "R"}
    routes = [
        {"nameHash": 10, "segment_path": path, "segments": [segment, segment]},
        {"nameHash": 20, "segment_path": path, "segments": [dict(segment, distance_m=210, max_distance_m=210)]},
        {"nameHash": 30, "segment_path": path, "segments": [dict(segment, distance_m=50)]},
        {"nameHash": 40, "segment_path": path, "segments": [dict(segment, hash=123, direction="F")]},
    ]
    monkeypatch.setattr(surface_map, "_WORLD_NAMES", {1: "Watopia"})
    monkeypatch.setattr(surface_map, "_load_route_world", lambda _: {"routes": routes})
    monkeypatch.setattr(surface_map, "get_world_altitude_scale", lambda _: 1)
    pacing_planner._segment_catalog.cache_clear()
    try:
        catalog = pacing_planner._segment_catalog()
        assert catalog["1:-123"]["route_hashes"] == [10, 20]
        assert catalog["1:123"]["route_hashes"] == [40]
        assert catalog["1:-123"]["length"] == 210
    finally:
        pacing_planner._segment_catalog.cache_clear()


def test_matching_routes_use_first_free_ride_crossing_and_report_unavailable_starts(monkeypatch):
    monkeypatch.setattr(app_module, "load_segment_geometry", lambda _: {
        "mapID": 1, "matching_route_hashes": [10, 20, 30, 40, 50],
    })
    def options(map_id, route_hash):
        assert map_id == 1
        return {"available_modes": ["race"] if route_hash == 30 else ["race", "free_ride"],
                "leadin_by_mode": {"free_ride": {"distance_m": None if route_hash == 40 else 100}}}
    monkeypatch.setattr(surface_map, "get_route_start_options", options)
    calls = []
    def route(map_id, route_hash, mode):
        calls.append((map_id, route_hash, mode))
        markers = {
            10: [{"hash": -123, "start_distance_m": 5000},
                 {"hash": 123, "start_distance_m": 5},
                 {"hash": -123, "start_distance_m": 2100}],
            20: [{"hash": -123, "start_distance_m": None}],
            50: [{"hash": 123, "start_distance_m": 100}],
        }
        return {"segments": markers[route_hash]}
    monkeypatch.setattr(surface_map, "get_route", route)
    response = app_module.app.test_client().get("/api/surface_map/segment/1:-123/routes")
    assert response.status_code == 200
    assert response.get_json()["routes"] == [
        {"nameHash": 10, "start_distance_m": 2100, "status": "available"},
        {"nameHash": 20, "start_distance_m": None, "status": "start_unavailable"},
        {"nameHash": 30, "start_distance_m": None, "status": "free_ride_unavailable"},
        {"nameHash": 40, "start_distance_m": None, "status": "leadin_unavailable"},
        {"nameHash": 50, "start_distance_m": None, "status": "not_on_free_ride"},
    ]
    assert calls == [(1, 10, "free_ride"), (1, 20, "free_ride"), (1, 50, "free_ride")]


def test_matching_route_start_includes_complete_free_ride_leadin_segment():
    response = app_module.app.test_client().get(
        "/api/surface_map/segment/9:-9223372031503620760/routes")
    assert response.status_code == 200
    routes = {row["nameHash"]: row for row in response.get_json()["routes"]}
    assert routes[1454553567]["status"] == "available"
    assert routes[1454553567]["start_distance_m"] == pytest.approx(3119.5, abs=2)


@pytest.mark.parametrize("leadin_starts, expected_start, expected_status", [
    ([900], 900, "available"), (None, None, "start_unavailable"),
])
def test_first_route_start_requires_checking_the_free_ride_leadin(
        monkeypatch, leadin_starts, expected_start, expected_status):
    monkeypatch.setattr(app_module, "load_segment_geometry", lambda _: {
        "mapID": 1, "matching_route_hashes": [10],
    })
    monkeypatch.setattr(surface_map, "get_route_start_options", lambda *args: {
        "available_modes": ["free_ride"], "leadin_by_mode": {"free_ride": {"distance_m": 1000}},
    })
    monkeypatch.setattr(surface_map, "get_route", lambda *args, **kwargs: {
        "segments": [{"hash": -123, "start_distance_m": 2100}],
    })
    monkeypatch.setattr(surface_map, "segment_starts_on_free_ride_leadin", lambda *args: leadin_starts)
    response = app_module.app.test_client().get("/api/surface_map/segment/1:-123/routes")
    assert response.get_json()["routes"] == [{
        "nameHash": 10, "start_distance_m": expected_start, "status": expected_status,
    }]
