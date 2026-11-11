import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

import app as app_module
from shared import segment_prs


def frame_from_xy(points, power=250):
    points = np.asarray(points, dtype=float)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return pd.DataFrame({
        "time_sec": np.arange(len(points), dtype=float),
        "lat": points[:, 0] / 111_320, "lng": points[:, 1] / 111_320,
        "distance_km": distance / 1000, "speed_kmh": np.full(len(points), 36.0),
        "altitude_m": -np.arange(len(points), dtype=float),
        "power_watts": np.full(len(points), power, dtype=float),
    })


def segment(points, name="Test", identifier="1:123"):
    return {"segment_id": identifier, "name": name,
            "latlng": (np.asarray(points, dtype=float) / 111_320).tolist()}


def test_interpolated_boundaries_and_all_metrics_preserve_raw_data():
    frame = frame_from_xy([(x, 0) for x in range(-5, 106, 10)], power=1500)
    rows = segment_prs.find_traversals(
        frame, [segment([(0, 0), (100, 0)])], 75, 180, app_module._get_efficiency_setup(),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["start_time_sec"] == pytest.approx(.5)
    assert row["end_time_sec"] == pytest.approx(10.5)
    assert row["total_time_sec"] == pytest.approx(10)
    assert row["distance_m"] == pytest.approx(100)
    assert row["elevation_change_m"] == pytest.approx(-10)
    assert row["avg_power"] == pytest.approx(1500)
    assert row["normalized_power"] == pytest.approx(1500)
    assert row["initial_speed_kmh"] == pytest.approx(36)
    assert row["avg_draft_watts"] < 0
    assert (frame["power_watts"] == 1500).all()


def test_short_segment_np_uses_complete_interpolated_interval():
    frame = frame_from_xy([(x, 0) for x in range(-5, 106, 10)])
    frame["power_watts"] = 100 + frame["time_sec"] * 20
    row = segment_prs.find_traversals(frame, [segment([(0, 0), (100, 0)])])[0]
    assert row["avg_power"] == pytest.approx(210)
    assert row["normalized_power"] == pytest.approx(210)


def test_repeated_closed_laps_are_distinct_attempts():
    path = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    points = [(-10, 0), (0, 0), *path[1:], *path[1:], (10, 0)]
    rows = segment_prs.find_traversals(frame_from_xy(points), [segment(path)])
    assert [row["pass"] for row in rows] == [1, 2]
    assert [row["start_time_sec"] for row in rows] == [1, 5]
    assert [row["end_time_sec"] for row in rows] == [5, 9]
    assert all(row["distance_m"] == pytest.approx(400) for row in rows)


@pytest.mark.parametrize("points", [
    [(x, 0) for x in range(105, -6, -10)],  # reverse
    [(x, 0) for x in range(15, 106, 10)],  # begins within the segment
    [(x, 0) for x in range(-5, 86, 10)],  # ends before the line
    [(x, 80) for x in range(-5, 106, 10)],  # parallel road
])
def test_reverse_partial_and_parallel_passes_are_not_records(points):
    assert segment_prs.find_traversals(frame_from_xy(points), [segment([(0, 0), (100, 0)])]) == []


@pytest.mark.parametrize("offset, accepted", [
    (19.9, True), (20, True), (20.1, False), (35, False), (50, False),
    (-19.9, True), (-20, True), (-20.1, False),
])
def test_crossing_corridor_accepts_only_offsets_within_twenty_metres(offset, accepted):
    path = [segment([(0, 0), (100, 0)])]
    offset_frame = frame_from_xy([(x, offset) for x in range(-5, 106, 10)])
    rows = segment_prs.find_traversals(offset_frame, path)
    assert len(rows) == int(accepted)
    if accepted:
        assert rows[0]["distance_m"] == pytest.approx(100)


def test_interior_crossings_use_the_same_twenty_metre_corridor():
    points = [(x, 25 if 80 < x < 220 else 0) for x in range(-5, 306, 10)]
    assert segment_prs.find_traversals(
        frame_from_xy(points), [segment([(0, 0), (300, 0)])],
    ) == []


@pytest.mark.parametrize("offset, accepted", [(10, True), (35, False)])
def test_molten_momentum_volcano_kom_matches_with_offset_gps_samples(offset, accepted):
    surface_map = segment_prs.surface_map
    route = surface_map.get_route(1, 121619062)
    parts = [part for part in (route.get("leadin"), route.get("route")) if part]
    xy = np.column_stack([
        np.concatenate([part["local_x"] for part in parts]),
        np.concatenate([part["local_z"] for part in parts]),
    ])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    valid = np.r_[True, np.diff(distance) > 0]
    gps = np.asarray(surface_map.local_to_latlng(1, xy[:, 0], xy[:, 1]))
    axis = np.arange(0, distance[-1], 10)
    lat = np.interp(axis, distance[valid], gps[valid, 0])
    lng = np.interp(axis, distance[valid], gps[valid, 1])
    lng += offset / (111_320 * np.cos(np.radians(lat.mean())))
    frame = pd.DataFrame({
        "time_sec": axis / 10, "distance_km": axis / 1000, "lat": lat, "lng": lng,
    })
    climb = next(row for row in segment_prs.world_segments(1, segment_prs.catalog_version())
                 if row["segment_id"] == "1:8176359221329222504")
    rows = segment_prs.find_traversals(frame, [climb])
    assert len(rows) == int(accepted)
    if accepted:
        assert rows[0]["distance_m"] == pytest.approx(3684, abs=100)
        assert rows[0]["total_time_sec"] == pytest.approx(368.4, abs=10)


def test_wrong_route_with_matching_endpoints_and_distance_is_rejected():
    path = [(0, 0), (100, 0), (100, 100), (200, 100), (200, 0), (300, 0)]
    shortcut = [(-10, 0), (0, 0), (100, 0), (100, -100), (200, -100), (200, 0), (300, 0), (310, 0)]
    assert segment_prs.find_traversals(frame_from_xy(shortcut), [segment(path)]) == []


def test_richmond_kom_matches_recorded_cobbled_climbs_attempt():
    frame = pd.DataFrame([
        [536, 5.185259817, 37.524112344, -77.418323447],
        [538, 5.206312737, 37.524290494, -77.418410747],
        [540, 5.227864571, 37.524472094, -77.418485198],
        [542, 5.249288522, 37.524659944, -77.418533199],
        [544, 5.271222156, 37.524848094, -77.418534801],
        [546, 5.292298714, 37.525033344, -77.418517501],
        [548, 5.312469669, 37.525208895, -77.418496651],
        [550, 5.332348009, 37.525379295, -77.418480950],
        [552, 5.350923415, 37.525541845, -77.418465950],
        [554, 5.368491180, 37.525697245, -77.418456150],
        [556, 5.385320587, 37.525847095, -77.418466549],
        [558, 5.401445382, 37.525988346, -77.418509449],
        [560, 5.416964029, 37.526125046, -77.418548999],
        [562, 5.432236294, 37.526259896, -77.418586549],
        [564, 5.447082621, 37.526390746, -77.418620399],
        [566, 5.461754450, 37.526520046, -77.418652549],
        [568, 5.476658627, 37.526650596, -77.418685399],
        [570, 5.491247524, 37.526777746, -77.418718399],
        [572, 5.506072841, 37.526906746, -77.418751399],
        [574, 5.521395620, 37.527039795, -77.418784699],
        [576, 5.538537699, 37.527190145, -77.418821849],
        [578, 5.557881255, 37.527357445, -77.418864898],
        [580, 5.582346404, 37.527521145, -77.418910249],
        [582, 5.602270885, 37.527665799, -77.418885806],
        [584, 5.619124092, 37.527634553, -77.418729104],
        [586, 5.635829620, 37.527544803, -77.418606353],
        [588, 5.650994618, 37.527486449, -77.418482504],
        [590, 5.665517958, 37.527571996, -77.418367352],
        [592, 5.680329848, 37.527695196, -77.418316051],
        [594, 5.695410899, 37.527827896, -77.418304250],
        [596, 5.713298794, 37.527959446, -77.418306999],
        [598, 5.731786615, 37.528086746, -77.418346100],
        [600, 5.747094921, 37.528178199, -77.418266605],
        [602, 5.760720406, 37.528141353, -77.418139652],
        [604, 5.772833530, 37.528037353, -77.418087102],
        [606, 5.785159333, 37.527935803, -77.418028703],
        [608, 5.799476464, 37.527848452, -77.417923504],
        [610, 5.815583176, 37.527779652, -77.417775155],
        [612, 5.830684859, 37.527700004, -77.417649900],
        [614, 5.843436842, 37.527585203, -77.417690548],
        [616, 5.856002470, 37.527485253, -77.417780996],
        [618, 5.870549130, 37.527386254, -77.417905497],
        [620, 5.886827226, 37.527243705, -77.417973850],
        [622, 5.904474792, 37.527091954, -77.417943403],
        [624, 5.922843725, 37.526977252, -77.417826105],
    ], columns=["time_sec", "distance_km", "lat", "lng"])
    kom = next(row for row in segment_prs.world_segments(2, segment_prs.catalog_version())
               if row["segment_id"] == "2:9651348518")
    rows = segment_prs.find_traversals(frame, [kom])
    assert len(rows) == 1
    assert rows[0]["total_time_sec"] == pytest.approx(79, abs=1)
    assert rows[0]["distance_m"] == pytest.approx(657, abs=10)
    assert segment_prs.find_traversals(frame[frame["time_sec"] >= 542], [kom]) == []
    assert segment_prs.find_traversals(frame[frame["time_sec"] <= 616], [kom]) == []


def test_richmond_calibration_aligns_held_out_worlds_second_lap():
    samples = np.asarray([
        [37.544743314, -77.440272947],
        [37.547156845, -77.448902158],
        [37.552619079, -77.457668736],
        [37.557909220, -77.466622924],
        [37.553462155, -77.459825842],
        [37.546769132, -77.459103537],
        [37.546513276, -77.448838718],
        [37.544756733, -77.440666912],
        [37.538672006, -77.437124645],
        [37.532670522, -77.431680654],
        [37.527860663, -77.422531076],
        [37.520511733, -77.416241217],
        [37.527844243, -77.418275258],
        [37.529876971, -77.422708308],
        [37.534908631, -77.425781229],
        [37.537319845, -77.432317507],
    ])
    surface_map = segment_prs.surface_map
    route = next(row for row in surface_map._load_route_world(2)["routes"]
                 if row["nameHash"] == 2196019512)["route"]
    distance = np.arange(0, route["d"][-1], 1)
    path = np.asarray(surface_map.local_to_latlng(
        2, np.interp(distance, route["d"], route["x"]),
        np.interp(distance, route["d"], route["z"]),
    ))
    scale = np.asarray([111_320, 111_320 * np.cos(np.radians(samples[:, 0].mean()))])
    errors = np.linalg.norm((path[:, None, :] - samples[None, :, :]) * scale, axis=2).min(axis=0)
    assert errors.max() < 8


def test_irregular_samples_are_time_weighted_and_missing_metrics_are_null():
    frame = frame_from_xy([(x, 0) for x in range(-5, 106, 10)])
    frame["time_sec"] = np.arange(len(frame), dtype=float) ** 2
    frame["power_watts"] = 100 + frame["time_sec"] * 2
    rows = segment_prs.find_traversals(frame, [segment([(0, 0), (100, 0)])])
    row = rows[0]
    assert row["avg_power"] == pytest.approx(100 + row["start_time_sec"] + row["end_time_sec"])
    assert row["avg_draft_watts"] is None
    frame = frame.drop(columns=["power_watts", "altitude_m", "speed_kmh"])
    row = segment_prs.find_traversals(frame, [segment([(0, 0), (100, 0)])])[0]
    for field in ("avg_power", "normalized_power", "elevation_change_m", "initial_speed_kmh", "avg_draft_watts"):
        assert row[field] is None


def test_invalid_matching_telemetry_is_reported():
    frame = frame_from_xy([(0, 0), (10, 0)])
    with pytest.raises(ValueError, match="GPS"):
        segment_prs.find_traversals(frame.drop(columns=["lat"]), [])
    frame["time_sec"] = [1, 1]
    with pytest.raises(ValueError, match="strictly increasing"):
        segment_prs.find_traversals(frame, [])


def mock_profile(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    profile = {"id": 7, "weight": 75000, "height": 1800}
    monkeypatch.setattr(app_module, "_request_with_retry",
                        lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: profile))
    monkeypatch.setattr(app_module, "catalog_version", lambda: "test-v1")
    return profile


def test_segment_page_requires_login_and_is_discoverable():
    client = app_module.app.test_client()
    assert client.get("/segment-prs").status_code == 302
    assert b'href="/segment-prs" class="tool-card"' in client.get("/").data
    with client.session_transaction() as session:
        session["tokens"] = {}
    response = client.get("/segment-prs")
    assert response.status_code == 200
    assert b'id="segmentSearch"' in response.data
    assert b'id="worldSelect"' in response.data
    assert b'<option value="">All worlds</option>' in response.data
    assert b'<option value="2">Richmond</option>' in response.data
    assert b'id="segmentSelect"' in response.data


def test_recent_listing_reuses_90_day_pagination_and_isolates_profile(monkeypatch):
    mock_profile(monkeypatch)
    now = datetime.now(timezone.utc)
    calls = []

    def fetch(headers, profile_id, days, page_size):
        calls.append((profile_id, days, page_size))
        return [{"id": "11", "name": "Ride", "start_dt": now,
                 "end_dt": now + timedelta(seconds=100), "duration_sec": 100}]

    monkeypatch.setattr(app_module, "_fetch_recent_activities_for_profile", fetch)
    data = app_module.app.test_client().get("/api/segment_prs/activities").get_json()
    assert calls == [(7, 90, 50)]
    assert data["profile_id"] == "7"
    assert data["weight_kg"] == 75
    assert data["height_cm"] == 180
    assert data["catalog_version"] == "test-v1"
    assert data["activities"][0]["activity_id"] == "11"


def test_activity_api_records_traversals_with_current_measurements(monkeypatch):
    mock_profile(monkeypatch)
    frame = frame_from_xy([(x, 0) for x in range(-5, 106, 10)])
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (
        {"test": True}, {"profileId": 7, "worldId": 1, "eventInfo": {"eventSubGroupId": 8}}, None,
    ))
    monkeypatch.setattr(app_module, "convert_telemetry_to_dataframe", lambda _: frame)
    monkeypatch.setattr(app_module, "world_segments", lambda *args: [segment([(0, 0), (100, 0)])])
    response = app_module.app.test_client().get("/api/segment_prs/activity/11")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["traversals"]) == 1
    assert data["profile_id"] == "7"
    assert data["is_race"] is True
    assert data["weight_kg"] == 75
    assert data["height_cm"] == 180
    assert data["traversals"][0]["avg_draft_watts"] is not None


def test_activity_api_can_match_without_optional_sensors(monkeypatch):
    profile = mock_profile(monkeypatch)
    profile.pop("height")
    x = np.arange(-5, 106, 10)
    telemetry = {
        "timeInSec": list(range(len(x))), "distanceInCm": ((x + 5) * 100).tolist(),
        "latitude": (x / 111_320).tolist(), "longitude": [0] * len(x),
    }
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (
        telemetry, {"profileId": 7, "worldId": 1}, None,
    ))
    monkeypatch.setattr(app_module, "world_segments", lambda *args: [segment([(0, 0), (100, 0)])])
    response = app_module.app.test_client().get("/api/segment_prs/activity/11")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["traversals"]) == 1
    assert data["traversals"][0]["avg_power"] is None
    assert data["traversals"][0]["avg_draft_watts"] is None
    assert "height" in data["warning"]


@pytest.mark.parametrize("world_id", [14, "14", 1000, None])
def test_video_activity_reports_unavailable_timed_segments_not_a_load_failure(monkeypatch, world_id):
    mock_profile(monkeypatch)
    monkeypatch.setattr(app_module, "fetch_rider_telemetry", lambda *args: (
        {"timeInSec": [0, 1], "distanceInCm": [0, 1000], "latitude": [56.63, 56.631],
         "longitude": [-4.89, -4.89]},
        {"profileId": 7, "worldId": world_id, "routeId": 2010849247}, None,
    ))
    response = app_module.app.test_client().get("/api/segment_prs/activity/11")
    assert response.status_code == 200
    data = response.get_json()
    assert data["traversals"] == []
    assert data["segment_status"] == "unavailable"
    assert "video routes" in data["warning"]
    assert "timed segments" in data["warning"]


def test_video_world_resolves_without_route_id_and_preserves_virtual_worlds():
    assert segment_prs.segment_map_id({"worldId": 14}) == 1000
    assert segment_prs.segment_map_id({"worldId": 1, "routeId": 999}) == 1
    with pytest.raises(ValueError, match="no world ID"):
        segment_prs.segment_map_id({"routeId": 999})


def test_video_catalog_changes_invalidate_cached_results(tmp_path, monkeypatch):
    monkeypatch.setattr(segment_prs.surface_map, "ROUTE_DIR", tmp_path)
    path = tmp_path / "video_routes.json"
    path.write_text(json.dumps({"routes": []}), encoding="utf-8")
    previous = segment_prs.catalog_version()
    path.write_text(json.dumps({"routes": [{"nameHash": 1, "source_mapID": 14}]}), encoding="utf-8")
    assert segment_prs.catalog_version() != previous
    segment_prs._catalog_version.cache_clear()


@pytest.mark.parametrize("owner,error,status", [(8, None, 403), (7, "Telemetry error: 503", 502)])
def test_activity_api_does_not_cache_failures_or_other_users(monkeypatch, owner, error, status):
    mock_profile(monkeypatch)
    monkeypatch.setattr(app_module, "fetch_rider_telemetry",
                        lambda *args: ({"test": True}, {"profileId": owner, "worldId": 1}, error))
    response = app_module.app.test_client().get("/api/segment_prs/activity/11")
    assert response.status_code == status
    assert "error" in response.get_json()
    assert "traversals" not in response.get_json()


def test_segment_api_authentication_and_request_failure(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: None)
    client = app_module.app.test_client()
    assert client.get("/api/segment_prs/activity/11").status_code == 401
    assert client.get("/api/segment_prs/activities").status_code == 401
    mock_profile(monkeypatch)
    assert client.get("/api/segment_prs/activity/invalid").status_code == 400

    def fail(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(app_module, "fetch_rider_telemetry", fail)
    assert client.get("/api/segment_prs/activity/11").status_code == 502


def test_real_catalog_deduplicates_directional_segments():
    catalog = segment_prs.world_segments(1, segment_prs.catalog_version())
    assert catalog
    assert len({row["segment_id"] for row in catalog}) == len(catalog)
    assert any("sprint" in row["name"].casefold() for row in catalog)
    assert all(len(row["latlng"]) >= 2 and row["path_distance_m"] > 0 for row in catalog)


def test_same_name_forward_reverse_segments_get_distinct_labels_without_merging_geometry():
    catalog = segment_prs.world_segments(13, segment_prs.catalog_version())
    variants = {row["segment_id"]: row for row in catalog if "Breakaway Brae" in row["name"]}
    forward = variants["13:18228230436"]
    reverse = variants["13:-9223372018626545372"]
    assert forward["name"] == "Breakaway Brae (Forward)"
    assert reverse["name"] == "Breakaway Brae (Reverse)"
    assert forward["direction"] == "F"
    assert reverse["direction"] == "R"
    assert forward["path_distance_m"] == pytest.approx(399.72, abs=.1)
    assert reverse["path_distance_m"] == pytest.approx(624.32, abs=.1)


def test_croissant_sprint_ignores_nearby_false_finish_crossings():
    surface_map = segment_prs.surface_map
    route = surface_map.get_route(10, 1600198346)
    parts = [part for part in (route.get("leadin"), route.get("route")) if part]
    xy = np.column_stack([
        np.concatenate([part["local_x"] for part in parts]),
        np.concatenate([part["local_z"] for part in parts]),
    ])
    # Ride beyond the race finish so the final sprint's line is fully crossed.
    direction = xy[-1] - xy[-2]
    direction /= np.linalg.norm(direction)
    xy = np.vstack([xy, xy[-1] + direction * 12, xy[-1] + direction * 24])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    valid = np.r_[True, np.diff(distance) > 0]
    gps = np.asarray(surface_map.local_to_latlng(10, xy[:, 0], xy[:, 1]))
    frame = pd.DataFrame({
        "time_sec": distance[valid] / 12, "distance_km": distance[valid] / 1000,
        "lat": gps[valid, 0], "lng": gps[valid, 1],
    })
    catalog = segment_prs.world_segments(10, segment_prs.catalog_version())
    sprint = next(row for row in catalog if row["segment_id"] == "10:-9223372035801291814")
    starts = list(segment_prs._crossings(frame, sprint["latlng"], 0))
    ends = list(segment_prs._crossings(frame, sprint["latlng"], len(sprint["latlng"]) - 1))
    assert ends[0] - starts[0] < 5  # A nearby road crosses the line before the actual finish.
    rows = segment_prs.find_traversals(frame, [sprint])
    assert len(rows) == 2
    assert [row["pass"] for row in rows] == [1, 2]
    assert all(row["distance_m"] == pytest.approx(749, abs=5) for row in rows)
    assert all(row["total_time_sec"] == pytest.approx(749 / 12, abs=1) for row in rows)
