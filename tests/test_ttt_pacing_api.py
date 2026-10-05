import numpy as np
import pytest
import json
import threading

import app as app_module
from bike_comparison.pacing_planner import RouteProfile
from bike_comparison.ttt_planner import TTTPlanResult


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "{}"

    def json(self):
        return self._payload


def test_ttt_page_renders():
    response = app_module.app.test_client().get("/ttt-pacing")

    assert response.status_code == 200
    assert b"TTT Pacing Planner" in response.data
    assert b'id="maxPowerPct" value="150"' in response.data
    assert b'id="draftSecondPct" value="25"' in response.data
    assert b'id="draftRestPct" value="40"' in response.data


def test_rider_lookup_requires_login(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: None)

    response = app_module.app.test_client().post(
        "/api/ttt_pacing/riders", json={"zwift_ids": ["123"]})

    assert response.status_code == 401


def test_rider_lookup_uses_profile_height_weight_and_zftp(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "Bearer x"})
    profiles = {
        "111": FakeResponse(200, {"firstName": "Ann", "lastName": "A", "height": 1750,
                                  "weight": 68000, "ftp": 290}),
        "222": FakeResponse(404, {}),
    }
    monkeypatch.setattr(app_module.requests, "get",
                        lambda url, **kwargs: profiles[url.rsplit("/", 1)[-1]])

    response = app_module.app.test_client().post(
        "/api/ttt_pacing/riders", json={"zwift_ids": ["111", " 222 ", "111"]})

    riders = response.get_json()["riders"]
    assert [r["zwift_id"] for r in riders] == ["111", "222"]
    assert riders[0] == {"zwift_id": "111", "name": "Ann A", "height_cm": 175.0,
                         "weight_kg": 68.0, "ftp_w": 290, "w_prime_kj": 20.0}
    assert riders[1]["ftp_w"] is None
    assert "error" in riders[1]


def test_rider_lookup_rejects_non_numeric_ids(monkeypatch):
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "Bearer x"})

    response = app_module.app.test_client().post(
        "/api/ttt_pacing/riders", json={"zwift_ids": ["12a"]})

    assert response.status_code == 400


@pytest.mark.parametrize("max_power_pct", [None, 125.0])
def test_plan_endpoint_returns_team_plan(monkeypatch, max_power_pct):
    route = RouteProfile(name="Flat", distance_m=np.array([0.0, 4000.0]),
                         altitude_m=np.array([0.0, 0.0]))
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: route)
    db = app_module.get_db()
    frame = next(iter(db.frames))
    riders = [{"name": f"R{i}", "height_cm": 180, "weight_kg": 75, "cp_w": 300 - 10 * i,
               "w_prime_kj": 20} for i in range(4)]

    payload = {
        "route_name": "Flat", "frame_id": frame, "upgrade_level": 0,
        "riders": riders, "allow_drops": False,
    }
    if max_power_pct is not None:
        payload["max_power_pct"] = max_power_pct
    response = app_module.app.test_client().post("/api/ttt_pacing_plan", json=payload)

    data = response.get_json()
    assert response.status_code == 200, data
    assert data["team_size"] == 4
    assert data["scoring_rider_count"] == 3
    assert data["feasible"]
    assert [r["name"] for r in data["riders"]] == ["R0", "R1", "R2", "R3"]
    assert set(data["profile"]["wbal_j"]) == {"R0", "R1", "R2", "R3"}
    assert all(row["min_wbal_j"] >= 0 for row in data["riders"])
    for spec in riders:
        power = [value for value in data["profile"]["power_w"][spec["name"]]
                 if value is not None]
        assert max(power) <= (max_power_pct or 150.0) / 100.0 * spec["cp_w"] + 0.5
        peak = [value for value in data["profile"]["peak_power_w"][spec["name"]]
            if value is not None]
        assert max(peak) <= (max_power_pct or 150.0) / 100.0 * spec["cp_w"] + 0.5
        assert data["profile"]["entry_speed_kph"][1:] == data["profile"]["exit_speed_kph"][:-1]


def test_plan_endpoint_rejects_wrong_team_size(monkeypatch):
    route = RouteProfile(name="Flat", distance_m=np.array([0.0, 4000.0]),
                         altitude_m=np.array([0.0, 0.0]))
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: route)
    frame = next(iter(app_module.get_db().frames))
    riders = [{"name": "Solo", "height_cm": 180, "weight_kg": 75, "cp_w": 300, "w_prime_kj": 20}]

    response = app_module.app.test_client().post("/api/ttt_pacing_plan", json={
        "route_name": "Flat", "frame_id": frame, "riders": riders,
    })

    assert response.status_code == 400
    assert "4-8 riders" in response.get_json()["error"]


def stream_request(monkeypatch):
    route = RouteProfile(name="Flat", distance_m=np.array([0.0, 4000.0]),
                         altitude_m=np.array([0.0, 0.0]))
    monkeypatch.setattr(app_module, "load_route_profile", lambda *args, **kwargs: route)
    return {
        "route_name": "Flat", "frame_id": next(iter(app_module.get_db().frames)),
        "stream": True,
        "riders": [{"name": f"R{index}", "height_cm": 180, "weight_kg": 75,
                    "cp_w": 300, "w_prime_kj": 20} for index in range(4)],
    }


def stream_plan():
    return TTTPlanResult(
        route_name="Flat", team_size=4, scoring_rider_count=3, feasible=True,
        total_time_seconds=360.0, total_distance_km=4.0, total_ascent_m=0.0,
        avg_speed_kph=40.0)


def test_plan_stream_outputs_progress_before_completion(monkeypatch):
    release = threading.Event()
    result = stream_plan()

    def optimizer(route, riders, *, progress_callback, **kwargs):
        progress_callback({"iteration": 1, "stage": "Team together", "step": "power search",
                           "candidate_iteration": 1, "candidate_iterations": 12,
                           "candidate_feasible": True, "improved": True, "plan": result})
        assert release.wait(timeout=5.0)
        return result

    monkeypatch.setattr(app_module, "plan_ttt_pacing", optimizer)
    response = app_module.app.test_client().post(
        "/api/ttt_pacing_plan", json=stream_request(monkeypatch), buffered=False)
    try:
        assert response.mimetype == "text/event-stream"
        chunks = iter(response.response)
        assert json.loads(next(chunks).decode()[6:])["type"] == "started"
        update = json.loads(next(chunks).decode()[6:])
        assert update["type"] == "progress"
        assert update["iteration"] == 1
        assert update["plan"]["feasible"]
        assert not release.is_set()
        release.set()
        complete = json.loads(next(chunks).decode()[6:])
        assert complete["type"] == "complete"
        assert complete["plan"] == update["plan"]
        assert list(chunks) == []
    finally:
        release.set()
        response.close()


def test_plan_stream_reports_worker_errors(monkeypatch):
    def optimizer(*args, **kwargs):
        raise RuntimeError("Optimization failed")

    monkeypatch.setattr(app_module, "plan_ttt_pacing", optimizer)
    response = app_module.app.test_client().post(
        "/api/ttt_pacing_plan", json=stream_request(monkeypatch), buffered=True)

    events = [json.loads(block[6:]) for block in response.get_data(as_text=True).split("\n\n")
              if block.startswith("data: ")]
    assert [event["type"] for event in events] == ["started", "error"]
    assert events[-1]["error"] == "Optimization failed"


def test_plan_stream_rejects_invalid_input_before_starting(monkeypatch):
    payload = stream_request(monkeypatch)
    payload["riders"] = payload["riders"][:1]
    response = app_module.app.test_client().post("/api/ttt_pacing_plan", json=payload)

    assert response.status_code == 400
    assert "4-8 riders" in response.get_json()["error"]


@pytest.mark.parametrize("stream", [False, True])
def test_plan_endpoint_passes_custom_draft_percentages(monkeypatch, stream):
    payload = stream_request(monkeypatch)
    payload.update(stream=stream, draft_second_pct=10.0, draft_rest_pct=15.0)
    received = []

    def optimizer(route, riders, **kwargs):
        received.append(kwargs)
        return stream_plan()

    monkeypatch.setattr(app_module, "plan_ttt_pacing", optimizer)
    response = app_module.app.test_client().post("/api/ttt_pacing_plan", json=payload,
                                                buffered=True)

    assert response.status_code == 200
    assert received[0]["draft_second_pct"] == 10.0
    assert received[0]["draft_rest_pct"] == 15.0


@pytest.mark.parametrize("field", ["draft_second_pct", "draft_rest_pct"])
@pytest.mark.parametrize("value", [-1.0, 100.0, "nan", "inf", "invalid"])
def test_plan_endpoint_rejects_invalid_draft_percentages(monkeypatch, field, value):
    payload = stream_request(monkeypatch)
    payload[field] = value
    response = app_module.app.test_client().post("/api/ttt_pacing_plan", json=payload)

    assert response.status_code == 400
    assert field in response.get_json()["error"] or "Invalid request" in response.get_json()["error"]


def test_plan_stream_runs_real_optimizer(monkeypatch):
    optimizer = app_module.plan_ttt_pacing

    def quick_optimizer(route, riders, **kwargs):
        return optimizer(route, riders, **kwargs, refine_pulls=False,
                         min_pull_s=20.0, max_pull_s=20.0, speed_step_mps=1.0,
                         search_iterations=2, final_iterations=2)

    monkeypatch.setattr(app_module, "plan_ttt_pacing", quick_optimizer)
    payload = stream_request(monkeypatch)
    payload["allow_drops"] = False
    response = app_module.app.test_client().post(
        "/api/ttt_pacing_plan", json=payload, buffered=True)
    events = [json.loads(block[6:]) for block in response.get_data(as_text=True).split("\n\n")
              if block.startswith("data: ")]
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "complete"
    updates = [event for event in events if event["type"] == "progress"]
    assert len(updates) >= 4
    plans = [event["plan"] for event in updates if event["plan"] is not None]
    assert plans
    for plan in plans + [events[-1]["plan"]]:
        assert plan["feasible"]
        assert all(row["min_wbal_j"] >= 0 for row in plan["riders"])
        assert all(value <= 450.5 for series in plan["profile"]["power_w"].values()
                   for value in series if value is not None)


def test_closing_plan_stream_cancels_worker_at_next_update(monkeypatch):
    release = threading.Event()
    stopped = threading.Event()

    def optimizer(route, riders, *, progress_callback, **kwargs):
        try:
            progress_callback({"iteration": 1, "plan": None})
            assert release.wait(timeout=5.0)
            progress_callback({"iteration": 2, "plan": None})
            pytest.fail("Disconnected optimization was not cancelled")
        finally:
            stopped.set()

    monkeypatch.setattr(app_module, "plan_ttt_pacing", optimizer)
    response = app_module.app.test_client().post(
        "/api/ttt_pacing_plan", json=stream_request(monkeypatch), buffered=False)
    try:
        next(iter(response.response))
        response.close()
    finally:
        release.set()
        response.close()
    assert stopped.wait(timeout=5.0)
