import threading

import pytest
import requests

from tools import fetch_dirt_stats as stats


def payload():
    return (
        {"riderId": 7681059, "velo": {"race": 639.123456, "factors": {
            "endurance": 548, "pursuit": 730, "sprint": 693.5,
            "punch": 853.8, "climb": 684.5, "timeTrial": 602,
        }}},
        {"riderId": 7681059, "curve": [{
            "durationSec": 300,
            "live": {"w": 326.36, "wkg": 4.94},
            "d90": {"w": 300, "wkg": 4.51},
        }]},
    )


def test_maps_overall_six_scores_and_seven_intervals_without_deriving_or_rounding():
    rider, power = payload()
    result = stats.parse_stats(rider, power, 7681059)
    assert result["velo"]["overall"] == 639.123456
    assert result["velo"]["sprint"] == 693.5
    assert len(result["velo"]) == 7
    assert len(result["power"]) == 7
    assert result["power"]["300"] == {
        "live": {"w": 326.36, "wkg": 4.94},
        "best_90d": {"w": 300, "wkg": 4.51},
    }
    assert result["power"]["5"]["live"]["w"] is None


def test_missing_values_are_not_replaced_with_zero_and_raw_extremes_are_preserved():
    rider, power = payload()
    rider["velo"]["factors"] = {"sprint": -18.123456, "punch": 0}
    power["curve"][0]["live"] = {"w": 9999.123456, "wkg": -1.5}
    result = stats.parse_stats(rider, power, 7681059)
    assert result["velo"]["sprint"] == -18.123456
    assert result["velo"]["punch"] == 0
    assert result["velo"]["endurance"] is None
    assert result["power"]["300"]["live"]["w"] == 9999.123456
    assert result["power"]["300"]["live"]["wkg"] == -1.5


@pytest.mark.parametrize("value", [None, 0, -18.123456, 1007.8017236673211])
def test_overall_uses_only_the_source_race_score_and_preserves_raw_values(value):
    rider, power = payload()
    rider["velo"]["race"] = value
    assert stats.parse_stats(rider, power, 7681059)["velo"]["overall"] == value


def test_missing_overall_is_not_inferred_from_category_scores():
    rider, power = payload()
    del rider["velo"]["race"]
    assert stats.parse_stats(rider, power, 7681059)["velo"]["overall"] is None
    rider["velo"] = None
    assert all(value is None for value in stats.parse_stats(rider, power, 7681059)["velo"].values())


def test_rejects_mismatched_rider_and_missing_power_loader():
    rider, power = payload()
    with pytest.raises(ValueError, match="rider ID"):
        stats.parse_stats(rider, power, 123)
    with pytest.raises(ValueError, match="curve"):
        stats.parse_stats(rider, {"riderId": 7681059}, 7681059)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "300"])
@pytest.mark.parametrize("field", ["overall", "factor"])
def test_invalid_metrics_are_rejected(value, field):
    rider, power = payload()
    if field == "overall":
        rider["velo"]["race"] = value
    else:
        rider["velo"]["factors"]["punch"] = value
    with pytest.raises(ValueError, match="numeric"):
        stats.parse_stats(rider, power, 7681059)


def roster():
    return {"season": 14, "complete": True, "teams": [
        {"riders": [{"zwift_id": 7681059}]},
        {"riders": [{"zwift_id": 7681059}, {"zwift_id": 8072730}]},
    ]}


def test_collection_requests_each_unique_id_once_and_resumes(tmp_path):
    path = tmp_path / "stats.json"
    collection = stats.StatsCollection(roster(), path)
    task = collection.next_task()
    assert task["zwift_id"] == 7681059
    rider, power = payload()
    collection.accept({"url": task["url"], "final_url": task["url"], "status": 200,
                       "rider": rider, "power": power}, stats.RequestPacer(0.5))
    resumed = stats.StatsCollection(roster(), path)
    assert resumed.next_task()["zwift_id"] == 8072730
    assert resumed.snapshot()["rider_count"] == 2
    assert resumed.snapshot()["fetched_count"] == 1


def test_old_checkpoints_refetch_records_without_overall(tmp_path):
    path = tmp_path / "stats.json"
    collection = stats.StatsCollection(roster(), path)
    collection.riders = {"7681059": {"status": "ok", "velo": {"sprint": 693.5}}}
    collection.save()
    assert stats.StatsCollection(roster(), path).next_task()["zwift_id"] == 7681059


def test_unavailable_profile_is_explicit_and_does_not_invent_metrics(tmp_path):
    collection = stats.StatsCollection(roster(), tmp_path / "stats.json", rider_id=7681059)
    task = collection.next_task()
    collection.accept({"url": task["url"], "status": 404}, stats.RequestPacer(0.5))
    assert collection.next_task() is None
    snapshot = collection.snapshot()
    assert snapshot["complete"] is True
    assert snapshot["unavailable_count"] == 1
    assert "power" not in snapshot["riders"]["7681059"]


def test_authentication_redirect_stops_collection_without_marking_rider_complete(tmp_path):
    collection = stats.StatsCollection(roster(), tmp_path / "stats.json")
    task = collection.next_task()
    with pytest.raises(ValueError, match="sign-in"):
        collection.accept({"url": task["url"], "final_url": "https://www.zwiftracing.app/login", "status": 200}, stats.RequestPacer(0.5))
    assert collection.snapshot()["fetched_count"] == 0


def test_stats_relay_serves_saved_worker_and_rejects_other_origins(tmp_path):
    collection = stats.StatsCollection(roster(), tmp_path / "stats.json")
    with stats.make_server(collection, stats.RequestPacer(0.5), 0,
                           origin=stats.BASE_URL, worker=stats.BROWSER_WORKER) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            relay = f"http://127.0.0.1:{server.server_port}"
            response = requests.get(f"{relay}/worker", timeout=5)
            assert response.json()["worker"] == stats.BROWSER_WORKER
            assert response.headers["Access-Control-Allow-Origin"] == stats.BASE_URL
            response = requests.post(f"{relay}/page", headers={"Origin": "https://example.com"}, json={}, timeout=5)
            assert response.status_code == 403
        finally:
            server.shutdown()
            thread.join()