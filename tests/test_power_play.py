import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import app as app_module
from shared import power_play
from tools.extract_zwift_routes import derive_arch_passages, load_multiroot, parse_arch_definitions


def arch(duration=5):
    return {
        "name": f"{duration} Sec", "duration_sec": duration, "road_id": 1,
        "latlng": [[0, 0], [10 / 111_320, 0]],
    }


def frame(times=None, power=1500):
    times = np.arange(32, dtype=float) if times is None else np.asarray(times, dtype=float)
    return pd.DataFrame({
        "time_sec": times, "lat": (times - .5) * 10 / 111_320,
        "lng": np.zeros(len(times)), "power_watts": np.full(len(times), power),
        "distance_km": times * .01, "speed_kmh": np.full(len(times), 36),
    })


def test_exact_interpolated_duration_not_distance_or_rolling_peak():
    samples = frame()
    samples["power_watts"] = samples["time_sec"] * 100
    efforts, incomplete = power_play.best_efforts(samples, [arch(5), arch(10), arch(20)])
    assert incomplete == 0
    assert efforts[5]["start_time_sec"] == pytest.approx(.5)
    assert efforts[5]["end_time_sec"] == pytest.approx(5.5)
    assert efforts[5]["avg_power"] == pytest.approx(300)
    assert efforts[10]["avg_power"] == pytest.approx(550)
    assert efforts[20]["avg_power"] == pytest.approx(1050)


def test_best_of_multiple_arch_crossings_and_raw_power_preserved():
    samples = frame()
    samples["lat"] = ((samples["time_sec"] % 10) - .5) * 10 / 111_320
    samples.loc[samples["time_sec"] >= 20, "power_watts"] = 2400
    efforts, incomplete = power_play.best_efforts(samples, [arch()])
    assert efforts[5]["avg_power"] == 2400
    assert efforts[5]["start_time_sec"] == pytest.approx(20.5)
    assert incomplete == 1
    assert samples["power_watts"].max() == 2400


def test_reverse_and_lateral_passages_do_not_score():
    backwards = frame()
    backwards["lat"] *= -1
    assert power_play.best_efforts(backwards, [arch()])[0] == {}
    off_road = frame()
    off_road["lng"] = 40 / 111_320
    assert power_play.best_efforts(off_road, [arch()])[0] == {}


@pytest.mark.parametrize("kind", ["cutoff", "missing_power", "missing_gps", "gap", "race_finish"])
def test_incomplete_windows_are_not_extrapolated_or_filled(kind):
    samples = frame()
    end = None
    if kind == "cutoff":
        samples = samples.iloc[:5]
    elif kind == "missing_power":
        samples.loc[3, "power_watts"] = np.nan
    elif kind == "missing_gps":
        samples.loc[3, "lat"] = np.nan
    elif kind == "gap":
        samples = samples.drop(index=[2, 3])
    else:
        end = 5
    efforts, incomplete = power_play.best_efforts(samples, [arch()], end_time=end)
    assert efforts == {}
    assert incomplete == 1


def test_pre_race_and_negative_power():
    assert power_play.best_efforts(frame(), [arch()], start_time=1)[0] == {}
    efforts, _ = power_play.best_efforts(frame(power=-50), [arch()])
    assert efforts[5]["avg_power"] == -50


def test_arch_extraction_preserves_native_duration_hash_and_start_geometry():
    definitions = parse_arch_definitions(b'''
        <ent type="ENTITY_TYPE_POWERARCH" m_Name="5 Sec" m_roadId="9"
             m_roadTime="0.5" m_powerArchNetworkHash="9007199254740993" />
        <ent type="ENTITY_TYPE_POWERARCH" m_Name="Power" m_roadId="9"
             m_roadTime="0.8" m_segmentDurationSecs="20" />
    ''')
    root = load_multiroot(b'''<highrescheckpoint>
        <entry road="9" time="0" dir="1" x="0" y="0" z="0" />
        <entry road="9" time="1" dir="1" x="10000" y="0" z="0" />
        </highrescheckpoint>''')
    passages = derive_arch_passages(root, definitions)
    assert passages[0]["duration_sec"] == 5
    assert passages[0]["network_hash"] == "9007199254740993"
    assert passages[0]["start_line"] == [[50, 0], [60, 0]]
    assert passages[1]["duration_sec"] == 20


def test_shipped_power_play_route_has_six_directional_starts():
    arches = power_play.power_arches(4, 2898229208)
    assert sorted(arch["duration_sec"] for arch in arches) == [5, 5, 10, 10, 20, 20]
    assert all(arch["enabled"] and len(arch["latlng"]) == 2 for arch in arches)


def write_race(root, race_id="race_data_123", title="Stage 1 - Zwift Power Play - Test", weight=True):
    directory = root / race_id
    directory.mkdir()
    (directory / "race_meta.json").write_text(json.dumps({
        "race_name": title, "world_id": 4, "route_id": 2898229208,
        "event_id": 456,
        "source_activity_id": "9007199254740993",
        "race_start_time": "2026-11-19T12:00:00Z",
    }))
    summary = pd.DataFrame([
        {"rank": 1, "name": "<Fast>", "activity_id": "9007199254740993",
         "player_id": "9007199254740995", "weight_kg": 100,
         "weight_is_event_recorded": weight, "status": "SUCCESS"},
        {"rank": 2, "name": "Light", "activity_id": "9007199254740994",
         "player_id": "9007199254740996", "weight_kg": 50,
         "weight_is_event_recorded": weight, "status": "SUCCESS"},
    ])
    summary["activity_start_time"] = "2026-11-19T12:00:00Z"
    summary["elapsed_ms"] = 31000
    summary.to_csv(directory / "complete_race_summary.csv", index=False)
    for row in summary.to_dict("records"):
        stem = f"rank{row['rank']}_{row['activity_id']}"
        frame(power=1500 if row["rank"] == 1 else 1000).to_csv(directory / f"{stem}.csv", index=False)
        (directory / f"{stem}_raw.json").write_text("{}")
    return directory


def test_leaderboard_uses_raw_roster_and_recorded_weights(tmp_path, monkeypatch):
    directory = write_race(tmp_path)
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch(5), arch(10), arch(20)])
    data = power_play.build_leaderboard(directory)
    assert len(data["rows"]) == 6
    fast = next(row for row in data["rows"] if row["name"] == "<Fast>")
    light = next(row for row in data["rows"] if row["name"] == "Light")
    assert fast["avg_power"] == 1500
    assert fast["watts_per_kg"] == 15
    assert light["watts_per_kg"] == 20
    assert fast["activity_id"] == "9007199254740993"
    assert fast["player_id"] == "9007199254740995"
    assert data["warnings"] == []


def test_missing_weight_does_not_use_replay_default(tmp_path, monkeypatch):
    directory = write_race(tmp_path, weight=False)
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch()])
    data = power_play.build_leaderboard(directory)
    assert all(row["watts_per_kg"] is None and row["weight_kg"] is None for row in data["rows"])
    assert len(data["warnings"]) == 2
    assert all("event weight unavailable" in warning for warning in data["warnings"])


def test_leaderboard_excludes_warmup_and_post_race_arch_crossings(tmp_path, monkeypatch):
    directory = write_race(tmp_path)
    meta = json.loads((directory / "race_meta.json").read_text())
    meta["race_start_time"] = "2026-11-19T12:00:10Z"
    (directory / "race_meta.json").write_text(json.dumps(meta))
    summary = pd.read_csv(directory / "complete_race_summary.csv", dtype={"activity_id": str, "player_id": str})
    summary["elapsed_ms"] = 10000
    summary.to_csv(directory / "complete_race_summary.csv", index=False)
    for row in summary.to_dict("records"):
        samples = frame()
        samples["lat"] = ((samples["time_sec"] % 10) - .5) * 10 / 111_320
        samples["power_watts"] = np.where((samples["time_sec"] >= 10) & (samples["time_sec"] < 20), 300, 2400)
        samples.to_csv(directory / f"rank{row['rank']}_{row['activity_id']}.csv", index=False)
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch()])
    data = power_play.build_leaderboard(directory)
    assert len(data["rows"]) == 2
    assert all(row["avg_power"] == 300 for row in data["rows"])
    assert all(row["start_time_sec"] == pytest.approx(10.5) for row in data["rows"])


def test_late_recording_does_not_extend_official_race_finish(tmp_path, monkeypatch):
    directory = write_race(tmp_path)
    summary = pd.read_csv(directory / "complete_race_summary.csv", dtype={"activity_id": str, "player_id": str})
    summary["activity_start_time"] = "2026-11-19T12:00:10Z"
    summary["elapsed_ms"] = 20000
    summary.to_csv(directory / "complete_race_summary.csv", index=False)
    for row in summary.to_dict("records"):
        samples = frame()
        samples["lat"] = ((samples["time_sec"] % 10) - .5) * 10 / 111_320
        samples["power_watts"] = np.where(samples["time_sec"] < 10, 300, 2400)
        samples.to_csv(directory / f"rank{row['rank']}_{row['activity_id']}.csv", index=False)
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch()])
    data = power_play.build_leaderboard(directory)
    assert len(data["rows"]) == 2
    assert all(row["avg_power"] == 300 for row in data["rows"])
    assert all(row["start_time_sec"] == pytest.approx(.5) for row in data["rows"])


def test_standalone_api_and_title_rejection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "race_data"
    root.mkdir()
    write_race(root)
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch()])
    client = app_module.app.test_client()
    page = client.get("/power-play")
    assert page.status_code == 200
    assert b"<title>Stage 1 - Zwift Power Play (Archived)</title>" in page.data
    assert b'href="/?tab=archived" class="back-link"' in page.data
    assert b"<th>Category</th>" not in page.data
    assert b'data-power-sort="avg_power"' in page.data
    assert b'data-power-sort="watts_per_kg"' in page.data
    assert b'data-power-sort="watts_rank"' in page.data
    assert b'data-power-sort="wkg_rank"' in page.data
    assert b'id="rank-unit"' not in page.data
    response = client.get("/api/power-play/leaderboard/race_data_123")
    assert response.status_code == 200
    assert len(response.get_json()["rows"]) == 2
    write_race(root, race_id="race_data_124", title="Other race")
    response = client.get("/api/power-play/leaderboard/race_data_124")
    assert response.status_code == 400
    assert "only available for Stage 1" in response.get_json()["error"]
    assert client.get("/api/power-play/leaderboard/other").status_code == 400


def test_combined_event_keeps_best_per_rider_and_category(tmp_path, monkeypatch):
    first = write_race(tmp_path)
    second = write_race(tmp_path, race_id="race_data_124")
    faster = second / "rank1_9007199254740993.csv"
    samples = pd.read_csv(faster)
    samples["power_watts"] = 2000
    samples.to_csv(faster, index=False)
    event = tmp_path / "race_event_1"
    event.mkdir()
    (event / "event_manifest.json").write_text(json.dumps({
        "race_name": "STAGE 1 - ZWIFT POWER PLAY Test", "event_id": 456,
        "subgroups": [{"race_id": first.name, "label": "A"}, {"race_id": second.name, "label": "B"}],
    }))
    monkeypatch.setattr(power_play, "power_arches", lambda *_: [arch()])
    data = power_play.build_leaderboard(event)
    assert len(data["rows"]) == 2
    fast = next(row for row in data["rows"] if row["name"] == "<Fast>")
    assert fast["avg_power"] == 2000
    assert fast["category"] == "B"
    assert data["source_activity_id"] == "9007199254740993"


@pytest.mark.parametrize("title,expected", [
    ("Stage 1 - Zwift Power Play", True),
    ("Series: STAGE 1 - ZWIFT POWER PLAY - Test", True),
    ("Zwift Power Play - Stage 1", True),
    ("Stage 2 - Zwift Power Play", False),
    ("Stage 10 - Zwift Power Play", False),
    ("Stage 11 - Zwift Power Play", False),
    ("Zwift Power Play", False),
    ("Stage 1 - Other race", False),
])
def test_only_stage_one_is_supported(title, expected):
    assert power_play.is_power_play(title) is expected


def test_event_link_resolves_cached_pen_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "race_data"
    root.mkdir()
    write_race(root)
    headers = Mock(side_effect=AssertionError("Cached event must resolve offline"))
    monkeypatch.setattr(app_module, "get_headers", headers)
    response = app_module.app.test_client().get("/api/power-play/resolve/456")
    assert response.status_code == 200
    assert response.get_json() == {
        "race_id": "race_data_123", "source_activity_id": "9007199254740993",
    }


@pytest.mark.parametrize("title", [
    "Stage 2 - Zwift Power Play", "Stage 10 - Zwift Power Play", "Zwift Power Play",
])
def test_cached_other_stages_are_rejected_by_both_endpoints(tmp_path, monkeypatch, title):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "race_data"
    root.mkdir()
    write_race(root, title=title)
    client = app_module.app.test_client()
    for url in ("/api/power-play/resolve/456", "/api/power-play/leaderboard/race_data_123"):
        response = client.get(url)
        assert response.status_code == 400
        assert "only available for Stage 1" in response.get_json()["error"]


def test_uncached_event_resolves_source_from_its_only_pen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "get_headers", lambda: {"Authorization": "test"})
    event = {"id": 456, "name": "Stage 1 - Zwift Power Play", "eventSubgroups": [{"id": 123}]}
    request = Mock(return_value=Mock(status_code=200, json=lambda: event))
    monkeypatch.setattr(app_module, "_request_with_retry", request)
    monkeypatch.setattr(app_module.blob_storage, "race_exists_in_blob", lambda _: False)
    entries = Mock(return_value=([{"activity_id": "9007199254740993"}], None))
    monkeypatch.setattr(app_module, "get_race_entries", entries)
    response = app_module.app.test_client().get("/api/power-play/resolve/456")
    assert response.status_code == 200
    assert response.get_json() == {"race_id": None, "source_activity_id": "9007199254740993"}
    assert request.call_args.args[1].endswith("/events/456")
    entries.assert_called_once_with("123", {"Authorization": "test"})
    event["name"] = "Stage 2 - Zwift Power Play"
    entries.reset_mock()
    assert app_module.app.test_client().get("/api/power-play/resolve/456").status_code == 400
    entries.assert_not_called()
    event["name"] = "Stage 1 - Zwift Power Play"
    event["eventSubgroups"].append({"id": 124})
    response = app_module.app.test_client().get("/api/power-play/resolve/456")
    assert response.status_code == 400
    assert "exactly one pen" in response.get_json()["error"]
