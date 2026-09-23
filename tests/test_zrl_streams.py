from unittest.mock import patch

import app as app_module


PARTICIPANTS = [
    {
        "player_id": 3201227,
        "end_date": "2026-09-23T10:35:00Z",
        "elapsed_ms": 2_100_000,
    },
    {
        "player_id": 123,
        "end_date": "2026-09-23T10:34:50Z",
        "elapsed_ms": 2_090_000,
    },
]


def test_zrl_stream_context_uses_race_clock_and_unique_player_ids():
    context = app_module._zrl_stream_context(PARTICIPANTS)

    assert context == {
        "race_start_time": "2026-09-23T10:00:00Z",
        "race_duration_sec": 2100.0,
        "player_ids": [123, 3201227],
    }


def test_zrl_stream_endpoint_caches_matching_streams():
    stream = {
        "streamer_name": "Njål Pedersen",
        "youtube_url": "https://www.youtube.com/watch?v=test",
        "stream_title": "ZRL",
        "offset_seconds": 742,
    }
    app_module._zrl_stream_cache.clear()
    with (
        patch.object(
            app_module,
            "get_headers",
            return_value={"Authorization": "Bearer test"},
        ),
        patch.object(
            app_module,
            "get_race_entries",
            return_value=(PARTICIPANTS, None),
        ),
        patch("shared.youtube_streams.get_api_key", return_value="test-key"),
        patch(
            "shared.youtube_streams.find_matching_streams",
            return_value=[stream],
        ) as matcher,
    ):
        client = app_module.app.test_client()
        first = client.get("/api/zrl/streams?subgroup_id=7354731")
        second = client.get("/api/zrl/streams?subgroup_id=7354731")

    assert first.status_code == 200
    assert first.get_json()["streams"] == [stream]
    assert second.get_json() == first.get_json()
    assert matcher.call_count == 1