import app as app_module
import numpy as np


def test_parse_ttt_manual_activity_ids_accepts_ids_and_urls():
    raw = "123\nhttps://www.zwift.com/activity/456?foo=bar, 123"

    assert app_module._parse_ttt_manual_activity_ids(raw) == ["123", "456"]


def test_parse_ttt_manual_activity_ids_rejects_invalid_value():
    try:
        app_module._parse_ttt_manual_activity_ids("123, not-an-activity")
    except ValueError as error:
        assert str(error) == "Invalid Zwift activity ID or URL: not-an-activity"
    else:
        raise AssertionError("Expected invalid manual activity to be rejected")


def test_load_ttt_manual_participants_hydrates_and_skips_result_activity(monkeypatch):
    activity_data = {
        "profile": {
            "id": 42,
            "firstName": "Dana",
            "lastName": "Rider [RTB]",
            "weightInGrams": 67890,
            "heightInCentimeters": 171,
        }
    }
    requested = []

    def fake_get_activity_details(activity_id, headers):
        requested.append((activity_id, headers))
        return activity_data, None

    monkeypatch.setattr(app_module, "get_activity_details", fake_get_activity_details)

    participants, error = app_module._load_ttt_manual_participants(
        "111, 222", {"Authorization": "Bearer test"}, ["111"]
    )

    assert error is None
    assert requested == [("222", {"Authorization": "Bearer test"})]
    assert participants == [
        {
            "rank": None,
            "name": "Dana Rider [RTB]",
            "activity_id": "222",
            "weight_kg": 67.9,
            "weight_is_event_recorded": True,
            "player_id": 42,
            "is_manual": True,
            "height_cm": 171,
        }
    ]


def test_load_ttt_manual_participants_reports_activity_error(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_activity_details",
        lambda activity_id, headers: (None, "Error fetching activity: 404"),
    )

    participants, error = app_module._load_ttt_manual_participants("999", {}, [])

    assert participants is None
    assert error == "Could not add activity 999: Error fetching activity: 404"


def test_load_ttt_manual_participant_fetches_missing_profile_measurements(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_activity_details",
        lambda activity_id, headers: ({
            "profile": {"id": 42, "firstName": "Tim", "lastName": "Johnson"}
        }, None),
    )

    class ProfileResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"weight": 70123, "height": 1800}

    requested = []

    def fake_request(method, url, **kwargs):
        requested.append((method, url, kwargs))
        return ProfileResponse()

    monkeypatch.setattr(app_module, "_request_with_retry", fake_request)

    participants, error = app_module._load_ttt_manual_participants(
        "222", {"Authorization": "Bearer test"}, []
    )

    assert error is None
    assert participants[0]["weight_kg"] == 70.1
    assert participants[0]["height_cm"] == 180.0
    assert participants[0]["weight_is_event_recorded"] is False
    assert requested == [(
        "GET",
        f"{app_module.BASE_URL}/profiles/42",
        {"headers": {"Authorization": "Bearer test"}, "timeout": 10},
    )]


def test_apply_ttt_time_offset_uses_base_clock_without_stacking():
    rider = {
        "activity_id": "222",
        "is_manual": True,
        "time_sec": np.array([10.0, 11.0, 12.0]),
        "base_time_sec": np.array([10.0, 11.0, 12.0]),
        "time_offset_sec": 0.0,
        "calculated_time_offset_sec": 0.0,
        "time_offset_source": "none",
    }
    processed = {"222": rider}

    assert app_module._apply_ttt_time_offsets(processed, {"222": 15}) is None
    np.testing.assert_array_equal(rider["time_sec"], [25.0, 26.0, 27.0])
    assert rider["time_offset_sec"] == 15.0

    assert app_module._apply_ttt_time_offsets(processed, {"222": -4}) is None
    np.testing.assert_array_equal(rider["time_sec"], [6.0, 7.0, 8.0])
    assert rider["time_offset_sec"] == -4.0


def test_apply_ttt_time_offset_allows_result_activity():
    processed = {
        "111": {
            "activity_id": "111",
            "is_manual": False,
            "time_sec": np.array([10.0]),
            "base_time_sec": np.array([8.0]),
            "time_offset_sec": 2.0,
        }
    }

    error = app_module._apply_ttt_time_offsets(processed, {"111": -3})

    assert error is None
    np.testing.assert_array_equal(processed["111"]["time_sec"], [5.0])


def test_rider_summary_exposes_calculated_offset():
    summary = app_module._ttt_rider_summary({
        "activity_id": "111",
        "name": "Finished Rider",
        "weight_kg": 75.0,
        "is_manual": False,
        "time_offset_sec": -27.346,
        "calculated_time_offset_sec": -27.346,
        "time_offset_source": "calculated",
    })

    assert summary["time_offset_sec"] == -27.346
    assert summary["calculated_time_offset_sec"] == -27.346
    assert summary["time_offset_source"] == "calculated"


def test_reassign_endpoint_applies_and_returns_manual_offset(monkeypatch):
    rider = {
        "activity_id": "222",
        "name": "Dana Rider",
        "weight_kg": 67.9,
        "is_manual": True,
        "time_sec": np.array([10.0, 11.0]),
        "base_time_sec": np.array([10.0, 11.0]),
        "time_offset_sec": 0.0,
        "calculated_time_offset_sec": 0.0,
        "time_offset_source": "none",
    }
    cache_key = 987654
    monkeypatch.setitem(
        app_module._ttt_cache,
        cache_key,
        {
            "event_name": "Test TTT",
            "ttt_bike": "Test bike",
            "all_processed": {"222": rider},
            "team_tags": ["RTB"],
            "race_distance_km": 20,
            "team_assignments": {"222": "RTB"},
        },
    )

    def fake_build(all_processed, assignments, ordered_tags):
        np.testing.assert_array_equal(all_processed["222"]["time_sec"], [22.0, 23.0])
        return [
            {
                "label": "RTB",
                "riders": [app_module._ttt_rider_summary(all_processed["222"])],
            }
        ]

    monkeypatch.setattr(app_module, "_build_ttt_team_results", fake_build)

    response = app_module.app.test_client().post(
        "/api/ttt/reassign",
        json={
            "subgroup_id": cache_key,
            "assignments": {"222": "RTB"},
            "time_offsets": {"222": 12},
        },
    )

    assert response.status_code == 200
    rider_json = response.get_json()["teams"][0]["riders"][0]
    assert rider_json["is_manual"] is True
    assert rider_json["time_offset_sec"] == 12.0