from contextlib import nullcontext
import csv
from email.utils import formatdate
import json
from unittest.mock import Mock

import pytest
import requests

from shared.zwift_auth import ZwiftTokens
from tools import fetch_tour_of_watopia as tour


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(tour.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tour.time, "time", lambda: 1_800_000_000 + now[0])
    monkeypatch.setattr(tour.time, "sleep", sleep)
    return now, sleeps


def tokens():
    return ZwiftTokens("access", "refresh", float("inf"))


def response(payload, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload).encode()
    return result


def test_pagination_preserves_raw_scores_and_server_capped_pages():
    pages = [
        {"entries": [{"id": 1, "racingScore": -12.5}], "total": 2},
        {"entries": [{"id": 2, "racingScore": 12345}], "total": 2},
    ]
    session = Mock()
    session.get.side_effect = [response(page) for page in pages] + [
        response({"entries": []})
    ]
    assert tour.fetch_classification(
        session, tokens(), "tourofwatopia2026", tour.DEFAULT_CLASSIFICATION
    ) == pages
    assert [call.kwargs["params"]["start"] for call in session.get.call_args_list] == [
        0, 1, 2
    ]
    request = session.get.call_args_list[0]
    assert request.args[0].endswith(
        "/classifications/%E2%80%9Coverall%E2%80%9D/entries"
    )
    assert request.kwargs["headers"]["Authorization"] == "Bearer access"
    assert request.kwargs["timeout"] == 30
    assert request.kwargs["allow_redirects"] is False


def test_accepts_bare_entry_array():
    session = Mock()
    session.get.side_effect = [response([{"id": 1}]), response([])]
    assert tour.fetch_classification(session, tokens(), "league", "overall") == [
        [{"id": 1}]
    ]


@pytest.mark.parametrize("payload", [{}, {"entries": None}, {"entries": [1]}, "html"])
def test_unknown_schema_is_an_error(payload):
    session = Mock()
    session.get.return_value = response(payload)
    with pytest.raises(ValueError, match="Unexpected API schema"):
        tour.fetch_classification(session, tokens(), "league", "overall")


def test_repeated_page_is_an_error():
    session = Mock()
    session.get.return_value = response([{"id": 1}])
    with pytest.raises(ValueError, match="repeated a page"):
        tour.fetch_classification(session, tokens(), "league", "overall")
    assert session.get.call_count == 2


def test_refreshes_expired_token(monkeypatch):
    refresh = Mock(return_value=tokens())
    monkeypatch.setattr(tour, "refresh_access_token", refresh)
    session = Mock()
    session.get.return_value = response([])
    tour.fetch_classification(session, ZwiftTokens("old", "refresh", 0), "l", "c")
    refresh.assert_called_once_with("refresh")


def test_refreshes_after_unauthorized(monkeypatch):
    refresh = Mock(return_value=ZwiftTokens("new", "refresh2", float("inf")))
    monkeypatch.setattr(tour, "refresh_access_token", refresh)
    session = Mock()
    session.get.side_effect = [response({}, 401), response([])]
    assert tour.fetch_classification(session, tokens(), "l", "c") == []
    refresh.assert_called_once_with("refresh")
    assert session.get.call_args.kwargs["headers"]["Authorization"] == "Bearer new"


@pytest.mark.parametrize("status,delay", [(429, 30), (503, 3)])
def test_retries_transient_errors(clock, status, delay):
    session = Mock()
    session.get.side_effect = [
        response({}, status, {"Retry-After": "3"}), response([])
    ]
    assert tour.fetch_classification(session, tokens(), "l", "c") == []
    assert clock[1] == [delay]


def test_retry_exhaustion_raises():
    session = Mock()
    session.get.return_value = response({}, 503)
    with pytest.raises(requests.HTTPError):
        tour.fetch_classification(session, tokens(), "l", "c")
    assert session.get.call_count == 4


def test_main_uses_environment_and_saves_raw_pages(monkeypatch, tmp_path):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    login = Mock(return_value=tokens())
    monkeypatch.setattr(tour, "get_token_with_password", login)
    pages = [{"entries": [{"id": 1, "racingScore": -20}]}]
    monkeypatch.setattr(tour, "fetch_classification", Mock(return_value=pages))
    output = tmp_path / "nested" / "results.json"
    assert tour.main(["--classification-only", "--output", str(output)]) == 0
    login.assert_called_once_with("rider@example.test", "test-password")
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pages"] == pages
    assert result["classification"] == tour.DEFAULT_CLASSIFICATION
    assert "test-password" not in output.read_text(encoding="utf-8")
    log = output.with_suffix(".log").read_text(encoding="utf-8")
    assert "Saved 1 classification pages" in log
    assert "test-password" not in log
    assert "rider@example.test" not in log


def test_main_keeps_login_prompts_visible_but_out_of_log(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ZWIFT_USERNAME", raising=False)
    monkeypatch.delenv("ZWIFT_PASSWORD", raising=False)

    def email(prompt):
        print(prompt, end="")
        return "rider@example.test"

    def password(prompt):
        print(prompt, end="")
        return "secret-password"

    monkeypatch.setattr("builtins.input", email)
    monkeypatch.setattr(tour.getpass, "getpass", password)
    login = Mock(return_value=tokens())
    monkeypatch.setattr(tour, "get_token_with_password", login)
    monkeypatch.setattr(tour, "fetch_classification", lambda *args, **kwargs: [])
    output = tmp_path / "results.json"
    log_path = tmp_path / "logs" / "custom.log"
    assert tour.main([
        "--classification-only", "--output", str(output), "--log-file", str(log_path),
    ]) == 0
    login.assert_called_once_with("rider@example.test", "secret-password")
    terminal = capsys.readouterr().out
    assert "Zwift email:" in terminal
    assert "Zwift password:" in terminal
    log = log_path.read_text(encoding="utf-8")
    assert "Saved 0 classification pages" in log
    for private in ["Zwift email:", "Zwift password:", "rider@example.test", "secret-password"]:
        assert private not in log
    assert not output.with_suffix(".log").exists()


def test_failure_preserves_existing_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    monkeypatch.setattr(
        tour, "fetch_classification", Mock(side_effect=ValueError("Unexpected API schema"))
    )
    output = tmp_path / "results.json"
    output.write_text("previous results", encoding="utf-8")
    log_path = tmp_path / "collection.log"
    assert tour.main([
        "--refresh-classification", "--output", str(output), "--log-file", str(log_path),
    ]) == 1
    assert output.read_text(encoding="utf-8") == "previous results"
    assert "Unexpected API schema" in capsys.readouterr().err
    assert "Unexpected API schema" in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["get_token_with_password", "refresh_access_token"])
def test_auth_requests_have_timeout(monkeypatch, name):
    from shared import zwift_auth

    post = Mock(return_value=response({
        "access_token": "access", "refresh_token": "refresh", "expires_in": 3600
    }))
    monkeypatch.setattr(zwift_auth.requests, "post", post)
    if name == "get_token_with_password":
        zwift_auth.get_token_with_password("email", "password")
    else:
        zwift_auth.refresh_access_token("refresh")
    assert post.call_args.kwargs["timeout"] == 30


def history(profile=1, event=100, subgroup=1000, score=660.11, rank=21):
    return {
        "profileId": profile, "eventId": event, "eventSubgroupId": subgroup,
        "rank": rank, "scoreHistory": {"previousScore": score, "newScore": 655.18},
    }


def test_full_export_cached_classification_and_single_bulk_fetch(monkeypatch, tmp_path):
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({
        "league": "tourofwatopia2026",
        "classification": tour.DEFAULT_CLASSIFICATION,
        "pages": [{"entries": [{"profileId": "1"}, {"profileId": "2"}, {"profileId": "1"}]}],
    }), encoding="utf-8")
    session = Mock()
    session.get.side_effect = [
        response({"entries": [history(score=-12.5), history(event=200)]}),
        response({"entries": [history(event=101, subgroup=1001, score=None, rank=None)]}),
        response({"entries": []}),
        response([history(profile=2, score=12345, rank=0)]),
        response([]),
    ]
    session.post.return_value = response([
        {"eventId": 100, "tags": ["tourofwatopia2026"]},
        {"eventId": 101, "tags": ["tourofwatopia2026"]},
        {"eventId": 200, "tags": ["tourofwatopia2026race"]},
    ])
    monkeypatch.setattr(tour.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    output = tmp_path / "results.csv"
    args = [
        "--classification-file", str(classification), "--csv-output", str(output),
        "--history-cache", str(tmp_path / "cache"),
        "--event-cache", str(tmp_path / "events"),
        "--since", "2026-10-01", "--until", "2026-10-19",
    ]
    assert tour.main(args) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == tour.CSV_FIELDS
        rows = list(reader)
    assert rows == [
        {"profileId": "1", "eventId": "100", "eventSubgroupId": "1000",
         "previousScore": "-12.5", "rank": "21", "newScore": "655.18"},
        {"profileId": "1", "eventId": "101", "eventSubgroupId": "1001",
         "previousScore": "", "rank": "", "newScore": "655.18"},
        {"profileId": "2", "eventId": "100", "eventSubgroupId": "1000",
         "previousScore": "12345", "rank": "0", "newScore": "655.18"},
    ]
    session.post.assert_called_once()
    assert session.post.call_args.args[0] == f"{tour.BASE_URL}/events/bulk-fetch"
    assert session.post.call_args.kwargs["json"] == [
        {"eventId": 100}, {"eventId": 101}, {"eventId": 200}
    ]
    assert session.get.call_args_list[0].args[0].endswith(
        "/race-results/profile/1/entries"
    )
    assert [call.kwargs["params"]["start"] for call in session.get.call_args_list] == [
        0, 2, 3, 0, 1
    ]
    assert all(
        call.kwargs["params"]["from"] == "2026-10-01"
        and call.kwargs["params"]["to"] == "2026-10-19"
        for call in session.get.call_args_list
    )
    session.get.reset_mock()
    session.post.reset_mock()
    assert tour.main(args) == 0
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_history_failure_resumes_completed_riders(monkeypatch, tmp_path):
    client = Mock()
    client.request.side_effect = [
        [history()], [], requests.ConnectionError("interrupted")
    ]
    with pytest.raises(requests.ConnectionError):
        tour.collect_histories(client, [1, 2], "2026-10-01", "2026-10-19", 50, tmp_path)
    assert (tmp_path / "1_2026-10-01_2026-10-19.json").exists()
    assert not (tmp_path / "2_2026-10-01_2026-10-19.json").exists()
    client.request.reset_mock()
    client.request.side_effect = [[history(profile=2)], []]
    records = tour.collect_histories(
        client, [1, 2], "2026-10-01", "2026-10-19", 50, tmp_path
    )
    assert [record["profileId"] for record in records] == [1, 2]
    assert client.request.call_count == 2
    assert client.request.call_args.args[1] == "race-results/profile/2/entries"


@pytest.mark.parametrize("status", [403, 404, 410])
def test_unavailable_event_stops_strict_export_but_preserves_available_cache(tmp_path, status):
    client = Mock()
    client.request.side_effect = [
        [{"eventId": 100, "tags": []}], requests.HTTPError(response=response({}, status))
    ]
    with pytest.raises(ValueError, match="1 events unavailable"):
        tour.fetch_events(client, [history(), history(event=200)], cache_dir=tmp_path)
    assert (tmp_path / "100.json").exists()
    assert not (tmp_path / "200.json").exists()


def test_no_histories_writes_header_only_without_bulk_fetch(tmp_path):
    client = Mock()
    assert tour.fetch_events(client, []) == {}
    client.request.assert_not_called()
    output = tmp_path / "empty.csv"
    tour.write_csv(output, [])
    assert output.read_text().strip() == ",".join(tour.CSV_FIELDS)


def test_missing_scores_are_retained_with_warning(capsys):
    record = history()
    del record["scoreHistory"]
    rows = tour.race_rows([record], {100: {"tags": ["tourofwatopia2026"]}},
                          "tourofwatopia2026")
    assert rows[0]["previousScore"] is None
    assert rows[0]["newScore"] is None
    assert rows[0]["rank"] == 21
    warning = capsys.readouterr().err
    assert "1 rows without previousScore" in warning
    assert "1 rows without newScore" in warning


@pytest.mark.parametrize("new_score", [0, -12.5, 12345.67, None])
def test_post_race_score_preserved_in_csv(tmp_path, capsys, new_score):
    record = history()
    record["scoreHistory"]["newScore"] = new_score
    rows = tour.race_rows(
        [record], {100: {"tags": ["tourofwatopia2026"]}}, "tourofwatopia2026"
    )
    assert rows[0]["newScore"] == new_score
    assert rows[0]["previousScore"] == 660.11
    output = tmp_path / "results.csv"
    tour.write_csv(output, rows)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "profileId", "eventId", "eventSubgroupId", "previousScore", "rank", "newScore"
        ]
        row = next(reader)
    assert row["newScore"] == ("" if new_score is None else str(new_score))
    assert ("1 rows without newScore" in capsys.readouterr().err) == (new_score is None)


def test_missing_post_race_score_does_not_substitute_previous_score(capsys):
    record = history()
    del record["scoreHistory"]["newScore"]
    rows = tour.race_rows(
        [record], {100: {"tags": ["tourofwatopia2026"]}}, "tourofwatopia2026"
    )
    assert rows[0]["newScore"] is None
    assert rows[0]["previousScore"] == 660.11
    assert "1 rows without newScore" in capsys.readouterr().err


def test_profile_mismatch_does_not_cache_history(tmp_path):
    client = Mock()
    client.request.side_effect = [[history(profile=2)], []]
    with pytest.raises(ValueError, match="profile mismatch"):
        tour.collect_histories(client, [1], "2026-10-01", "2026-10-19", 50, tmp_path)
    assert not list(tmp_path.iterdir())


def test_tokens_remain_refreshed_across_get_and_post(monkeypatch):
    refreshed = ZwiftTokens("new", "refresh2", float("inf"))
    refresh = Mock(return_value=refreshed)
    monkeypatch.setattr(tour, "refresh_access_token", refresh)
    session = Mock()
    session.get.return_value = response([])
    session.post.return_value = response([])
    client = tour.ZwiftClient(session, ZwiftTokens("old", "refresh", 0))
    client.request("GET", "test")
    client.request("POST", "events/bulk-fetch", json=[])
    refresh.assert_called_once_with("refresh")
    assert client.tokens is refreshed
    assert session.post.call_args.kwargs["headers"]["Authorization"] == "Bearer new"


def test_bulk_failure_keeps_existing_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    monkeypatch.setattr(tour, "fetch_classification", Mock(return_value=[
        {"entries": [{"profileId": "1"}]}
    ]))
    monkeypatch.setattr(tour, "collect_histories", Mock(return_value=[history()]))
    monkeypatch.setattr(tour, "fetch_events", Mock(side_effect=ValueError("omitted events")))
    output = tmp_path / "results.csv"
    output.write_text("previous output", encoding="utf-8")
    assert tour.main([
        "--output", str(tmp_path / "classification.json"), "--csv-output", str(output),
        "--history-cache", str(tmp_path / "cache"), "--event-cache", str(tmp_path / "events"),
    ]) == 1
    assert output.read_text(encoding="utf-8") == "previous output"


def test_reversed_date_range_rejected_before_login(monkeypatch):
    login = Mock()
    monkeypatch.setattr(tour, "get_token_with_password", login)
    with pytest.raises(SystemExit, match="2"):
        tour.main(["--since", "2026-10-19", "--until", "2026-10-01"])
    login.assert_not_called()


def test_successful_requests_are_paced_across_get_and_post(clock):
    session = Mock()
    starts = []

    def success(*args, **kwargs):
        starts.append(clock[0][0])
        return response([])

    session.get.side_effect = success
    session.post.side_effect = success
    client = tour.ZwiftClient(session, tokens())
    client.request("GET", "classification")
    client.request("GET", "history")
    client.request("POST", "events/bulk-fetch", json=[])
    assert starts == [0, 2, 4]


def test_rate_limit_slows_subsequent_requests_even_after_success(clock):
    session = Mock()
    session.get.side_effect = [response({}, 429), response([]), response([])]
    session.post.return_value = response([])
    client = tour.ZwiftClient(session, tokens())
    client.request("GET", "history")
    client.request("GET", "another-history")
    client.request("POST", "events/bulk-fetch", json=[])
    assert clock[1] == [30, 4, 4]


@pytest.mark.parametrize("kind", ["seconds", "date"])
def test_long_retry_after_is_never_shortened(clock, kind):
    header = "180" if kind == "seconds" else formatdate(
        tour.time.time() + 180, usegmt=True
    )
    session = Mock()
    session.get.side_effect = [response({}, 429, {"Retry-After": header}), response([])]
    client = tour.ZwiftClient(session, tokens())
    client.request("GET", "history")
    assert clock[1] == [180]


@pytest.mark.parametrize("header", ["broken", "-1", "NaN", "Infinity"])
def test_invalid_retry_after_warns_and_uses_safe_cooldown(clock, capsys, header):
    session = Mock()
    session.get.side_effect = [response({}, 429, {"Retry-After": header}), response([])]
    tour.ZwiftClient(session, tokens()).request("GET", "history")
    assert clock[1] == [30]
    assert "Invalid Retry-After" in capsys.readouterr().err


def test_repeated_429s_increase_cooldown(clock):
    session = Mock()
    session.get.side_effect = [
        response({}, 429), response({}, 429), response({}, 429), response([])
    ]
    session.post.return_value = response([])
    client = tour.ZwiftClient(session, tokens())
    client.request("GET", "history")
    client.request("POST", "events/bulk-fetch", json=[])
    assert clock[1] == [30, 60, 120, 16]


def test_request_interval_is_configurable_and_counts_network_time(clock):
    session = Mock()

    def slow_response(*args, **kwargs):
        clock[0][0] += 3
        return response([])

    session.get.side_effect = slow_response
    client = tour.ZwiftClient(session, tokens(), request_interval=5)
    client.request("GET", "first")
    client.request("GET", "second")
    assert clock[1] == [2]


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_request_interval_rejected_before_login(monkeypatch, interval):
    login = Mock()
    monkeypatch.setattr(tour, "get_token_with_password", login)
    with pytest.raises(SystemExit, match="2"):
        tour.main(["--request-interval", interval])
    login.assert_not_called()


def test_main_shares_custom_pacing_across_all_phases(monkeypatch, tmp_path, clock):
    session = Mock()
    starts = []
    payloads = iter([
        [{"profileId": "1"}], [], [history()], [],
        [{"eventId": 100, "tags": ["tourofwatopia2026"]}],
    ])

    def next_response(*args, **kwargs):
        starts.append(clock[0][0])
        return response(next(payloads))

    session.get.side_effect = next_response
    session.post.side_effect = next_response
    monkeypatch.setattr(tour.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    assert tour.main([
        "--request-interval", "7",
        "--output", str(tmp_path / "classification.json"),
        "--csv-output", str(tmp_path / "results.csv"),
        "--history-cache", str(tmp_path / "cache"),
        "--event-cache", str(tmp_path / "events"),
    ]) == 0
    assert starts == [0, 7, 14, 21, 28]


def test_adaptive_interval_stops_increasing_at_60_seconds(clock):
    session = Mock()
    session.get.side_effect = [
        response({}, 429), response([]), response({}, 429), response([]), response([])
    ]
    client = tour.ZwiftClient(session, tokens(), request_interval=40)
    client.request("GET", "first")
    client.request("GET", "second")
    client.request("GET", "third")
    assert clock[1] == [60, 60, 60, 60]


def test_past_retry_after_date_does_not_remove_local_cooldown(clock):
    session = Mock()
    session.get.side_effect = [
        response({}, 429, {"Retry-After": formatdate(tour.time.time() - 60, usegmt=True)}),
        response([]),
    ]
    tour.ZwiftClient(session, tokens()).request("GET", "history")
    assert clock[1] == [30]


def test_token_expiring_during_cooldown_is_refreshed(monkeypatch, clock):
    refresh = Mock(return_value=tokens())
    monkeypatch.setattr(tour, "refresh_access_token", refresh)
    session = Mock()
    session.get.side_effect = [response({}, 429), response([])]
    expiring = ZwiftTokens("old", "refresh", tour.time.time() + 70)
    tour.ZwiftClient(session, expiring).request("GET", "history")
    refresh.assert_called_once_with("refresh")
    assert session.get.call_args.kwargs["headers"]["Authorization"] == "Bearer access"


@pytest.mark.parametrize("status", [400, 413])
def test_bulk_fetch_splits_rejected_requests_and_caches_results(tmp_path, status):
    client = Mock()

    def limited_request(method, path, **kwargs):
        ids = [entry["eventId"] for entry in kwargs["json"]]
        if len(ids) > 2:
            raise requests.HTTPError(response=response({}, status))
        return [{"eventId": event_id, "tags": ["tourofwatopia2026"]} for event_id in ids]

    client.request.side_effect = limited_request
    records = [history(event=event_id) for event_id in range(100, 105)]
    events = tour.fetch_events(client, records, batch_size=5, cache_dir=tmp_path)
    assert set(events) == set(range(100, 105))
    assert all((tmp_path / f"{event_id}.json").exists() for event_id in events)
    client.request.reset_mock()
    assert tour.fetch_events(client, records, cache_dir=tmp_path) == events
    client.request.assert_not_called()


def test_bulk_fetch_failure_resumes_successful_batches(tmp_path):
    client = Mock()
    client.request.side_effect = [
        [{"eventId": 100, "tags": []}, {"eventId": 101, "tags": []}],
        requests.HTTPError(response=response({}, 503)),
    ]
    records = [history(event=event_id) for event_id in range(100, 104)]
    with pytest.raises(requests.HTTPError):
        tour.fetch_events(client, records, batch_size=2, cache_dir=tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["100.json", "101.json"]
    client.request.reset_mock()
    client.request.side_effect = [[
        {"eventId": 102, "tags": []}, {"eventId": 103, "tags": []}
    ]]
    assert len(tour.fetch_events(client, records, batch_size=2, cache_dir=tmp_path)) == 4
    client.request.assert_called_once()
    assert client.request.call_args.kwargs["json"] == [{"eventId": 102}, {"eventId": 103}]


def test_single_rejected_event_is_reported_not_skipped(tmp_path):
    client = Mock()
    client.request.side_effect = requests.HTTPError(response=response({}, 400))
    with pytest.raises(requests.HTTPError, match="event 100"):
        tour.fetch_events(client, [history()], cache_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_default_until_is_pinned_across_reruns(monkeypatch, tmp_path):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({
        "league": "tourofwatopia2026", "classification": tour.DEFAULT_CLASSIFICATION,
        "pages": [{"entries": [{"profileId": "1"}]}],
    }), encoding="utf-8")
    histories = Mock(return_value=[])
    monkeypatch.setattr(tour, "collect_histories", histories)
    args = [
        "--classification-file", str(classification),
        "--history-cache", str(tmp_path / "cache"),
        "--event-cache", str(tmp_path / "events"),
        "--csv-output", str(tmp_path / "results.csv"),
    ]
    assert tour.main([*args, "--until", "2026-10-20"]) == 0
    assert tour.main(args) == 0
    assert [call.args[3] for call in histories.call_args_list] == [
        "2026-10-20", "2026-10-20"
    ]


def test_default_rerun_reuses_saved_classification(monkeypatch, tmp_path):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({
        "league": "tourofwatopia2026", "classification": tour.DEFAULT_CLASSIFICATION,
        "pages": [{"entries": [{"profileId": "1"}]}],
    }), encoding="utf-8")
    download = Mock()
    monkeypatch.setattr(tour, "fetch_classification", download)
    monkeypatch.setattr(tour, "collect_histories", Mock(return_value=[]))
    assert tour.main([
        "--output", str(classification),
        "--history-cache", str(tmp_path / "cache"), "--event-cache", str(tmp_path / "events"),
        "--csv-output", str(tmp_path / "results.csv"),
    ]) == 0
    download.assert_not_called()


def test_bulk_fetch_bounds_request_size_without_losing_events():
    client = Mock()
    client.request.side_effect = lambda method, path, **kwargs: [
        {"eventId": item["eventId"], "tags": []} for item in kwargs["json"]
    ]
    records = [history(event=event_id) for event_id in range(1, 124)]
    assert len(tour.fetch_events(client, records)) == 123
    assert [len(call.kwargs["json"]) for call in client.request.call_args_list] == [50, 50, 23]


def test_explicit_until_overrides_saved_run_but_other_scopes_do_not_reuse_it(tmp_path):
    tour.write_json(tmp_path / "download_state.json", {
        "league": "tourofwatopia2026", "classification": tour.DEFAULT_CLASSIFICATION,
        "from": "2026-10-01", "to": "2026-10-20",
    })
    assert tour.resolve_until(
        tmp_path, "tourofwatopia2026", tour.DEFAULT_CLASSIFICATION,
        tour.date(2026, 10, 1), tour.date(2026, 10, 21),
    ) == tour.date(2026, 10, 21)
    assert tour.resolve_until(
        tmp_path, "another-league", tour.DEFAULT_CLASSIFICATION,
        tour.date(2026, 10, 1), None,
    ) == tour.datetime.now(tour.timezone.utc).date() + tour.timedelta(days=1)


def test_bulk_http_failure_names_failed_phase(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    monkeypatch.setattr(tour, "fetch_classification", Mock(return_value=[
        {"entries": [{"profileId": "1"}]}
    ]))
    monkeypatch.setattr(tour, "collect_histories", Mock(return_value=[history()]))
    monkeypatch.setattr(tour, "fetch_events", Mock(side_effect=requests.HTTPError(
        "events/bulk-fetch: event 100", response=response({}, 400)
    )))
    assert tour.main([
        "--output", str(tmp_path / "classification.json"),
        "--history-cache", str(tmp_path / "cache"), "--event-cache", str(tmp_path / "events"),
        "--csv-output", str(tmp_path / "results.csv"),
    ]) == 1
    message = capsys.readouterr().err
    assert "event detail lookup (events/bulk-fetch)" in message
    assert "HTTP 400" in message
    assert "event 100" in message


@pytest.mark.parametrize("id_field", ["id", "eventId"])
def test_bulk_omissions_recovered_by_individual_endpoint(tmp_path, id_field):
    client = Mock()
    client.request.side_effect = [
        [{"eventId": 100, "tags": []}],
        {id_field: 200, "tags": ["tourofwatopia2026"]},
    ]
    events = tour.fetch_events(client, [history(), history(event=200)], cache_dir=tmp_path)
    assert set(events) == {100, 200}
    assert events[200]["tags"] == ["tourofwatopia2026"]
    assert events[200]["eventId"] == 200
    assert client.request.call_args.args == ("GET", "events/200")
    assert json.loads((tmp_path / "200.json").read_text())["eventId"] == 200
    client.request.reset_mock()
    assert tour.fetch_events(client, [history(), history(event=200)], cache_dir=tmp_path) == events
    client.request.assert_not_called()


@pytest.mark.parametrize("status", [403, 404, 410])
def test_skip_unavailable_writes_auditable_report_and_continues_other_batches(tmp_path, status):
    client = Mock()
    client.request.side_effect = [
        [{"eventId": 100, "tags": ["tourofwatopia2026"]}],
        requests.HTTPError(response=response({}, status)),
        [{"eventId": 300, "tags": ["tourofwatopia2026"]}],
    ]
    report = tmp_path / "excluded.json"
    records = [history(), history(event=200), history(profile=2, event=200), history(event=300)]
    events = tour.fetch_events(
        client, records, batch_size=2, cache_dir=tmp_path / "events",
        skip_unavailable=True, report_path=report,
    )
    assert set(events) == {100, 300}
    data = json.loads(report.read_text())
    assert data["excludedResultCount"] == 2
    assert data["unavailableEvents"][0]["eventId"] == 200
    assert data["unavailableEvents"][0]["httpStatus"] == status
    assert data["unavailableEvents"][0]["affectedResults"] == 2
    assert "unknown" in data["unavailableEvents"][0]["reason"]
    assert not (tmp_path / "events" / "200.json").exists()


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_skip_unavailable_does_not_hide_other_individual_failures(tmp_path, status):
    client = Mock()
    client.request.side_effect = [[], requests.HTTPError(response=response({}, status))]
    with pytest.raises(requests.HTTPError, match="event 100"):
        tour.fetch_events(
            client, [history()], skip_unavailable=True, report_path=tmp_path / "excluded.json"
        )


@pytest.mark.parametrize(
    "event", [{"id": 201, "tags": []}, {"id": 200}, {"id": 201, "eventId": 200, "tags": []}]
)
def test_invalid_individual_event_data_is_not_treated_as_non_tour(event):
    client = Mock()
    client.request.side_effect = [[], event]
    with pytest.raises(ValueError):
        tour.fetch_events(client, [history(event=200)])


@pytest.mark.parametrize("status", [403, 404, 410])
def test_skip_unavailable_cli_writes_csv_without_unresolved_events(monkeypatch, tmp_path, status):
    monkeypatch.setenv("ZWIFT_USERNAME", "rider@example.test")
    monkeypatch.setenv("ZWIFT_PASSWORD", "test-password")
    monkeypatch.setattr(tour, "get_token_with_password", Mock(return_value=tokens()))
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({
        "league": "tourofwatopia2026", "classification": tour.DEFAULT_CLASSIFICATION,
        "pages": [{"entries": [{"profileId": "1"}]}],
    }))
    monkeypatch.setattr(tour, "collect_histories", Mock(return_value=[
        history(), history(event=200)
    ]))
    session = Mock()
    session.post.return_value = response([{"eventId": 100, "tags": ["tourofwatopia2026"]}])
    session.get.return_value = response({}, status)
    monkeypatch.setattr(tour.requests, "Session", lambda: nullcontext(session))
    output = tmp_path / "results.csv"
    assert tour.main([
        "--classification-file", str(classification), "--csv-output", str(output),
        "--history-cache", str(tmp_path / "histories"), "--event-cache", str(tmp_path / "events"),
        "--skip-unavailable-events",
    ]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["eventId"] for row in rows] == ["100"]
    report = json.loads((tmp_path / "results.excluded_events.json").read_text())
    assert report["excludedResultCount"] == 1
    assert report["unavailableEvents"][0]["eventId"] == 200
    assert report["unavailableEvents"][0]["httpStatus"] == status


def test_exclusions_report_is_cleared_when_event_becomes_available(tmp_path):
    report = tmp_path / "excluded.json"
    report.write_text('{"unavailableEvents": [{"eventId": 100}]}')
    client = Mock()
    client.request.return_value = [{"eventId": 100, "tags": []}]
    tour.fetch_events(client, [history()], report_path=report)
    assert json.loads(report.read_text())["unavailableEvents"] == []


def test_classification_deduplicates_and_reports_unidentified_entries(capsys):
    classification = {"pages": [
        {"entries": [{"profileId": "2"}, {"profileId": "1"}, {"profileId": "2"}]},
        {"entries": [{}, {"profileId": None}, {"profileId": ""}]},
    ]}
    assert tour.participant_ids(classification) == [1, 2]
    output = capsys.readouterr().err
    assert "3 entries without profileId" in output
    assert "deduplicated 1 repeated rider entries" in output


def test_classification_with_no_identifiable_riders_is_an_error():
    with pytest.raises(ValueError, match="no identifiable riders"):
        tour.participant_ids({"pages": [{"entries": [{}]}]})


def test_classification_invalid_nonempty_id_is_not_silently_skipped():
    with pytest.raises(ValueError):
        tour.participant_ids({"pages": [{"entries": [{"profileId": "invalid"}]}]})
