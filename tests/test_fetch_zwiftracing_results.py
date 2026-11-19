from contextlib import nullcontext
import csv
from email.utils import formatdate
import json
from unittest.mock import Mock

import pytest
import requests

from shared import http_pacing
from tools import fetch_zwiftracing_results as zr


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(http_pacing.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(http_pacing.time, "time", lambda: 1_800_000_000 + now[0])
    monkeypatch.setattr(http_pacing.time, "sleep", sleep)
    return now, sleeps


def detail(event=100):
    return {
        "eventId": str(event), "resultsFinalized": True,
        "pens": [
            {
                "id": "1000", "results": {"count": 2},
                "riders": [
                    {
                        "riderId": 1, "position": 2, "expectedPosition": 1,
                        "ratingBefore": 1479.7131210117332, "rating": 1460.54321,
                        "velo": {"elo": {"before": 593.4714, "after": 580}},
                    },
                    {
                        "riderId": 2, "position": 1, "expectedPosition": 2,
                        "ratingBefore": 520.6942590120161, "rating": 540.234567,
                    },
                ],
            },
            {"id": "1001", "results": {"count": 1}, "riders": [
                {
                    "riderId": 3, "position": 1, "expectedPosition": 1,
                    "ratingBefore": 0, "rating": -12.5,
                }
            ]},
        ],
    }


def html(data):
    return (
        '<html><script>not_json()</script><script type="application/json" '
        'id="__NEXT_DATA__">'
        + json.dumps({"props": {"pageProps": {"detail": data}}})
        + "</script></html>"
    )


def response(data=None, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result.encoding = "utf-8"
    result._content = html(data if data is not None else detail()).encode()
    return result


def test_extracts_embedded_data_and_all_riders_from_all_pens():
    rows = zr.event_rows(zr.parse_detail(html(detail())), 100)
    assert rows == [
        {"profileId": 1, "eventId": 100, "eventSubgroupId": 1000, "position": 2,
         "expectedPosition": 1, "ratingBefore": 1479.7131210117332,
         "ratingAfter": 1460.54321},
        {"profileId": 2, "eventId": 100, "eventSubgroupId": 1000, "position": 1,
         "expectedPosition": 2, "ratingBefore": 520.6942590120161,
         "ratingAfter": 540.234567},
        {"profileId": 3, "eventId": 100, "eventSubgroupId": 1001, "position": 1,
         "expectedPosition": 1, "ratingBefore": 0, "ratingAfter": -12.5},
    ]


def test_missing_fields_are_not_substituted_from_nested_velo(capsys):
    data = detail()
    rider = data["pens"][0]["riders"][0]
    for field in ["position", "expectedPosition", "ratingBefore", "rating"]:
        del rider[field]
    row = zr.event_rows(data, 100)[0]
    assert all(row[field] is None for field in zr.CSV_FIELDS[3:])
    assert "1 riders without ratingAfter" in capsys.readouterr().err


def test_unfinalized_results_warn_but_are_retained(capsys):
    data = detail()
    data["resultsFinalized"] = False
    assert len(zr.event_rows(data, 100)) == 3
    assert "not finalized" in capsys.readouterr().err


@pytest.mark.parametrize("dnf", [True, False, None, "true"])
def test_missing_expected_position_warning_excludes_only_explicit_dnf(capsys, dnf):
    data = detail()
    rider = data["pens"][0]["riders"][0]
    rider["didNotFinish"] = dnf
    rider["expectedPosition"] = None
    rows = zr.event_rows(data, 100)
    assert len(rows) == 3
    assert rows[0]["expectedPosition"] is None
    assert ("1 riders without expectedPosition" in capsys.readouterr().err) == (dnf is not True)


def test_mixed_dnf_and_finisher_warning_counts_only_finisher(capsys):
    data = detail()
    riders = data["pens"][0]["riders"]
    riders[0].update(didNotFinish=True, expectedPosition=None, rating=None)
    riders[1].update(didNotFinish=False, expectedPosition=None)
    rows = zr.event_rows(data, 100)
    assert len(rows) == 3
    warning = capsys.readouterr().err
    assert "1 riders without expectedPosition" in warning
    assert "2 riders without expectedPosition" not in warning
    assert "1 riders without ratingAfter" in warning


def test_challenge_page_is_an_error():
    with pytest.raises(ValueError, match="no __NEXT_DATA__"):
        zr.parse_detail("<html>Checking your browser</html>")


def test_malformed_json_is_an_error():
    with pytest.raises(ValueError):
        zr.parse_detail('<script id="__NEXT_DATA__">{"invalid"</script>')


def test_event_mismatch_is_an_error():
    with pytest.raises(ValueError, match="does not match"):
        zr.event_rows(detail(200), 100)


def test_incomplete_embedded_results_warn_and_preserve_rows_and_positions(capsys):
    data = detail()
    data["pens"][0]["results"]["count"] = 100
    data["pens"][0]["riders"][1]["position"] = 7
    rows = zr.event_rows(data, 100)
    assert len(rows) == 3
    assert [row["position"] for row in rows] == [2, 7, 1]
    warning = capsys.readouterr().err
    assert "subgroup 1000" in warning
    assert "source reports 100 riders but supplies 2" in warning
    assert "Dataset may be incomplete" in warning


def test_incomplete_event_is_cached_and_exported_without_synthetic_rows(tmp_path, capsys):
    data = detail()
    data["pens"][0]["results"]["count"] = 3
    session = Mock()
    session.get.return_value = response(data)
    output = tmp_path / "out.csv"
    cache = tmp_path / "cache"
    assert zr.export_results(session, [100], output, cache) == 3
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert json.loads((cache / "100.json").read_text()) == data
    assert "source reports 3 riders but supplies 2" in capsys.readouterr().err


@pytest.mark.parametrize("pens", [None, {}, [{"id": 1000}]])
def test_unknown_pen_schema_is_an_error(pens):
    data = detail()
    data["pens"] = pens
    with pytest.raises(ValueError, match="Missing or invalid"):
        zr.event_rows(data, 100)


def test_reads_unique_event_ids_from_zrs_csv_with_bom(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(
        "profileId,eventId,eventSubgroupId\n1,200,2000\n2,100,1000\n3,200,2000\n",
        encoding="utf-8-sig",
    )
    assert zr.event_ids(path) == [100, 200]


@pytest.mark.parametrize("content", ["profileId\n1\n", "eventId\n", "eventId\ninvalid\n"])
def test_bad_input_csv_is_an_error(tmp_path, content):
    path = tmp_path / "input.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        zr.event_ids(path)


def test_export_paces_pages_and_reuses_cache_without_network(tmp_path, clock):
    session = Mock()
    session.get.side_effect = [response(detail(100)), response(detail(200))]
    output = tmp_path / "results.csv"
    cache = tmp_path / "cache"
    assert zr.export_results(session, [100, 200], output, cache, 0.5) == 6
    assert clock[1] == [0.5]
    with output.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == [
            "profileId", "eventId", "eventSubgroupId", "position", "expectedPosition",
            "ratingBefore", "ratingAfter",
        ]
        rows = list(reader)
    assert len(rows) == 6
    assert rows[0]["ratingBefore"] == "1479.7131210117332"
    assert rows[2]["ratingAfter"] == "-12.5"
    request = session.get.call_args_list[0]
    assert request.args[0] == "https://www.zwiftracing.app/events/100"
    assert "Authorization" not in request.kwargs["headers"]
    assert request.kwargs["timeout"] == 30
    session.get.reset_mock()
    assert zr.export_results(session, [100, 200], output, cache) == 6
    session.get.assert_not_called()


def test_failure_preserves_output_and_resumes_completed_events(tmp_path):
    session = Mock()
    session.get.side_effect = [response(detail(100)), response(status=404)]
    output = tmp_path / "results.csv"
    output.write_text("previous results", encoding="utf-8")
    cache = tmp_path / "cache"
    with pytest.raises(requests.HTTPError):
        zr.export_results(session, [100, 200], output, cache)
    assert output.read_text(encoding="utf-8") == "previous results"
    assert (cache / "100.json").exists()
    assert not (cache / "200.json").exists()
    assert not (tmp_path / "results.csv.tmp").exists()
    session.get.reset_mock()
    session.get.side_effect = [response(detail(200))]
    assert zr.export_results(session, [100, 200], output, cache) == 6
    session.get.assert_called_once()
    assert session.get.call_args.args[0].endswith("/200")


def test_refresh_replaces_cached_event_data(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "100.json").write_text(json.dumps(detail()), encoding="utf-8")
    updated = detail()
    updated["pens"][0]["riders"][0]["rating"] = 1555
    session = Mock()
    session.get.return_value = response(updated)
    zr.export_results(session, [100], tmp_path / "out.csv", cache, refresh=True)
    session.get.assert_called_once()
    saved = json.loads((cache / "100.json").read_text(encoding="utf-8"))
    assert saved["pens"][0]["riders"][0]["rating"] == 1555


@pytest.mark.parametrize("header", ["180", "date"])
def test_retry_after_and_adaptive_pacing_are_shared_between_events(clock, header):
    if header == "date":
        header = formatdate(http_pacing.time.time() + 180, usegmt=True)
    session = Mock()
    session.get.side_effect = [
        response(status=429, headers={"Retry-After": header}),
        response(detail(100)), response(detail(200)),
    ]
    pacer = zr.RequestPacer()
    zr.download_detail(session, pacer, 100)
    zr.download_detail(session, pacer, 200)
    assert clock[1] == [180, 4]


def test_retry_exhaustion_stops_after_four_attempts(clock):
    session = Mock()
    session.get.return_value = response(status=429)
    with pytest.raises(requests.HTTPError):
        zr.download_detail(session, zr.RequestPacer(), 100)
    assert session.get.call_count == 4
    assert clock[1] == [30, 60, 120]


def test_main_uses_csv_ids_and_writes_expected_output(monkeypatch, tmp_path):
    input_csv = tmp_path / "zrs.csv"
    input_csv.write_text("eventId\n100\n100\n", encoding="utf-8")
    session = Mock()
    session.get.return_value = response()
    monkeypatch.setattr(zr.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setattr(zr, "browser_session", lambda *args: nullcontext(session))
    output = tmp_path / "results.csv"
    assert zr.main([
        "--input", str(input_csv), "--output", str(output),
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    session.get.assert_called_once()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    log = output.with_suffix(".log").read_text(encoding="utf-8")
    assert "Event 1/1: 100" in log
    assert "Saved 3 riders" in log


def test_main_explicit_event_needs_no_input_csv(monkeypatch, tmp_path):
    session = Mock()
    session.get.return_value = response()
    monkeypatch.setattr(zr.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setattr(zr, "browser_session", lambda *args: nullcontext(session))
    assert zr.main([
        "--event-id", "100", "--event-id", "100",
        "--output", str(tmp_path / "results.csv"), "--cache-dir", str(tmp_path / "cache"),
        "--log-file", str(tmp_path / "custom.log"),
    ]) == 0
    session.get.assert_called_once()
    assert "Saved 3 riders" in (tmp_path / "custom.log").read_text(encoding="utf-8")
    assert not (tmp_path / "results.log").exists()


def test_main_reports_http_error_without_writing_output(monkeypatch, tmp_path, capsys):
    session = Mock()
    session.get.return_value = response(status=429)
    monkeypatch.setattr(zr.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setattr(zr, "browser_session", lambda *args: nullcontext(session))
    output = tmp_path / "results.csv"
    assert zr.main([
        "--event-id", "100", "--output", str(output),
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 1
    assert "HTTP 429" in capsys.readouterr().err
    assert "HTTP 429" in output.with_suffix(".log").read_text(encoding="utf-8")
    assert not output.exists()


def browser_page(status=200, url="https://www.zwiftracing.app/events/100"):
    page = Mock()
    page.url = url
    page.content.return_value = html(detail())
    page.goto.return_value = Mock(
        status=status, headers={"content-type": "text/html", "retry-after": "180"}
    )
    return page


def test_browser_transport_reads_authenticated_page_without_copying_cookies():
    page = browser_page()
    session = zr.BrowserSession(page)
    result = session.get(
        "https://www.zwiftracing.app/events/100",
        headers={"User-Agent": "unused", "Accept": "text/html"},
        timeout=30, allow_redirects=False,
    )
    page.goto.assert_called_once_with(
        "https://www.zwiftracing.app/events/100",
        wait_until="domcontentloaded", timeout=30_000,
    )
    assert result.status_code == 200
    assert zr.parse_detail(result.text)["eventId"] == "100"
    assert result.headers["Retry-After"] == "180"
    page.context.cookies.assert_not_called()


@pytest.mark.parametrize(
    "status,url",
    [
        (200, "https://www.zwiftracing.app/sign-in"),
        (200, "https://identity.example.test/login"),
        (401, "https://www.zwiftracing.app/events/100"),
        (403, "https://www.zwiftracing.app/events/100"),
    ],
)
def test_browser_login_redirects_or_denials_are_explicit(status, url):
    page = browser_page(status, url)
    with pytest.raises(zr.LoginRequired, match="--login"):
        zr.download_detail(zr.BrowserSession(page), zr.RequestPacer(), 100)


def test_browser_rate_limit_keeps_existing_backoff(clock):
    page = browser_page()
    page.goto.side_effect = [
        Mock(status=429, headers={"Retry-After": "180"}),
        Mock(status=200, headers={}),
    ]
    assert zr.download_detail(zr.BrowserSession(page), zr.RequestPacer(), 100)["eventId"] == "100"
    assert clock[1] == [180]


def test_interactive_login_waits_for_user_and_checks_event(monkeypatch):
    page = browser_page()
    confirm = Mock(return_value="")
    monkeypatch.setattr("builtins.input", confirm)
    zr.interactive_login(page, 100)
    confirm.assert_called_once()
    page.content.assert_called_once()


def test_interactive_login_does_not_accept_login_page_as_results(monkeypatch):
    page = browser_page(url="https://www.zwiftracing.app/sign-in")
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(zr.LoginRequired, match="Complete sign-in"):
        zr.interactive_login(page, 100)


@pytest.mark.parametrize("existing,force", [(False, False), (True, False), (True, True)])
def test_browser_profile_reuse_and_cleanup(monkeypatch, tmp_path, existing, force):
    api = pytest.importorskip("playwright.sync_api")
    profile = tmp_path / "browser"
    if existing:
        profile.mkdir()
    page = browser_page()
    context = Mock()
    context.pages = [page]
    runtime = Mock()
    runtime.chromium.launch_persistent_context.return_value = context
    monkeypatch.setattr(api, "sync_playwright", lambda: nullcontext(runtime))
    login = Mock()
    monkeypatch.setattr(zr, "interactive_login", login)
    with zr.browser_session(profile, 100, force) as session:
        assert session.page is page
    assert login.call_count == int(force or not existing)
    runtime.chromium.launch_persistent_context.assert_called_once_with(
        str(profile.resolve()), headless=False
    )
    context.close.assert_called_once()


def test_browser_context_closes_after_export_error(monkeypatch, tmp_path):
    api = pytest.importorskip("playwright.sync_api")
    profile = tmp_path / "browser"
    profile.mkdir()
    context = Mock()
    context.pages = [browser_page()]
    runtime = Mock()
    runtime.chromium.launch_persistent_context.return_value = context
    monkeypatch.setattr(api, "sync_playwright", lambda: nullcontext(runtime))
    with pytest.raises(ValueError, match="bad event"):
        with zr.browser_session(profile, 100):
            raise ValueError("bad event")
    context.close.assert_called_once()


def test_fully_cached_export_does_not_start_browser(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "100.json").write_text(json.dumps(detail()), encoding="utf-8")
    launch = Mock()
    monkeypatch.setattr(zr, "browser_session", launch)
    assert zr.main([
        "--event-id", "100", "--output", str(tmp_path / "results.csv"),
        "--cache-dir", str(cache),
    ]) == 0
    launch.assert_not_called()


def test_main_passes_explicit_login_and_profile(monkeypatch, tmp_path):
    session = Mock()
    session.get.return_value = response()
    launch = Mock(return_value=nullcontext(session))
    monkeypatch.setattr(zr, "browser_session", launch)
    profile = tmp_path / "private-profile"
    assert zr.main([
        "--login", "--browser-profile", str(profile), "--event-id", "100",
        "--output", str(tmp_path / "results.csv"), "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    launch.assert_called_once_with(profile, 100, True)


def test_current_router_event_data_used_when_next_data_is_absent():
    page = browser_page()
    page.content.return_value = "<html><body>Current event page</body></html>"
    handle = page.wait_for_function.return_value
    handle.json_value.return_value = detail()
    assert zr.download_detail(zr.BrowserSession(page), zr.RequestPacer(), 100) == detail()
    assert page.wait_for_function.call_args.kwargs == {"arg": 100, "timeout": 15_000}
    handle.dispose.assert_called_once()


def test_legacy_embedded_data_does_not_wait_for_router():
    page = browser_page()
    assert zr.browser_detail(page, 100) == detail()
    page.wait_for_function.assert_not_called()


def test_interactive_login_accepts_current_router_results(monkeypatch):
    page = browser_page()
    page.content.return_value = "<html>Current results</html>"
    page.wait_for_function.return_value.json_value.return_value = detail()
    monkeypatch.setattr("builtins.input", lambda _: "")
    zr.interactive_login(page, 100)
    page.wait_for_function.assert_called_once()


def test_invalid_router_data_is_an_error():
    page = browser_page()
    page.content.return_value = "<html></html>"
    handle = page.wait_for_function.return_value
    handle.json_value.return_value = []
    with pytest.raises(ValueError, match="Invalid event loader data"):
        zr.browser_detail(page, 100)
    handle.dispose.assert_called_once()


@pytest.mark.parametrize("raw_id", ["", None])
def test_missing_subgroup_id_uses_exact_cached_label(raw_id, capsys):
    data = detail()
    data["pens"][0].update(id=raw_id, name="A")
    rows = zr.event_rows(data, 100, {"A": 1234})
    assert [row["eventSubgroupId"] for row in rows] == [1234, 1234, 1001]
    assert "using cached Zwift subgroup ID 1234" in capsys.readouterr().err


def test_unresolved_subgroup_retains_rows_as_blank_csv_cells(tmp_path, capsys):
    data = detail()
    data["pens"][0].update(id="", name="A")
    session = Mock()
    session.get.return_value = response(data)
    output = tmp_path / "results.csv"
    assert zr.export_results(
        session, [100], output, tmp_path / "zr-cache",
        zwift_event_cache=tmp_path / "missing-cache",
    ) == 3
    with output.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["eventSubgroupId"] for row in rows] == ["", "", "1001"]
    assert "blank eventSubgroupId" in capsys.readouterr().err


def test_export_recovers_subgroups_from_zwift_cache_without_changing_raw_cache(tmp_path):
    cache = tmp_path / "zwift"
    cache.mkdir()
    (cache / "100.json").write_text(json.dumps({
        "eventId": 100,
        "eventSubgroups": [{"subgroupLabel": "A", "eventSubgroupId": 1234}],
    }))
    data = detail()
    data["pens"][0].update(id="", name="A")
    session = Mock()
    session.get.return_value = response(data)
    output = tmp_path / "results.csv"
    zr_cache = tmp_path / "zr"
    assert zr.export_results(session, [100], output, zr_cache, zwift_event_cache=cache) == 3
    with output.open(newline="") as source:
        assert next(csv.DictReader(source))["eventSubgroupId"] == "1234"
    assert json.loads((zr_cache / "100.json").read_text())["pens"][0]["id"] == ""


def test_ambiguous_cached_label_is_an_error(tmp_path):
    (tmp_path / "100.json").write_text(json.dumps({
        "eventId": 100,
        "eventSubgroups": [
            {"subgroupLabel": "A", "eventSubgroupId": 1234},
            {"subgroupLabel": "A", "eventSubgroupId": 1235},
        ],
    }))
    with pytest.raises(ValueError, match="Ambiguous"):
        zr.cached_subgroup_ids(tmp_path, 100)
