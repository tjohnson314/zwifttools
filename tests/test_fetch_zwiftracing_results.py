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


def test_challenge_page_is_an_error():
    with pytest.raises(ValueError, match="no __NEXT_DATA__"):
        zr.parse_detail("<html>Checking your browser</html>")


def test_malformed_json_is_an_error():
    with pytest.raises(ValueError):
        zr.parse_detail('<script id="__NEXT_DATA__">{"invalid"</script>')


def test_event_mismatch_is_an_error():
    with pytest.raises(ValueError, match="does not match"):
        zr.event_rows(detail(200), 100)


def test_incomplete_embedded_results_are_an_error():
    data = detail()
    data["pens"][0]["results"]["count"] = 100
    with pytest.raises(ValueError, match="incomplete"):
        zr.event_rows(data, 100)


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
    output = tmp_path / "results.csv"
    assert zr.main([
        "--input", str(input_csv), "--output", str(output),
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    session.get.assert_called_once()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


def test_main_explicit_event_needs_no_input_csv(monkeypatch, tmp_path):
    session = Mock()
    session.get.return_value = response()
    monkeypatch.setattr(zr.requests, "Session", lambda: nullcontext(session))
    assert zr.main([
        "--event-id", "100", "--event-id", "100",
        "--output", str(tmp_path / "results.csv"), "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    session.get.assert_called_once()


def test_main_reports_http_error_without_writing_output(monkeypatch, tmp_path, capsys):
    session = Mock()
    session.get.return_value = response(status=429)
    monkeypatch.setattr(zr.requests, "Session", lambda: nullcontext(session))
    output = tmp_path / "results.csv"
    assert zr.main([
        "--event-id", "100", "--output", str(output),
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 1
    assert "HTTP 429" in capsys.readouterr().err
    assert not output.exists()
