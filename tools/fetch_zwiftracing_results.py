"""Export all riders' ZwiftRacing event results from embedded HTML page data."""

from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.http_pacing import RequestPacer  # noqa: E402
from tools.fetch_tour_of_watopia import (  # noqa: E402
    identifier,
    positive_float,
    positive_int,
    write_json,
)

BASE_URL = "https://www.zwiftracing.app/events"
CSV_FIELDS = [
    "profileId", "eventId", "eventSubgroupId", "position", "expectedPosition",
    "ratingBefore", "ratingAfter",
]


class NextDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.in_data = False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.in_data = True

    def handle_data(self, data):
        if self.in_data:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False


def parse_detail(html: str) -> dict:
    parser = NextDataParser()
    parser.feed(html)
    parser.close()
    if not parser.parts:
        raise ValueError("Page has no __NEXT_DATA__ JSON; it may be an error or challenge page.")
    payload = json.loads("".join(parser.parts))
    detail = payload["props"]["pageProps"]["detail"]
    if not isinstance(detail, dict):
        raise ValueError("Missing or invalid event detail in embedded page data.")
    return detail


def event_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "eventId" not in reader.fieldnames:
            raise ValueError(f"Input CSV must have an eventId column: {path}")
        ids = {identifier(row["eventId"], "eventId") for row in reader}
    if not ids:
        raise ValueError(f"Input CSV contains no event IDs: {path}")
    return sorted(ids)


def event_rows(detail: dict, event_id: int) -> list[dict]:
    if identifier(detail["eventId"], "eventId") != event_id:
        raise ValueError(f"Page eventId does not match requested event {event_id}.")
    if detail.get("resultsFinalized") is False:
        print(
            f"Warning: event {event_id} results are not finalized; "
            "use --refresh later to update cached results.",
            file=sys.stderr,
        )
    pens = detail.get("pens")
    if not isinstance(pens, list):
        raise ValueError(f"Missing or invalid pens for event {event_id}.")
    rows = []
    missing = {key: 0 for key in CSV_FIELDS[3:]}
    for pen in pens:
        if not isinstance(pen, dict) or not isinstance(pen.get("riders"), list):
            raise ValueError(f"Missing or invalid pen riders for event {event_id}.")
        subgroup_id = identifier(pen["id"], "eventSubgroupId")
        count = pen.get("results", {}).get("count")
        if isinstance(count, int) and count > len(pen["riders"]):
            raise ValueError(
                f"Event {event_id}, subgroup {subgroup_id}: embedded rider list "
                f"is incomplete ({len(pen['riders'])} of {count})."
            )
        for rider in pen["riders"]:
            if not isinstance(rider, dict):
                raise ValueError(f"Invalid rider for event {event_id}.")
            row = {
                "profileId": identifier(rider["riderId"], "riderId"),
                "eventId": event_id,
                "eventSubgroupId": subgroup_id,
                "position": rider.get("position"),
                "expectedPosition": rider.get("expectedPosition"),
                "ratingBefore": rider.get("ratingBefore"),
                "ratingAfter": rider.get("rating"),
            }
            for field in missing:
                missing[field] += row[field] is None
            rows.append(row)
    for field, count in missing.items():
        if count:
            print(
                f"Warning: event {event_id}: {count} riders without {field}; "
                "retained as blank CSV cells.",
                file=sys.stderr,
            )
    return rows


def download_detail(
    session: requests.Session, pacer: RequestPacer, event_id: int
) -> dict:
    for attempt in range(4):
        pacer.wait()
        pacer.started()
        response = session.get(
            f"{BASE_URL}/{event_id}",
            headers={
                "Accept": "text/html",
                "User-Agent": "ZwiftTools/1.0 (personal race analysis)",
            },
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            pacer.defer(response, attempt)
            if attempt < 3:
                continue
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError(f"Event {event_id} redirected instead of returning results.")
        return parse_detail(response.text)
    raise RuntimeError("Request retry loop exhausted unexpectedly.")


def export_results(
    session: requests.Session, ids: list[int], output: Path, cache: Path,
    request_interval: float = 2, refresh: bool = False,
) -> int:
    pacer = RequestPacer(request_interval)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for index, event_id in enumerate(ids, 1):
                path = cache / f"{event_id}.json"
                cached = path.exists() and not refresh
                detail = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if cached else download_detail(session, pacer, event_id)
                )
                rows = event_rows(detail, event_id)
                if not cached:
                    write_json(path, detail)
                writer.writerows(rows)
                count += len(rows)
                print(
                    f"Event {index}/{len(ids)}: {event_id}, {len(rows)} riders "
                    f"({'cached' if cached else 'downloaded'}).",
                    file=sys.stderr,
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", type=Path, default=Path("race_data") / "tour_of_watopia_2026.csv",
        help="ZRS CSV containing eventId values (default: race_data Tour CSV).",
    )
    source.add_argument(
        "--event-id", type=positive_int, action="append",
        help="Download a specific event instead of reading the CSV; repeat for multiple events.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("race_data") / "tour_of_watopia_2026_velo.csv"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("race_data") / "zwiftracing_events"
    )
    parser.add_argument("--request-interval", type=positive_float, default=2)
    parser.add_argument("--refresh", action="store_true", help="Redownload cached event pages.")
    args = parser.parse_args(argv)
    try:
        ids = sorted(set(args.event_id)) if args.event_id else event_ids(args.input)
        with requests.Session() as session:
            count = export_results(
                session, ids, args.output, args.cache_dir, args.request_interval, args.refresh
            )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(
            f"ZwiftRacing HTTP {status}; export stopped without replacing the final CSV. "
            "For 429s, allow a cooldown and increase --request-interval; "
            "completed event caches are preserved.",
            file=sys.stderr,
        )
        return 1
    except requests.RequestException as exc:
        print(f"ZwiftRacing network request failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, OSError, EOFError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {count} riders across {len(ids)} events to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
