"""Export Tour of Watopia race ranks and pre-race racing scores to CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.zwift_auth import (  # noqa: E402
    ZwiftTokens,
    get_token_with_password,
    refresh_access_token,
)
from shared.http_pacing import RequestPacer  # noqa: E402

BASE_URL = "https://us-or-rly101.zwift.com/api"
# Match the curly quotes in the supplied classification URL.
DEFAULT_CLASSIFICATION = "\u201coverall\u201d"
CSV_FIELDS = ["profileId", "eventId", "eventSubgroupId", "previousScore", "rank", "newScore"]


class ZwiftClient:
    """Share tokens, pacing, and rate-limit cooldowns across the whole download."""

    def __init__(
        self, session: requests.Session, tokens: ZwiftTokens, request_interval: float = 2
    ):
        self.session = session
        self.tokens = tokens
        self.pacer = RequestPacer(request_interval)

    def request(self, method: str, path: str, **kwargs):
        for attempt in range(4):
            self.pacer.wait()
            if self.tokens.is_expired:
                self.tokens = refresh_access_token(self.tokens.refresh_token)
            self.pacer.started()
            request = self.session.get if method == "GET" else self.session.post
            response = request(
                f"{BASE_URL}/{path}",
                headers={
                    "Authorization": f"Bearer {self.tokens.access_token}",
                    "Accept": "application/json",
                    "User-Agent": "Zwift/115 CFNetwork/758.0.2 Darwin/15.0.0",
                },
                timeout=30,
                allow_redirects=False,
                **kwargs,
            )
            if response.status_code == 401 and attempt == 0:
                self.tokens = refresh_access_token(self.tokens.refresh_token)
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                self.pacer.defer(response, attempt)
                if attempt < 3:
                    continue
            response.raise_for_status()
            if response.is_redirect:
                raise ValueError("API redirected instead of returning JSON.")
            return response.json()
        raise RuntimeError("Request retry loop exhausted unexpectedly.")


def page_entries(payload) -> list[dict]:
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(
            "Unexpected API schema: expected an array of entry objects or an "
            "object with an 'entries' array."
        )
    return entries


def fetch_pages(
    client: ZwiftClient, path: str, page_size: int, params: dict | None = None
) -> list:
    if page_size <= 0:
        raise ValueError("Page size must be positive.")
    pages = []
    seen_pages = set()
    start = 0
    while True:
        payload = client.request(
            "GET", path, params={**(params or {}), "start": start, "limit": page_size}
        )
        entries = page_entries(payload)
        if not entries:
            break
        signature = json.dumps(entries, sort_keys=True)
        if signature in seen_pages:
            raise ValueError("API repeated a page; pagination may be ignoring 'start'.")
        seen_pages.add(signature)
        pages.append(payload)
        start += len(entries)
        print(f"Loaded {start} entries from {path}.", file=sys.stderr)
    return pages


def fetch_classification(
    session: requests.Session,
    tokens: ZwiftTokens,
    league: str,
    classification: str,
    page_size: int = 50,
    *,
    client: ZwiftClient | None = None,
) -> list:
    """Keep complete page payloads; fail explicitly on an unknown entries schema."""
    path = (
        f"leagues/league/{quote(league, safe='')}"
        f"/classifications/{quote(classification, safe='')}/entries"
    )
    return fetch_pages(
        client if client is not None else ZwiftClient(session, tokens), path, page_size
    )


def identifier(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Invalid {label}: {value!r}")
    number = int(value)
    if number <= 0:
        raise ValueError(f"Invalid {label}: {value!r}")
    return number


def participant_ids(classification: dict) -> list[int]:
    ids = set()
    for page in classification["pages"]:
        for entry in page_entries(page):
            ids.add(identifier(entry["profileId"], "profileId"))
    return sorted(ids)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_histories(
    client: ZwiftClient,
    profiles: list[int],
    since: str,
    until: str,
    page_size: int,
    cache_dir: Path,
) -> list[dict]:
    records = []
    for index, profile_id in enumerate(profiles, 1):
        path = cache_dir / f"{profile_id}_{since}_{until}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (
                cached.get("profileId") != profile_id
                or cached.get("from") != since
                or cached.get("to") != until
            ):
                raise ValueError(f"History cache metadata mismatch: {path}")
            pages = cached["pages"]
        else:
            pages = fetch_pages(
                client, f"race-results/profile/{profile_id}/entries", page_size,
                {"from": since, "to": until},
            )
        rider_records = []
        for page in pages:
            for entry in page_entries(page):
                if identifier(entry["profileId"], "profileId") != profile_id:
                    raise ValueError(f"History profile mismatch for rider {profile_id}.")
                identifier(entry["eventId"], "eventId")
                rider_records.append(entry)
        if not path.exists():
            write_json(path, {
                "profileId": profile_id, "from": since, "to": until, "pages": pages
            })
        records.extend(rider_records)
        print(
            f"History {index}/{len(profiles)}: rider {profile_id}, "
            f"{len(rider_records)} results.",
            file=sys.stderr,
        )
    return records


def fetch_events(client: ZwiftClient, records: list[dict]) -> dict[int, dict]:
    ids = sorted({identifier(entry["eventId"], "eventId") for entry in records})
    if not ids:
        return {}
    payload = client.request(
        "POST", "events/bulk-fetch", json=[{"eventId": event_id} for event_id in ids]
    )
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError("Unexpected bulk-fetch schema: expected an array of events.")
    by_id = {}
    for event in events:
        event_id = identifier(event["eventId"], "eventId")
        if not isinstance(event.get("tags"), list):
            raise ValueError(f"Missing or invalid tags for event {event_id}.")
        by_id[event_id] = event
    missing = set(ids) - by_id.keys()
    if missing:
        raise ValueError(
            f"Bulk-fetch omitted {len(missing)} events; cannot safely filter results. "
            f"First missing event: {min(missing)}."
        )
    return by_id


def race_rows(records: list[dict], events: dict[int, dict], tag: str) -> list[dict]:
    rows = []
    missing_scores = missing_new_scores = missing_ranks = 0
    for entry in records:
        event_id = identifier(entry["eventId"], "eventId")
        if tag not in events[event_id]["tags"]:
            continue
        score_history = entry.get("scoreHistory")
        if score_history is not None and not isinstance(score_history, dict):
            raise ValueError(f"Invalid scoreHistory for event {event_id}.")
        score = score_history.get("previousScore") if score_history else None
        new_score = score_history.get("newScore") if score_history else None
        rank = entry.get("rank")
        missing_scores += score is None
        missing_new_scores += new_score is None
        missing_ranks += rank is None
        rows.append({
            "profileId": identifier(entry["profileId"], "profileId"),
            "eventId": event_id,
            "eventSubgroupId": identifier(entry["eventSubgroupId"], "eventSubgroupId"),
            "previousScore": score,
            "rank": rank,
            "newScore": new_score,
        })
    if missing_scores or missing_new_scores or missing_ranks:
        print(
            f"Warning: retained {missing_scores} rows without previousScore, "
            f"{missing_new_scores} rows without newScore, and "
            f"{missing_ranks} rows without rank as blank CSV cells.",
            file=sys.stderr,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="tourofwatopia2026")
    parser.add_argument(
        "--classification",
        default=DEFAULT_CLASSIFICATION,
        help="Exact classification identifier (default matches the supplied curly-quoted URL).",
    )
    parser.add_argument("--page-size", type=positive_int, default=50)
    parser.add_argument(
        "--request-interval", type=positive_float, default=2,
        help="Minimum seconds between API request starts (default: 2; increases after 429s).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("race_data") / "tour_of_watopia_2026.json"
    )
    parser.add_argument(
        "--classification-file", type=Path,
        help="Use a previously downloaded classification JSON instead of refetching it.",
    )
    parser.add_argument(
        "--classification-only", action="store_true",
        help="Only download classification JSON (the original script behavior).",
    )
    parser.add_argument("--since", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument(
        "--until", type=date.fromisoformat,
        default=datetime.now(timezone.utc).date() + timedelta(days=1),
        help="API 'to' date (default: tomorrow UTC).",
    )
    parser.add_argument(
        "--history-cache", type=Path, default=Path("race_data") / "tow_histories"
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("race_data") / "tour_of_watopia_2026.csv"
    )
    args = parser.parse_args(argv)
    if args.since > args.until:
        parser.error("--since must not be after --until")

    try:
        classification = None
        if args.classification_file:
            classification = json.loads(args.classification_file.read_text(encoding="utf-8"))
            if (
                classification["league"] != args.league
                or classification["classification"] != args.classification
            ):
                raise ValueError("Classification file does not match --league/--classification.")
            participant_ids(classification)
        username = os.environ.get("ZWIFT_USERNAME") or input("Zwift email: ").strip()
        password = os.environ.get("ZWIFT_PASSWORD") or getpass.getpass("Zwift password: ")
        if not username.strip() or not password:
            raise ValueError("Zwift email and password must not be empty.")
        tokens = get_token_with_password(username, password)
        del password
        with requests.Session() as session:
            client = ZwiftClient(session, tokens, args.request_interval)
            if classification is None:
                pages = fetch_classification(
                    session, tokens, args.league, args.classification, args.page_size,
                    client=client,
                )
                classification = {
                    "league": args.league,
                    "classification": args.classification,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "pages": pages,
                }
                write_json(args.output, classification)
                print(f"Saved {len(pages)} classification pages to {args.output}.")
            if not args.classification_only:
                profiles = participant_ids(classification)
                print(f"Collecting histories for {len(profiles)} riders.", file=sys.stderr)
                records = collect_histories(
                    client, profiles, args.since.isoformat(), args.until.isoformat(),
                    args.page_size, args.history_cache,
                )
                events = fetch_events(client, records)
                rows = race_rows(records, events, args.league)
                write_csv(args.csv_output, rows)
                print(f"Saved {len(rows)} race results to {args.csv_output}.")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 429:
            print(
                "Zwift rate limit persisted after retries. Stop and allow a cooldown, "
                "then resume with the same dates and a larger --request-interval; "
                "completed history caches are preserved.",
                file=sys.stderr,
            )
        else:
            print(
                f"Zwift request failed (HTTP {status}). Check credentials, permissions, "
                "and the exact league/classification identifier.",
                file=sys.stderr,
            )
        return 1
    except requests.exceptions.JSONDecodeError:
        print("Zwift returned invalid JSON; no final CSV was written.", file=sys.stderr)
        return 1
    except requests.RequestException:
        print("Could not connect to Zwift; check your connection and retry.", file=sys.stderr)
        return 1
    except (ValueError, KeyError, OSError, EOFError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
