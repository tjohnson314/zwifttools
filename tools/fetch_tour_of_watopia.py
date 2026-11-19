"""Export Tour of Watopia race ranks and pre-race racing scores to CSV."""

from __future__ import annotations

import argparse
from collections import Counter
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
from shared.console_log import run_logged, terminal_only  # noqa: E402

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
    unidentified = duplicates = 0
    for page in classification["pages"]:
        for entry in page_entries(page):
            profile_id = entry.get("profileId")
            if profile_id is None or profile_id == "":
                unidentified += 1
                continue
            profile_id = identifier(profile_id, "profileId")
            duplicates += profile_id in ids
            ids.add(profile_id)
    if unidentified:
        print(
            f"Warning: classification contains {unidentified} entries without profileId; "
            "cannot fetch their histories. Raw entries remain in the classification snapshot.",
            file=sys.stderr,
        )
    if duplicates:
        print(
            f"Classification: deduplicated {duplicates} repeated rider entries.",
            file=sys.stderr,
        )
    if not ids:
        raise ValueError("Classification contains no identifiable riders.")
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
        was_cached = path.exists()
        if was_cached:
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
        if not was_cached:
            write_json(path, {
                "profileId": profile_id, "from": since, "to": until, "pages": pages
            })
        records.extend(rider_records)
        print(
            f"History {index}/{len(profiles)}: rider {profile_id}, "
            f"{len(rider_records)} results ({'cached' if was_cached else 'downloaded'}).",
            file=sys.stderr,
        )
    return records


def fetch_events(
    client: ZwiftClient, records: list[dict], batch_size: int = 50,
    cache_dir: Path | None = None,
    *, skip_unavailable: bool = False, report_path: Path | None = None,
) -> dict[int, dict]:
    if batch_size <= 0:
        raise ValueError("Event batch size must be positive.")
    if skip_unavailable and report_path is None:
        raise ValueError("Skipping unavailable events requires an excluded-events report path.")
    ids = sorted({identifier(entry["eventId"], "eventId") for entry in records})
    by_id = {}
    unavailable = {}

    def validate_event(event: dict, expected_id: int | None = None) -> int:
        event_id = identifier(event["eventId"], "eventId")
        if expected_id is not None and event_id != expected_id:
            raise ValueError(f"Event cache metadata mismatch for event {expected_id}.")
        if not isinstance(event.get("tags"), list):
            raise ValueError(f"Missing or invalid tags for event {event_id}.")
        return event_id

    def save_event(event: dict, event_id: int) -> None:
        if cache_dir is not None:
            write_json(cache_dir / f"{event_id}.json", event)
        by_id[event_id] = event

    def fetch_individual(event_id: int) -> None:
        try:
            event = client.request("GET", f"events/{event_id}")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in (403, 404, 410):
                raise requests.HTTPError(
                    f"Individual lookup for event {event_id} failed (HTTP {status}).",
                    response=exc.response, request=exc.request,
                ) from exc
            unavailable[event_id] = status
            print(
                f"Warning: event {event_id} omitted by bulk-fetch and unavailable "
                f"from individual lookup (HTTP {status}); Tour membership is unknown.",
                file=sys.stderr,
            )
            return
        if not isinstance(event, dict):
            raise ValueError(f"Unexpected individual-event schema for event {event_id}.")
        # The individual endpoint uses id; the bulk endpoint uses eventId.
        event = {**event, "eventId": event.get("eventId", event.get("id"))}
        validate_event(event, event_id)
        if "id" in event and identifier(event["id"], "id") != event_id:
            raise ValueError(f"Individual lookup returned the wrong ID for event {event_id}.")
        save_event(event, event_id)

    def fetch_batch(batch: list[int]) -> None:
        try:
            payload = client.request(
                "POST", "events/bulk-fetch", json=[{"eventId": event_id} for event_id in batch]
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (400, 413) and len(batch) > 1:
                print(
                    f"HTTP {status} from events/bulk-fetch for {len(batch)} events; "
                    "splitting into smaller batches.",
                    file=sys.stderr,
                )
                middle = len(batch) // 2
                fetch_batch(batch[:middle])
                fetch_batch(batch[middle:])
                return
            raise requests.HTTPError(
                f"events/bulk-fetch failed for {len(batch)} events "
                f"(first event {batch[0]}, HTTP {status}).",
                response=exc.response, request=exc.request,
            ) from exc
        events = payload.get("events") if isinstance(payload, dict) else payload
        if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
            raise ValueError("Unexpected bulk-fetch schema: expected an array of events.")
        fetched = {validate_event(event): event for event in events}
        unexpected = fetched.keys() - set(batch)
        if unexpected:
            raise ValueError(f"Bulk-fetch returned an unrequested event: {min(unexpected)}.")
        for event_id, event in fetched.items():
            save_event(event, event_id)
        missing = set(batch) - fetched.keys()
        if missing:
            print(
                f"Bulk-fetch omitted {len(missing)} events; trying individual lookups.",
                file=sys.stderr,
            )
            for event_id in sorted(missing):
                fetch_individual(event_id)
        print(f"Event details: {len(by_id)}/{len(ids)} loaded.", file=sys.stderr)

    pending = []
    for event_id in ids:
        path = cache_dir / f"{event_id}.json" if cache_dir is not None else None
        if path is not None and path.exists():
            event = json.loads(path.read_text(encoding="utf-8"))
            validate_event(event, event_id)
            by_id[event_id] = event
        else:
            pending.append(event_id)
    if ids:
        print(
            f"Event details: {len(by_id)} cached, {len(pending)} to fetch "
            f"in batches of at most {batch_size}.",
            file=sys.stderr,
        )
    for start in range(0, len(pending), batch_size):
        fetch_batch(pending[start:start + batch_size])
    counts = Counter(identifier(entry["eventId"], "eventId") for entry in records)
    excluded_rows = sum(counts[event_id] for event_id in unavailable)
    if report_path is not None:
        write_json(report_path, {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "excludedResultCount": excluded_rows,
            "unavailableEvents": [
                {
                    "eventId": event_id, "httpStatus": status,
                    "affectedResults": counts[event_id],
                    "reason": "Omitted by bulk-fetch and unavailable individually; Tour membership unknown.",
                }
                for event_id, status in sorted(unavailable.items())
            ],
        })
    if unavailable:
        if not skip_unavailable:
            raise ValueError(
                f"{len(unavailable)} events unavailable from both endpoints "
                f"(first event {min(unavailable)}). Valid event details are cached. "
                "Use --skip-unavailable-events to export available results with an exclusions report."
            )
        print(
            f"Warning: excluding {excluded_rows} race results across {len(unavailable)} "
            f"unavailable events; the export may be incomplete. Report: {report_path}",
            file=sys.stderr,
        )
    return by_id


def resolve_until(
    cache_dir: Path, league: str, classification: str, since: date,
    requested: date | None,
) -> date:
    if requested is not None:
        return requested
    path = cache_dir / "download_state.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state["league"] == league
            and state["classification"] == classification
            and state["from"] == since.isoformat()
        ):
            until = date.fromisoformat(state["to"])
            print(f"Resuming saved history date range: {since} to {until}.", file=sys.stderr)
            return until
    return datetime.now(timezone.utc).date() + timedelta(days=1)


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
    parser.add_argument(
        "--refresh-classification", action="store_true",
        help="Redownload classification instead of reusing an existing --output file.",
    )
    parser.add_argument("--since", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument(
        "--until", type=date.fromisoformat,
        help="API 'to' date (default: saved run date, or tomorrow UTC for a new run).",
    )
    parser.add_argument(
        "--history-cache", type=Path, default=Path("race_data") / "tow_histories"
    )
    parser.add_argument(
        "--event-cache", type=Path, default=Path("race_data") / "tow_events"
    )
    parser.add_argument("--event-batch-size", type=positive_int, default=50)
    parser.add_argument(
        "--skip-unavailable-events", action="store_true",
        help="Exclude events returning 403/404/410 individually, with warnings and a JSON report.",
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("race_data") / "tour_of_watopia_2026.csv"
    )
    parser.add_argument(
        "--log-file", type=Path,
        help="Append terminal output to this file (default: output path with .log suffix).",
    )
    args = parser.parse_args(argv)
    if args.until is not None and args.since > args.until:
        parser.error("--since must not be after --until")
    output = args.output if args.classification_only else args.csv_output
    return run_logged(args.log_file or output.with_suffix(".log"), lambda: run(args))


def run(args: argparse.Namespace) -> int:
    phase = "loading saved data"
    try:
        classification = None
        classification_path = args.classification_file
        if (
            classification_path is None and not args.classification_only
            and not args.refresh_classification and args.output.exists()
        ):
            classification_path = args.output
        if classification_path is not None:
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            if (
                classification["league"] != args.league
                or classification["classification"] != args.classification
            ):
                raise ValueError("Classification file does not match --league/--classification.")
            participant_ids(classification)
            print(f"Using saved classification: {classification_path}.", file=sys.stderr)
        until = resolve_until(
            args.history_cache, args.league, args.classification, args.since, args.until
        ) if not args.classification_only else args.until
        if until is not None and args.since > until:
            raise ValueError("--since must not be after the saved --until date.")
        phase = "authentication"
        with terminal_only():
            username = os.environ.get("ZWIFT_USERNAME") or input("Zwift email: ").strip()
            password = os.environ.get("ZWIFT_PASSWORD") or getpass.getpass("Zwift password: ")
        if not username.strip() or not password:
            raise ValueError("Zwift email and password must not be empty.")
        tokens = get_token_with_password(username, password)
        del password
        with requests.Session() as session:
            client = ZwiftClient(session, tokens, args.request_interval)
            if classification is None:
                phase = "classification download"
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
                assert until is not None
                write_json(args.history_cache / "download_state.json", {
                    "league": args.league, "classification": args.classification,
                    "from": args.since.isoformat(), "to": until.isoformat(),
                })
                profiles = participant_ids(classification)
                print(
                    f"Loading histories for {len(profiles)} riders, "
                    f"{args.since} to {until}; completed caches will be reused.",
                    file=sys.stderr,
                )
                phase = "rider history download"
                records = collect_histories(
                    client, profiles, args.since.isoformat(), until.isoformat(),
                    args.page_size, args.history_cache,
                )
                phase = "event detail lookup (events/bulk-fetch)"
                events = fetch_events(
                    client, records, batch_size=args.event_batch_size, cache_dir=args.event_cache,
                    skip_unavailable=args.skip_unavailable_events,
                    report_path=args.csv_output.with_suffix(".excluded_events.json"),
                )
                phase = "CSV export"
                if args.skip_unavailable_events:
                    records = [
                        entry for entry in records
                        if identifier(entry["eventId"], "eventId") in events
                    ]
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
                f"Zwift request failed during {phase} (HTTP {status}): {exc}",
                file=sys.stderr,
            )
        return 1
    except requests.exceptions.JSONDecodeError:
        print("Zwift returned invalid JSON; no final CSV was written.", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(
            f"Zwift request failed during {phase}: {exc}. Check your connection and retry.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, KeyError, OSError, EOFError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
