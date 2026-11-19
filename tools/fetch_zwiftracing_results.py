"""Export all riders' ZwiftRacing event results from embedded HTML page data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Iterator
from urllib.parse import urlsplit

import requests

if TYPE_CHECKING:
    from playwright.sync_api import Page

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.http_pacing import RequestPacer  # noqa: E402
from shared.console_log import run_logged, terminal_only  # noqa: E402
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
DEFAULT_BROWSER_PROFILE = Path.home() / ".zwifttools" / "zwiftracing-browser"


class LoginRequired(ValueError):
    pass


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


def browser_detail(page: Page, event_id: int) -> dict:
    html = page.content()
    parser = NextDataParser()
    parser.feed(html)
    if parser.parts:
        return parse_detail(html)
    # Current pages use TanStack Router; read its decoded event loader rather
    # than evaluating the serialized hydration scripts ourselves.
    handle = page.wait_for_function(
        """eventId => {
            const matches = window.__TSR_ROUTER__?.state?.matches;
            const match = matches?.find(({loaderData}) =>
                loaderData && String(loaderData.eventId) === String(eventId)
                && Array.isArray(loaderData.pens));
            return match?.loaderData || false;
        }""",
        arg=event_id,
        timeout=15_000,
    )
    try:
        detail = handle.json_value()
    finally:
        handle.dispose()
    if not isinstance(detail, dict):
        raise ValueError(f"Invalid event loader data for event {event_id}.")
    return detail


def is_event_page(current: str, requested: str) -> bool:
    actual = urlsplit(current)
    expected = urlsplit(requested)
    return (
        actual.scheme == expected.scheme
        and actual.netloc.removeprefix("www.") == expected.netloc.removeprefix("www.")
        and actual.path.rstrip("/") == expected.path.rstrip("/")
    )


class BrowserSession:
    """Navigate in the authenticated browser; never export cookies into requests."""

    def __init__(self, page: Page):
        self.page = page

    def get(
        self, url: str, *, headers: dict[str, str], timeout: float, allow_redirects: bool
    ) -> requests.Response:
        # Navigation uses the browser's own headers and follows sign-in redirects.
        navigation = self.page.goto(
            url, wait_until="domcontentloaded", timeout=timeout * 1000
        )
        if navigation is None:
            raise ValueError("Browser navigation returned no HTTP response.")
        if not is_event_page(self.page.url, url) or navigation.status in (401, 403):
            raise LoginRequired(
                "ZwiftRacing sign-in is required or event access was denied. "
                "Rerun with --login and sign in using the browser window; "
                "completed event caches are preserved."
            )
        response = requests.Response()
        response.status_code = navigation.status
        response.headers.update(navigation.headers)
        response.url = self.page.url
        response.encoding = "utf-8"
        response._content = self.page.content().encode("utf-8")
        return response


def interactive_login(page: Page, event_id: int) -> None:
    url = f"{BASE_URL}/{event_id}"
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    print(
        "Sign in to ZwiftRacing.app in the browser window, then open the event "
        "results there. Do not enter credentials in this terminal.",
        file=sys.stderr,
    )
    with terminal_only():
        input("When the event results are visible, press Enter here: ")
    if not is_event_page(page.url, url):
        raise LoginRequired(
            "The browser is not on the requested event page. Complete sign-in "
            "and rerun with --login."
        )
    try:
        detail = browser_detail(page, event_id)
    except (ValueError, KeyError) as exc:
        raise LoginRequired(
            "The browser page does not contain event results yet. Complete "
            "sign-in and rerun with --login."
        ) from exc
    if identifier(detail["eventId"], "eventId") != event_id or not isinstance(
        detail.get("pens"), list
    ):
        raise LoginRequired("The requested event results are not available after sign-in.")


@contextmanager
def browser_session(
    profile: Path, event_id: int, login: bool = False
) -> Iterator[BrowserSession]:
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("playwright"):
            raise
        raise ValueError(
            "Browser dependencies are missing. Run "
            "'python -m pip install -r requirements-scraping.txt', then "
            "'python -m playwright install chromium'."
        ) from exc
    new_profile = not profile.exists()
    profile.mkdir(parents=True, exist_ok=True)
    print(
        f"Using local browser profile: {profile}. "
        "This contains sensitive session data; do not share or commit it.",
        file=sys.stderr,
    )
    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                str(profile.resolve()), headless=False
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                if login or new_profile:
                    interactive_login(page, event_id)
                yield BrowserSession(page)
            finally:
                context.close()
        except PlaywrightError as exc:
            raise ValueError(
                "Browser operation failed. Ensure Chromium is installed with "
                "'python -m playwright install chromium', close other copies "
                "using this browser profile, and retry with --login. "
                "Completed event caches are preserved."
            ) from exc


def event_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "eventId" not in reader.fieldnames:
            raise ValueError(f"Input CSV must have an eventId column: {path}")
        ids = {identifier(row["eventId"], "eventId") for row in reader}
    if not ids:
        raise ValueError(f"Input CSV contains no event IDs: {path}")
    return sorted(ids)


def event_rows(
    detail: dict, event_id: int, subgroup_ids: dict[str, int] | None = None
) -> list[dict]:
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
        raw_id = pen.get("id")
        if raw_id is None or raw_id == "":
            label = pen.get("name")
            subgroup_id = (subgroup_ids or {}).get(label)
            if subgroup_id is None:
                print(
                    f"Warning: event {event_id}, pen {label!r}: missing subgroup ID; "
                    "retained riders with blank eventSubgroupId cells.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Event {event_id}, pen {label!r}: using cached Zwift subgroup "
                    f"ID {subgroup_id}.",
                    file=sys.stderr,
                )
        else:
            subgroup_id = identifier(raw_id, "eventSubgroupId")
        count = pen.get("results", {}).get("count")
        if isinstance(count, int) and count > len(pen["riders"]):
            print(
                f"Warning: event {event_id}, subgroup {subgroup_id}: source reports "
                f"{count} riders but supplies {len(pen['riders'])}; exporting all "
                "available riders with their original positions. Dataset may be incomplete.",
                file=sys.stderr,
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
                if field == "expectedPosition" and rider.get("didNotFinish") is True:
                    continue
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


def cached_subgroup_ids(cache: Path, event_id: int) -> dict[str, int]:
    path = cache / f"{event_id}.json"
    if not path.exists():
        return {}
    event = json.loads(path.read_text(encoding="utf-8"))
    if identifier(event["eventId"], "eventId") != event_id:
        raise ValueError(f"Cached Zwift event ID mismatch: {path}")
    result = {}
    for subgroup in event["eventSubgroups"]:
        label = subgroup["subgroupLabel"]
        if not isinstance(label, str):
            raise ValueError(f"Invalid cached Zwift subgroup label for event {event_id}.")
        subgroup_id = identifier(
            subgroup.get("eventSubgroupId", subgroup.get("id")), "eventSubgroupId"
        )
        if label in result and result[label] != subgroup_id:
            raise ValueError(f"Ambiguous cached Zwift subgroup label {label!r} for {event_id}.")
        result[label] = subgroup_id
    return result


def download_detail(
    session: requests.Session | BrowserSession, pacer: RequestPacer, event_id: int
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
        if isinstance(session, BrowserSession):
            return browser_detail(session.page, event_id)
        return parse_detail(response.text)
    raise RuntimeError("Request retry loop exhausted unexpectedly.")


def export_results(
    session: requests.Session | BrowserSession, ids: list[int], output: Path, cache: Path,
    request_interval: float = 2, refresh: bool = False,
    zwift_event_cache: Path = Path("race_data") / "tow_events",
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
                pens = detail.get("pens")
                needs_ids = isinstance(pens, list) and any(
                    isinstance(pen, dict) and pen.get("id") in (None, "") for pen in pens
                )
                subgroup_ids = cached_subgroup_ids(zwift_event_cache, event_id) if needs_ids else {}
                rows = event_rows(detail, event_id, subgroup_ids)
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
    parser.add_argument(
        "--zwift-event-cache", type=Path, default=Path("race_data") / "tow_events",
        help="Cached Zwift event details used to recover missing subgroup IDs by exact pen label.",
    )
    parser.add_argument(
        "--login", action="store_true",
        help="Pause for sign-in in the browser, even if a saved profile exists.",
    )
    parser.add_argument(
        "--browser-profile", type=Path, default=DEFAULT_BROWSER_PROFILE,
        help="Dedicated local browser profile directory containing sensitive session data.",
    )
    parser.add_argument(
        "--log-file", type=Path,
        help="Append terminal output to this file (default: output path with .log suffix).",
    )
    args = parser.parse_args(argv)
    return run_logged(args.log_file or args.output.with_suffix(".log"), lambda: run(args))


def run(args: argparse.Namespace) -> int:
    try:
        ids = sorted(set(args.event_id)) if args.event_id else event_ids(args.input)
        pending = [
            event_id for event_id in ids
            if args.refresh or not (args.cache_dir / f"{event_id}.json").exists()
        ]
        transport = (
            browser_session(args.browser_profile, pending[0] if pending else ids[0], args.login)
            if pending or args.login else requests.Session()
        )
        with transport as session:
            count = export_results(
                session, ids, args.output, args.cache_dir, args.request_interval, args.refresh,
                args.zwift_event_cache,
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
