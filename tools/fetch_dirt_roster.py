"""Collect public DIRT Racing Series team assignments and Zwift profile IDs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.http_pacing import RequestPacer


BASE_URL = "https://dirtracingseries.com"
TEAMS_URL = f"{BASE_URL}/riders/teams/all/"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dirt_s14_roster.json"

BROWSER_WORKER = """async relay => {
    for (let pageNumber = 0; pageNumber < 100; pageNumber++) {
        const next = await fetch(relay + '/next');
        const task = await next.json();
        if (!next.ok) throw new Error(task.error);
        if (task.done) return task;
        const response = await fetch(task.url, {credentials: 'include'});
        const result = await fetch(relay + '/page', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url: task.url, final_url: response.url,
                status: response.status, retry_after: response.headers.get('Retry-After'),
                html: await response.text()})
        });
        const saved = await result.json();
        if (!result.ok) throw new Error(saved.error);
    }
    return {done: false, pages: 100};
}"""


def parse_teams(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#teams-table")
    if table is None:
        raise ValueError("DIRT team table is missing; refusing an empty roster.")
    teams = []
    seen = set()
    for row in table.select('tr[data-in-series="true"]'):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 5:
            raise ValueError("Current-series team row has fewer than five columns.")
        link = cells[0].find("a", href=re.compile(r"^/riders/\d+/$"))
        if link is None:
            raise ValueError("Current-series team has no team-page link.")
        team_id = int(link["href"].strip("/").split("/")[-1])
        if team_id in seen:
            raise ValueError(f"Duplicate team ID {team_id} in DIRT team table.")
        seen.add(team_id)
        teams.append({
            "team_id": team_id,
            "name": link.get_text(" ", strip=True),
            "url": urljoin(BASE_URL, link["href"]),
            "time_zone": cells[2].get_text(" ", strip=True),
            "league": cells[3].get_text(" ", strip=True),
            "category": cells[4].get_text(" ", strip=True),
        })
    if not teams:
        raise ValueError("No teams are marked as participating in the current series.")
    return teams


def page_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one('input[type="password"]') is not None:
        raise ValueError("DIRT sign-in is required; sign in directly in the browser.")
    return soup


def parse_roster(html: str) -> list[dict]:
    soup = page_soup(html)
    table = next((
        table for table in soup.find_all("table")
        if [cell.get_text(" ", strip=True) for cell in table.select("thead th")][:2]
        == ["Rider Name", "Team Role"]
    ), None)
    if table is None:
        if re.search(r"No riders|Total:\s*0 rider", soup.get_text(" ", strip=True), re.I):
            return []
        raise ValueError("DIRT roster table is missing.")
    riders = []
    seen = set()
    for row in table.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue
        link = cells[0].find("a", href=re.compile(r"^/riders/profile/\d+/$"))
        if link is None:
            if re.search(r"No riders", row.get_text(" ", strip=True), re.I):
                continue
            raise ValueError("Roster row has no rider-profile link.")
        rider_id = int(link["href"].strip("/").split("/")[-1])
        if rider_id in seen:
            raise ValueError(f"Duplicate rider {rider_id} in a team roster.")
        seen.add(rider_id)
        riders.append({
            "rider_id": rider_id,
            "name": link.get_text(" ", strip=True),
            "url": urljoin(BASE_URL, link["href"]),
            "role": cells[1].get_text(" ", strip=True),
            "zwift_id": None,
            "profile_fetched": False,
        })
    total = re.search(r"Total:\s*(\d+) rider", soup.get_text(" ", strip=True))
    if total and int(total[1]) != len(riders):
        raise ValueError("Parsed rider count does not match the DIRT roster total.")
    return riders


def parse_rider(html: str, season: int = 14) -> dict:
    soup = page_soup(html)
    text = soup.get_text(" ", strip=True)
    registration = re.search(r"Series\s+(\d+)\s+Registration", text)
    if registration is None or int(registration[1]) != season:
        raise ValueError(f"Rider page does not identify Series {season} registration.")
    match = re.search(r"\bZwift ID:\s*(\d+)\b", text)
    return {"zwift_id": int(match[1]) if match else None, "profile_fetched": True}


class Collection:
    def __init__(self, teams: list[dict], output: Path, refresh: bool = False):
        self.output = output
        self.teams = teams
        self.task = None
        self.profiles = {}
        self.attempts = {}
        self.error = None
        self.done = False
        previous = {}
        if output.exists() and not refresh:
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved.get("season") != 14:
                raise ValueError("The existing checkpoint is not for season 14.")
            previous = {team["team_id"]: team for team in saved["teams"]}
        for team in teams:
            old = previous.get(team["team_id"], {})
            team["roster_fetched"] = old.get("roster_fetched", False)
            team["riders"] = old.get("riders", [])
            for rider in team["riders"]:
                if rider["profile_fetched"]:
                    self.profiles[rider["rider_id"]] = {
                        "zwift_id": rider["zwift_id"], "profile_fetched": True,
                    }

    def next_task(self) -> dict | None:
        for team in self.teams:
            if not team["roster_fetched"]:
                self.task = {"url": team["url"], "team": team, "rider": None}
                return self.task
            for rider in team["riders"]:
                if not rider["profile_fetched"]:
                    cached = self.profiles.get(rider["rider_id"])
                    if cached is not None:
                        rider.update(cached)
                        continue
                    self.task = {"url": rider["url"], "team": team, "rider": rider}
                    return self.task
        self.task = None
        return None

    def accept(self, payload: dict, pacer: RequestPacer) -> None:
        if self.task is None or payload.get("url") != self.task["url"]:
            raise ValueError("Browser result does not match the pending page.")
        status = payload["status"]
        if status == 429 or status >= 500:
            attempt = self.attempts.get(payload["url"], 0)
            self.attempts[payload["url"]] = attempt + 1
            if attempt >= 3:
                raise ValueError(f"HTTP {status}; retry limit reached for {payload['url']}.")
            response = requests.Response()
            response.status_code = status
            if payload.get("retry_after"):
                response.headers["Retry-After"] = payload["retry_after"]
            pacer.defer(response, attempt)
            return
        if status != 200 or payload.get("final_url") != self.task["url"]:
            raise ValueError("DIRT access failed or sign-in expired; checkpoint preserved.")
        team, rider = self.task["team"], self.task["rider"]
        if rider is None:
            team["riders"] = parse_roster(payload["html"])
            team["roster_fetched"] = True
            print(f"{team['name']}: {len(team['riders'])} riders", flush=True)
        else:
            profile = parse_rider(payload["html"])
            self.profiles[rider["rider_id"]] = profile
            rider.update(profile)
        self.save()

    def snapshot(self) -> dict:
        riders = [rider for team in self.teams for rider in team["riders"]]
        unique = {rider["rider_id"]: rider for rider in riders}
        return {
            "season": 14,
            "source_url": TEAMS_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "complete": all(team["roster_fetched"] for team in self.teams)
                and all(rider["profile_fetched"] for rider in riders),
            "team_count": len(self.teams),
            "rosters_fetched": sum(team["roster_fetched"] for team in self.teams),
            "assignment_count": len(riders),
            "rider_count": len(unique),
            "profiles_fetched": sum(rider["profile_fetched"] for rider in unique.values()),
            "missing_zwift_ids": sum(rider["profile_fetched"] and rider["zwift_id"] is None for rider in unique.values()),
            "teams": self.teams,
        }

    def save(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(self.output)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * (attempt + 1))


def make_server(collection: Collection, pacer: RequestPacer, port: int, origin: str = BASE_URL, worker: str | None = None) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.reply({})

        def do_GET(self):
            if self.path == "/worker" and worker is not None:
                self.reply({"worker": worker})
                return
            if self.path != "/next":
                self.reply({"error": "Not found"}, 404)
                return
            task = collection.next_task()
            if task is None:
                collection.save()
                self.reply({"done": True, **{key: value for key, value in collection.snapshot().items() if key not in ("teams", "riders")}})
                collection.done = True
                return
            pacer.wait()
            pacer.started()
            self.reply({"url": task["url"]})

        def do_POST(self):
            if self.path != "/page" or self.headers.get("Origin") != origin:
                self.reply({"error": "Only the authenticated source browser page may submit results."}, 403)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4_000_000 or self.headers.get_content_type() != "application/json":
                self.reply({"error": "Expected a JSON page result under 4 MB."}, 400)
                return
            try:
                collection.accept(json.loads(self.rfile.read(size)), pacer)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                collection.error = str(exc)
                self.reply({"error": collection.error}, 400)
                return
            self.reply({"saved": True})

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(server: HTTPServer, collection: Collection) -> None:
    while not collection.done and collection.error is None:
        server.handle_request()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", help="Test one exact team name instead of the full current series.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Refetch all rosters and profiles.")
    parser.add_argument("--request-interval", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--browser-relay", action="store_true", help="Use an already signed-in DIRT browser page.")
    mode.add_argument("--login", action="store_true", help="Open a persistent browser and sign in directly there.")
    args = parser.parse_args()
    try:
        pacer = RequestPacer(args.request_interval)
        response = requests.get(TEAMS_URL, timeout=60)
        response.raise_for_status()
        teams = parse_teams(response.text)
        total_teams = len(teams)
        if args.team:
            if args.output == DEFAULT_OUTPUT:
                raise ValueError("Use --output with --team to avoid replacing the full roster with a sample.")
            teams = [team for team in teams if team["name"].casefold() == args.team.casefold()]
            if not teams:
                raise ValueError(f"Current-series team not found: {args.team}")
        collection = Collection(teams, args.output, args.refresh)
        collection.save()
        with make_server(collection, pacer, args.port) as server:
            relay = f"http://127.0.0.1:{server.server_port}"
            print(f"Found {total_teams} current-series teams; collecting {len(teams)}. Relay: {relay}", flush=True)
            if args.browser_relay:
                print("Run BROWSER_WORKER from this module in the authenticated DIRT page, passing the relay URL.", flush=True)
                serve(server, collection)
            else:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as playwright:
                    profile = Path.home() / ".zwifttools" / "dirt-browser"
                    context = playwright.chromium.launch_persistent_context(str(profile), headless=False)
                    try:
                        page = context.pages[0] if context.pages else context.new_page()
                        page.goto(teams[0]["url"])
                        input("Sign in directly in the browser, then press Enter here (no credentials): ")
                        worker = threading.Thread(target=serve, args=(server, collection), daemon=True)
                        worker.start()
                        while not collection.done and collection.error is None:
                            result = page.evaluate(BROWSER_WORKER, relay)
                            if not result["done"]:
                                page.reload()
                        worker.join()
                    finally:
                        context.close()
        if collection.error:
            raise ValueError(collection.error)
        print(json.dumps({key: value for key, value in collection.snapshot().items() if key != "teams"}), flush=True)
        return 0
    except (ValueError, requests.RequestException) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())