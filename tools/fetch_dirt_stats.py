"""Collect ZwiftRacing category ratings and power metrics for the DIRT roster."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fetch_dirt_roster import Collection, make_server, serve
from shared.http_pacing import RequestPacer


VELO_CATEGORIES = ("endurance", "pursuit", "sprint", "punch", "climb", "timeTrial")
POWER_INTERVALS = (5, 15, 30, 60, 120, 300, 1200)
BASE_URL = "https://zwiftracing.app"
DEFAULT_ROSTER = Path(__file__).resolve().parents[1] / "dirt_s14_roster.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dirt_s14_stats.json"

BROWSER_WORKER = """async relay => {
    for (let index = 0; index < 60; index++) {
        const next = await fetch(relay + '/next');
        const task = await next.json();
        if (!next.ok) throw new Error(task.error);
        if (task.done) return task;
        const router = window.__TSR_ROUTER__;
        if (!router) throw new Error('ZwiftRacing router is unavailable.');
        const matches = await router.preloadRoute({to: new URL(task.url).pathname});
        const rider = matches?.find(match => match.routeId === '/riders/$riderId')?.loaderData;
        const power = matches?.find(match => match.routeId === '/riders/$riderId/power')?.loaderData;
        const failure = matches?.find(match => match.error)?.error;
        const failedStatus = failure?.status || failure?.statusCode;
        let payload;
        if ([404, 429, 500, 502, 503, 504].includes(failedStatus)) {
            payload = {url: task.url, status: failedStatus};
        } else {
            if (!rider || !power || !Array.isArray(power.curve)) {
                throw new Error('Rider data unavailable or sign-in expired for ' + task.url);
            }
            const values = value => value == null ? null : {w: value.w ?? null, wkg: value.wkg ?? null};
            payload = {url: task.url, final_url: task.url, status: 200,
                rider: {riderId: rider.riderId, velo: {factors: rider.velo?.factors ?? null}},
                power: {riderId: power.riderId, curve: power.curve.map(row => ({
                    durationSec: row.durationSec, live: values(row.live), d90: values(row.d90)
                }))}};
        }
        const response = await fetch(relay + '/page', {method: 'POST',
            headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        const saved = await response.json();
        if (!response.ok) throw new Error(saved.error);
    }
    return {done: false, pages: 60};
}"""


def metric(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric metric or null, received {value!r}.")
    return value


def parse_stats(rider: dict, power: dict, zwift_id: int) -> dict:
    if rider.get("riderId") != zwift_id or power.get("riderId") != zwift_id:
        raise ValueError("ZwiftRacing rider ID does not match the requested rider.")
    factors = (rider.get("velo") or {}).get("factors")
    if factors is not None and not isinstance(factors, dict):
        raise ValueError("Invalid vELO category scores.")
    curve = power.get("curve")
    if not isinstance(curve, list):
        raise ValueError("ZwiftRacing power curve is missing.")
    intervals = {}
    for row in curve:
        duration = row.get("durationSec")
        if duration not in POWER_INTERVALS:
            continue
        if str(duration) in intervals:
            raise ValueError(f"Duplicate power interval {duration}.")
        intervals[str(duration)] = {
            source: {unit: metric((row.get(field) or {}).get(unit)) for unit in ("w", "wkg")}
            for source, field in (("live", "live"), ("best_90d", "d90"))
        }
    return {
        "zwift_id": zwift_id,
        "velo": {category: metric((factors or {}).get(category)) for category in VELO_CATEGORIES},
        "power": {
            str(duration): intervals.get(str(duration), {
                "live": {"w": None, "wkg": None},
                "best_90d": {"w": None, "wkg": None},
            }) for duration in POWER_INTERVALS
        },
    }


class StatsCollection(Collection):
    def __init__(self, roster: dict, output: Path, refresh: bool = False, rider_id: int | None = None):
        if roster.get("season") != 14 or not roster.get("complete"):
            raise ValueError("A complete season 14 roster is required.")
        ids = {rider["zwift_id"] for team in roster["teams"] for rider in team["riders"] if rider.get("zwift_id") is not None}
        if rider_id is not None and rider_id not in ids:
            raise ValueError(f"Rider {rider_id} is not in the season 14 roster.")
        self.ids = sorted(ids if rider_id is None else {rider_id})
        self.output = output
        self.task = None
        self.error = None
        self.done = False
        self.attempts = {}
        self.riders = {}
        if output.exists() and not refresh:
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved.get("season") != 14 or saved.get("schema_version") != 1:
                raise ValueError("Existing stats checkpoint has an unexpected season or schema.")
            self.riders = {key: value for key, value in saved["riders"].items() if int(key) in self.ids}

    def next_task(self) -> dict | None:
        self.task = next((
            {"url": f"{BASE_URL}/riders/{rider_id}/power", "zwift_id": rider_id}
            for rider_id in self.ids if str(rider_id) not in self.riders
        ), None)
        return self.task

    def accept(self, payload: dict, pacer: RequestPacer) -> None:
        if self.task is None or payload.get("url") != self.task["url"]:
            raise ValueError("Browser result does not match the pending rider.")
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
        rider_id = self.task["zwift_id"]
        if status == 404:
            record = {"zwift_id": rider_id, "status": "unavailable", "reason": "ZwiftRacing profile not found."}
        else:
            if status != 200 or payload.get("final_url") != self.task["url"]:
                raise ValueError("ZwiftRacing access failed or sign-in expired; checkpoint preserved.")
            record = {**parse_stats(payload["rider"], payload["power"], rider_id), "status": "ok"}
        record["fetched_at"] = datetime.now(timezone.utc).isoformat()
        self.riders[str(rider_id)] = record
        self.save()
        print(f"Rider {rider_id}: {record['status']} ({len(self.riders)}/{len(self.ids)})", flush=True)

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "season": 14,
            "source_url": BASE_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "complete": len(self.riders) == len(self.ids),
            "rider_count": len(self.ids),
            "fetched_count": len(self.riders),
            "available_count": sum(rider["status"] == "ok" for rider in self.riders.values()),
            "unavailable_count": sum(rider["status"] == "unavailable" for rider in self.riders.values()),
            "riders": self.riders,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rider-id", type=int, help="Test one roster rider; requires a separate --output.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--browser-relay", action="store_true")
    mode.add_argument("--login", action="store_true")
    args = parser.parse_args()
    try:
        if args.rider_id is not None and args.output == DEFAULT_OUTPUT:
            raise ValueError("Use --output with --rider-id to protect the full stats snapshot.")
        pacer = RequestPacer(args.request_interval)
        roster = json.loads(args.roster.read_text(encoding="utf-8"))
        collection = StatsCollection(roster, args.output, args.refresh, args.rider_id)
        collection.save()
        with make_server(collection, pacer, args.port, origin=BASE_URL, worker=BROWSER_WORKER) as server:
            relay = f"http://127.0.0.1:{server.server_port}"
            print(f"Collecting {len(collection.ids)} unique riders. Relay: {relay}", flush=True)
            if args.browser_relay:
                print("Run BROWSER_WORKER in the authenticated ZwiftRacing page; reload between batches.", flush=True)
                serve(server, collection)
            else:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as playwright:
                    profile = Path.home() / ".zwifttools" / "zwiftracing-browser"
                    context = playwright.chromium.launch_persistent_context(str(profile), headless=False)
                    try:
                        page = context.pages[0] if context.pages else context.new_page()
                        page.goto(f"{BASE_URL}/riders/7681059/power")
                        input("Sign in directly in the browser, then press Enter here (no credentials): ")
                        worker = threading.Thread(target=serve, args=(server, collection), daemon=True)
                        worker.start()
                        while not collection.done and collection.error is None:
                            result = page.evaluate(f"({BROWSER_WORKER})({json.dumps(relay)})")
                            if not result["done"]:
                                page.reload()
                                page.wait_for_function("Boolean(window.__TSR_ROUTER__)")
                        worker.join()
                    finally:
                        context.close()
        if collection.error:
            raise ValueError(collection.error)
        print(json.dumps({key: value for key, value in collection.snapshot().items() if key != "riders"}), flush=True)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())