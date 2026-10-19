"""Shared pacing and retry cooldowns for sequential data downloads."""

from email.utils import parsedate_to_datetime
import math
import sys
import time

import requests


def retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0
    try:
        value = value.strip()
        if value.isdecimal():
            seconds = float(value)
            if math.isfinite(seconds):
                return seconds
            raise ValueError("Non-finite delay")
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            raise ValueError("HTTP-date must include a timezone")
        return max(0, deadline.timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        print(
            f"Invalid Retry-After header {value!r}; using local backoff.",
            file=sys.stderr,
        )
        return 0


class RequestPacer:
    def __init__(self, request_interval: float = 2):
        if not math.isfinite(request_interval) or request_interval <= 0:
            raise ValueError("Request interval must be positive and finite.")
        self.request_interval = request_interval
        self._max_interval = max(60, request_interval)
        self._next_request_at = 0.0

    def wait(self) -> None:
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def started(self) -> None:
        self._next_request_at = time.monotonic() + self.request_interval

    def defer(self, response: requests.Response, attempt: int) -> None:
        delay = retry_after_seconds(response.headers.get("Retry-After"))
        if response.status_code == 429:
            self.request_interval = min(self.request_interval * 2, self._max_interval)
            delay = max(delay, 30 * 2 ** attempt, self.request_interval)
            print(
                f"HTTP 429; cooldown {delay:g}s; subsequent request interval "
                f"{self.request_interval:g}s.",
                file=sys.stderr,
            )
        else:
            delay = max(delay, 2 ** attempt)
        self._next_request_at = max(
            self._next_request_at, time.monotonic() + delay
        )
        if attempt < 3:
            delay = self._next_request_at - time.monotonic()
            print(
                f"HTTP {response.status_code}; retrying in {delay:g}s.",
                file=sys.stderr,
            )
