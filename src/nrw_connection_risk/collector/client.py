"""HTTP client for the DB Timetables API with rate limiting and retries."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` in any `window_s` seconds."""

    def __init__(self, max_calls: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_calls = max_calls
        self.window_s = window_s
        self.clock = clock
        self.sleep = sleep
        self.calls: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.window_s:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            self.sleep(self.window_s - (now - self.calls[0]) + 0.01)


@dataclass
class ApiResponse:
    url: str
    status: int
    body: str
    collected_at: datetime          # UTC, timezone-aware, when the response arrived
    duration_ms: float


class TimetablesClient:
    def __init__(self, base_url: str, client_id: str, api_key: str,
                 max_calls_per_minute: int = 50, timeout_s: float = 30,
                 max_retries: int = 3, session: requests.Session | None = None,
                 limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({
            "DB-Client-Id": client_id,
            "DB-Api-Key": api_key,
            "accept": "application/xml",
        })
        self.limiter = limiter or RateLimiter(max_calls_per_minute)
        self.sleep = sleep
        self.calls_total = 0
        self.errors_total = 0

    def plan(self, eva: str, yymmdd: str, hh: str) -> ApiResponse | None:
        return self._get(f"/plan/{eva}/{yymmdd}/{hh}")

    def fchg(self, eva: str) -> ApiResponse | None:
        return self._get(f"/fchg/{eva}")

    def rchg(self, eva: str) -> ApiResponse | None:
        return self._get(f"/rchg/{eva}")

    def _get(self, path: str) -> ApiResponse | None:
        """Returns the response, or None after all retries failed (logged, never raised)."""
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            self.limiter.acquire()
            self.calls_total += 1
            t0 = time.perf_counter()
            try:
                r = self.session.get(url, timeout=self.timeout_s)
            except requests.RequestException as exc:
                log.warning("GET %s failed (attempt %d/%d): %s", path, attempt, self.max_retries, exc)
                self._backoff(attempt, None)
                continue
            duration_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return ApiResponse(url=url, status=200, body=r.text,
                                   collected_at=datetime.now(timezone.utc),
                                   duration_ms=duration_ms)
            if r.status_code in RETRYABLE_STATUS:
                log.warning("GET %s -> HTTP %d (attempt %d/%d)", path, r.status_code, attempt, self.max_retries)
                self._backoff(attempt, r.headers.get("Retry-After"))
                continue
            # 4xx other than 429: not retryable (bad key, unknown station, empty past plan ...)
            log.error("GET %s -> HTTP %d: %s", path, r.status_code, r.text[:200])
            self.errors_total += 1
            return None
        self.errors_total += 1
        log.error("GET %s gave up after %d attempts", path, self.max_retries)
        return None

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after and retry_after.isdigit():
            wait = min(float(retry_after), 120.0)
        else:
            wait = min(2 ** attempt, 30)
        self.sleep(wait)
