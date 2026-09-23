"""Heartbeat file and optional external ping (e.g. healthchecks.io)."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class Health:
    def __init__(self, heartbeat_file: Path, ping_url: str | None, ping_interval_s: int):
        self.file = heartbeat_file
        self.ping_url = ping_url
        self.ping_interval_s = ping_interval_s
        self.started_at = datetime.now(timezone.utc)
        self.last_success: dict[str, str] = {}
        self.success_count: dict[str, int] = {}
        self.failure_count: dict[str, int] = {}
        self._last_ping = 0.0
        self._last_rchg_ok = 0.0

    def record(self, source: str, ok: bool) -> None:
        if ok:
            self.last_success[source] = datetime.now(timezone.utc).isoformat()
            self.success_count[source] = self.success_count.get(source, 0) + 1
            if source == "rchg":
                self._last_rchg_ok = time.monotonic()
        else:
            self.failure_count[source] = self.failure_count.get(source, 0) + 1

    def write(self, extra: dict | None = None) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "started_at": self.started_at.isoformat(),
            "last_success": self.last_success,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            **(extra or {}),
        }
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.file)

    def maybe_ping(self) -> None:
        """Ping only while rchg succeeds, so a silent failure stops the pings and triggers an alert."""
        if not self.ping_url:
            return
        now = time.monotonic()
        if now - self._last_ping < self.ping_interval_s or now - self._last_rchg_ok > 300:
            return
        try:
            requests.get(self.ping_url, timeout=10)
            self._last_ping = now
        except requests.RequestException as exc:
            log.warning("health ping failed: %s", exc)
