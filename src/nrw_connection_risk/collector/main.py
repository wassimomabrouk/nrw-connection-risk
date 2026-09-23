"""Collector entry point.

Run from the repo root:
    py -m nrw_connection_risk.collector.main                 # run until stopped (Ctrl+C)
    py -m nrw_connection_risk.collector.main --duration-min 10
    py -m nrw_connection_risk.collector.main --once           # every job once, then exit
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from .client import TimetablesClient
from .config import Settings, load_settings
from .health import Health
from .parse import BERLIN, parse_timetable
from .scheduler import Scheduler
from .storage import ParsedStore, RawStore

log = logging.getLogger("collector")


class Collector:
    def __init__(self, settings: Settings, client: TimetablesClient | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        self.s = settings
        self.client = client or TimetablesClient(
            settings.base_url, settings.client_id, settings.api_key,
            max_calls_per_minute=settings.max_calls_per_minute,
            timeout_s=settings.timeout_s, max_retries=settings.max_retries)
        self.raw = RawStore(settings.data_dir)
        self.parsed = ParsedStore(settings.data_dir)
        self.health = Health(settings.heartbeat_file, settings.healthcheck_url, settings.ping_interval_s)
        self.clock, self.sleep = clock, sleep
        self.scheduler = Scheduler(clock)
        self.stop = False
        self._build_jobs()

    # ---------- jobs ----------
    def _build_jobs(self) -> None:
        n = len(self.s.stations)
        for i, st in enumerate(self.s.stations):
            # stagger stations so calls are spread evenly, start with full state and plans
            self.scheduler.add(f"fchg:{st.eva}", self.s.fchg_interval_s,
                               lambda e=st.eva: self._fetch("fchg", e), offset_s=i * 2)
            self.scheduler.add(f"plan:{st.eva}", self.s.plan_interval_s,
                               lambda e=st.eva: self._plan(e), offset_s=n * 2 + i * 2)
            self.scheduler.add(f"rchg:{st.eva}", self.s.rchg_interval_s,
                               lambda e=st.eva: self._fetch("rchg", e),
                               offset_s=n * 4 + i * self.s.rchg_interval_s / n)

    def _store(self, source: str, eva: str, resp) -> None:
        self.raw.write(source, eva, resp)
        try:
            rows = parse_timetable(resp.body, source, eva, resp.collected_at)
            self.parsed.add(rows)
        except Exception:  # raw is kept, so a parse bug never loses data
            log.exception("parse failed for %s %s (raw kept)", source, eva)

    def _fetch(self, source: str, eva: str) -> None:
        resp = self.client.fchg(eva) if source == "fchg" else self.client.rchg(eva)
        self.health.record(source, resp is not None)
        if resp is not None:
            self._store(source, eva, resp)

    def _plan(self, eva: str) -> None:
        now_utc = datetime.now(timezone.utc)
        for h in range(-self.s.plan_hours_behind, self.s.plan_hours_ahead + 1):
            local = (now_utc + timedelta(hours=h)).astimezone(BERLIN)
            resp = self.client.plan(eva, f"{local:%y%m%d}", f"{local:%H}")
            self.health.record("plan", resp is not None)
            if resp is not None:
                self._store("plan", eva, resp)

    # ---------- loop ----------
    def run(self, duration_s: float | None = None, once: bool = False) -> None:
        start = self.clock()
        last_flush = start
        log.info("collector started: %d stations, %d jobs", len(self.s.stations), len(self.scheduler.jobs))
        try:
            if once:
                for job in sorted(self.scheduler.jobs, key=lambda j: j.next_run):
                    self.scheduler.run_job(job)
                return
            while not self.stop:
                if duration_s is not None and self.clock() - start >= duration_s:
                    log.info("duration reached, stopping")
                    break
                job = self.scheduler.next_job()
                wait = job.next_run - self.clock()
                if wait > 0:
                    self.sleep(min(wait, 1.0))   # short naps keep Ctrl+C and SIGTERM responsive
                    continue
                self.scheduler.run_job(job)
                self.health.write(self._status())
                self.health.maybe_ping()
                if self.clock() - last_flush >= self.s.flush_interval_s:
                    self.parsed.flush()
                    last_flush = self.clock()
        finally:
            self.parsed.flush()
            self.health.write(self._status())
            log.info("collector stopped (API calls: %d, failed requests: %d)",
                     self.client.calls_total, self.client.errors_total)

    def _status(self) -> dict:
        return {"api_calls_total": self.client.calls_total,
                "api_errors_total": self.client.errors_total,
                "parsed_rows_pending": self.parsed.pending()}


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = RotatingFileHandler(log_dir / "collector.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file.setFormatter(fmt)
    root.handlers[:] = [console, file]


def main() -> None:
    ap = argparse.ArgumentParser(description="DB Timetables collector")
    ap.add_argument("--config", type=Path, default=Path("config/collector.toml"))
    ap.add_argument("--duration-min", type=float, default=None)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    root = args.config.resolve().parents[1]
    load_dotenv(root / ".env")
    settings = load_settings(args.config, root)
    setup_logging(settings.data_dir / "logs")

    collector = Collector(settings)

    def _stop(signum, frame):
        log.info("signal %s received, finishing current job", signum)
        collector.stop = True

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    collector.run(duration_s=args.duration_min * 60 if args.duration_min else None, once=args.once)


if __name__ == "__main__":
    main()
