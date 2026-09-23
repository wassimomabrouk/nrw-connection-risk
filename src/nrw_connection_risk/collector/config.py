"""Collector settings: TOML file for behaviour, environment for secrets."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Station:
    eva: str
    name: str


@dataclass(frozen=True)
class Settings:
    base_url: str
    max_calls_per_minute: int
    timeout_s: float
    max_retries: int
    rchg_interval_s: int
    fchg_interval_s: int
    plan_interval_s: int
    plan_hours_behind: int
    plan_hours_ahead: int
    data_dir: Path
    flush_interval_s: int
    heartbeat_file: Path
    ping_interval_s: int
    stations: tuple[Station, ...]
    client_id: str
    api_key: str
    healthcheck_url: str | None


def load_settings(config_path: Path, root: Path | None = None) -> Settings:
    """Read the TOML config; secrets come from the environment (.env loaded by main)."""
    root = root or config_path.resolve().parents[1]
    with open(config_path, "rb") as f:
        c = tomllib.load(f)

    client_id = os.getenv("DB_CLIENT_ID", "")
    api_key = os.getenv("DB_API_KEY", "")
    if not client_id or not api_key:
        raise RuntimeError("DB_CLIENT_ID and DB_API_KEY must be set (see .env.example).")

    stations = tuple(Station(eva=str(s["eva"]), name=s["name"]) for s in c["stations"])
    if not stations:
        raise RuntimeError("No stations configured.")
    if len({s.eva for s in stations}) != len(stations):
        raise RuntimeError("Duplicate EVA numbers in config.")

    def path(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    return Settings(
        base_url=c["api"]["base_url"].rstrip("/"),
        max_calls_per_minute=int(c["api"]["max_calls_per_minute"]),
        timeout_s=float(c["api"]["timeout_s"]),
        max_retries=int(c["api"]["max_retries"]),
        rchg_interval_s=int(c["schedule"]["rchg_interval_s"]),
        fchg_interval_s=int(c["schedule"]["fchg_interval_s"]),
        plan_interval_s=int(c["schedule"]["plan_interval_s"]),
        plan_hours_behind=int(c["schedule"]["plan_hours_behind"]),
        plan_hours_ahead=int(c["schedule"]["plan_hours_ahead"]),
        data_dir=path(c["storage"]["data_dir"]),
        flush_interval_s=int(c["storage"]["flush_interval_s"]),
        heartbeat_file=path(c["health"]["heartbeat_file"]),
        ping_interval_s=int(c["health"]["ping_interval_s"]),
        stations=stations,
        client_id=client_id,
        api_key=api_key,
        healthcheck_url=os.getenv("HEALTHCHECK_URL") or None,
    )
