"""Serving settings from config/serving.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingConfig:
    parsed: Path = Path("data/collector/parsed")
    raw: Path = Path("data/collector/raw")
    lookback_h: float = 6
    interval_s: float = 60
    min_minutes_ahead: float = 2
    max_minutes_ahead: float = 65
    max_collector_gap_min: float = 45
    models_dir: Path = Path("models")
    bundle: str = ""
    log_dir: Path = Path("data/serving/predictions")
    flush_interval_s: float = 600
    max_late_min: float = 5
    max_data_age_s: float = 300
    monitoring_dir: Path = Path("data/monitoring")


def load_serving_config(path: Path, root: Path) -> ServingConfig:
    """Relative paths are resolved against `root` (the repo root)."""
    with open(path, "rb") as f:
        c = tomllib.load(f)
    p = lambda v: (root / v) if not Path(v).is_absolute() else Path(v)   # noqa: E731
    return ServingConfig(
        parsed=p(c["data"]["parsed"]), raw=p(c["data"]["raw"]), lookback_h=float(c["data"]["lookback_h"]),
        interval_s=float(c["scoring"]["interval_s"]), min_minutes_ahead=float(c["scoring"]["min_minutes_ahead"]),
        max_minutes_ahead=float(c["scoring"]["max_minutes_ahead"]),
        max_collector_gap_min=float(c["scoring"]["max_collector_gap_min"]),
        models_dir=p(c["model"]["models_dir"]), bundle=str(c["model"]["bundle"]),
        log_dir=p(c["log"]["dir"]), flush_interval_s=float(c["log"]["flush_interval_s"]),
        max_late_min=float(c["log"]["max_late_min"]), max_data_age_s=float(c["health"]["max_data_age_s"]),
        monitoring_dir=p(c.get("monitoring", {}).get("dir", "data/monitoring")),
    )
