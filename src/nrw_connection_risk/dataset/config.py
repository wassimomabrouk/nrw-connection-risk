"""Dataset builder settings from config/dataset.toml, stations from config/collector.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    hubs: dict[str, str]                      # EVA number -> station name (role "hub" only)
    builder_version: int = 1
    min_parser_version: int = 2
    min_transfer_min: int = 4
    max_slack_min: int = 30
    start_hour_local: int = 4
    cutoffs_min: tuple[int, ...] = (60, 30, 10)
    max_collector_gap_min: int = 45
    label_horizon_h: int = 6

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def load_config(path: Path) -> DatasetConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    with open(path.parent / "collector.toml", "rb") as f:
        stations = tomllib.load(f)["stations"]
    return DatasetConfig(
        hubs={str(s["eva"]): s["name"] for s in stations if s.get("role", "hub") == "hub"},
        builder_version=int(c["builder_version"]),
        min_parser_version=int(c["min_parser_version"]),
        min_transfer_min=int(c["candidates"]["min_transfer_min"]),
        max_slack_min=int(c["candidates"]["max_slack_min"]),
        start_hour_local=int(c["service_day"]["start_hour_local"]),
        cutoffs_min=tuple(int(x) for x in c["cutoffs"]["minutes_before_arrival"]),
        max_collector_gap_min=int(c["quality"]["max_collector_gap_min"]),
        label_horizon_h=int(c["quality"]["label_horizon_h"]),
    )
