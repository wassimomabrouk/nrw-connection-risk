"""Feature settings from config/features.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .columns import feature_columns


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: int = 1
    groups: tuple[str, ...] = ("db", "hub", "freshness", "context")
    grid_min: int = 5
    window_min: int = 30
    line_lookback_min: int = 60
    late_threshold_min: int = 5
    holidays: frozenset[date] = field(default_factory=frozenset)

    def columns(self) -> list[str]:
        return feature_columns(list(self.groups))

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        """Inverse of as_dict (used to restore the settings a model was trained with)."""
        d = dict(d)
        d["groups"] = tuple(d["groups"])
        d["holidays"] = frozenset(date.fromisoformat(x) for x in d["holidays"])
        return cls(**d)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["groups"] = list(self.groups)
        d["holidays"] = sorted(x.isoformat() for x in self.holidays)
        return d


def load_feature_config(path: Path) -> FeatureConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    cfg = FeatureConfig(
        feature_version=int(c["feature_version"]),
        groups=tuple(c["groups"]),
        grid_min=int(c["hub"]["grid_min"]),
        window_min=int(c["hub"]["window_min"]),
        line_lookback_min=int(c["hub"]["line_lookback_min"]),
        late_threshold_min=int(c["hub"]["late_threshold_min"]),
        holidays=frozenset(date.fromisoformat(x) for x in c["context"]["holidays"]),
    )
    cfg.columns()                      # validates group names
    return cfg
