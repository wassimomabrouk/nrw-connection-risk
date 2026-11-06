"""Training settings from config/training.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

BASELINES = ("B0", "B1", "B2", "B3-lin", "B3")
MODELS = ("lr", "gbm")


@dataclass(frozen=True)
class Splits:
    train: tuple[date, date] = (date(2026, 9, 24), date(2026, 11, 8))
    validation: tuple[date, date] = (date(2026, 11, 9), date(2026, 11, 22))
    test: tuple[date, date] = (date(2026, 11, 23), date(2026, 12, 12))
    robustness_from: date = date(2026, 12, 13)


@dataclass(frozen=True)
class TrainingConfig:
    splits: Splits = field(default_factory=Splits)
    min_train_days: int = 14
    fold_days: int = 7
    run: tuple[str, ...] = BASELINES + MODELS
    cutoff_mode: str = "separate"
    calibration: str = "none"
    calibration_folds: int = 5
    lr: dict = field(default_factory=lambda: {"C": 1.0, "clip_quantile": 0.005})
    gbm: dict = field(default_factory=lambda: {
        "learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 31, "min_samples_leaf": 100,
        "l2_regularization": 1.0, "validation_fraction": 0.15, "n_iter_no_change": 20})
    b3: dict = field(default_factory=dict)           # B3's own GBM settings where tuning chose differently
    first_fit_day: date | None = None                # fit only on days from here (first day with all features)
    fit_window: dict = field(default_factory=dict)   # model -> days of history it is fit on (0 or absent: all)
    slack_buckets: tuple[int, ...] = (4, 6, 8, 10, 15, 20, 31)
    smoothing: float = 20.0
    primary_cutoff: int = 30
    headline_baseline: str = "B3"
    slices: tuple[str, ...] = ("segment_a", "segment_b", "hub")
    bootstrap_n: int = 1000
    bootstrap_n_auc: int = 200
    calibration_bins: int = 10
    min_days_for_ci: int = 10
    selection_tie: float = 0.01
    seed: int = 42

    def __post_init__(self):
        unknown = set(self.run) - set(BASELINES + MODELS)
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        if self.cutoff_mode not in ("separate", "feature"):
            raise ValueError("cutoff_mode must be 'separate' or 'feature'")
        if self.calibration not in ("none", "platt", "isotonic"):
            raise ValueError("calibration must be 'none', 'platt' or 'isotonic'")
        bad = {k: v for k, v in self.fit_window.items() if k not in BASELINES + MODELS or int(v) < 0}
        if bad:
            raise ValueError(f"fit_window needs known models and days >= 0: {bad}")

    def window(self, model: str) -> int:
        """Days of history `model` is fit on; 0 = every day from first_fit_day."""
        return int(self.fit_window.get(model, 0))

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "splits"}
        d["first_fit_day"] = self.first_fit_day.isoformat() if self.first_fit_day else None
        s = self.splits
        d["splits"] = {"train": [x.isoformat() for x in s.train],
                       "validation": [x.isoformat() for x in s.validation],
                       "test": [x.isoformat() for x in s.test], "robustness_from": s.robustness_from.isoformat()}
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def load_training_config(path: Path) -> TrainingConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    s, m, e = c["splits"], c["models"], c["evaluation"]
    pair = lambda v: (date.fromisoformat(v[0]), date.fromisoformat(v[1]))   # noqa: E731
    return TrainingConfig(
        splits=Splits(train=pair(s["train"]), validation=pair(s["validation"]), test=pair(s["test"]),
                      robustness_from=date.fromisoformat(s["robustness_from"])),
        min_train_days=int(c["cv"]["min_train_days"]), fold_days=int(c["cv"]["fold_days"]),
        run=tuple(m["run"]), cutoff_mode=m["cutoff_mode"], calibration=m["calibration"],
        calibration_folds=int(m["calibration_folds"]),
        lr=dict(c["lr"]), gbm=dict(c["gbm"]), b3=dict(c.get("b3", {})),
        first_fit_day=date.fromisoformat(m["first_fit_day"]) if m.get("first_fit_day") else None,
        fit_window={k: int(v) for k, v in c.get("fit_window", {}).items()},
        slack_buckets=tuple(int(x) for x in c["history"]["slack_buckets"]),
        smoothing=float(c["history"]["smoothing"]),
        primary_cutoff=int(e["primary_cutoff"]), headline_baseline=e["headline_baseline"],
        slices=tuple(e["slices"]), bootstrap_n=int(e["bootstrap_n"]), bootstrap_n_auc=int(e["bootstrap_n_auc"]),
        calibration_bins=int(e["calibration_bins"]), min_days_for_ci=int(e["min_days_for_ci"]),
        selection_tie=float(e["selection_tie"]), seed=int(e["seed"]),
    )
