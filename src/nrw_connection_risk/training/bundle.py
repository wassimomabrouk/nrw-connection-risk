"""A deployable model bundle: the selected model plus B3, fitted once and saved together
with a model card (model.json).

B3 travels with the model so that the live service scores every connection with both:
monitoring can then compare model and DB baseline on live data, not only offline.

Layout: models/<model_id>/bundle.joblib and models/<model_id>/model.json
"""
from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .models import Model

COMPANION = "B3"


class BundleVersionError(RuntimeError):
    """The bundle was saved with an incompatible library version."""


@dataclass
class Bundle:
    models: dict[str, Model]      # "model" (the selected one) and "B3"
    meta: dict

    @property
    def model_id(self) -> str:
        return self.meta["model_id"]

    @property
    def cutoffs(self) -> list[int]:
        return list(self.meta["cutoffs"])

    def predict(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        """P(fail) of every model in the bundle; rows need `cutoff_min` in the bundle's cutoffs."""
        unknown = set(rows.cutoff_min.unique()) - set(self.cutoffs)
        if unknown:
            raise ValueError(f"no model for cutoffs {sorted(unknown)}")
        return {name: m.predict(rows) for name, m in self.models.items()}


def versions() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def save_bundle(bundle: Bundle, models_root: Path) -> Path:
    out = models_root / bundle.model_id
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle.models, out / "bundle.joblib", compress=3)
    (out / "model.json").write_text(json.dumps(bundle.meta, indent=2, default=str), encoding="utf-8")
    return out


def _minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def load_bundle(path: Path) -> Bundle:
    """Load a bundle folder. Refuses a scikit-learn minor version other than the one it was
    saved with: pickled models are only guaranteed to work with the same version."""
    meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
    saved, now = meta["versions"], versions()
    if _minor(saved["scikit-learn"]) != _minor(now["scikit-learn"]):
        raise BundleVersionError(f"bundle {meta['model_id']} was saved with scikit-learn {saved['scikit-learn']}, "
                                 f"this environment has {now['scikit-learn']}. Install the same minor version "
                                 f"or refit the bundle.")
    for lib in ("pandas", "numpy"):
        if saved[lib].split(".")[0] != now[lib].split(".")[0]:
            warnings.warn(f"bundle saved with {lib} {saved[lib]}, running {now[lib]}")
    return Bundle(models=joblib.load(path / "bundle.joblib"), meta=meta)


DATASET_KEYS = ("builder_version", "min_transfer_min", "max_slack_min", "hubs")


def dataset_mismatch(meta: dict, current: dict) -> list[str]:
    """Dataset settings that differ between the model's training data and the live service
    (they decide which connections exist and how they are labelled)."""
    saved = meta.get("dataset_config")
    if saved is None:
        return ["model card has no dataset settings"]
    return [f"{k}: model {saved.get(k)!r}, service {current.get(k)!r}" for k in DATASET_KEYS
            if saved.get(k) != current.get(k)]


def latest_bundle(models_root: Path) -> Path | None:
    """Newest bundle folder (model ids start with a sortable timestamp)."""
    dirs = sorted(p for p in models_root.glob("*") if (p / "model.json").exists()) if models_root.exists() else []
    return dirs[-1] if dirs else None


def new_model_id(name: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{name}"
