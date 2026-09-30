"""Reference profile of the training data and population stability (PSI).

The profile is stored in the model card when a bundle is fitted: for every input column
the share of rows per decile bin (numeric) or per value (categorical), plus the share of
missing values. Live rows are binned the same way; PSI measures how far apart the two
distributions are. Usual reading: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.columns import CATEGORICAL

MISSING = "<missing>"
EPS = 1e-4


def _numeric_bins(x: pd.Series, edges: list[float]) -> np.ndarray:
    """Share per bin: len(edges) + 1 value bins, then one bin for missing values."""
    v = x.to_numpy(dtype=float)
    miss = np.isnan(v)
    idx = np.searchsorted(np.asarray(edges, dtype=float), v[~miss], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, miss.sum())
    return counts / max(len(v), 1)


def build_profile(rows: pd.DataFrame, cols: list[str], bins: int = 10) -> dict:
    prof = {}
    for c in cols:
        if c in CATEGORICAL:
            v = rows[c].astype("string").fillna(MISSING)
            prof[c] = {"kind": "categorical", "shares": {str(k): float(s) for k, s in
                                                         v.value_counts(normalize=True).items()}}
        else:
            x = pd.to_numeric(rows[c], errors="coerce").astype(float)
            present = x.dropna()
            edges = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1])).tolist() if len(present) else []
            prof[c] = {"kind": "numeric", "edges": edges, "shares": _numeric_bins(x, edges).tolist()}
    return prof


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = np.maximum(np.asarray(expected, float), EPS), np.maximum(np.asarray(actual, float), EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(profile: dict, rows: pd.DataFrame) -> dict[str, float]:
    """PSI of every profiled column that the rows contain."""
    out = {}
    for c, p in profile.items():
        if c not in rows:
            continue
        if p["kind"] == "categorical":
            live = rows[c].astype("string").fillna(MISSING).value_counts(normalize=True)
            cats = sorted(set(p["shares"]) | set(live.index))
            out[c] = psi([p["shares"].get(k, 0.0) for k in cats], [float(live.get(k, 0.0)) for k in cats])
        else:
            x = pd.to_numeric(rows[c], errors="coerce").astype(float)
            out[c] = psi(p["shares"], _numeric_bins(x, p["edges"]))
    return out


def level(value: float, moderate: float = 0.1, large: float = 0.25) -> str:
    return "large" if value > large else "moderate" if value > moderate else "stable"
