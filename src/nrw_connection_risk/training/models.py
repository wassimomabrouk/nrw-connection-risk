"""Baselines and models behind one interface: fit(rows) and predict(rows) -> P(fail).

Rows are feature-table rows (docs/FEATURES.md) with `label_fail`, `service_day` and
`cutoff_min`. B3 and gbm share one implementation and one set of settings, so their
difference measures the value of the extra information, not of the model class.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features.columns import CATEGORICAL, GROUPS
from .config import TrainingConfig

EPS = 1e-6


class Model(Protocol):
    def fit(self, rows: pd.DataFrame) -> "Model": ...
    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...


def _y(rows: pd.DataFrame) -> np.ndarray:
    return rows.label_fail.astype(int).to_numpy()


# ---------------------------------------------------------------- learned models

class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column at quantiles of the training data (robust linear inputs)."""

    def __init__(self, q: float = 0.005):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanquantile(X, self.q, axis=0), np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class LogReg:
    """Logistic regression: numeric inputs clipped, imputed with a missing indicator and
    standardised; categorical inputs one-hot encoded."""

    def __init__(self, cols: list[str], C: float = 1.0, clip_quantile: float = 0.005):
        self.cols, self.C, self.q = list(cols), C, clip_quantile

    def fit(self, rows):
        num = [c for c in self.cols if c not in CATEGORICAL]
        cat = [c for c in self.cols if c in CATEGORICAL]
        parts = []
        if num:
            parts.append(("num", make_pipeline(QuantileClipper(self.q),
                                               SimpleImputer(strategy="median", add_indicator=True,
                                                             keep_empty_features=True),
                                               StandardScaler()), num))
        if cat:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        self.pipe_ = make_pipeline(ColumnTransformer(parts), LogisticRegression(C=self.C, max_iter=5000))
        self.pipe_.fit(self._frame(rows), _y(rows))
        return self

    def _frame(self, rows):
        X = rows[self.cols].copy()
        for c in self.cols:
            X[c] = X[c].astype(str) if c in CATEGORICAL else X[c].astype(float)
        return X

    def predict(self, rows):
        return self.pipe_.predict_proba(self._frame(rows))[:, 1]


class GBM:
    """Histogram gradient boosting with native missing values and categorical inputs."""

    def __init__(self, cols: list[str], seed: int = 0, **params):
        self.cols, self.seed, self.params = list(cols), seed, params

    def fit(self, rows):
        self.categories_ = {c: sorted(rows[c].dropna().astype(str).unique()) for c in self.cols
                            if c in CATEGORICAL}
        self.clf_ = HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=True,
                                                   random_state=self.seed, **self.params)
        self.clf_.fit(self._frame(rows), _y(rows))
        return self

    def _frame(self, rows):
        X = rows[self.cols].copy()
        for c in self.cols:
            if c in CATEGORICAL:      # categories fixed at fit time; unseen values become missing
                v = X[c].astype("string")
                X[c] = pd.Categorical(v.where(v.isin(self.categories_[c])), categories=self.categories_[c])
            else:
                X[c] = X[c].astype(float)
        return X

    def predict(self, rows):
        return self.clf_.predict_proba(self._frame(rows))[:, 1]


# ---------------------------------------------------------------- non-learned baselines

class HistoryRate:
    """B1: failure rate per hub, segment pair and planned-slack bucket on the training
    days, shrunk towards hub x bucket and then bucket rates when a cell is small."""

    def __init__(self, buckets: tuple[int, ...], smoothing: float):
        self.buckets, self.m = list(buckets), smoothing

    def _cells(self, rows):
        b = pd.cut(rows.planned_slack_min, self.buckets, right=False, labels=False)
        return pd.DataFrame({"b": b.fillna(-1).astype(int).to_numpy(), "hub": rows.hub.astype(str).to_numpy(),
                             "seg": (rows.segment_a.astype(str) + ">" + rows.segment_b.astype(str)).to_numpy()})

    def fit(self, rows):
        c = self._cells(rows).assign(y=_y(rows))
        self.base_ = c.y.mean()
        lvl1 = c.groupby("b").y.agg(["sum", "count"])
        self.r1_ = (lvl1["sum"] + self.m * self.base_) / (lvl1["count"] + self.m)
        lvl2 = c.groupby(["b", "hub"]).y.agg(["sum", "count"]).reset_index()
        lvl2["prior"] = lvl2.b.map(self.r1_)
        lvl2["r"] = (lvl2["sum"] + self.m * lvl2.prior) / (lvl2["count"] + self.m)
        self.r2_ = lvl2.set_index(["b", "hub"]).r
        lvl3 = c.groupby(["b", "hub", "seg"]).y.agg(["sum", "count"]).reset_index()
        lvl3["prior"] = self.r2_.reindex(pd.MultiIndex.from_frame(lvl3[["b", "hub"]])).to_numpy()
        lvl3["r"] = (lvl3["sum"] + self.m * lvl3.prior) / (lvl3["count"] + self.m)
        self.r3_ = lvl3.set_index(["b", "hub", "seg"]).r
        return self

    def predict(self, rows):
        c = self._cells(rows)
        p3 = self.r3_.reindex(pd.MultiIndex.from_frame(c[["b", "hub", "seg"]])).to_numpy()
        p2 = self.r2_.reindex(pd.MultiIndex.from_frame(c[["b", "hub"]])).to_numpy()
        p1 = self.r1_.reindex(c.b).to_numpy()
        return np.where(~np.isnan(p3), p3, np.where(~np.isnan(p2), p2, np.where(~np.isnan(p1), p1, self.base_)))


class DBRule:
    """B2: DB's own verdict, binary. Fails if DB's predicted slack is below the minimum
    transfer time or B's cancellation is known."""

    def __init__(self, min_transfer_min: float = 4.0):
        self.min = min_transfer_min

    def fit(self, rows):
        return self

    def predict(self, rows):
        return ((rows.db_slack_min < self.min) | (rows.b_cancel_known.astype(int) == 1)).astype(float).to_numpy()


# ---------------------------------------------------------------- wrappers

class Calibrated:
    """Probability calibration without looking at the evaluation data: out-of-fold
    predictions over blocks of whole training days, a calibrator fit on them, then the
    base model refit on all training days. Needs at least two training days."""

    def __init__(self, make: Callable[[], Model], method: str, folds: int = 5):
        self.make, self.method, self.folds = make, method, folds

    def fit(self, rows):
        days = np.array(sorted(rows.service_day.unique()))
        self.skipped_ = len(days) < 2
        if not self.skipped_:
            blocks = np.array_split(days, min(self.folds, len(days)))
            oof = np.empty(len(rows))
            for block in blocks:
                held = rows.service_day.isin(block).to_numpy()
                oof[held] = self.make().fit(rows[~held]).predict(rows[held])
            y = _y(rows)
            if self.method == "isotonic":
                self.cal_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(oof, y)
            else:
                z = np.log(np.clip(oof, EPS, 1 - EPS) / (1 - np.clip(oof, EPS, 1 - EPS)))
                self.cal_ = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
        self.base_ = self.make().fit(rows)
        self.make = None                    # fitted: keep only fitted parts, so the model can be saved
        return self

    def predict(self, rows):
        p = self.base_.predict(rows)
        if self.skipped_:
            return p
        if self.method == "isotonic":
            return self.cal_.predict(p)
        z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
        return self.cal_.predict_proba(z.reshape(-1, 1))[:, 1]


class PerCutoff:
    """One model per prediction cutoff."""

    def __init__(self, make: Callable[[], Model]):
        self.make = make

    def fit(self, rows):
        self.models_ = {L: self.make().fit(g) for L, g in rows.groupby("cutoff_min")}
        self.make = None                    # fitted: keep only fitted parts, so the model can be saved
        return self

    def predict(self, rows):
        p = np.full(len(rows), np.nan)
        for L, idx in rows.groupby("cutoff_min").indices.items():
            if L not in self.models_:
                raise ValueError(f"no model for cutoff {L}")
            p[idx] = self.models_[L].predict(rows.iloc[idx])
        return p


# ---------------------------------------------------------------- factory

def inputs(name: str, model_features: list[str]) -> list[str]:
    """Input columns of each model (before an optional cutoff column)."""
    return {"B0": ["planned_slack_min"], "B3-lin": GROUPS["db"], "B3": GROUPS["db"],
            "lr": model_features, "gbm": model_features}.get(name, [])


def make_model(name: str, cfg: TrainingConfig, model_features: list[str]) -> Model:
    """The full model as evaluated: base learner, calibration and cutoff handling."""
    if name == "B2":
        return DBRule()
    cols = inputs(name, model_features) + (["cutoff_min"] if cfg.cutoff_mode == "feature" else [])

    def base() -> Model:
        if name == "B1":
            return HistoryRate(cfg.slack_buckets, cfg.smoothing)
        if name in ("B0", "B3-lin", "lr"):
            return LogReg(cols, **cfg.lr)
        return GBM(cols, seed=cfg.seed, **cfg.gbm)

    def calibrated() -> Model:
        if cfg.calibration == "none" or name == "B1":      # B1 is a rate, calibrated by construction
            return base()
        return Calibrated(base, cfg.calibration, cfg.calibration_folds)

    if cfg.cutoff_mode == "separate" or name == "B1":
        return PerCutoff(calibrated)
    return calibrated()
