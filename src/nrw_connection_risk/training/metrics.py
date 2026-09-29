"""Metrics and the day-block bootstrap (DESIGN.md section 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve

EPS = 1e-6


def pointwise_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def is_binary(p: np.ndarray) -> bool:
    return bool(np.isin(p, (0.0, 1.0)).all())


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) formula, ties averaged; equal to
    sklearn's roc_auc_score but fast enough for the bootstrap."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_recall_at(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and false-alarm rate (share of holding connections flagged)."""
    tp = (flagged & (y == 1)).sum()
    prec = tp / flagged.sum() if flagged.sum() else np.nan
    rec = tp / (y == 1).sum() if (y == 1).sum() else np.nan
    far = (flagged & (y == 0)).sum() / (y == 0).sum() if (y == 0).sum() else np.nan
    return float(prec), float(rec), float(far)


def recall_at_precision(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float, float]:
    """Highest recall at a threshold whose precision is at least `target`, and the
    threshold. NaN if no threshold reaches the target."""
    if np.isnan(target) or y.sum() == 0:
        return np.nan, np.nan
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= target
    if not ok.any():
        return np.nan, np.nan
    i = np.argmax(np.where(ok, rec[:-1], -1))
    return float(rec[i]), float(thr[i])


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Point metrics for one model on one set of rows. Log loss is undefined (NaN) for a
    binary rule such as B2."""
    binary = is_binary(p)
    return {"n": int(len(y)), "fail_rate": float(y.mean()) if len(y) else np.nan,
            "mean_p": float(p.mean()) if len(y) else np.nan,
            "log_loss": np.nan if binary or not len(y) else float(pointwise_log_loss(y, p).mean()),
            "brier": float(((p - y) ** 2).mean()) if len(y) else np.nan,
            "auc": np.nan if binary else auc(y, p)}


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    """Mean predicted vs observed failure rate in equal-width probability bins."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"),
                                                                     observed=("y", "mean"))
    df.index = [f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in df.index]
    return df


# ---------------------------------------------------------------- day-block bootstrap

def _draws(n_days: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_days, size=(n, n_days))


def bootstrap_vs(days: np.ndarray, y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray,
                 n: int, n_auc: int, seed: int) -> dict:
    """Model vs baseline with 95% intervals from resampling whole service days (rows on
    one day share disruptions and are not independent).

    log_loss_gain / brier_gain: relative reduction vs the baseline (positive = better).
    auc_diff: model AUC minus baseline AUC."""
    uniq, d = np.unique(days, return_inverse=True)
    k = len(uniq)
    cnt = np.bincount(d, minlength=k).astype(float)
    ll_m = np.bincount(d, pointwise_log_loss(y, p_model), minlength=k)
    ll_b = np.bincount(d, pointwise_log_loss(y, p_base), minlength=k)
    br_m = np.bincount(d, (p_model - y) ** 2, minlength=k)
    br_b = np.bincount(d, (p_base - y) ** 2, minlength=k)
    gain = lambda m, b: 1 - m / b   # noqa: E731
    out = {"days": k,
           "log_loss_gain": gain(ll_m.sum(), ll_b.sum()), "brier_gain": gain(br_m.sum(), br_b.sum()),
           "auc_diff": auc(y, p_model) - auc(y, p_base)}
    draws = _draws(k, n, seed)
    w = np.stack([np.bincount(r, minlength=k) for r in draws]).astype(float)      # times each day is drawn
    ll = gain(w @ ll_m / (w @ cnt), w @ ll_b / (w @ cnt))
    br = gain(w @ br_m / (w @ cnt), w @ br_b / (w @ cnt))
    out["log_loss_gain_ci"] = tuple(np.nanquantile(ll, [0.025, 0.975]).tolist())
    out["brier_gain_ci"] = tuple(np.nanquantile(br, [0.025, 0.975]).tolist())
    rows_of = [np.flatnonzero(d == i) for i in range(k)]
    diffs = []
    for r in draws[:n_auc]:
        idx = np.concatenate([rows_of[i] for i in r])
        diffs.append(auc(y[idx], p_model[idx]) - auc(y[idx], p_base[idx]))
    out["auc_diff_ci"] = tuple(np.nanquantile(diffs, [0.025, 0.975]).tolist())
    return out
