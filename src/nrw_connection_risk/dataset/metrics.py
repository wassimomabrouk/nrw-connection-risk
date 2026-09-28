"""Small metric helpers without extra dependencies."""
from __future__ import annotations

import pandas as pd


def roc_auc(score: pd.Series, y: pd.Series) -> float:
    """ROC AUC by the rank-sum formula; a higher score must mean higher risk."""
    y = y.astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = score.rank(method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
