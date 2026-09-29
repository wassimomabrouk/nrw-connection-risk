"""Time-based splits. Whole service days only; test and robustness days are locked."""
from __future__ import annotations

from datetime import date

from .config import Splits


class LockedPeriodError(RuntimeError):
    """Code tried to read a locked day (test or robustness period) without unlocking it."""


def period_of(day: date, s: Splits) -> str:
    if s.train[0] <= day <= s.train[1]:
        return "train"
    if s.validation[0] <= day <= s.validation[1]:
        return "validation"
    if s.test[0] <= day <= s.test[1]:
        return "test"
    if day >= s.robustness_from:
        return "robustness"
    return "outside"


def guard(days: list[date], s: Splits, unlocked: frozenset[str] = frozenset()) -> None:
    """Raise if any day belongs to a locked period that was not explicitly unlocked."""
    for d in days:
        p = period_of(d, s)
        if p in ("test", "robustness") and p not in unlocked:
            raise LockedPeriodError(f"{d} is in the locked {p} period")


def days_in(period: str, available: list[date], s: Splits) -> list[date]:
    return sorted(d for d in available if period_of(d, s) == period)


def rolling_origin(days: list[date], min_train_days: int, fold_days: int = 1) -> list[tuple[list[date], list[date]]]:
    """Expanding window: the first fold fits on the first `min_train_days` days and
    evaluates on the next `fold_days`; each later fold adds the previous block to the fit."""
    days = sorted(days)
    return [(days[:k], days[k:k + fold_days]) for k in range(min_train_days, len(days), fold_days)]
