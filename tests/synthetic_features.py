"""Synthetic feature tables with a known truth, for training-pipeline tests.

Failure depends on DB's predicted slack AND on the hub state (information DB's
prognosis lacks), so a model with the hub group must beat B3 on DB inputs only."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nrw_connection_risk.features.columns import ALL_FEATURES

HUBS = ["Köln Hbf", "Düsseldorf Hbf", "Essen Hbf"]
SEGS = ["long-distance", "regional", "s-bahn"]


def make_day(day: date, n: int, rng: np.random.Generator, hub_effect: float = 1.5) -> pd.DataFrame:
    cut = rng.choice([60, 30, 10], n)
    planned = rng.uniform(4, 30, n)
    day_shock = rng.normal(0, 1.5)                       # shared disruption level of the day
    hub_delay = np.clip(rng.gamma(2, 2, n) + day_shock, 0, None)
    delay_a = rng.exponential(3, n) + 0.5 * hub_delay
    delay_b = rng.exponential(1, n)
    db_slack = planned + delay_b - delay_a
    b_cancel = rng.random(n) < 0.02
    # the truth: DB's slack, sharpened closer to the event, plus the hub state DB does not use
    sharp = np.where(cut == 10, 0.6, np.where(cut == 30, 0.4, 0.25))
    logit = -sharp * (db_slack - 4) + hub_effect * (hub_delay - 4) / 4 - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))) | b_cancel
    df = pd.DataFrame({
        "service_day": day.isoformat(), "cutoff_min": cut,
        "t_cut": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rng.integers(300, 1400, n), unit="m"),
        "eva": "8000207", "stop_id_a": [f"a{i}" for i in range(n)], "stop_id_b": [f"b{i}" for i in range(n)],
        "label_fail": y, "fail_reason": np.where(b_cancel, "b_cancelled", np.where(y, "delay", "none")),
        "db_slack_min": db_slack, "db_delay_a_min": delay_a, "db_delay_b_min": delay_b,
        "b_cancel_known": b_cancel.astype(int),
        "hub_mean_delay": hub_delay, "hub_share_late5": np.clip(hub_delay / 10, 0, 1),
        "hub_share_cancel": rng.uniform(0, 0.05, n), "line_recent_delay_a": np.where(rng.random(n) < 0.3, np.nan,
                                                                                      hub_delay + rng.normal(0, 1, n)),
        "age_a_min": rng.uniform(0, 5, n), "age_b_min": np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0, 30, n)),
        "planned_slack_min": planned, "hour_sin": rng.uniform(-1, 1, n), "hour_cos": rng.uniform(-1, 1, n),
        "day_type": "saturday" if day.weekday() == 5 else ("sunday_holiday" if day.weekday() == 6 else "weekday"),
        "n_stations_before_a": rng.integers(0, 20, n), "same_platform": rng.integers(0, 2, n),
        "segment_a": rng.choice(SEGS, n), "segment_b": rng.choice(SEGS, n), "hub": rng.choice(HUBS, n),
        "trend_a_15": rng.normal(0, 1, n), "trend_a_30": rng.normal(0, 1, n), "trend_b_15": rng.normal(0, 1, n),
        "n_delay_codes_a": rng.integers(0, 3, n), "n_quality_a": 0, "n_delay_codes_b": 0,
        "h_notice_a": 0, "h_notice_b": 0, "c_notice_a": 0,
        "corridor_delay_a": np.where(rng.random(n) < 0.5, np.nan, rng.gamma(2, 2, n)),     # noise: not in the truth
        "corridor_line_delay_a": np.where(rng.random(n) < 0.6, np.nan, rng.gamma(2, 2, n)),
        "corridor_delay_b": np.where(rng.random(n) < 0.6, np.nan, rng.gamma(2, 2, n)),
    })
    assert set(ALL_FEATURES) <= set(df.columns)
    return df


def write_days(root: Path, first: date, n_days: int, rows_per_day: int = 1500, seed: int = 0,
               **kw) -> list[date]:
    rng = np.random.default_rng(seed)
    days = [first + timedelta(days=i) for i in range(n_days)]
    for d in days:
        out = root / f"service_day={d}"
        out.mkdir(parents=True, exist_ok=True)
        make_day(d, rows_per_day, rng, **kw).to_parquet(out / "part-0.parquet", index=False)
        (out / "_meta.json").write_text(json.dumps({"service_day": str(d), "git_commit": "synthetic"}))
    return days
