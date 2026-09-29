"""Prediction log for monitoring: each connection is logged once per cutoff (60, 30, 10
minutes before arrival), at the first scoring run at or after the cutoff moment. That
mirrors the training table, so live predictions can later be joined to the observed
outcome and evaluated exactly like the offline evaluation (phase 7).

Layout: <dir>/date=YYYY-MM-DD/part-HHMMSS-<id>.parquet, flushed every few minutes.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.columns import ALL_FEATURES

log = logging.getLogger(__name__)

COLUMNS = ["scored_at", "data_as_of", "model_id", "cutoff_min", "minutes_to_arrival", "lag_s", "hub", "eva",
           "stop_id_a", "stop_id_b", "train_a", "train_b", "pt_a", "pt_b", "a_ct_cut", "b_ct_cut",
           "db_rule_fail", "p_model", "p_B3"] + [c for c in ALL_FEATURES if c not in ("hub",)]


class PredictionLog:
    def __init__(self, root: Path, cutoffs: list[int], max_late_min: float):
        self.root, self.cutoffs, self.max_late = root, sorted(cutoffs), max_late_min
        self.logged: dict[tuple[str, str, int], pd.Timestamp] = {}     # key -> planned arrival
        self.buffer: list[pd.DataFrame] = []

    def restore(self, now: pd.Timestamp) -> int:
        """Remember what today's and yesterday's log files already hold, so a restart does
        not log the same connection and cutoff twice."""
        for day in (now - pd.Timedelta(days=1), now):
            folder = self.root / f"date={day.date()}"
            if not folder.is_dir():
                continue
            for f in folder.glob("*.parquet"):
                try:
                    d = pd.read_parquet(f, columns=["stop_id_a", "stop_id_b", "cutoff_min", "pt_a"])
                except Exception:
                    continue
                pt = d.pt_a.dt.tz_convert(None) if d.pt_a.dt.tz is not None else d.pt_a
                for a, b, L, t in zip(d.stop_id_a, d.stop_id_b, d.cutoff_min, pt):
                    if t > now - pd.Timedelta(hours=2):
                        self.logged[(a, b, int(L))] = t
        return len(self.logged)

    def select(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Rows that just reached one of the cutoffs and were not logged for it yet."""
        ok = scores[scores.status.eq("ok")] if len(scores) else scores
        parts = []
        for L in self.cutoffs:
            m = ok.minutes_to_arrival
            hit = ok[(m <= L) & (m > L - self.max_late)]
            keys = list(zip(hit.stop_id_a, hit.stop_id_b, [L] * len(hit)))
            new = np.array([k not in self.logged for k in keys], dtype=bool)
            hit = hit.loc[new].assign(cutoff_min=L)
            for k, pt in zip((k for k, n in zip(keys, new) if n), hit.pt_a):
                self.logged[k] = pt
            parts.append(hit)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(out):
            out["lag_s"] = (out.cutoff_min - out.minutes_to_arrival) * 60
        return out.reindex(columns=COLUMNS) if len(out) else out

    def add(self, scores: pd.DataFrame, now: pd.Timestamp) -> int:
        rows = self.select(scores)
        if len(rows):
            self.buffer.append(rows)
        # forget connections whose arrival is long past
        self.logged = {k: pt for k, pt in self.logged.items() if pt > now - pd.Timedelta(hours=2)}
        return len(rows)

    def pending(self) -> int:
        return sum(len(b) for b in self.buffer)

    def flush(self) -> int:
        if not self.buffer:
            return 0
        df = pd.concat(self.buffer, ignore_index=True)
        for c in ("scored_at", "data_as_of", "pt_a", "pt_b", "a_ct_cut", "b_ct_cut"):
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC")
        written = 0
        for day, g in df.groupby(df.scored_at.dt.date):
            folder = self.root / f"date={day}"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%H%M%S")
            g.to_parquet(folder / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet", index=False)
            written += len(g)
        self.buffer = []
        log.info("logged %d predictions", written)
        return written
