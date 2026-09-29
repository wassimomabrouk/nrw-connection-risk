"""Score every upcoming transfer candidate at the hubs, now.

The rows are built exactly like a dataset row at cutoff t_cut = now (same candidate
rules, same point-in-time state, same feature functions), so a live prediction equals
what the training pipeline would compute for that moment. tests/test_serving.py checks
this against the dataset and feature builders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset.candidates import build_candidates
from ..dataset.config import DatasetConfig
from ..dataset.state import last_poll_at, state_at
from ..dataset.timeutil import is_ambiguous_local
from ..features.columns import ALL_FEATURES
from ..features.compute import compute_all
from ..features.config import FeatureConfig
from ..training.bundle import COMPANION, Bundle
from .live import LiveSnapshot

def _minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def _train(cat, line, num) -> str:
    cat = cat if isinstance(cat, str) else ""
    if isinstance(line, str) and line:
        return line if line.startswith(cat) else f"{cat} {line}".strip()
    return f"{cat} {num}".strip() if isinstance(num, str) else cat


def candidate_rows(snap: LiveSnapshot, dcfg: DatasetConfig, min_ahead: float, max_ahead: float,
                   max_gap_min: float, cutoffs: list[int]) -> pd.DataFrame:
    """Transfer candidates whose arrival is min_ahead to max_ahead minutes away, with the
    realtime state as known now. `cutoff_min` is the trained cutoff nearest to the time
    left until arrival; `status` says whether the row can be scored."""
    now = snap.now
    t0, t1 = now + pd.Timedelta(minutes=min_ahead), now + pd.Timedelta(minutes=max_ahead)
    cands, _ = build_candidates(snap.plan, t0, t1, dcfg.hubs, dcfg.min_transfer_min, dcfg.max_slack_min)
    if cands.empty:
        return cands
    t_cut = pd.Series(now, index=cands.index)
    sa = state_at(cands.stop_id_a + "|ar", t_cut, snap.obs)
    sb = state_at(cands.stop_id_b + "|dp", t_cut, snap.obs)
    d = cands.copy()
    d["t_cut"] = t_cut
    d["a_ct_cut"], d["a_cs_cut"], d["a_obs_cut"] = sa.ct, sa.cs, sa.obs
    d["b_ct_cut"], d["b_cs_cut"], d["b_obs_cut"] = sb.ct, sb.cs, sb.obs
    a_pred, b_pred = sa.ct.fillna(cands.pt_a), sb.ct.fillna(cands.pt_b)
    d["db_delay_a_min"] = _minutes(a_pred - cands.pt_a)
    d["db_delay_b_min"] = _minutes(b_pred - cands.pt_b)
    d["db_slack_min"] = _minutes(b_pred - a_pred)
    d["b_cancel_known"] = sb.cs.eq("c")
    d["minutes_to_arrival"] = _minutes(cands.pt_a - now)
    cut = np.array(sorted(cutoffs))
    d["cutoff_min"] = cut[np.abs(d.minutes_to_arrival.to_numpy()[:, None] - cut[None, :]).argmin(axis=1)]
    poll = last_poll_at(cands.eva, t_cut, snap.polls)
    gap = poll.isna() | (_minutes(now - poll) > max_gap_min)
    ambiguous = pd.Series([is_ambiguous_local(a) or is_ambiguous_local(b)
                           for a, b in zip(cands.pt_raw_a, cands.pt_raw_b)], index=cands.index)
    # same exclusions as the dataset (DESIGN.md section 2) where they can be known now
    d["status"] = np.select([sa.cs.eq("c").to_numpy(), gap.to_numpy(), ambiguous.to_numpy()],
                            ["a_cancelled", "collector_gap", "dst_ambiguous"], "ok")
    d["db_rule_fail"] = (d.db_slack_min < dcfg.min_transfer_min) | d.b_cancel_known
    d["train_a"] = [_train(c, l, n) for c, l, n in zip(d.cat_a, d.line_a, d.num_a)]
    d["train_b"] = [_train(c, l, n) for c, l, n in zip(d.cat_b, d.line_b, d.num_b)]
    return d.reset_index(drop=True)


def score(snap: LiveSnapshot, bundle: Bundle, dcfg: DatasetConfig, min_ahead: float, max_ahead: float,
          max_gap_min: float) -> pd.DataFrame:
    """Candidates with P(fail) from the bundle's model and from B3 (NaN where status != ok)."""
    fcfg = FeatureConfig.from_dict(bundle.meta["feature_config"])
    rows = candidate_rows(snap, dcfg, min_ahead, max_ahead, max_gap_min, bundle.cutoffs)
    if rows.empty:
        return rows
    feats = compute_all(rows, snap.plan, snap.obs, snap.event_msgs, snap.stop_msgs, fcfg)
    out = pd.concat([rows.drop(columns=[c for c in ALL_FEATURES if c in rows.columns]), feats], axis=1)
    out["p_model"] = np.nan
    out[f"p_{COMPANION}"] = np.nan
    ok = out.status.eq("ok").to_numpy()
    if ok.any():
        preds = bundle.predict(out[ok].reset_index(drop=True))
        out.loc[ok, "p_model"] = preds["model"]
        out.loc[ok, f"p_{COMPANION}"] = preds[COMPANION]
    out["scored_at"] = snap.now
    out["data_as_of"] = snap.data_as_of
    out["model_id"] = bundle.model_id
    return out
