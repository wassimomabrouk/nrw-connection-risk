"""Training/serving skew on production traffic.

For every logged prediction, the features are recomputed offline from the collected
data exactly as it stood at the moment of scoring (`scored_at`), with the training
pipeline's own code and data window, and compared with the features the API logged.
They should be identical; the share of identical rows is monitored daily. Known,
expected sources of small differences: the live service reads the last 6 hours while
the offline builder reads from the day before (an event not updated for 6 hours), and
a response that arrived within the same second as the scoring run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.config import DatasetConfig
from ..dataset.load import load_window
from ..dataset.state import state_at
from ..dataset.timeutil import service_day_bounds
from ..features.columns import ALL_FEATURES, CATEGORICAL
from ..features.compute import compute_all
from ..features.config import FeatureConfig
from ..features.load import load_messages

PLANNED = ["eva", "hub", "stop_id_a", "stop_id_b", "pt_a", "pt_b", "line_a", "pp_a", "pp_b",
           "n_stations_before_a", "planned_slack_min", "segment_a", "segment_b"]
TOL = 1e-6


def _minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def recompute(logged: pd.DataFrame, dataset: pd.DataFrame, day, parsed: Path, dcfg: DatasetConfig,
              fcfg: FeatureConfig) -> pd.DataFrame:
    """Offline features at each logged row's scoring time (rows the dataset knows)."""
    planned = dataset.drop_duplicates(["stop_id_a", "stop_id_b"])[PLANNED]
    rows = logged[["stop_id_a", "stop_id_b", "cutoff_min", "scored_at"]].merge(
        planned, on=["stop_id_a", "stop_id_b"], how="inner").reset_index(drop=True)
    if rows.empty:
        return rows
    rows["t_cut"] = rows.scored_at
    t0, _ = service_day_bounds(day, dcfg.start_hour_local)
    t_from, t_to = t0 - pd.Timedelta(days=1), rows.t_cut.max()
    hubs = list(dcfg.hubs)
    w = load_window(parsed, t_from, t_to, dcfg.min_parser_version, stations=fcfg.stations(hubs))
    ev, st = load_messages(parsed, t_from, t_to, hubs)
    sa = state_at(rows.stop_id_a + "|ar", rows.t_cut, w.obs)
    sb = state_at(rows.stop_id_b + "|dp", rows.t_cut, w.obs)
    a_pred, b_pred = sa.ct.fillna(rows.pt_a), sb.ct.fillna(rows.pt_b)
    rows["a_obs_cut"], rows["b_obs_cut"] = sa.obs, sb.obs
    rows["db_delay_a_min"] = _minutes(a_pred - rows.pt_a)
    rows["db_delay_b_min"] = _minutes(b_pred - rows.pt_b)
    rows["db_slack_min"] = _minutes(b_pred - a_pred)
    rows["b_cancel_known"] = sb.cs.eq("c")
    feats = compute_all(rows, w.plan, w.obs, ev, st, fcfg)
    return pd.concat([rows[["stop_id_a", "stop_id_b", "cutoff_min"]], feats], axis=1)


def compare(live: pd.DataFrame, offline: pd.DataFrame, cols: list[str]) -> dict:
    """Per feature: share of rows with the same value (NaN equals NaN) and the mean
    absolute difference where they differ; overall share of fully identical rows."""
    key = ["stop_id_a", "stop_id_b", "cutoff_min"]
    m = live[key + cols].merge(offline[key + cols], on=key, suffixes=("_live", "_offline"))
    if m.empty:
        return {"rows": 0}
    same_all = np.ones(len(m), dtype=bool)
    per = {}
    for c in cols:
        a, b = m[f"{c}_live"], m[f"{c}_offline"]
        if c in CATEGORICAL:
            same = (a.astype("string").fillna("<na>") == b.astype("string").fillna("<na>")).to_numpy()
            diff = None
        else:
            x, y = pd.to_numeric(a, errors="coerce").to_numpy(float), pd.to_numeric(b, errors="coerce").to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            same = both_nan | (np.abs(x - y) <= TOL)
            d = np.abs(x - y)[~same & ~np.isnan(x) & ~np.isnan(y)]
            diff = round(float(d.mean()), 4) if len(d) else None
        same_all &= same
        per[c] = {"equal_share": round(float(same.mean()), 5), "mean_abs_diff_where_different": diff}
    worst = sorted(per.items(), key=lambda kv: kv[1]["equal_share"])[:5]
    return {"rows": int(len(m)), "identical_share": round(float(same_all.mean()), 5),
            "per_feature": per, "least_equal": [k for k, v in worst if v["equal_share"] < 1]}


def skew_check(j: pd.DataFrame, dataset: pd.DataFrame, day, parsed: Path, dcfg: DatasetConfig, models: Path) -> dict:
    res = {}
    for model_id, g in j.groupby("model_id"):
        card = models / str(model_id) / "model.json"
        if not card.exists():
            res[model_id] = {"available": False, "reason": "model card not found"}
            continue
        fcfg = FeatureConfig.from_dict(json.loads(card.read_text(encoding="utf-8"))["feature_config"])
        offline = recompute(g, dataset, day, parsed, dcfg, fcfg)
        cols = [c for c in ALL_FEATURES if c in g.columns and c in offline.columns]
        res[model_id] = {"available": True, **compare(g, offline, cols)}
    return res
