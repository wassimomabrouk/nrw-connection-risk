"""Daily evaluation of the live service for one service day:

1. read the predictions the API logged for connections arriving that day,
2. label them with the dataset builder (the same labels and exclusions as training),
3. score the model, B3 and the DB rule on the live predictions,
4. compare every input's live distribution with the training data (PSI),
5. operational figures: coverage, logging lag, data age.

Output: data/monitoring/daily/service_day=YYYY-MM-DD.json and joined rows in
data/monitoring/joined/service_day=YYYY-MM-DD.parquet; then the summary over all days.

Run from the repo root (on the server, daily by nrw-monitor.timer):
    python -m nrw_connection_risk.monitoring.daily              # every finished day not yet evaluated
    python -m nrw_connection_risk.monitoring.daily --day 2026-10-01
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.build import IncompleteDay, build_day as build_dataset_day
from ..dataset.config import DatasetConfig, load_config
from ..dataset.load import ParsedLayerError
from ..dataset.timeutil import service_day_bounds, to_naive_utc
from ..training.metrics import precision_recall_at, recall_at_precision, reliability, scores
from .profile import drift, level
from .skew import skew_check

ROOT = Path(__file__).resolve().parents[3]
KEY = ["stop_id_a", "stop_id_b", "cutoff_min"]
TIME_COLS = ("scored_at", "data_as_of", "pt_a", "pt_b", "a_ct_cut", "b_ct_cut")


@dataclass(frozen=True)
class MonitoringConfig:
    parsed: Path
    predictions: Path
    models: Path
    out: Path
    primary_cutoff: int = 30
    calibration_bins: int = 10
    bootstrap_n: int = 1000
    bootstrap_n_auc: int = 200
    seed: int = 42
    psi_moderate: float = 0.1
    psi_large: float = 0.25
    min_drift_rows: int = 500


def load_monitoring_config(path: Path, root: Path) -> MonitoringConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    p = lambda v: (root / v) if not Path(v).is_absolute() else Path(v)   # noqa: E731
    d, e, dr = c["data"], c["evaluation"], c["drift"]
    return MonitoringConfig(parsed=p(d["parsed"]), predictions=p(d["predictions"]), models=p(d["models"]),
                            out=p(d["out"]), primary_cutoff=int(e["primary_cutoff"]),
                            calibration_bins=int(e["calibration_bins"]), bootstrap_n=int(e["bootstrap_n"]),
                            bootstrap_n_auc=int(e["bootstrap_n_auc"]), seed=int(e["seed"]),
                            psi_moderate=float(dr["moderate"]), psi_large=float(dr["large"]),
                            min_drift_rows=int(dr["min_rows"]))


# ---------------------------------------------------------------- inputs

def load_predictions(root: Path, day: date, start_hour: int = 4) -> pd.DataFrame:
    """Logged predictions for connections whose arrival A falls into the service day,
    one row per connection and cutoff (the first logged, should a restart repeat one)."""
    t0, t1 = service_day_bounds(day, start_hour)
    parts = []
    for d in pd.date_range(t0.normalize() - pd.Timedelta(days=1), t1.normalize()).date:
        folder = root / f"date={d}"
        parts += [pd.read_parquet(f) for f in sorted(folder.glob("*.parquet"))] if folder.is_dir() else []
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    for c in TIME_COLS:
        if c in df:
            df[c] = to_naive_utc(df[c])
    df = df[(df.pt_a >= t0) & (df.pt_a < t1)]
    df = df.sort_values("scored_at", kind="stable").drop_duplicates(KEY, keep="first")
    df["cutoff_min"] = df.cutoff_min.astype(int)
    return df.reset_index(drop=True)


def outcomes(day: date, parsed: Path, dcfg: DatasetConfig, dataset: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels and exclusions exactly as in the training table (dataset builder)."""
    df = dataset if dataset is not None else build_dataset_day(day, parsed, dcfg)[0]
    df = df[KEY + ["label_fail", "fail_reason", "eligible", "exclusion_reason"]].copy()
    df["cutoff_min"] = df.cutoff_min.astype(int)
    return df


def join(pred: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Each logged prediction with its outcome. `outcome` says why a row is not evaluated."""
    j = pred.merge(out, on=KEY, how="left", indicator=True)
    j["outcome"] = np.select([j._merge.eq("left_only").to_numpy(), ~j.eligible.fillna(False).astype(bool).to_numpy()],
                             ["not_a_candidate", "excluded"], "evaluated")
    return j.drop(columns="_merge")


# ---------------------------------------------------------------- metrics

def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def performance(ev: pd.DataFrame, bins: int, primary: int) -> dict:
    """Model vs B3 vs DB rule per cutoff on evaluated rows."""
    out = {}
    for L, g in ev.groupby("cutoff_min"):
        y = g.label_fail.astype(int).to_numpy()
        pm, pb = g.p_model.to_numpy(float), g.p_B3.to_numpy(float)
        sm, sb = scores(y, pm), scores(y, pb)
        prec, rec, far = precision_recall_at(y, g.db_rule_fail.astype(bool).to_numpy())
        entry = {"rows": int(len(g)), "fail_rate": _r(y.mean()),
                 "model": {k: _r(v) for k, v in sm.items() if k != "n"},
                 "B3": {k: _r(v) for k, v in sb.items() if k != "n"},
                 "log_loss_gain_vs_B3": _r(1 - sm["log_loss"] / sb["log_loss"]) if sb["log_loss"] else None,
                 "db_rule": {"precision": _r(prec), "recall": _r(rec), "false_alarm_rate": _r(far)},
                 "model_recall_at_db_precision": _r(recall_at_precision(y, pm, prec)[0]),
                 "B3_recall_at_db_precision": _r(recall_at_precision(y, pb, prec)[0])}
        if int(L) == primary:
            entry["calibration"] = {name: reliability(y, p, bins).reset_index(names="bin").round(4).to_dict("records")
                                    for name, p in (("model", pm), ("B3", pb))}
        out[str(int(L))] = entry
    return out


def operations(j: pd.DataFrame, out: pd.DataFrame) -> dict:
    """Coverage of the eligible connections, logging lag and data age at scoring time."""
    elig = out[out.eligible].groupby("cutoff_min").size()
    got = j[j.outcome.eq("evaluated")].groupby("cutoff_min").size()
    age = (j.scored_at - j.data_as_of).dt.total_seconds()
    q = lambda s, p: _r(np.nanquantile(s, p), 1) if len(s) else None   # noqa: E731
    return {"logged": int(len(j)), "by_outcome": {k: int(v) for k, v in j.outcome.value_counts().items()},
            "excluded_reasons": {k: int(v) for k, v in
                                 j.loc[j.outcome.eq("excluded"), "exclusion_reason"].value_counts().items()},
            "coverage": {str(int(L)): _r(got.get(L, 0) / n, 4) for L, n in elig.items()},
            "lag_s": {"p50": q(j.lag_s, 0.5), "p95": q(j.lag_s, 0.95), "max": q(j.lag_s, 1.0)},
            "data_age_s": {"p50": q(age, 0.5), "p95": q(age, 0.95), "max": q(age, 1.0)}}


def feature_drift(j: pd.DataFrame, models: Path, cfg: MonitoringConfig) -> dict:
    """PSI per input against the training profile stored in each model's card."""
    res = {}
    for model_id, g in j.groupby("model_id"):
        card = models / str(model_id) / "model.json"
        meta = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
        prof = meta.get("reference_profile")
        if not prof:
            res[model_id] = {"available": False, "reason": "model card has no reference profile"}
            continue
        if len(g) < cfg.min_drift_rows:              # PSI on a handful of rows is noise
            res[model_id] = {"available": False, "reason": f"only {len(g)} logged rows (< {cfg.min_drift_rows})"}
            continue
        values = drift(prof, g)
        ref_fail = meta.get("reference_fail_rate", {})
        ev = g[g.outcome.eq("evaluated")]
        res[model_id] = {
            "available": True, "rows": int(len(g)),
            "psi": {k: _r(v) for k, v in sorted(values.items(), key=lambda kv: -kv[1])},
            "level": {k: level(v, cfg.psi_moderate, cfg.psi_large) for k, v in values.items()},
            "fail_rate": {str(L): {"live": _r(ev[ev.cutoff_min == int(L)].label_fail.mean()),
                                   "training": _r(r)} for L, r in ref_fail.items()},
        }
    return res


# ---------------------------------------------------------------- one day

def evaluate_day(day: date, cfg: MonitoringConfig, dcfg: DatasetConfig) -> dict:
    pred = load_predictions(cfg.predictions, day, dcfg.start_hour_local)
    report = {"service_day": day.isoformat(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if pred.empty:
        return {**report, "status": "no_predictions"}
    try:
        dataset = build_dataset_day(day, cfg.parsed, dcfg)[0]
    except (IncompleteDay, ParsedLayerError) as e:
        return {**report, "status": "no_outcomes", "reason": f"{type(e).__name__}: {e}"}
    out = outcomes(day, cfg.parsed, dcfg, dataset)
    j = join(pred, out)
    (cfg.out / "joined").mkdir(parents=True, exist_ok=True)
    j.to_parquet(cfg.out / "joined" / f"service_day={day}.parquet", index=False)
    ev = j[j.outcome.eq("evaluated")]
    try:
        skew = skew_check(j, dataset, day, cfg.parsed, dcfg, cfg.models)
    except Exception as e:                    # the skew check must never block the daily report
        skew = {"error": f"{type(e).__name__}: {e}"}
    return {**report, "status": "ok", "model_ids": sorted(j.model_id.astype(str).unique()),
            "performance": performance(ev, cfg.calibration_bins, cfg.primary_cutoff),
            "operations": operations(j, out), "drift": feature_drift(j, cfg.models, cfg), "skew": skew}


def finished_days(cfg: MonitoringConfig, dcfg: DatasetConfig, now: datetime) -> list[date]:
    """Service days with logged predictions whose label horizon has passed."""
    dates = sorted(date.fromisoformat(p.name.split("=", 1)[1]) for p in cfg.predictions.glob("date=*"))
    if not dates:
        return []
    days, d = [], dates[0] - timedelta(days=1)
    while d <= dates[-1]:
        t1 = service_day_bounds(d, dcfg.start_hour_local)[1]
        if t1 + pd.Timedelta(hours=dcfg.label_horizon_h) <= pd.Timestamp(now).tz_convert(None):
            days.append(d)
        d += timedelta(days=1)
    return days


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the live service for finished service days")
    ap.add_argument("--day", type=date.fromisoformat, help="one day; default: every finished day not yet evaluated")
    ap.add_argument("--redo", action="store_true", help="also re-evaluate days that already have a report")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "monitoring.toml")
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "config" / "dataset.toml")
    args = ap.parse_args(argv)

    cfg = load_monitoring_config(args.config, ROOT)
    dcfg = load_config(args.dataset_config)
    daily = cfg.out / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    days = [args.day] if args.day else finished_days(cfg, dcfg, datetime.now(timezone.utc))
    todo = [d for d in days if args.redo or args.day or not (daily / f"service_day={d}.json").exists()]
    written = 0
    for d in todo:
        rep = evaluate_day(d, cfg, dcfg)
        if rep["status"] == "no_predictions":        # e.g. days before the API ran: no report
            continue
        written += 1
        (daily / f"service_day={d}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        line = f"{d}: {rep['status']}"
        if rep["status"] == "ok":
            p = rep["performance"].get(str(cfg.primary_cutoff), {})
            line += (f", {rep['operations']['logged']:,} logged, at {cfg.primary_cutoff} min: "
                     f"{p.get('rows', 0):,} rows, log loss model {p.get('model', {}).get('log_loss')} "
                     f"vs B3 {p.get('B3', {}).get('log_loss')}")
        else:
            line += f" ({rep.get('reason')})"
        print(line)
    from .summary import write_summary
    s = write_summary(cfg)
    print(f"{written} new report(s); summary over {s['days']} evaluated day(s): {cfg.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
