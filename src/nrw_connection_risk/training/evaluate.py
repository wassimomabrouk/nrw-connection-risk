"""Fit and evaluate every baseline and model on time-based splits.

Modes (DESIGN.md section 6):
    cv        rolling origin over the training period: fold k fits on all earlier days
    validate  fit on the training period, evaluate on the validation period
    test      fit on training + validation, evaluate on the locked test period (once;
              needs --final and is logged in reports/test_log.jsonl)

Run from the repo root:
    python -m nrw_connection_risk.training.evaluate --mode cv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.build import _git_commit
from ..features.config import load_feature_config
from .config import MODELS, Splits, TrainingConfig, load_training_config
from .metrics import bootstrap_vs, precision_recall_at, recall_at_precision, reliability, scores
from .models import make_model
from .report import write_report
from .splits import days_in, guard, rolling_origin

ROOT = Path(__file__).resolve().parents[3]
KEY_COLS = ["service_day", "cutoff_min", "eva", "hub", "stop_id_a", "stop_id_b", "segment_a", "segment_b",
            "label_fail", "fail_reason"]


# ---------------------------------------------------------------- data

def available_days(features_root: Path) -> list[date]:
    return sorted(date.fromisoformat(p.name.split("=", 1)[1])
                  for p in features_root.glob("service_day=*") if (p / "part-0.parquet").exists())


def load_rows(features_root: Path, days: list[date], splits: Splits,
              unlocked: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Feature rows of the given days. Refuses locked days (test, robustness) unless unlocked."""
    guard(days, splits, unlocked)
    parts = [pd.read_parquet(features_root / f"service_day={d}" / "part-0.parquet") for d in days]
    rows = pd.concat(parts, ignore_index=True)
    rows["service_day"] = pd.to_datetime(rows.service_day.astype(str)).dt.date
    return rows


# ---------------------------------------------------------------- fitting

def predict_folds(rows: pd.DataFrame, folds: list[tuple[list[date], list[date]]], cfg: TrainingConfig,
                  model_features: list[str], log=print) -> pd.DataFrame:
    """Out-of-sample predictions of every model: each fold fits on its fit days only."""
    out = []
    for i, (fit_days, eval_days) in enumerate(folds):
        fit = rows[rows.service_day.isin(fit_days)].reset_index(drop=True)
        ev = rows[rows.service_day.isin(eval_days)].reset_index(drop=True)
        pred = ev[KEY_COLS].copy()
        pred["fold"] = i
        for name in cfg.run:
            pred[f"p_{name}"] = make_model(name, cfg, model_features).fit(fit).predict(ev)
        out.append(pred)
        log(f"fold {i + 1}/{len(folds)}: fit {len(fit_days)} day(s), {len(fit):,} rows -> "
            f"evaluate {', '.join(map(str, eval_days))} ({len(ev):,} rows)")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- evaluation

def metric_table(pred: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """Point metrics per model, cutoff and slice (overall and per slice value).
    `recall_at_db_precision`: recall of the model at the precision the DB rule (B2) reaches."""
    recs = []
    for L, g in pred.groupby("cutoff_min"):
        for col in ("all",) + tuple(cfg.slices):
            groups = [("all", g)] if col == "all" else g.groupby(col)
            for val, s in groups:
                y = s.label_fail.astype(int).to_numpy()
                db_prec = np.nan
                if "p_B2" in s:
                    db_prec, db_rec, db_far = precision_recall_at(y, s.p_B2.to_numpy() == 1)
                for name in cfg.run:
                    p = s[f"p_{name}"].to_numpy()
                    r = {"model": name, "cutoff": int(L), "slice": col, "value": str(val), **scores(y, p)}
                    if name == "B2":
                        r.update(precision=db_prec, recall=db_rec, false_alarm_rate=db_far)
                    else:
                        r["recall_at_db_precision"] = recall_at_precision(y, p, db_prec)[0]
                    recs.append(r)
    return pd.DataFrame(recs)


def comparisons(pred: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """Every probabilistic model vs the headline baseline, with day-block bootstrap intervals."""
    base, recs = cfg.headline_baseline, []
    if base not in cfg.run:
        return pd.DataFrame()
    for L, g in pred.groupby("cutoff_min"):
        y, days = g.label_fail.astype(int).to_numpy(), g.service_day.astype(str).to_numpy()
        for name in cfg.run:
            if name in (base, "B2"):
                continue
            b = bootstrap_vs(days, y, g[f"p_{name}"].to_numpy(), g[f"p_{base}"].to_numpy(),
                             cfg.bootstrap_n, cfg.bootstrap_n_auc, cfg.seed)
            recs.append({"model": name, "vs": base, "cutoff": int(L), **b})
    return pd.DataFrame(recs)


def per_day(pred: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    g = pred[pred.cutoff_min == cfg.primary_cutoff]
    recs = []
    for day, s in g.groupby("service_day"):
        y = s.label_fail.astype(int).to_numpy()
        recs.append({"day": str(day), "rows": len(s), "fail_rate": y.mean(),
                     **{m: scores(y, s[f"p_{m}"].to_numpy())["log_loss"] for m in cfg.run if m != "B2"}})
    return pd.DataFrame(recs)


def select_model(table: pd.DataFrame, cfg: TrainingConfig) -> str | None:
    """Best log loss at the primary cutoff among the models (not baselines); a simpler
    model (earlier in the run order) within the tie margin wins."""
    overall = table[(table.slice == "all") & (table.cutoff == cfg.primary_cutoff)].set_index("model").log_loss
    cands = [m for m in cfg.run if m in MODELS and m in overall and not np.isnan(overall[m])]
    if not cands:
        return None
    best = min(cands, key=lambda m: overall[m])
    for m in cands:                         # run order = simplest first
        if overall[m] <= overall[best] * (1 + cfg.selection_tie):
            return m
    return best


def decision_layer(pred: pd.DataFrame, cfg: TrainingConfig, model: str) -> dict | None:
    """At the primary cutoff: the threshold at which the model is as precise as the DB rule,
    and what share of failing connections each flags, at what false-alarm rate."""
    g = pred[pred.cutoff_min == cfg.primary_cutoff]
    if "p_B2" not in g or model not in cfg.run:
        return None
    y = g.label_fail.astype(int).to_numpy()
    prec, rec, far = precision_recall_at(y, g.p_B2.to_numpy() == 1)
    m_rec, thr = recall_at_precision(y, g[f"p_{model}"].to_numpy(), prec)
    out = {"cutoff": cfg.primary_cutoff, "db_rule": {"precision": prec, "recall": rec, "false_alarm_rate": far},
           "model": model, "threshold": thr}
    if not np.isnan(thr):
        out["model_at_threshold"] = dict(zip(("precision", "recall", "false_alarm_rate"),
                                             precision_recall_at(y, g[f"p_{model}"].to_numpy() >= thr)))
    return out


# ---------------------------------------------------------------- test lock

def check_test_log(log_path: Path, final: bool, rerun_reason: str | None) -> list[dict]:
    """The locked test is evaluated once. Every evaluation is appended to a log kept in git;
    a second one needs a reason, which is recorded next to the first."""
    if not final:
        raise SystemExit("The test period is locked. Evaluating it is final and happens once, after "
                         "DESIGN.md is locked and the model is selected on validation. Pass --final to proceed.")
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if log_path.exists() else []
    if entries and not rerun_reason:
        raise SystemExit(f"The test period was already evaluated ({entries[-1]['at']}). A rerun must be "
                         "justified: pass --rerun-reason \"...\"; it is logged permanently.")
    return entries


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate baselines and models on time-based splits")
    ap.add_argument("--mode", choices=("cv", "validate", "test"), default="cv")
    ap.add_argument("--features", type=Path, default=None, help="default: data/features/v{feature_version}")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "training.toml")
    ap.add_argument("--features-config", type=Path, default=ROOT / "config" / "features.toml")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--min-train-days", type=int, help="override [cv] min_train_days")
    ap.add_argument("--fold-days", type=int, help="override [cv] fold_days")
    ap.add_argument("--models", nargs="+", help="override [models] run")
    ap.add_argument("--groups", nargs="+", help="override the feature groups of lr and gbm")
    ap.add_argument("--calibration", choices=("none", "platt", "isotonic"))
    ap.add_argument("--cutoff-mode", choices=("separate", "feature"))
    ap.add_argument("--final", action="store_true", help="required for --mode test")
    ap.add_argument("--rerun-reason")
    ap.add_argument("--test-log", type=Path, default=ROOT / "reports" / "test_log.jsonl")
    args = ap.parse_args(argv)

    cfg = load_training_config(args.config)
    over = {k: v for k, v in {"min_train_days": args.min_train_days, "fold_days": args.fold_days, "calibration": args.calibration,
                              "cutoff_mode": args.cutoff_mode,
                              "run": tuple(args.models) if args.models else None}.items() if v is not None}
    cfg = replace(cfg, **over)
    fcfg = load_feature_config(args.features_config)
    if args.groups:
        fcfg = replace(fcfg, groups=tuple(args.groups))
    model_features = fcfg.columns()
    features_root = args.features or ROOT / "data" / "features" / f"v{fcfg.feature_version}"

    avail = available_days(features_root)
    s = cfg.splits
    train, val = days_in("train", avail, s), days_in("validation", avail, s)
    unlocked: frozenset[str] = frozenset()
    if args.mode == "cv":
        folds = rolling_origin(train, cfg.min_train_days, cfg.fold_days)
        if not folds:
            raise SystemExit(f"{len(train)} training day(s) built; rolling origin needs more than "
                             f"min_train_days={cfg.min_train_days} (override with --min-train-days).")
    elif args.mode == "validate":
        if not train or not val:
            raise SystemExit("validate needs built days in both the training and the validation period")
        folds = [(train, val)]
    else:
        entries = check_test_log(args.test_log, args.final, args.rerun_reason)
        test = days_in("test", avail, s)
        unlocked = frozenset({"test"})
        if not test:
            raise SystemExit("no built days in the test period")
        folds = [(train + val, test)]
    all_days = sorted({d for f, e in folds for d in f + e})
    rows = load_rows(features_root, all_days, s, unlocked)
    print(f"mode {args.mode}: {len(all_days)} day(s), {len(rows):,} rows, models {', '.join(cfg.run)}, "
          f"features {', '.join(fcfg.groups)}, calibration {cfg.calibration}, cutoffs {cfg.cutoff_mode}")
    pred = predict_folds(rows, folds, cfg, model_features)

    table = metric_table(pred, cfg)
    comp = comparisons(pred, cfg)
    days_tab = per_day(pred, cfg)
    selected = select_model(table, cfg)
    decision = decision_layer(pred, cfg, selected) if selected else None
    rel = {m: reliability(pred.loc[pred.cutoff_min == cfg.primary_cutoff, "label_fail"].astype(int).to_numpy(),
                          pred.loc[pred.cutoff_min == cfg.primary_cutoff, f"p_{m}"].to_numpy(),
                          cfg.calibration_bins)
           for m in (cfg.headline_baseline, selected) if m in cfg.run}

    eval_days = sorted({d for _, e in folds for d in e})
    warnings = []
    if len(eval_days) < cfg.min_days_for_ci:
        warnings.append(f"Only {len(eval_days)} evaluation day(s): intervals from resampling days are "
                        f"unreliable below {cfg.min_days_for_ci} days.")
    if cfg.calibration != "none" and min(len(f) for f, _ in folds) < 2:
        warnings.append("Calibration needs at least 2 fit days; folds with fewer were left uncalibrated.")
    gaps = [str(d) for d in pd.date_range(min(all_days), max(all_days)).date if d not in set(all_days)]
    if gaps:
        warnings.append(f"Days missing inside the range (not built): {', '.join(gaps)}")

    stamp, k = datetime.now().strftime("%Y%m%d-%H%M%S"), 1
    run_id = f"{stamp}-{args.mode}"
    while (args.out / run_id).exists():          # two runs within one second
        k += 1
        run_id = f"{stamp}-{args.mode}-{k}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    table.to_csv(run_dir / "metrics.csv", index=False)
    comp.to_csv(run_dir / "comparisons.csv", index=False)
    feature_meta = json.loads((features_root / f"service_day={all_days[0]}" / "_meta.json").read_text(encoding="utf-8"))
    info = {"run_id": run_id, "mode": args.mode, "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "folds": [{"fit": [str(d) for d in f], "evaluate": [str(d) for d in e]} for f, e in folds],
            "rows": len(pred), "config": cfg.as_dict(), "features": fcfg.as_dict(),
            "feature_columns": model_features, "feature_build_commit": feature_meta.get("git_commit"),
            "selected_model": selected, "decision": decision, "warnings": warnings}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    write_report(run_dir / "report.md", info, table, comp, days_tab, rel, cfg)

    if args.mode == "test":
        args.test_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.test_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": info["created_at"], "run_id": run_id, "git_commit": info["git_commit"],
                                "selected_model": selected, "rerun_reason": args.rerun_reason,
                                "previous_runs": len(entries)}) + "\n")

    head = table[(table.slice == "all") & (table.cutoff == cfg.primary_cutoff)].set_index("model")
    print(f"\nlog loss at {cfg.primary_cutoff} min: "
          + ", ".join(f"{m} {head.log_loss[m]:.4f}" for m in cfg.run if m != "B2"))
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
