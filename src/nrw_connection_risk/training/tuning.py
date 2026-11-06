"""Tuning with a pre-registered rule (phase 9, DESIGN.md section 4, config/tuning.toml).

Each model class is tuned on its own and with the same budget: the model (gbm), its headline
benchmark B3 (same implementation, on DB's four inputs) and the logistic regression (lr).
A candidate is a training window (how many days of history each fit uses) plus settings of
the learner. Every candidate is fitted on the rolling-origin folds of DESIGN.md section 6
(weekly blocks after 14 days) and compared with the current settings; it replaces them only
if it beats them by more than the noise of refitting with another seed.

Run from the repo root, once the training period is built:
    python -m nrw_connection_risk.training.tuning
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tomllib
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.build import _git_commit
from ..features.config import load_feature_config
from .config import TrainingConfig, load_training_config
from .evaluate import available_days, load_rows
from .feature_selection import _frame_table, _pct, compare, log_loss_at, model_config, per_slice, predict_variant
from .report import _table
from .splits import fit_days_for, period_of, rolling_origin

ROOT = Path(__file__).resolve().parents[3]
TUNED = ("gbm", "B3", "lr")                 # gbm and B3 share the grid of the gradient boosting
DEFAULT, NOISE = "current settings", "current settings, other seed"


# ---------------------------------------------------------------- settings

@dataclass(frozen=True)
class TuningRule:
    primary_cutoff: int = 30
    min_gain: float = 0.002            # a candidate must beat the current settings by at least this ...
    noise_multiple: float = 2.0        # ... and by this multiple of the seed noise
    max_loss_other: float = 0.002      # largest loss allowed at the other cutoffs (never below their noise)


@dataclass(frozen=True)
class TuningConfig:
    first_day: date
    last_day: date
    min_train_days: int = 14
    fold_days: int = 7
    windows: tuple[int, ...] = (0,)
    gbm_grid: dict = field(default_factory=dict)       # setting -> values, for gbm and B3
    lr_grid: dict = field(default_factory=dict)
    rule: TuningRule = field(default_factory=TuningRule)
    name: str = "e08"
    report: str = "exploration/out/e08_tuning.md"

    def __post_init__(self):
        if self.last_day <= self.first_day:
            raise ValueError("last_day must be after first_day")
        if not self.windows or any(int(w) < 0 for w in self.windows) or 0 not in self.windows:
            raise ValueError("windows must include 0 (every day) and no negative values")
        bad = set(self.gbm_grid) - {"max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization",
                                    "max_iter"}
        if bad or set(self.lr_grid) - {"C"}:
            raise ValueError(f"unknown settings in the grid: {sorted(bad | (set(self.lr_grid) - {'C'}))}")


def load_tuning_config(path: Path) -> TuningConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    d, g, out = c["data"], c["grid"], c.get("output", {})
    return TuningConfig(
        first_day=date.fromisoformat(d["first_day"]), last_day=date.fromisoformat(d["last_day"]),
        min_train_days=int(d["min_train_days"]), fold_days=int(d["fold_days"]),
        windows=tuple(int(w) for w in g["windows"]),
        gbm_grid={k: tuple(v) for k, v in g.get("gbm", {}).items()},
        lr_grid={k: tuple(v) for k, v in g.get("lr", {}).items()},
        rule=TuningRule(**c["rule"]), name=out.get("name", "e08"), report=out.get("report", "exploration/out/e08_tuning.md"))


# ---------------------------------------------------------------- candidates

@dataclass(frozen=True)
class Candidate:
    name: str
    window: int = 0                    # days of history per fit (0 = every day since first_day)
    params: tuple = ()                 # ((setting, value), ...) over the current settings
    seed_offset: int = 0


def current(cfg: TrainingConfig, model: str) -> dict:
    """The model's settings as configured now."""
    if model == "lr":
        return dict(cfg.lr)
    return {**cfg.gbm, **cfg.b3} if model == "B3" else dict(cfg.gbm)


def candidates(tc: TuningConfig, cfg: TrainingConfig, model: str) -> list[Candidate]:
    """The current settings (with another seed for the noise floor, where the learner has one),
    then every combination of window and grid values that differs from them."""
    grid = tc.lr_grid if model == "lr" else tc.gbm_grid
    now = current(cfg, model)
    out = [Candidate(DEFAULT, cfg.window(model))]
    if model != "lr":                  # logistic regression is deterministic: no seed noise
        out.append(Candidate(NOISE, cfg.window(model), seed_offset=1))
    keys = sorted(grid)
    for w in tc.windows:
        for values in itertools.product(*(grid[k] for k in keys)):
            params = tuple(zip(keys, values))
            if w == cfg.window(model) and all(now.get(k) == v for k, v in params):
                continue               # identical to the current settings
            label = ", ".join([f"window {w} d" if w else "all days"] + [f"{k} {v}" for k, v in params])
            out.append(Candidate(label, w, params))
    return out


def with_candidate(cfg: TrainingConfig, model: str, c: Candidate) -> TrainingConfig:
    """Training settings in which `model` uses the candidate's learner settings."""
    p = dict(c.params)
    cfg = replace(cfg, seed=cfg.seed + c.seed_offset)
    if model == "lr":
        return replace(cfg, lr={**cfg.lr, **p})
    if model == "B3":
        return replace(cfg, gbm={**cfg.gbm, **cfg.b3, **p}, b3={})
    return replace(cfg, gbm={**cfg.gbm, **p})


def windowed(folds, window: int):
    return [(fit_days_for(fit, None, window), ev) for fit, ev in folds]


# ---------------------------------------------------------------- rule

def thresholds(rule: TuningRule, noise: dict[int, dict] | None) -> tuple[float, dict[int, float]]:
    """Minimum gain at the primary cutoff and the largest loss allowed at each other cutoff."""
    n = {L: abs(r["gain"]) for L, r in (noise or {}).items()}
    thr = max(rule.min_gain, rule.noise_multiple * n.get(rule.primary_cutoff, 0.0))
    tol = {L: max(rule.max_loss_other, rule.noise_multiple * v) for L, v in n.items()}
    return thr, tol


def passes(res: dict[int, dict], rule: TuningRule, thr: float, tol: dict[int, float]) -> tuple[bool, list[str]]:
    """Does a candidate beat the current settings clearly enough? Returns the verdict and the
    conditions it failed."""
    P = rule.primary_cutoff
    if P not in res:
        return False, [f"no evaluation at {P} min"]
    r, failed = res[P], []
    if r["gain"] < thr:
        failed.append(f"gain at {P} min {r['gain']:+.2%} < {thr:.2%}")
    if r["ci"][0] <= 0:
        failed.append(f"95% interval at {P} min includes 0")
    for L, o in res.items():
        t = tol.get(L, rule.max_loss_other)
        if L != P and o["gain"] < -t:
            failed.append(f"{o['gain']:+.2%} at {L} min (more than {t:.2%} worse)")
    return not failed, failed


def decide(rows: pd.DataFrame, preds: dict[str, np.ndarray], cands: list[Candidate], rule: TuningRule,
           n: int, seed: int) -> dict:
    """Every candidate against the current settings; the passing one with the largest gain at
    the primary cutoff wins, otherwise the current settings stay."""
    noise = compare(rows, preds[NOISE], preds[DEFAULT], n, seed) if NOISE in preds else None
    thr, tol = thresholds(rule, noise)
    results = []
    for c in cands:
        if c.name in (DEFAULT, NOISE):
            continue
        res = compare(rows, preds[c.name], preds[DEFAULT], n, seed)
        ok, failed = passes(res, rule, thr, tol)
        results.append({"candidate": c.name, "window": c.window, "params": dict(c.params), "passes": ok,
                        "failed": failed, "result": {str(k): v for k, v in res.items()}})
    P = str(rule.primary_cutoff)
    ok = [r for r in results if r["passes"]]
    winner = max(ok, key=lambda r: r["result"][P]["gain"]) if ok else None
    return {"noise": {str(k): v for k, v in noise.items()} if noise else None, "threshold": thr,
            "tolerance": {str(k): v for k, v in tol.items()}, "results": results,
            "winner": winner["candidate"] if winner else DEFAULT}


# ---------------------------------------------------------------- report

def config_lines(model: str, cfg: TrainingConfig, cands: list[Candidate], winner: str) -> list[str]:
    """What config/training.toml says after the run."""
    c = next(x for x in cands if x.name == winner)
    section = {"gbm": "[gbm]", "B3": "[b3]", "lr": "[lr]"}[model]
    lines = [f"[fit_window] {model} = {c.window}"]
    lines += [f"{section} {k} = {v}" for k, v in c.params] or [f"{section}: unchanged"]
    return lines


def write_report(path: Path, info: dict, per_model: dict, folds_tab: pd.DataFrame, hubs_tab: pd.DataFrame,
                 vs_b3: pd.DataFrame) -> None:
    P = info["rule"]["primary_cutoff"]
    L = [f"# {info['name']}: tuning", "",
         f"Service days {info['window'][0]} to {info['window'][1]} ({info['days']} days, {info['rows']:,} rows), "
         f"rolling origin with weekly blocks (DESIGN.md section 6): {len(info['folds'])} folds, "
         f"{info['eval_days']} evaluation days ({info['first_eval']} to {info['last_eval']}). One model per cutoff, "
         f"no calibration (chosen on validation). Rule: {info['config_path']} (sha256 {info['config_sha256'][:12]}), "
         f"DESIGN.md section 4. Code {info['git_commit']}.", ""]
    for w in info["warnings"]:
        L.append(f"> **Warning:** {w}")
    L += ["", "Folds:", ""] + [f"- fit {f['fit'][0]} to {f['fit'][-1]} ({len(f['fit'])} days), evaluate "
                                f"{f['evaluate'][0]} to {f['evaluate'][-1]}" for f in info["folds"]]
    L += ["", "## Decisions", "",
          f"Gain = relative reduction of log loss against the current settings (positive = better). A candidate "
          f"replaces them only if its gain at {P} min reaches the threshold, its 95% interval (resampling whole "
          f"days) lies above 0, and it is not worse at the other cutoffs by more than the tolerance.", "",
          _table(["Model", "Noise (other seed) 60 / 30 / 10", f"Threshold at {P}", "Winner", "Gain 30 min",
                  "Settings in config/training.toml"],
                 [[m, " / ".join(_pct(d["noise"][c]["gain"]) for c in ("60", "30", "10")) if d["noise"] else "none "
                   "(deterministic)", f"{d['threshold']:.2%}", f"**{d['winner']}**",
                   _pct(next((r["result"][str(P)]["gain"] for r in d["results"] if r["candidate"] == d["winner"]),
                             np.nan)) or "", "; ".join(d["config_lines"])] for m, d in per_model.items()]), ""]
    for m, d in per_model.items():
        rows = sorted(d["results"], key=lambda r: -r["result"][str(P)]["gain"])
        L += [f"## {m}: every candidate against the current settings", "",
              _table(["Candidate", "Gain 60", f"Gain {P}", f"95% interval ({P})", "Gain 10", f"Days better ({P})",
                      "Passes"],
                     [[r["candidate"], _pct(r["result"]["60"]["gain"]), _pct(r["result"][str(P)]["gain"]),
                       f"{_pct(r['result'][str(P)]['ci'][0])} to {_pct(r['result'][str(P)]['ci'][1])}",
                       _pct(r["result"]["10"]["gain"]), f"{r['result'][str(P)]['day_share']:.0%}",
                       "yes" if r["passes"] else "no"] for r in rows]), ""]
    L += [f"## Winners per fold and per hub at {P} min (for information)", "",
          "Gain of each model's winner over its current settings. The winner was chosen as the best of many "
          "candidates on these days, so its gain is optimistic; validation is the first test on new days.", "",
          _frame_table(folds_tab), "", _frame_table(hubs_tab), "",
          "## Against B3 (for information)", "",
          "Log loss of each setting and its gain over B3 with B3's settings of the same row, on the same rows.", "",
          _frame_table(vs_b3), ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main

def run(tc: TuningConfig, cfg: TrainingConfig, features_root: Path, out: Path, publish: Path | None,
        model_features: list[str], allow_missing: bool = False, config_text: bytes = b"", log=print,
        config_path: str = "config/tuning.toml") -> dict:
    for d in (tc.first_day, tc.last_day):
        if period_of(d, cfg.splits) != "train":
            raise SystemExit(f"{d} is not in the training period: tuning uses training days only")
    wanted = [d.date() for d in pd.date_range(tc.first_day, tc.last_day)]
    have = set(available_days(features_root))
    missing = [d for d in wanted if d not in have]
    warnings = []
    if missing:
        msg = f"days not built: {', '.join(map(str, missing))}"
        if not allow_missing:
            raise SystemExit(f"The tuning window is incomplete ({msg}). Build them, or run with --allow-missing "
                             "(recorded in the report).")
        warnings.append(f"Run with missing {msg}.")
    days = [d for d in wanted if d in have]
    rows = load_rows(features_root, days, cfg.splits)
    folds = rolling_origin(days, tc.min_train_days, tc.fold_days)
    if not folds:
        raise SystemExit(f"{len(days)} day(s): need more than min_train_days={tc.min_train_days}")
    mcfg = replace(model_config(cfg), first_fit_day=None)
    log(f"{tc.name}: {len(days)} days, {len(rows):,} rows, {len(folds)} folds, models {', '.join(TUNED)}")

    preds, per_model, cands_by_model = {}, {}, {}
    for model in TUNED:
        cands = candidates(tc, mcfg, model)
        cands_by_model[model] = cands
        p = {}
        for c in cands:
            log(f"{model}: {c.name}")
            p[c.name] = predict_variant(rows, windowed(folds, c.window), with_candidate(mcfg, model, c), model,
                                        model_features, log)
        d = decide(rows, p, cands, tc.rule, cfg.bootstrap_n, cfg.seed)
        d["config_lines"] = config_lines(model, mcfg, cands, d["winner"])
        per_model[model] = d
        preds[model] = {k: v for k, v in p.items() if k in (DEFAULT, d["winner"])}   # free the others

    P = tc.rule.primary_cutoff
    fold_of = {d: f"fold {i + 1}" for i, (_, ev) in enumerate(folds) for d in ev}
    fold_col = rows.service_day.map(fold_of)
    rec_f, rec_h = {}, {}
    for model, d in per_model.items():
        if d["winner"] == DEFAULT:
            continue
        pw, pd0 = preds[model][d["winner"]], preds[model][DEFAULT]
        rec_f[model] = {k: _pct(v) for k, v in
                        per_slice(rows.assign(fold=fold_col), pw, pd0, "fold", P).items()}
        rec_h[model] = {h: _pct(v) for h, v in per_slice(rows, pw, pd0, "hub", P).items()}
    folds_tab = pd.DataFrame(rec_f).T.rename_axis("model").reset_index() if rec_f else pd.DataFrame(
        {"model": ["(every model keeps its current settings)"]})
    hubs_tab = pd.DataFrame(rec_h).T.rename_axis("model").reset_index() if rec_h else pd.DataFrame(
        {"model": ["(every model keeps its current settings)"]})

    recs, b3 = [], preds["B3"]
    for model in ("gbm", "lr"):
        for which in (DEFAULT, per_model[model]["winner"]):
            for b3_which in (DEFAULT, per_model["B3"]["winner"]):
                rec = {"model": model, "settings": which, "B3 settings": b3_which}
                for L in sorted(rows.cutoff_min.unique()):
                    a, b = log_loss_at(rows, preds[model][which], L), log_loss_at(rows, b3[b3_which], L)
                    rec[f"log loss {int(L)}"], rec[f"vs B3 {int(L)}"] = a, _pct(1 - a / b)
                recs.append(rec)
    vs_b3 = pd.DataFrame(recs).drop_duplicates(["model", "settings", "B3 settings"])

    ev_days = sorted({d for _, e in folds for d in e})
    stamp, k = datetime.now().strftime("%Y%m%d-%H%M%S"), 1
    run_dir = out / f"{stamp}-{tc.name}"
    while run_dir.exists():
        k += 1
        run_dir = out / f"{stamp}-{tc.name}-{k}"
    run_dir.mkdir(parents=True)
    info = {"run_id": run_dir.name, "name": tc.name, "config_path": config_path, "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_sha256": hashlib.sha256(config_text).hexdigest(), "rule": tc.rule.__dict__,
            "window": [str(tc.first_day), str(tc.last_day)], "days": len(days), "rows": len(rows),
            "folds": [{"fit": [str(x) for x in f], "evaluate": [str(x) for x in e]} for f, e in folds],
            "eval_days": len(ev_days), "first_eval": str(ev_days[0]), "last_eval": str(ev_days[-1]),
            "current": {m: {"window": mcfg.window(m), "settings": current(mcfg, m)} for m in TUNED},
            "models": per_model, "warnings": warnings}
    (run_dir / "decision.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    keep = {f"p_{m}_{'current' if w == DEFAULT else 'winner'}": preds[m][w]
            for m in TUNED for w in {DEFAULT, per_model[m]["winner"]}}
    pd.DataFrame({**rows[["service_day", "cutoff_min", "stop_id_a", "stop_id_b", "label_fail"]], **keep}).to_parquet(
        run_dir / "predictions.parquet", index=False)
    write_report(run_dir / "report.md", info, per_model, folds_tab, hubs_tab, vs_b3)
    if publish:
        publish.parent.mkdir(parents=True, exist_ok=True)
        publish.write_text((run_dir / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    log("")
    for m, d in per_model.items():
        log(f"{m:<4} winner: {d['winner']}   ->  " + "; ".join(d["config_lines"]))
    log(f"report: {run_dir / 'report.md'}" + (f" (copied to {publish})" if publish else ""))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tuning with a pre-registered rule (phase 9)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "tuning.toml")
    ap.add_argument("--training-config", type=Path, default=ROOT / "config" / "training.toml")
    ap.add_argument("--features-config", type=Path, default=ROOT / "config" / "features.toml")
    ap.add_argument("--features", type=Path, default=None, help="default: data/features/v{feature_version}")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--publish", type=Path, default=None,
                    help="copy of the report kept in git (default: [output] report of the config)")
    ap.add_argument("--allow-missing", action="store_true", help="run although days of the window are not built")
    args = ap.parse_args(argv)
    tc = load_tuning_config(args.config)
    cfg = load_training_config(args.training_config)
    fcfg = load_feature_config(args.features_config)
    features_root = args.features or ROOT / "data" / "features" / f"v{fcfg.feature_version}"
    try:
        shown = args.config.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(args.config)
    run(tc, cfg, features_root, args.out, args.publish or ROOT / tc.report, fcfg.columns(), args.allow_missing,
        args.config.read_bytes(), config_path=shown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
