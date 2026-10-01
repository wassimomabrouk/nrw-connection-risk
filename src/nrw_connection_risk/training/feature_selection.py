"""Feature selection with a pre-registered rule: e06 (second feature exploration) and e07 (feeder group).

DESIGN.md section 3. Every group must earn its place under the same rule: groups of the
reference set are tested by leaving them out, other groups by adding them. All variants are
the model as deployed (gradient boosting, one model per cutoff, settings from
config/training.toml), fitted on expanding windows of whole days and evaluated on the next
day. Rule and evaluation window: config/feature_selection.toml.

Run from the repo root, once all days of the window are built:
    python -m nrw_connection_risk.training.feature_selection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tomllib
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.build import _git_commit
from ..features.columns import GROUPS, feature_columns
from ..features.config import load_feature_config
from .config import TrainingConfig, load_training_config
from .evaluate import available_days, load_rows
from .metrics import _draws, auc, pointwise_log_loss
from .models import make_model
from .report import _table
from .splits import period_of, rolling_origin

ROOT = Path(__file__).resolve().parents[3]
REF, NOISE = "reference", "reference, other seed"


# ---------------------------------------------------------------- settings

@dataclass(frozen=True)
class Rule:
    primary_cutoff: int = 30
    min_gain: float = 0.005
    noise_multiple: float = 2.0
    min_day_share: float = 0.6
    max_loss_other: float = 0.002


@dataclass(frozen=True)
class SelectionConfig:
    first_day: date
    last_day: date
    min_train_days: int
    reference: tuple[str, ...]
    drop: tuple[str, ...]
    add: tuple[str, ...]
    restricted: dict[str, date] = field(default_factory=dict)
    rule: Rule = field(default_factory=Rule)
    name: str = "e06"                  # run folder suffix and report title
    report: str = "exploration/out/e06_feature_selection.md"     # copy of the report kept in git
    reference_from: str = ""           # where the reference groups came from, if not listed

    def __post_init__(self):
        feature_columns(list(self.reference) + list(self.add) + list(self.restricted))   # unknown groups raise
        if not set(self.drop) <= set(self.reference):
            raise ValueError("groups to drop must be in the reference set")
        if set(self.add) & set(self.reference) or set(self.restricted) & set(self.reference):
            raise ValueError("groups to add must not be in the reference set")
        if "db" in self.drop:
            raise ValueError("the DB group is the benchmark's information and is never dropped")

    def as_dict(self) -> dict:
        return {"first_day": str(self.first_day), "last_day": str(self.last_day),
                "min_train_days": self.min_train_days, "reference": list(self.reference),
                "reference_from": self.reference_from, "name": self.name, "report": self.report,
                "drop": list(self.drop), "add": list(self.add),
                "restricted": {k: str(v) for k, v in self.restricted.items()}, "rule": self.rule.__dict__}


def load_selection_config(path: Path, root: Path = ROOT) -> SelectionConfig:
    """`[reference] groups = [...]`, or `[reference] from = "config/features.toml"` to take the
    groups a previous selection wrote there (e07 builds on the e06 result)."""
    with open(path, "rb") as f:
        c = tomllib.load(f)
    d, ref, out = c["data"], c["reference"], c.get("output", {})
    if "from" in ref:
        groups, source = load_feature_config(root / ref["from"]).groups, ref["from"]
    else:
        groups, source = ref["groups"], ""
    return SelectionConfig(
        first_day=date.fromisoformat(d["first_day"]), last_day=date.fromisoformat(d["last_day"]),
        min_train_days=int(d["min_train_days"]), reference=tuple(groups),
        drop=tuple(c["candidates"]["drop"]), add=tuple(c["candidates"]["add"]),
        restricted={k: date.fromisoformat(v) for k, v in c.get("restricted", {}).items()},
        rule=Rule(**c["rule"]), name=out.get("name", "e06"),
        report=out.get("report", "exploration/out/e06_feature_selection.md"), reference_from=source)


# ---------------------------------------------------------------- variants and tests

@dataclass(frozen=True)
class Variant:
    groups: tuple[str, ...]
    seed_offset: int = 0
    start: date | None = None          # restricted variants fit and evaluate only on days from here


@dataclass(frozen=True)
class Test:
    group: str
    kind: str                          # "drop", "add" or "restricted"
    with_group: str                    # variant that has the group
    without: str                       # variant that lacks it


def ordered(groups) -> tuple[str, ...]:
    return tuple(g for g in GROUPS if g in set(groups))


def variants_and_tests(sc: SelectionConfig) -> tuple[dict[str, Variant], list[Test]]:
    ref = ordered(sc.reference)
    v = {REF: Variant(ref), NOISE: Variant(ref, seed_offset=1)}
    tests = []
    for g in sc.drop:
        v[f"without {g}"] = Variant(ordered(set(ref) - {g}))
        tests.append(Test(g, "drop", REF, f"without {g}"))
    for g in sc.add:
        v[f"with {g}"] = Variant(ordered(set(ref) | {g}))
        tests.append(Test(g, "add", f"with {g}", REF))
    for g, start in sc.restricted.items():
        v[f"reference from {start}"] = Variant(ref, start=start)
        v[f"with {g}"] = Variant(ordered(set(ref) | {g}), start=start)
        tests.append(Test(g, "restricted", f"with {g}", f"reference from {start}"))
    return v, tests


# ---------------------------------------------------------------- fitting

def model_config(cfg: TrainingConfig) -> TrainingConfig:
    """The model as deployed, without calibration (chosen later on validation)."""
    return replace(cfg, cutoff_mode="separate", calibration="none")


def folds_for(days: list[date], start: date | None, min_train_days: int):
    return rolling_origin([d for d in days if start is None or d >= start], min_train_days, 1)


def predict_variant(rows: pd.DataFrame, folds, cfg: TrainingConfig, name: str, cols: list[str],
                    log=print) -> np.ndarray:
    """Out-of-sample P(fail) for every row of the evaluated days, NaN elsewhere."""
    p = np.full(len(rows), np.nan)
    t0 = time.time()
    for fit_days, eval_days in folds:
        fit = rows.service_day.isin(fit_days).to_numpy()
        ev = rows.service_day.isin(eval_days).to_numpy()
        p[ev] = make_model(name, cfg, cols).fit(rows[fit]).predict(rows[ev])
    log(f"  {len(folds)} folds in {time.time() - t0:.0f} s")
    return p


# ---------------------------------------------------------------- comparison

def ll_gain(days: np.ndarray, y: np.ndarray, p_with: np.ndarray, p_without: np.ndarray,
            n: int, seed: int) -> dict:
    """Relative log loss reduction of `with` over `without` (positive = the group helps), with a
    95% interval from resampling whole days, and the share of days on which it is lower."""
    uniq, d = np.unique(days, return_inverse=True)
    k = len(uniq)
    cnt = np.bincount(d, minlength=k).astype(float)
    a = np.bincount(d, pointwise_log_loss(y, p_with), minlength=k)
    b = np.bincount(d, pointwise_log_loss(y, p_without), minlength=k)
    w = np.stack([np.bincount(r, minlength=k) for r in _draws(k, n, seed)]).astype(float)
    boot = 1 - (w @ a / (w @ cnt)) / (w @ b / (w @ cnt))
    return {"days": k, "gain": float(1 - a.sum() / b.sum()),
            "ci": tuple(float(x) for x in np.nanquantile(boot, [0.025, 0.975])),
            "day_share": float(np.mean(a < b)), "auc_diff": float(auc(y, p_with) - auc(y, p_without)),
            "per_day": {str(u): float(1 - a[i] / b[i]) for i, u in enumerate(uniq)}}


def compare(rows: pd.DataFrame, p_with: np.ndarray, p_without: np.ndarray, n: int, seed: int) -> dict[int, dict]:
    ok = ~np.isnan(p_with) & ~np.isnan(p_without)
    out = {}
    for L in sorted(rows.cutoff_min.unique()):
        m = ok & (rows.cutoff_min.to_numpy() == L)
        if m.any():
            out[int(L)] = ll_gain(rows.service_day.astype(str).to_numpy()[m], rows.label_fail.astype(int).to_numpy()[m],
                                  p_with[m], p_without[m], n, seed)
    return out


def per_slice(rows: pd.DataFrame, p_with: np.ndarray, p_without: np.ndarray, col: str, L: int) -> dict[str, float]:
    """Relative log loss reduction per value of `col` at cutoff L (point estimates)."""
    if col not in rows:
        return {}
    m = ~np.isnan(p_with) & ~np.isnan(p_without) & (rows.cutoff_min.to_numpy() == L)
    y = rows.label_fail.astype(int).to_numpy()
    out = {}
    for v in sorted(rows.loc[m, col].astype(str).unique()):
        k = m & (rows[col].astype(str).to_numpy() == v)
        out[v] = float(1 - pointwise_log_loss(y[k], p_with[k]).mean() / pointwise_log_loss(y[k], p_without[k]).mean())
    return out


def threshold(rule: Rule, noise: dict[int, dict]) -> float:
    """Minimum gain at the primary cutoff."""
    return max(rule.min_gain, rule.noise_multiple * abs(noise[rule.primary_cutoff]["gain"]))


def tolerance(rule: Rule, noise: dict[int, dict] | None, L: int) -> float:
    """Largest loss allowed at another cutoff: never stricter than that cutoff's noise floor."""
    n = abs(noise[L]["gain"]) if noise and L in noise else 0.0
    return max(rule.max_loss_other, rule.noise_multiple * n)


def verdict(res: dict[int, dict], rule: Rule, thr: float,
            noise: dict[int, dict] | None = None) -> tuple[bool, list[str]]:
    """Does the group earn its place? Returns the verdict and the conditions it failed."""
    P = rule.primary_cutoff
    if P not in res:
        return False, [f"no evaluation at {P} min"]
    r, failed = res[P], []
    if r["gain"] < thr:
        failed.append(f"gain at {P} min {r['gain']:+.2%} < {thr:.2%}")
    if r["ci"][0] <= 0:
        failed.append(f"95% interval at {P} min includes 0 ({r['ci'][0]:+.2%} to {r['ci'][1]:+.2%})")
    if r["day_share"] < rule.min_day_share:
        failed.append(f"better on {r['day_share']:.0%} of days < {rule.min_day_share:.0%}")
    for L, o in res.items():
        tol = tolerance(rule, noise, L)
        if L != P and o["gain"] < -tol:
            failed.append(f"{o['gain']:+.2%} at {L} min (more than {tol:.2%} worse)")
    return not failed, failed


def log_loss_at(rows: pd.DataFrame, p: np.ndarray, L: int) -> float:
    m = ~np.isnan(p) & (rows.cutoff_min.to_numpy() == L)
    return float(pointwise_log_loss(rows.label_fail.astype(int).to_numpy()[m], p[m]).mean())


# ---------------------------------------------------------------- report

def _pct(x: float) -> str:
    return "" if x is None or np.isnan(x) else f"{x:+.2%}"


def _frame_table(df: pd.DataFrame) -> str:
    cell = lambda x: f"{x:.5f}" if isinstance(x, float) else ("" if x is None else str(x))   # noqa: E731
    return _table(list(df.columns), [[cell(x) for x in r] for r in df.itertuples(index=False)])


def write_report(path: Path, info: dict, results: list[dict], vs_b3: pd.DataFrame, daily: pd.DataFrame) -> None:
    P = info["rule"]["primary_cutoff"]
    title = {"e06": "second feature exploration", "e07": "feeder group"}.get(info["name"], "feature selection")
    L = [f"# {info['name']}: {title}", "",
         f"Service days {info['window'][0]} to {info['window'][1]} ({info['days']} days, "
         f"{info['rows']:,} rows), expanding window with daily folds: {info['eval_days']} evaluation days "
         f"({info['first_eval']} to {info['last_eval']}). Model: gradient boosting, one model per cutoff, "
         f"no calibration, settings from config/training.toml. Rule: {info['config_path']} "
         f"(sha256 {info['config_sha256'][:12]}), DESIGN.md section 3. Code {info['git_commit']}.", "",
         f"Reference groups: {', '.join(info['selection']['reference'])}"
         + (f" (from {info['selection']['reference_from']})" if info["selection"]["reference_from"] else "") + ".", ""]
    for w in info["warnings"]:
        L.append(f"> **Warning:** {w}")
    if info["warnings"]:
        L.append("")
    n = info["noise"]
    L += ["## Noise floor", "",
          "Refitting the reference with another seed changes log loss by "
          + ", ".join(f"{_pct(n[str(c)]['gain'])} at {c} min" for c in sorted(map(int, n))) + ". "
          f"Threshold at {P} min: max({info['rule']['min_gain']:.1%}, {info['rule']['noise_multiple']:g} x "
          f"{abs(n[str(P)]['gain']):.2%}) = **{info['threshold']:.2%}**. Largest loss allowed at the other "
          f"cutoffs: " + ", ".join(f"{t:.2%} at {c} min" for c, t in sorted(info["tolerance"].items(), key=lambda x: -int(x[0])))
          + ".", "",
          "## Decisions", "",
          "Gain = relative reduction of log loss by having the group (positive = the group helps).", "",
          "| Group | Test | Gain 60 min | Gain 30 min | 95% interval (30) | Gain 10 min | Days better (30) | AUC diff (30) | Decision |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        res = r["result"]
        g = lambda c: _pct(res[str(c)]["gain"]) if str(c) in res else ""   # noqa: E731
        ci = res.get(str(P), {}).get("ci", (np.nan, np.nan))
        test = {"drop": "leave out", "add": "add", "restricted": f"add, from {r.get('start')}"}[r["kind"]]
        L.append(f"| {r['group']} | {test} | {g(60)} | {g(30)} | {_pct(ci[0])} to {_pct(ci[1])} | {g(10)} | "
                 f"{res.get(str(P), {}).get('day_share', np.nan):.0%} | {res.get(str(P), {}).get('auc_diff', np.nan):+.4f} | "
                 f"**{'keep' if r['keep'] else 'drop'}** |")
    L.append("")
    for r in results:
        if r["failed"]:
            L.append(f"- {r['group']}: " + "; ".join(r["failed"]))
    hubs = sorted({h for r in results for h in r.get("per_hub", {})})
    if hubs:
        L += ["", f"Gain at {P} min per hub (for information):", "",
              _table(["Group"] + hubs, [[r["group"]] + [_pct(r["per_hub"].get(h, np.nan)) for h in hubs] for r in results])]
    L += ["", "## Result", "", f"**Feature groups: {', '.join(info['final_groups'])}**", ""]
    if info.get("combination"):
        c = info["combination"]
        L.append(f"Combination check: all changes together {c['combined']:.5f} log loss at {P} min, best single "
                 f"change ({c['best_single']}) {c['best_single_ll']:.5f}. {c['outcome']}")
        L.append("")
    L += ["## Every variant against B3", "",
          "Log loss and gain over B3 (DB's prognosis as a probability, same model class) fitted on the same days "
          "and evaluated on the same rows. "
          "For information only; the decision is made against the reference above.", "",
          _frame_table(vs_b3), "",
          f"## Gain per evaluation day at {P} min", "", _frame_table(daily), ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main

def run(sc: SelectionConfig, cfg: TrainingConfig, features_root: Path, out: Path, publish: Path | None,
        allow_missing: bool = False, config_text: bytes = b"", log=print,
        config_path: str = "config/feature_selection.toml") -> dict:
    rule = sc.rule
    for d in (sc.first_day, sc.last_day):
        if period_of(d, cfg.splits) != "train":
            raise SystemExit(f"{d} is not in the training period: feature selection uses training days only")
    wanted = [d.date() for d in pd.date_range(sc.first_day, sc.last_day)]
    have = set(available_days(features_root))
    missing = [d for d in wanted if d not in have]
    warnings = []
    if missing:
        msg = f"days not built: {', '.join(map(str, missing))}"
        if not allow_missing:
            raise SystemExit(f"The pre-registered window is incomplete ({msg}). Build them, or run with "
                             "--allow-missing (recorded in the report).")
        warnings.append(f"Run with missing {msg}.")
    days = [d for d in wanted if d in have]
    rows = load_rows(features_root, days, cfg.splits)
    variants, tests = variants_and_tests(sc)
    need = feature_columns(sorted({g for v in variants.values() for g in v.groups}))
    absent = [c for c in need if c not in rows.columns]
    if absent:
        raise SystemExit(f"feature table lacks columns {absent}: rebuild the features")
    mcfg = model_config(cfg)
    main_folds = folds_for(days, None, sc.min_train_days)
    if not main_folds:
        raise SystemExit(f"{len(days)} day(s): need more than min_train_days={sc.min_train_days}")
    log(f"{sc.name}: {len(days)} days, {len(rows):,} rows, {len(main_folds)} daily folds, {len(variants)} variants + B3")

    preds = {}
    log("B3")
    preds["B3"] = predict_variant(rows, main_folds, mcfg, "B3", [], log)
    for start in sorted(set(sc.restricted.values())):          # B3 on the same folds as restricted variants
        log(f"B3 from {start}")
        preds[f"B3 from {start}"] = predict_variant(rows, folds_for(days, start, sc.min_train_days), mcfg, "B3", [], log)
    for name, v in variants.items():
        log(f"{name}: {', '.join(v.groups)}")
        vcfg = replace(mcfg, seed=mcfg.seed + v.seed_offset)
        preds[name] = predict_variant(rows, folds_for(days, v.start, sc.min_train_days), vcfg, "gbm",
                                      feature_columns(list(v.groups)), log)

    noise = compare(rows, preds[NOISE], preds[REF], cfg.bootstrap_n, cfg.seed)
    thr = threshold(rule, noise)
    results, keep = [], {}
    for t in tests:
        res = compare(rows, preds[t.with_group], preds[t.without], cfg.bootstrap_n, cfg.seed)
        ok, failed = verdict(res, rule, thr, noise)
        per_hub = per_slice(rows, preds[t.with_group], preds[t.without], "hub", rule.primary_cutoff)
        keep[t.group] = ok
        results.append({"group": t.group, "kind": t.kind, "with": t.with_group, "without": t.without,
                        "start": str(sc.restricted.get(t.group)) if t.kind == "restricted" else None,
                        "keep": ok, "failed": failed, "result": {str(k): v for k, v in res.items()},
                        "per_hub": per_hub})

    # changes on the full window: removed reference groups and added groups
    changes = {f"without {t.group}": t for t in tests if t.kind == "drop" and not keep[t.group]}
    changes |= {f"with {t.group}": t for t in tests if t.kind == "add" and keep[t.group]}
    groups = [g for g in sc.reference if keep.get(g, True)] + [g for g in sc.add if keep[g]]
    combination = None
    if len(changes) > 1:
        P = rule.primary_cutoff
        name = "all changes"
        log(f"{name}: {', '.join(ordered(groups))}")
        preds[name] = predict_variant(rows, main_folds, mcfg, "gbm", feature_columns(list(ordered(groups))), log)
        variants[name] = Variant(ordered(groups))
        single = {c: log_loss_at(rows, preds[c], P) for c in changes}
        best = min(single, key=single.get)
        combined = log_loss_at(rows, preds[name], P)
        combination = {"combined": combined, "best_single": best, "best_single_ll": single[best]}
        if combined > single[best]:
            groups = list(variants[best].groups)
            combination["outcome"] = "The combination is worse, so only the best single change is applied."
        else:
            combination["outcome"] = "The combination is at least as good, so all changes are applied."
    groups += [g for g in sc.restricted if keep[g]]
    final = list(ordered(groups))

    # information: every variant against B3, on the rows both have
    recs = []
    for name, v in variants.items():
        rec = {"variant": name, "groups": ", ".join(v.groups)}
        b3 = preds["B3" if v.start is None else f"B3 from {v.start}"]
        for L in sorted(rows.cutoff_min.unique()):
            m = ~np.isnan(preds[name]) & ~np.isnan(b3) & (rows.cutoff_min.to_numpy() == L)
            y = rows.label_fail.astype(int).to_numpy()[m]
            a, b = pointwise_log_loss(y, preds[name][m]).mean(), pointwise_log_loss(y, b3[m]).mean()
            rec[f"log loss {int(L)}"], rec[f"vs B3 {int(L)}"] = a, f"{1 - a / b:+.2%}"
        recs.append(rec)
    vs_b3 = pd.DataFrame(recs)
    P = str(rule.primary_cutoff)
    daily = pd.DataFrame({r["group"] + (" (leave out)" if r["kind"] == "drop" else ""):
                          pd.Series(r["result"][P]["per_day"])
                          for r in results if P in r["result"]})
    daily.insert(0, "other seed", pd.Series(noise[int(P)]["per_day"]))
    daily = daily.map(_pct).rename_axis("day").reset_index()

    eval_days = sorted({d for _, e in main_folds for d in e})
    stamp, k = datetime.now().strftime("%Y%m%d-%H%M%S"), 1
    run_dir = out / f"{stamp}-{sc.name}"
    while run_dir.exists():
        k += 1
        run_dir = out / f"{stamp}-{sc.name}-{k}"
    run_dir.mkdir(parents=True)
    info = {"run_id": run_dir.name, "name": sc.name, "config_path": config_path, "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_sha256": hashlib.sha256(config_text).hexdigest(), "selection": sc.as_dict(),
            "rule": rule.__dict__, "window": [str(sc.first_day), str(sc.last_day)], "days": len(days),
            "rows": len(rows), "eval_days": len(eval_days), "first_eval": str(eval_days[0]),
            "last_eval": str(eval_days[-1]), "model": model_config(cfg).as_dict(),
            "noise": {str(k): v for k, v in noise.items()}, "threshold": thr,
            "tolerance": {str(L): tolerance(rule, noise, L) for L in noise if L != rule.primary_cutoff}, "tests": results,
            "combination": combination, "final_groups": final, "warnings": warnings}
    (run_dir / "decision.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({**rows[["service_day", "cutoff_min", "stop_id_a", "stop_id_b", "label_fail"]],
                  **{f"p_{k}": v for k, v in preds.items()}}).to_parquet(run_dir / "predictions.parquet", index=False)
    write_report(run_dir / "report.md", info, results, vs_b3, daily)
    if publish:
        publish.parent.mkdir(parents=True, exist_ok=True)
        publish.write_text((run_dir / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    log(f"\nfeature groups: {', '.join(final)}")
    for r in results:
        log(f"  {r['group']:<10} {'keep' if r['keep'] else 'drop'}" + (f"  ({'; '.join(r['failed'])})" if r["failed"] else ""))
    log(f"report: {run_dir / 'report.md'}" + (f" (copied to {publish})" if publish else ""))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Feature selection with a pre-registered rule (e06, e07)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "feature_selection.toml")
    ap.add_argument("--training-config", type=Path, default=ROOT / "config" / "training.toml")
    ap.add_argument("--features-config", type=Path, default=ROOT / "config" / "features.toml")
    ap.add_argument("--features", type=Path, default=None, help="default: data/features/v{feature_version}")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--publish", type=Path, default=None,
                    help="copy of the report kept in git (default: [output] report of the config)")
    ap.add_argument("--allow-missing", action="store_true", help="run although days of the window are not built")
    args = ap.parse_args(argv)
    sc = load_selection_config(args.config)
    cfg = load_training_config(args.training_config)
    fcfg = load_feature_config(args.features_config)
    features_root = args.features or ROOT / "data" / "features" / f"v{fcfg.feature_version}"
    publish = args.publish or ROOT / sc.report
    try:
        shown = args.config.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(args.config)
    run(sc, cfg, features_root, args.out, publish, args.allow_missing, args.config.read_bytes(), config_path=shown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
