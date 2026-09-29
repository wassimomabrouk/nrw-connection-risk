"""Markdown report of one evaluation run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainingConfig

NAMES = {"B0": "B0 timetable", "B1": "B1 history", "B2": "B2 DB rule", "B3-lin": "B3-lin DB linear",
         "B3": "B3 DB (GBM)", "lr": "logistic regression", "gbm": "gradient boosting"}


def _f(x, digits=4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f}%"


def _ci(ci, pct=True) -> str:
    lo, hi = ci
    return f"[{_pct(lo)}, {_pct(hi)}]" if pct else f"[{lo:+.3f}, {hi:+.3f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_report(path: Path, info: dict, table: pd.DataFrame, comp: pd.DataFrame, days: pd.DataFrame,
                 rel: dict[str, pd.DataFrame], cfg: TrainingConfig) -> None:
    L, base = cfg.primary_cutoff, cfg.headline_baseline
    overall = table[table.slice == "all"]
    out = [f"# Evaluation run {info['run_id']}", ""]
    folds = info["folds"]
    eval_days = sorted({d for f in folds for d in f["evaluate"]})
    out += [f"- Mode: **{info['mode']}**, {len(folds)} fold(s), evaluated on {len(eval_days)} day(s) "
            f"({eval_days[0]} to {eval_days[-1]}), {info['rows']:,} rows",
            f"- Model features: {', '.join(info['features']['groups'])} ({len(info['feature_columns'])} columns); "
            f"calibration {cfg.calibration}; cutoffs {cfg.cutoff_mode}",
            f"- Code {info['git_commit']}, features built at {info['feature_build_commit']}", ""]
    for w in info["warnings"]:
        out.append(f"> **Warning:** {w}")
    if info["warnings"]:
        out.append("")

    # ---- headline
    out += [f"## Headline: {L} minutes before arrival, against {NAMES.get(base, base)}", ""]
    h = overall[overall.cutoff == L].set_index("model")
    c = comp[comp.cutoff == L].set_index("model") if len(comp) else pd.DataFrame()
    rows = []
    for m in cfg.run:
        r = h.loc[m]
        if m == "B2":
            rows.append([NAMES[m], "n/a", "", "n/a", "", _f(r.brier), f"precision {_f(r.precision, 3)}, "
                         f"recall {_f(r.recall, 3)}"])
            continue
        gain = ci = auc_d = ""
        if m in c.index:
            gain, ci = _pct(c.log_loss_gain[m]), _ci(c.log_loss_gain_ci[m])
            auc_d = f"{c.auc_diff[m]:+.3f} {_ci(c.auc_diff_ci[m], pct=False)}"
        rows.append([NAMES[m], _f(r.log_loss), f"{gain} {ci}".strip(), _f(r.auc, 3), auc_d, _f(r.brier),
                     _f(r.recall_at_db_precision, 3)])
    out += [_table(["model", "log loss", f"gain vs {base} [95% CI]", "AUC", f"AUC vs {base} [95% CI]",
                    "Brier", "recall at DB-rule precision"], rows), ""]
    out += ["Gain = relative reduction of log loss (positive = better than the baseline). Intervals: "
            f"{cfg.bootstrap_n} resamples of whole service days ({cfg.bootstrap_n_auc} for AUC).", ""]
    if info.get("selected_model"):
        out += [f"Selected by validation rule (best log loss at {L} min, simpler model within "
                f"{100 * cfg.selection_tie:.0f}%): **{NAMES[info['selected_model']]}**", ""]

    # ---- all cutoffs
    out += ["## All cutoffs", ""]
    rows = []
    for m in cfg.run:
        if m == "B2":
            continue
        cells = []
        for cut in sorted(overall.cutoff.unique(), reverse=True):
            r = overall[(overall.model == m) & (overall.cutoff == cut)].iloc[0]
            g = ""
            if len(comp) and m != base:
                cc = comp[(comp.model == m) & (comp.cutoff == cut)]
                g = f" ({_pct(cc.log_loss_gain.iloc[0])})" if len(cc) else ""
            cells.append(f"{_f(r.log_loss)}{g} / {_f(r.auc, 3)}")
        rows.append([NAMES[m]] + cells)
    out += [_table(["model"] + [f"{cut} min: log loss (gain) / AUC" for cut in sorted(overall.cutoff.unique(),
                                                                                    reverse=True)], rows), ""]

    # ---- decision layer
    d = info.get("decision")
    if d:
        db, mm = d["db_rule"], d.get("model_at_threshold")
        out += [f"## Decision layer ({d['cutoff']} min)", ""]
        rows = [["DB rule (B2)", "", _f(db["precision"], 3), _f(db["recall"], 3), _f(db["false_alarm_rate"], 3)]]
        if mm:
            rows.append([NAMES[d["model"]], _f(d["threshold"], 3), _f(mm["precision"], 3), _f(mm["recall"], 3),
                         _f(mm["false_alarm_rate"], 3)])
        out += [_table(["", "threshold", "precision", "share of failures flagged", "false-alarm rate"], rows), "",
                "The model's threshold is set to match the DB rule's precision.", ""]

    # ---- slices
    sel = info.get("selected_model")
    if sel and base in cfg.run:
        out += [f"## Slices at {L} min: {NAMES[sel]} vs {NAMES[base]}", ""]
        sl = table[(table.cutoff == L) & (table.slice != "all")]
        rows = []
        for (col, val), g in sl.groupby(["slice", "value"]):
            g = g.set_index("model")
            if sel in g.index and base in g.index:
                lb, lm = g.log_loss[base], g.log_loss[sel]
                rows.append([col, val, f"{int(g.n[base]):,}", _f(g.fail_rate[base], 3), _f(lb), _f(lm),
                             _pct(1 - lm / lb), _f(g.auc[base], 3), _f(g.auc[sel], 3)])
        out += [_table(["slice", "value", "rows", "fail rate", f"log loss {base}", f"log loss {sel}", "gain",
                        f"AUC {base}", f"AUC {sel}"], rows), ""]

    # ---- calibration
    if rel:
        out += [f"## Calibration at {L} min", ""]
        for m, r in rel.items():
            out += [f"**{NAMES[m]}**", "", _table(["predicted", "rows", "mean predicted", "observed"],
                                                  [[i, f"{int(x.n):,}", _f(x.mean_p, 3), _f(x.observed, 3)]
                                                   for i, x in r.iterrows()]), ""]

    # ---- per day
    if len(days):
        out += [f"## Log loss per evaluation day ({L} min)", ""]
        ms = [m for m in cfg.run if m != "B2"]
        out += [_table(["day", "rows", "fail rate"] + ms,
                       [[r["day"], f"{r['rows']:,}", _f(r["fail_rate"], 3)] + [_f(r[m]) for m in ms]
                        for _, r in days.iterrows()]), ""]
    path.write_text("\n".join(out), encoding="utf-8")
