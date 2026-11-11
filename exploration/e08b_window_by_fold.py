"""e08b: where do the shorter training windows of the tuning run (e08) lose?

A shorter window was expected to help in the folds after the construction period around
Essen (DESIGN.md section 4); pooled over all evaluation days it was worse. This reads the
daily gains the run stored for every candidate and shows them per fold and per day, at the
30-minute cutoff. For information: it decides nothing, the rule was applied in e08.
Output: exploration/out/e08b_window_by_fold.txt

A window candidate is identical to "all days" in a fold whose fit has no more days than the
window; those days show a gain of exactly zero and count as "same".
Per-fold figures are means of the daily gains (every day weighs the same), so they differ
slightly from the pooled gains of the e08 report, which weigh every row the same.

Reads runs/<run>/decision.json (not in the repository). Run from the repo root:
    py exploration/e08b_window_by_fold.py runs/20261111-025102-e08/decision.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

MODELS = ("gbm", "B3", "lr")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def daily(result: dict, cutoff: str) -> pd.Series:
    """Gain per service day in percent (positive = better than the current settings)."""
    return pd.Series(result["result"][cutoff]["per_day"]).sort_index() * 100


def same_learner(result: dict, now: dict) -> bool:
    return all(now.get(k) == v for k, v in result["params"].items())


def fold_cell(g: pd.Series) -> str:
    if (g == 0).all():
        return "same"
    return f"{g.mean():+.2f} ({int((g > 0).sum())}/{len(g)})"


def main(path: str) -> None:
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    P = str(info["rule"]["primary_cutoff"])
    folds = info["folds"]
    fold_of = {d: i + 1 for i, f in enumerate(folds) for d in f["evaluate"]}
    print(f"e08b: shorter training windows per fold, run {info['run_id']} (code {info['git_commit']}), "
          f"cutoff {P} min")
    print("Gain = reduction of log loss against the current settings, in percent (positive = better).\n")

    print("1. Folds")
    for i, f in enumerate(folds, 1):
        print(f"   fold {i}: fit {f['fit'][0]} to {f['fit'][-1]} ({len(f['fit'])} days), "
              f"evaluate {f['evaluate'][0]} to {f['evaluate'][-1]} ({len(f['evaluate'])} days)")

    print("\n2. Every candidate per fold: mean daily gain (days better / days in the fold)")
    for m in MODELS:
        rows = {}
        for r in info["models"][m]["results"]:
            g = daily(r, P)
            fold = g.index.map(fold_of)
            rows[r["candidate"]] = {**{f"fold {k}": fold_cell(g[fold == k]) for k in sorted(set(fold))},
                                    "all folds": f"{g.mean():+.2f} ({int((g > 0).sum())}/{len(g)})",
                                    "pooled (e08)": f"{100 * r['result'][P]['gain']:+.2f}"}
        print(f"\n{m}")
        print(pd.DataFrame(rows).T.to_string())

    print("\n3. The windows with the current learner settings, per day")
    cols = {}
    for m in MODELS:
        now = info["current"][m]["settings"]
        for r in info["models"][m]["results"]:
            if r["window"] and same_learner(r, now):
                cols[f"{m} {r['window']} d"] = daily(r, P)
    t = pd.DataFrame(cols)
    t.insert(0, "weekday", pd.to_datetime(t.index).strftime("%a"))
    t.insert(0, "fold", t.index.map(fold_of))
    print(t.to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main(sys.argv[1])
