"""Exploration 04: how good is DB's own prognosis at predicting failed connections?

This measures the headroom for a model before any model is built. For every
transfer candidate (DESIGN.md section 1) and prediction cutoffs of 60, 30 and 10
minutes before A's planned arrival, the script reconstructs what was known at the
cutoff: the latest prognosis (ct) and cancellation status (cs) of A and B observed
by the collector up to that moment (planned time if no change was known yet).
It then compares three scores against the label (DESIGN.md section 2):

  planned slack    timetable only, the naive baseline
  DB slack         predicted slack from DB's prognosis at the cutoff
  DB rule          DB slack below the minimum transfer time, or a known cancellation

Ranking quality is measured with ROC AUC (higher = better separation of failing
and holding connections), the DB rule with precision and recall.

Run from the repo root:
    python exploration/e04_db_prognosis_baseline.py --raw data/restore/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e02_transfer_candidates import HUBS, Report, build_candidates, load_plan  # noqa: E402
from e03_label_feasibility import BERLIN, segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
CUTOFFS = (60, 30, 10)


def load_history(raw_root: Path) -> pd.DataFrame:
    """Every realtime observation of every event: (key, obs time, ct, cs)."""
    rows = []
    for f in sorted(raw_root.rglob("*.jsonl.gz")):
        src = f.parent.parent.name.replace("source=", "")
        if src not in ("fchg", "rchg"):
            continue
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                lines = list(fh)
        except (EOFError, OSError):
            continue
        for line in lines:
            try:
                rec = json.loads(line)
                root = ET.fromstring(rec["body"])
            except Exception:
                continue
            obs = datetime.fromisoformat(rec["collected_at"]).astimezone(BERLIN).replace(tzinfo=None)
            for s in root.findall("s"):
                for ev in ("ar", "dp"):
                    e = s.find(ev)
                    if e is None or not (e.get("ct") or e.get("cs")):
                        continue
                    rows.append((f"{s.get('id')}|{ev}", obs, e.get("ct"), e.get("cs")))
    h = pd.DataFrame(rows, columns=["key", "obs", "ct", "cs"])
    if h.empty:
        raise SystemExit(f"No realtime observations under {raw_root}")
    h["ct"] = pd.to_datetime(h.ct, format="%y%m%d%H%M", errors="coerce")
    return h.sort_values("obs").reset_index(drop=True)


def state_at(queries: pd.DataFrame, hist: pd.DataFrame, key_col: str, time_col: str, prefix: str) -> pd.DataFrame:
    """Latest observation of each event at or before the query time (point-in-time join)."""
    # one time resolution on both sides (pandas refuses to merge ns with us)
    q = queries[[key_col, time_col]].reset_index()
    q[time_col] = q[time_col].astype("datetime64[ns]")
    q = q.sort_values(time_col)
    h = hist.rename(columns={"key": key_col})
    h["obs"] = h["obs"].astype("datetime64[ns]")
    m = pd.merge_asof(q, h, left_on=time_col, right_on="obs",
                      by=key_col, direction="backward")
    return m.set_index("index")[["ct", "cs"]].add_prefix(prefix)


def auc(score: pd.Series, y: pd.Series) -> float:
    """ROC AUC via the rank-sum formula; a higher score must mean higher risk."""
    r = score.rank(method="average")
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "restore" / "raw")
    ap.add_argument("--day", default="2026-09-24")
    ap.add_argument("--mtt", type=int, default=4)
    args = ap.parse_args()

    rep = Report()
    events, _ = load_plan(args.raw)
    t0 = pd.Timestamp(args.day) + pd.Timedelta(hours=4)
    c, _, _, _ = build_candidates(events, t0, t0 + pd.Timedelta(days=1), args.mtt, 30)
    hist = load_history(args.raw)
    data_start = hist.obs.min()
    c = c.reset_index(drop=True)
    c["key_a"] = c.stop_id_a + "|ar"
    c["key_b"] = c.stop_id_b + "|dp"

    # label from the final state (same definition as e03)
    c["t_end"] = pd.Timestamp("2100-01-01")
    fin = state_at(c, hist, "key_a", "t_end", "fa_").join(state_at(c, hist, "key_b", "t_end", "fb_"))
    c = c.join(fin)
    c["observed"] = (c.fa_ct.notna() | c.fa_cs.eq("c")) & (c.fb_ct.notna() | c.fb_cs.eq("c"))
    c["real_slack"] = (c.fb_ct - c.fa_ct).dt.total_seconds() / 60
    c["fail"] = c.fb_cs.eq("c") | c.fa_cs.eq("c") | (c.real_slack < args.mtt)
    c = c[c.observed].copy()

    rep.h("1. Sanity check against e03")
    rep.say(f"candidates observed: {len(c):,}, failure rate: {c.fail.mean() * 100:.2f} %  (must equal e03)")
    rep.say(f"collector data starts at {data_start} local; cutoffs before that are excluded")

    rows, seg_rows = [], []
    for L in CUTOFFS:
        c["t_cut"] = c.pt_a - pd.Timedelta(minutes=L)
        d = c[c.t_cut >= data_start].copy()
        d = d.join(state_at(d, hist, "key_a", "t_cut", "a_")).join(state_at(d, hist, "key_b", "t_cut", "b_"))
        d = d[~d.a_cs.eq("c")]                        # A known cancelled: no prediction needed
        a_pred = d.a_ct.fillna(d.pt_a)
        b_pred = d.b_ct.fillna(d.pt_b)
        d["db_slack"] = (b_pred - a_pred).dt.total_seconds() / 60
        d["db_rule"] = d.b_cs.eq("c") | (d.db_slack < args.mtt)
        y = d.fail
        tp = int((d.db_rule & y).sum())
        rows.append({
            "cutoff_min": L, "candidates": len(d), "fail_pct": round(y.mean() * 100, 2),
            "auc_planned_slack": round(auc(-d.slack, y), 3),
            "auc_db_slack": round(auc(-d.db_slack.where(~d.b_cs.eq("c"), -999), y), 3),
            "db_rule_precision": round(tp / max(int(d.db_rule.sum()), 1), 3),
            "db_rule_recall": round(tp / max(int(y.sum()), 1), 3),
            "known_B_cancel_pct": round(d.b_cs.eq("c").mean() * 100, 2),
            "mae_a_arrival_min": round(((a_pred - d.fa_ct).dt.total_seconds().abs() / 60).mean(), 2),
        })
        if L == 30:
            d["segment"] = d.cat_a.map(segment) + " -> " + d.cat_b.map(segment)
            d["hub"] = d.eva.map(HUBS)
            for col in ("segment", "hub"):
                for k, g in d.groupby(col):
                    if len(g) < 200:
                        continue
                    seg_rows.append({"by": col, "group": k, "candidates": len(g),
                                     "fail_pct": round(g.fail.mean() * 100, 1),
                                     "auc_planned": round(auc(-g.slack, g.fail), 3),
                                     "auc_db": round(auc(-g.db_slack.where(~g.b_cs.eq("c"), -999), g.fail), 3)})

    rep.h("2. Planned slack vs DB prognosis, by prediction cutoff")
    rep.table(pd.DataFrame(rows))
    rep.say("AUC 0.5 = no better than chance, 1.0 = perfect ranking of failing before holding connections.")
    rep.say("mae_a_arrival_min: mean absolute error of DB's arrival prognosis for A at the cutoff vs the final time.")

    rep.h("3. At the 30-minute cutoff, by segment and hub (groups with 200+ candidates)")
    rep.table(pd.DataFrame(seg_rows))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "e04_db_prognosis_baseline.txt"
    p.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {p}")


if __name__ == "__main__":
    main()
