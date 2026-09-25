"""Exploration 03: can the label be observed, and how often do connections fail?

For the transfer candidates of one service day (DESIGN.md section 1), this script
reconstructs the actual arrival and departure times from the realtime responses
(fchg, rchg) and computes the label of DESIGN.md section 2:
  a connection fails if B is cancelled, A is cancelled, or the realised slack
  actual_dep(B) - actual_arr(A) is below the minimum transfer time.

Actual time = the prognosis (ct) of the last observation of that event. The script
checks that this last observation was made after the event (otherwise it would be a
stale prognosis, the problem found in the historical dataset).

Run from the repo root:
    python exploration/e03_label_feasibility.py --raw data/restore/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e02_transfer_candidates import HUBS, Report, build_candidates, load_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
BERLIN = ZoneInfo("Europe/Berlin")
LONG_DISTANCE = {"ICE", "IC", "EC", "ECE", "EUR", "FLX", "NJ", "TGV", "RJ", "RJX", "EN", "ES", "WB", "TRI"}


def segment(cat) -> str:
    """Train category -> segment. The tl@f flag is missing for a third of trains, so it is not used."""
    if not isinstance(cat, str):
        return "unknown"
    if cat == "S":
        return "S-Bahn"
    return "long-distance" if cat in LONG_DISTANCE else "regional"


def load_realtime(raw_root: Path) -> pd.DataFrame:
    """Last observation per (stop_id, event) from fchg and rchg, plus observation counts."""
    last: dict[tuple, dict] = {}
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
                    key = (s.get("id"), ev)
                    prev = last.get(key)
                    n = prev["n_obs"] + 1 if prev else 1
                    if prev is None or obs >= prev["obs"]:
                        last[key] = {"stop_id": key[0], "event": ev, "obs": obs, "n_obs": n,
                                     "ct": e.get("ct"), "cs": e.get("cs"),
                                     "first_obs": prev["first_obs"] if prev else obs}
                    else:
                        prev["n_obs"] = n
    rt = pd.DataFrame(last.values())
    if rt.empty:
        raise SystemExit(f"No realtime observations (fchg/rchg) under {raw_root}")
    rt["ct"] = pd.to_datetime(rt.ct, format="%y%m%d%H%M", errors="coerce")
    return rt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "restore" / "raw")
    ap.add_argument("--day", default="2026-09-24")
    ap.add_argument("--mtt", type=int, default=4, help="minimum transfer time in minutes")
    args = ap.parse_args()

    rep = Report()
    events, _ = load_plan(args.raw)
    t0 = pd.Timestamp(args.day) + pd.Timedelta(hours=4)
    cands, _, _, _ = build_candidates(events, t0, t0 + pd.Timedelta(days=1), args.mtt, 30)
    rt = load_realtime(args.raw)

    a = rt[rt.event == "ar"].add_suffix("_ra").rename(columns={"stop_id_ra": "stop_id_a"})
    b = rt[rt.event == "dp"].add_suffix("_rb").rename(columns={"stop_id_rb": "stop_id_b"})
    c = cands.merge(a, on="stop_id_a", how="left").merge(b, on="stop_id_b", how="left")

    c["a_cancel"] = c.cs_ra.eq("c")
    c["b_cancel"] = c.cs_rb.eq("c")
    c["observed"] = (c.ct_ra.notna() | c.a_cancel) & (c.ct_rb.notna() | c.b_cancel)
    c["delay_a"] = (c.ct_ra - c.pt_a).dt.total_seconds() / 60
    c["delay_b"] = (c.ct_rb - c.pt_b).dt.total_seconds() / 60
    c["real_slack"] = (c.ct_rb - c.ct_ra).dt.total_seconds() / 60
    c["stale_a"] = c.obs_ra < c.ct_ra
    c["stale_b"] = c.obs_rb < c.ct_rb

    def label(df: pd.DataFrame, mtt: int) -> pd.Series:
        return df.b_cancel | df.a_cancel | (df.real_slack < mtt)

    rep.h(f"1. Observability (service day {args.day}, {len(c):,} candidates)")
    rep.say(f"candidates with both events observed: {c.observed.mean() * 100:.1f} %")
    o = c[c.observed].copy()
    rep.say(f"last observation before the event (stale): arrivals {o.stale_a.mean() * 100:.2f} %, "
            f"departures {o.stale_b.mean() * 100:.2f} %")
    rep.say(f"observations per event, median: arrivals {o.n_obs_ra.median():.0f}, departures {o.n_obs_rb.median():.0f}")

    o["fail"] = label(o, args.mtt)
    o["fail_delay"] = ~o.a_cancel & ~o.b_cancel & (o.real_slack < args.mtt)
    rep.h(f"2. Label (minimum transfer time {args.mtt} min)")
    rep.say(f"connection fails: {o.fail.mean() * 100:.2f} %  "
            f"(by delay {o.fail_delay.mean() * 100:.2f} %, B cancelled {(o.b_cancel & ~o.a_cancel).mean() * 100:.2f} %, "
            f"A cancelled {o.a_cancel.mean() * 100:.2f} %)")
    rep.say("Sensitivity to the minimum transfer time (failure rate %):")
    rep.table(pd.DataFrame([{"mtt": m, "fail_pct": round(label(o, m).mean() * 100, 2)} for m in (3, 4, 5, 7)]))

    def by(col, name, order=None):
        g = o.groupby(col, observed=True).agg(candidates=("fail", "size"), fail_pct=("fail", "mean"),
                                              delay_fail_pct=("fail_delay", "mean"))
        g[["fail_pct", "delay_fail_pct"]] = (g[["fail_pct", "delay_fail_pct"]] * 100).round(2)
        g = g.reset_index().rename(columns={col: name})
        if order:
            g = g.set_index(name).reindex(order).dropna(how="all").reset_index()
        rep.table(g)

    rep.h("3. Failure rate by planned slack")
    o["slack_bucket"] = pd.cut(o.slack, [3.9, 5, 7, 9, 14, 19, 30],
                               labels=["4-5", "6-7", "8-9", "10-14", "15-19", "20-30"])
    by("slack_bucket", "planned_slack")

    rep.h("4. Failure rate by hub")
    o["hub"] = o.eva.map(HUBS)
    by("hub", "hub")

    rep.h("5. Failure rate by segment (A -> B)")
    o["segment"] = o.cat_a.map(segment) + " -> " + o.cat_b.map(segment)
    by("segment", "segment")

    rep.h("6. Failure rate by hour of planned arrival")
    o["hour"] = o.pt_a.dt.hour
    by("hour", "hour")

    rep.h("7. Delays (minutes, observed and not cancelled)")
    d = o[~o.a_cancel & ~o.b_cancel]
    rep.table(pd.DataFrame({
        "arrival_delay_A": d.drop_duplicates("stop_id_a").delay_a.describe(percentiles=[.5, .75, .9, .95, .99]),
        "departure_delay_B": d.drop_duplicates("stop_id_b").delay_b.describe(percentiles=[.5, .75, .9, .95, .99]),
    }).round(1).reset_index())

    rep.h("8. Connections that held although A was later than the planned slack allows")
    late = d[d.delay_a > d.slack - args.mtt]
    rep.say(f"candidates where A's delay alone would break the connection: {len(late):,}; "
            f"held because B was late too: {(~late.fail).mean() * 100:.1f} %")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "e03_label_feasibility.txt"
    p.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {p}")


if __name__ == "__main__":
    main()
