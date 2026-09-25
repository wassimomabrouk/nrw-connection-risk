"""Exploration 02: from raw arrival/departure pairs to plausible transfers.

DB's API delivers no managed connections (<conn> never occurs, see e01), so
transfers must be defined from the timetable. This script builds all pairs
(arrival A, departure B) at each hub within a time window and applies candidate
exclusion rules one by one, reporting how many pairs each rule removes, with
examples, so every rule can be judged before it is fixed in DESIGN.md.

Rules (applied in this order):
  R1 bus           A or B is a bus (decision D5)
  R2 same trip     B is the same run as A (the train simply continues)
  R3 transition    B is the run A turns into (tra attribute)
  R4 wings         A and B are coupled parts of one train (wings attribute)
  R5 backtrack     B next calls at the station A came from directly before the hub
  R6 redundant     A continues and B serves no station that A does not serve too
  R7 first-reach   B is not the earliest departure (with at least the minimum
                   transfer slack) to any station that A has not already served or
                   will not serve itself. A journey planner would never suggest it.

Also prints examples of stop-level messages of type "c" (44k in e01), whose
meaning must be checked before they are used.

Run from the repo root:
    python exploration/e02_transfer_candidates.py --raw data/restore/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
HUBS = {"8000207": "Köln Hbf", "8000085": "Düsseldorf Hbf", "8000086": "Duisburg Hbf",
        "8000098": "Essen Hbf", "8000001": "Aachen Hbf"}
TRIP_RE = re.compile(r"-\d+$")


def trip_key(stop_id) -> str | None:
    return TRIP_RE.sub("", stop_id) if isinstance(stop_id, str) and stop_id else None


def t_local(raw: str | None):
    return datetime.strptime(raw, "%y%m%d%H%M") if raw else None


def path(v) -> list[str]:
    """Missing values arrive as None or NaN (float) after pandas joins."""
    return [x for x in v.split("|") if x] if isinstance(v, str) and v else []


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def say(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def h(self, title):
        self.say("")
        self.say(f"== {title} ==")

    def table(self, df):
        self.say(df.to_string(index=False) if len(df) else "(empty)")


def load_plan(raw_root: Path) -> tuple[pd.DataFrame, list[str]]:  # noqa: C901
    """One row per (stop_id, event) from plan responses; last seen version wins."""
    rows: dict[tuple, dict] = {}
    c_examples: list[str] = []
    with_ct: set = set()
    for f in sorted(raw_root.rglob("*.jsonl.gz")):
        src = f.parent.parent.name.replace("source=", "")
        if src not in ("plan", "fchg"):
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
            if src == "fchg":
                for s_ in root.findall("s"):
                    for ev in ("ar", "dp"):
                        e = s_.find(ev)
                        if e is not None and e.get("ct"):
                            with_ct.add((s_.get("id"), ev))
                if len(c_examples) < 5:
                    for s in root.findall("s"):
                        if any(m.get("t") == "c" for m in s.findall("m")) and len(c_examples) < 5:
                            c_examples.append(ET.tostring(s, encoding="unicode")[:1200])
                continue
            eva = rec["eva"]
            for s in root.findall("s"):
                sid = s.get("id")
                tl = s.find("tl")
                for ev in ("ar", "dp"):
                    e = s.find(ev)
                    if e is None or not e.get("pt"):
                        continue
                    rows[(sid, ev)] = {
                        "eva": eva, "stop_id": sid, "trip": trip_key(sid), "event": ev,
                        "pt": t_local(e.get("pt")), "pp": e.get("pp"), "line": e.get("l") or e.get("fb"),
                        "path": e.get("ppth"), "wings": e.get("wings"), "tra": e.get("tra"),
                        "hidden": e.get("hi"),
                        "cat": tl.get("c") if tl is not None else None,
                        "num": tl.get("n") if tl is not None else None,
                        "flag": tl.get("f") if tl is not None else None,
                    }
    df = pd.DataFrame(rows.values())
    df["has_ct"] = [(a, b) in with_ct for a, b in zip(df.stop_id, df.event)]
    return df, c_examples


def build_candidates(df: pd.DataFrame, t0, t1, min_slack: int = 4, max_slack: int = 30):
    """Apply the transfer-candidate rules (DESIGN.md section 1) to planned events.

    Returns (candidates, arrivals, funnel rows, examples of removed pairs)."""
    dep_of_stop = df[df.event == "dp"].set_index("stop_id")["path"]
    A = df[(df.event == "ar") & (df.pt >= t0) & (df.pt < t1)].copy()
    B = df[df.event == "dp"].copy()
    A["cont_path"] = A.stop_id.map(dep_of_stop)

    # bucketed join: a departure within 30 min of an arrival lies in the same or next hour
    assert max_slack <= 60
    A["hkey"] = A.pt.dt.floor("h")
    A2 = pd.concat([A, A.assign(hkey=A.hkey + pd.Timedelta(hours=1))], ignore_index=True)
    B["hkey"] = B.pt.dt.floor("h")
    pairs = A2.merge(B, on=["eva", "hkey"], suffixes=("_a", "_b"))
    pairs["slack"] = (pairs.pt_b - pairs.pt_a).dt.total_seconds() / 60
    pairs = pairs[(pairs.slack >= min_slack) & (pairs.slack <= max_slack)].copy()

    def backtrack(r) -> bool:
        prev = path(r.path_a)
        return bool(prev) and prev[-1] in set(path(r.path_b))

    def redundant(r) -> bool:
        cont = set(path(r.cont_path))
        nxt = set(path(r.path_b))
        return bool(cont) and bool(nxt) and nxt <= cont

    def wings(r) -> bool:
        return (r.trip_b in set(path(r.wings_a))) or (r.trip_a in set(path(r.wings_b)))

    def transition(r) -> bool:
        return isinstance(r.tra_a, str) and (r.tra_a == r.stop_id_b or trip_key(r.tra_a) == r.trip_b)

    rules = [
        ("R1 bus", lambda r: r.cat_a == "Bus" or r.cat_b == "Bus"),
        ("R2 same trip", lambda r: r.trip_a == r.trip_b),
        ("R3 transition", transition),
        ("R4 wings", wings),
        ("R5 backtrack", backtrack),
        ("R6 redundant", redundant),
    ]
    funnel, examples = [], {}
    keep = pairs
    funnel.append({"step": "all pairs in window", **per_hub(keep)})
    for name, fn in rules:
        mask = keep.apply(fn, axis=1) if len(keep) else pd.Series(dtype=bool)
        examples[name] = keep[mask].sample(min(4, int(mask.sum())), random_state=0) if mask.any() else keep.head(0)
        keep = keep[~mask] if len(keep) else keep
        funnel.append({"step": f"after {name}", **per_hub(keep)})

    # R7 first reach: per arrival, walk departures in time order; keep B only if it is
    # the first to reach some station that A has neither passed nor will serve itself
    keep_idx, drop_idx = [], []
    for _, g in keep.sort_values("pt_b").groupby("stop_id_a", sort=False):
        first = g.iloc[0]
        covered = set(path(first.path_a)) | set(path(first.cont_path))
        for idx, r in g.iterrows():
            new = set(path(r.path_b)) - covered
            if new:
                keep_idx.append(idx)
                covered |= new
            else:
                drop_idx.append(idx)
    examples["R7 first-reach"] = keep.loc[drop_idx].sample(min(4, len(drop_idx)), random_state=0) if drop_idx else keep.head(0)
    keep = keep.loc[keep_idx]
    funnel.append({"step": "after R7 first-reach", **per_hub(keep)})
    return keep, A, funnel, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "restore" / "raw")
    ap.add_argument("--min-slack", type=int, default=4,
                    help="minimum planned transfer time in minutes (sensitivity: 3, 5, 7)")
    ap.add_argument("--day", default="2026-09-24",
                    help="service day to analyse: arrivals from 04:00 to 04:00 next day, local time")
    ap.add_argument("--max-slack", type=int, default=30)
    args = ap.parse_args()

    df, c_examples = load_plan(args.raw)
    rep = Report()
    rep.h("1. Planned events from plan responses")
    rep.say(f"rows: {len(df):,}, stops: {df.stop_id.nunique():,}, "
            f"planned times {df.pt.min()} to {df.pt.max()} (local)")
    cnt = df.groupby(["eva", "event"]).size().unstack(fill_value=0).reset_index()
    cnt.insert(1, "hub", cnt.eva.map(HUBS))
    rep.table(cnt)
    rep.say("Train categories (share of events):")
    rep.table(df.cat.value_counts(normalize=True).mul(100).round(1).rename("pct").reset_index().head(15))

    rep.h("1b. Realtime coverage: share of planned events that ever received a prognosis (ct) in fchg")
    t0 = pd.Timestamp(args.day) + pd.Timedelta(hours=4)
    t1 = t0 + pd.Timedelta(days=1)
    day_df = df[(df.pt >= t0) & (df.pt < t1)]
    cov = day_df.assign(is_bus=day_df.cat.eq("Bus")).groupby("is_bus").has_ct.mean().mul(100).round(1)
    rep.table(cov.rename("pct_with_ct").reset_index())

    keep, A, funnel, examples = build_candidates(df, t0, t1, args.min_slack, args.max_slack)
    rep.h(f"2. Funnel for service day {args.day} (window {args.min_slack}-{args.max_slack} min)")
    rep.table(pd.DataFrame(funnel))
    rep.say(f"Arrivals with at least one remaining candidate: {keep.stop_id_a.nunique():,} of {A.stop_id.nunique():,}")
    rep.say(f"Candidates per such arrival: median {keep.groupby('stop_id_a').size().median():.0f}")

    rep.h("3. Remaining candidates: slack distribution (minutes)")
    rep.table(keep.slack.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).rename("slack").reset_index())

    rep.h("4. Remaining candidates: category combinations (A -> B), share %")
    combo = (keep.cat_a.fillna("?") + " -> " + keep.cat_b.fillna("?")).value_counts(normalize=True).mul(100).round(1)
    rep.table(combo.rename("pct").reset_index().head(15))

    rep.h("5. Remaining candidates: most common line pairs per hub")
    for eva, hub in HUBS.items():
        k = keep[keep.eva == eva]
        if not len(k):
            continue
        top = (k.line_a.fillna("?") + " -> " + k.line_b.fillna("?")).value_counts().head(6)
        rep.say(f"{hub}: " + ", ".join(f"{p} ({n})" for p, n in top.items()))

    cols = ["eva", "pt_a", "cat_a", "line_a", "path_a", "pt_b", "cat_b", "line_b", "path_b", "slack"]
    rep.h("6. Examples removed by each rule (paths shortened)")
    for name, ex in examples.items():
        rep.say(f"--- {name}")
        rep.table(shorten(ex[cols]) if len(ex) else ex)
    rep.say("--- kept (random)")
    rep.table(shorten(keep.sample(min(6, len(keep)), random_state=1)[cols]) if len(keep) else keep)

    rep.h("7. Stop-level messages of type c (raw examples)")
    for i, x in enumerate(c_examples, 1):
        rep.say(f"--- example {i}")
        rep.say(x)
    if not c_examples:
        rep.say("none found")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "e02_transfer_candidates.txt"
    p.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {p}")


def per_hub(df: pd.DataFrame) -> dict:
    out = {HUBS[e]: int((df.eva == e).sum()) for e in HUBS}
    out["total"] = len(df)
    return out


def shorten(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["eva"] = d.eva.map(HUBS)
    for c in ("path_a", "path_b"):
        d[c] = d[c].fillna("").map(lambda v: (v.split("|")[-1] + " <-" if c == "path_a" else "-> " + "|".join(v.split("|")[:2]))
                                   if v else "")
    for c in ("pt_a", "pt_b"):
        d[c] = d[c].dt.strftime("%d %H:%M")
    return d


if __name__ == "__main__":
    main()
