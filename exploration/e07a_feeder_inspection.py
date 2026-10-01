"""e07a: what the feeder stations can tell us, before any feeder feature is designed.

Read-only. Uses the collector's parsed layer for one service day and answers:
  1. Volume: how much is collected at each feeder, and how often.
  2. Coverage: which share of hub arrivals pass a feeder before the hub (planned path).
  3. Join: can such an arrival be found at the feeder (same run id)?
  4. Lead time: how long before the hub arrival does the train pass the feeder, and has it
     passed it by the 60, 30 and 10-minute cutoffs?
  5. Information: at the cutoff, does the train's delay seen at the feeder say anything about
     its final delay at the hub that DB's own hub prognosis does not already say?
  6. Corridor traffic: how many trains a corridor-state feature at the feeder could average over.

Only data before the e07 evaluation days (from 2026-10-07) is used, and no connection labels.

Run on the server from the repo root:
    .venv/bin/python exploration/e07a_feeder_inspection.py --day 2026-09-30
"""
from __future__ import annotations

import argparse
import difflib
import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nrw_connection_risk.dataset.load import load_window
from nrw_connection_risk.dataset.state import final_state, state_at
from nrw_connection_risk.dataset.timeutil import service_day_bounds

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (60, 30, 10)
LAST_ALLOWED = date(2026, 10, 6)          # e07 evaluates from 2026-10-07


def stations(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with open(path, "rb") as f:
        st = tomllib.load(f)["stations"]
    hubs = {str(s["eva"]): s["name"] for s in st if s.get("role", "hub") == "hub"}
    feeders = {str(s["eva"]): s["name"] for s in st if s.get("role") == "feeder"}
    return hubs, feeders


def pct(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}%"


def q(s: pd.Series, qs=(0.1, 0.5, 0.9)) -> str:
    s = s.dropna()
    return "n/a" if s.empty else " / ".join(f"{s.quantile(v):.0f}" for v in qs)


def inspect(parsed: Path, day: date, hubs: dict[str, str], feeders: dict[str, str], out=print) -> dict:
    t0, t1 = service_day_bounds(day)
    w = load_window(parsed, t0 - pd.Timedelta(hours=6), t1 + pd.Timedelta(hours=6), 2,
                    stations=list(hubs) + list(feeders))
    plan, obs, polls = w.plan, w.obs, w.polls
    res: dict = {}
    out(f"e07a feeder inspection, service day {day} ({t0} to {t1} UTC), data {w.data_start} to {w.data_end}")
    out(f"{len(feeders)} feeders, {len(hubs)} hubs\n")

    # ---- 1. volume
    out("1. Volume per feeder (planned stops in the window, runs, realtime observations, median gap between polls)")
    vol = []
    for eva, name in feeders.items():
        p, o = plan[plan.eva == eva], obs[obs.eva == eva]
        t = polls.loc[polls.eva == eva, "t"].sort_values()
        gap = t.diff().dt.total_seconds().median() if len(t) > 1 else np.nan
        vol.append((name, len(p), p.trip.nunique(), len(o), gap))
        out(f"   {name:<24} stops {len(p):>6,}  runs {p.trip.nunique():>5,}  obs {len(o):>7,}  poll gap {gap:>5.0f} s")
    res["volume"] = vol
    if not plan.eva.isin(list(feeders)).any():
        out("\nNo feeder data in this window. Is the collector v1.1 running for this day?")
        return res

    # ---- hub arrivals of the service day (trains only, like rule T1)
    A = plan[plan.eva.isin(list(hubs)) & (plan.event == "ar") & (plan.pt >= t0) & (plan.pt < t1)
             & ~plan.cat.fillna("").str.lower().str.startswith("bus")].copy()
    A["hub"] = A.eva.map(hubs)
    name_to_eva = {n: e for e, n in feeders.items()}

    def on_path(path):
        names = path.split("|") if isinstance(path, str) else []
        return [name_to_eva[n] for n in names if n in name_to_eva]

    names_in_paths = {n for p in plan.path.dropna() for n in p.split("|")}
    for eva, name in feeders.items():
        if name not in names_in_paths:
            close = difflib.get_close_matches(name, names_in_paths, n=3, cutoff=0.6)
            out(f"   WARNING: feeder name {name!r} never appears in a planned path (similar: {close})")
    A["feeders_on_path"] = A.path.map(on_path)
    A["n_on_path"] = A.feeders_on_path.map(len)

    # ---- 2. coverage
    out(f"\n2. Coverage: {len(A):,} train arrivals at the hubs")
    share = (A.n_on_path > 0).mean()
    out(f"   pass at least one feeder before the hub (planned path): {pct(share)}")
    for hub, g in A.groupby("hub"):
        out(f"   {hub:<24} {pct((g.n_on_path > 0).mean()):>7} of {len(g):,}")
    res["coverage"] = share
    D = plan[plan.eva.isin(list(hubs)) & (plan.event == "dp") & (plan.pt >= t0) & (plan.pt < t1)
             & ~plan.cat.fillna("").str.lower().str.startswith("bus")]
    inbound = plan[(plan.event == "ar") & plan.eva.isin(list(hubs))].drop_duplicates("stop_id").set_index("stop_id").path
    d_cov = np.array([len(on_path(p)) > 0 for p in D.stop_id.map(inbound)], dtype=bool)
    res["coverage_b"] = float(d_cov.mean()) if len(D) else np.nan
    out(f"   departures (train B side): {pct(res['coverage_b'])} of {len(D):,} come through a feeder before the hub")

    # ---- 3. join: the same run at the feeder (departure if any, else arrival)
    F = plan[plan.eva.isin(list(feeders))].copy()
    F["rank"] = F.event.map({"dp": 0, "ar": 1})
    F = F.sort_values("rank").drop_duplicates(["trip", "eva"])
    pairs = A[A.n_on_path > 0].explode("feeders_on_path").rename(columns={"feeders_on_path": "f_eva"})
    pairs = pairs.merge(F[["trip", "eva", "stop_id", "event", "pt"]].rename(
        columns={"eva": "f_eva", "stop_id": "f_stop", "event": "f_event", "pt": "f_pt"}), on=["trip", "f_eva"], how="left")
    matched = pairs.f_pt.notna()
    out(f"\n3. Join: {len(pairs):,} (arrival, feeder on its path) pairs")
    out(f"   found at the feeder with the same run id: {pct(matched.mean())}")
    pairs = pairs[matched].copy()
    pairs["lead_min"] = (pairs.pt - pairs.f_pt).dt.total_seconds() / 60
    bad = (pairs.lead_min <= 0) | (pairs.lead_min > 240)
    out(f"   implausible times (feeder not 0 to 240 min before the hub): {pct(bad.mean())}")
    pairs = pairs[~bad]
    res["join"] = float(matched.mean())

    # ---- 4. lead time: the last feeder passed before the hub
    last = pairs.sort_values("lead_min").drop_duplicates("stop_id")      # closest feeder to the hub
    out(f"\n4. Lead time, closest feeder before the hub (minutes, p10 / median / p90): {q(last.lead_min)}")
    for L in CUTOFFS:
        passed = pairs[pairs.lead_min >= L].sort_values("lead_min").drop_duplicates("stop_id")
        out(f"   at the {L:>2}-min cutoff, already passed a feeder (planned): {pct(len(passed) / len(A))} of all "
            f"arrivals, lead {q(passed.lead_min)} min")
        res[f"passed_{L}"] = len(passed) / len(A)

    # ---- 5. information at the cutoff: delay seen at the feeder vs DB's hub prognosis
    out("\n5. Information at the cutoff (arrivals that passed a feeder before it; delays in minutes)")
    for L in CUTOFFS:
        d = pairs[pairs.lead_min >= L].sort_values("lead_min").drop_duplicates("stop_id").copy()
        if d.empty:
            continue
        t_cut = d.pt - pd.Timedelta(minutes=L)
        hub = state_at(d.stop_id + "|ar", t_cut, obs)
        fed = state_at(d.f_stop + "|" + d.f_event, t_cut, obs)
        end = final_state(d.stop_id + "|ar", obs)
        d["db"] = ((hub.ct - d.pt).dt.total_seconds() / 60).fillna(0)          # no change known = on time
        d["feeder"] = (fed.ct - d.f_pt).dt.total_seconds() / 60
        d["final"] = (end.ct - d.pt).dt.total_seconds() / 60
        d = d[d.final.notna() & (end.cs != "c")]
        seen = d.feeder.notna()
        dd = d[seen]
        if len(dd) < 30:
            out(f"   {L:>2} min: only {len(dd)} usable arrivals")
            continue
        err_db, err_f = (dd.final - dd.db).abs(), (dd.final - dd.feeder).abs()
        resid, gap = dd.final - dd.db, dd.feeder - dd.db
        corr = np.corrcoef(resid, gap)[0, 1] if gap.std() > 0 and resid.std() > 0 else np.nan
        out(f"   {L:>2} min: {len(d):,} arrivals, feeder realtime value known for {pct(seen.mean())}")
        out(f"          error of DB's hub prognosis {err_db.mean():.2f}, of the feeder delay {err_f.mean():.2f} (mean abs)")
        out(f"          feeder and DB differ by 2+ min in {pct((gap.abs() >= 2).mean())}")
        out(f"          corr(DB's error, feeder minus DB) = {corr:+.3f}   (0: DB already uses it; > 0: extra signal)")
        res[f"info_{L}"] = {"n": int(len(dd)), "mae_db": float(err_db.mean()), "mae_feeder": float(err_f.mean()),
                            "corr": float(corr)}

    # ---- 6. corridor traffic at the feeder in the 30 minutes before the 30-min cutoff
    Ff = F.copy()
    counts = []
    for eva in feeders:
        times = np.sort(Ff.loc[Ff.eva == eva, "pt"].to_numpy())
        sub = last[last.f_eva == eva]
        t = (sub.pt - pd.Timedelta(minutes=30)).to_numpy()
        counts.append(np.searchsorted(times, t, side="right") - np.searchsorted(times, t - np.timedelta64(30, "m")))
    c = pd.Series(np.concatenate(counts)) if counts else pd.Series(dtype=float)
    out(f"\n6. Trains at the arrival's closest feeder in the 30 min before the 30-min cutoff (p10 / median / p90): {q(c)}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="e07a: inspect the feeder data (read-only)")
    ap.add_argument("--day", type=date.fromisoformat, default=date(2026, 9, 30))
    ap.add_argument("--parsed", type=Path, default=ROOT / "data" / "collector" / "parsed")
    ap.add_argument("--stations", type=Path, default=ROOT / "config" / "collector.toml")
    args = ap.parse_args(argv)
    if args.day > LAST_ALLOWED:
        raise SystemExit(f"e07a only looks at days up to {LAST_ALLOWED}: later days are e07's evaluation data")
    hubs, feeders = stations(args.stations)
    inspect(args.parsed, args.day, hubs, feeders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
