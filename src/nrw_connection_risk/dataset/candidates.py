"""Transfer candidates at the hubs (DESIGN.md section 1, rules T1 to T7).

Ported from exploration/e02_transfer_candidates.py. Two deliberate differences:
times are naive UTC instead of local time, and T7 evaluates departures at the same
minute together (each is kept if it reaches a station no earlier departure reaches),
so the result does not depend on row order.
"""
from __future__ import annotations

import re

import pandas as pd

_TRIP_RE = re.compile(r"-\d+$")
LONG_DISTANCE = {"ICE", "IC", "EC", "ECE", "EUR", "FLX", "NJ", "TGV", "RJ", "RJX", "EN", "ES", "WB", "TRI"}


def trip_key(stop_id) -> str | None:
    return _TRIP_RE.sub("", stop_id) if isinstance(stop_id, str) and stop_id else None


def path(v) -> list[str]:
    """'A|B|C' -> ['A', 'B', 'C']; missing values (None or NaN after joins) -> []."""
    return [x for x in v.split("|") if x] if isinstance(v, str) and v else []


def segment(cat) -> str:
    """Train category -> segment. The tl@f flag is missing for a third of trains, so it is not used."""
    if not isinstance(cat, str):
        return "unknown"
    if cat == "S":
        return "S-Bahn"
    return "long-distance" if cat in LONG_DISTANCE else "regional"


def _count(df: pd.DataFrame, hubs: dict[str, str], step: str) -> dict:
    return {"step": step, **{name: int((df.eva == eva).sum()) for eva, name in hubs.items()}, "total": len(df)}


def build_candidates(plan: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, hubs: dict[str, str],
                     min_slack: int = 4, max_slack: int = 30) -> tuple[pd.DataFrame, list[dict]]:
    """Candidates for arrivals A with planned time in [t0, t1) at the hubs.

    plan: one row per (stop_id, event) with eva, trip, pt, pt_raw, pp, line, path,
    wings, tra, cat, num, first_seen. Returns (candidates, funnel)."""
    if max_slack > 60:
        raise ValueError("max_slack above 60 minutes is not supported by the hourly join")
    plan = plan[plan.eva.isin(hubs)]
    dep_path = plan[plan.event == "dp"].set_index("stop_id")["path"]
    A = plan[(plan.event == "ar") & (plan.pt >= t0) & (plan.pt < t1)].copy()
    B = plan[plan.event == "dp"].copy()
    A["cont_path"] = A.stop_id.map(dep_path)

    # a departure within max_slack of an arrival lies in the same or the next hour
    A["hkey"] = A.pt.dt.floor("h")
    A2 = pd.concat([A, A.assign(hkey=A.hkey + pd.Timedelta(hours=1))], ignore_index=True)
    B["hkey"] = B.pt.dt.floor("h")
    p = A2.merge(B.drop(columns=["event"]), on=["eva", "hkey"], suffixes=("_a", "_b"))
    p["slack"] = (p.pt_b - p.pt_a).dt.total_seconds() / 60
    p = p[(p.slack >= min_slack) & (p.slack <= max_slack)].reset_index(drop=True)

    funnel = [_count(p, hubs, "window")]

    def drop(mask, name):
        nonlocal p
        p = p[~pd.Series(mask, index=p.index, dtype=bool)].reset_index(drop=True)
        funnel.append(_count(p, hubs, name))

    drop(p.cat_a.eq("Bus") | p.cat_b.eq("Bus"), "T1 bus")
    drop(p.trip_a.eq(p.trip_b), "T2 same trip")
    drop([isinstance(t, str) and (t == sb or trip_key(t) == tb)
          for t, sb, tb in zip(p.tra_a, p.stop_id_b, p.trip_b)], "T3 transition")
    drop([(tb in set(path(wa))) or (ta in set(path(wb)))
          for ta, tb, wa, wb in zip(p.trip_a, p.trip_b, p.wings_a, p.wings_b)], "T4 wings")
    drop([bool(pa_) and pa_[-1] in set(path(pb))
          for pa_, pb in zip((path(x) for x in p.path_a), p.path_b)], "T5 backtrack")
    drop([bool(c) and bool(n) and n <= c
          for c, n in zip((set(path(x)) for x in p.cont_path), (set(path(x)) for x in p.path_b))], "T6 redundant")

    # T7 first reach: walk each arrival's departures in time order; keep B if it reaches a
    # station that A has neither passed nor will serve itself, and that no strictly earlier
    # departure reaches. Departures at the same minute are judged together (no tie order).
    p = p.sort_values(["stop_id_a", "pt_b", "stop_id_b"], kind="stable").reset_index(drop=True)
    keep = pd.Series(False, index=p.index)
    for _, g in p.groupby("stop_id_a", sort=False):
        covered = set(path(g.path_a.iat[0])) | set(path(g.cont_path.iat[0]))
        for _, same_minute in g.groupby("pt_b", sort=True):
            reached = [set(path(pb)) - covered for pb in same_minute.path_b]
            keep.loc[same_minute.index] = [bool(r) for r in reached]
            covered |= set().union(*reached)
    p = p[keep].reset_index(drop=True)
    funnel.append(_count(p, hubs, "T7 first reach"))

    pa_list = [path(x) for x in p.path_a]
    pb_list = [path(x) for x in p.path_b]
    out = pd.DataFrame({
        "eva": p.eva, "hub": p.eva.map(hubs),
        "stop_id_a": p.stop_id_a, "stop_id_b": p.stop_id_b, "trip_a": p.trip_a, "trip_b": p.trip_b,
        "pt_a": p.pt_a, "pt_b": p.pt_b, "pt_raw_a": p.pt_raw_a, "pt_raw_b": p.pt_raw_b,
        "planned_slack_min": p.slack,
        "cat_a": p.cat_a, "cat_b": p.cat_b, "num_a": p.num_a, "num_b": p.num_b,
        "line_a": p.line_a, "line_b": p.line_b,
        "segment_a": p.cat_a.map(segment), "segment_b": p.cat_b.map(segment),
        "pp_a": p.pp_a, "pp_b": p.pp_b,
        "origin_a": [x[0] if x else None for x in pa_list],
        "prev_station_a": [x[-1] if x else None for x in pa_list],
        "n_stations_before_a": [len(x) for x in pa_list],
        "next_station_b": [x[0] if x else None for x in pb_list],
        "destination_b": [x[-1] if x else None for x in pb_list],
        "n_stations_after_b": [len(x) for x in pb_list],
        "first_seen_a": p.first_seen_a, "first_seen_b": p.first_seen_b,
    })
    return out, funnel
