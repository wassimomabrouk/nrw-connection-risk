"""Exploration 05: which information available at the cutoff predicts failed
connections beyond DB's own prognosis?

Five candidate feature groups, all computed strictly from observations collected
at or before the cutoff (point in time):

  trend       change of DB's prognosis for A and B over the last 15 and 30 minutes
  freshness   minutes since the last update of A and B
  messages    delay cause codes and quality messages on A's arrival and B's departure,
              disruption notices (h) and connection notices (c) on the stops
  hub         mean delay, share of trains 5+ minutes late and share cancelled at the hub
              (planned within 30 minutes of now), mean delay of A's line in the last hour
  context     planned slack, hour, weekend, segments, hub, same planned platform

Evaluation: leave-one-service-day-out. Each group is added to B3 (logistic regression
on DB's prognosis, DESIGN.md section 4) and the change in log loss is reported per
held-out day. A gradient-boosting model on all features vs on DB features only
separates new information from mere nonlinearity; dropping one group at a time from
it shows what each group adds on top of all others.

Only training-period days (up to 2026-11-08) are used; validation and test stay untouched.

Run from the repo root (needs scikit-learn):
    python exploration/e05_features.py --dataset data/dataset/v1 --parsed data/restore/parsed
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nrw_connection_risk.dataset.load import _connect, load_window
from nrw_connection_risk.dataset.state import state_at
from nrw_connection_risk.dataset.timeutil import service_day_bounds, to_naive_utc

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
TRAIN_END = date(2026, 11, 8)          # DESIGN.md section 6: exploration never sees later days
GRID = pd.Timedelta(minutes=5)

DB = ["db_slack_min", "db_delay_a_min", "db_delay_b_min", "b_cancel_known"]
GROUPS = {
    "trend": ["trend_a_15", "trend_a_30", "trend_b_15"],
    "freshness": ["age_a_min", "age_b_min"],
    "messages": ["n_delay_codes_a", "n_quality_a", "n_delay_codes_b", "h_notice_a", "h_notice_b", "c_notice_a"],
    "hub": ["hub_mean_delay", "hub_share_late5", "hub_share_cancel", "line_recent_delay_a"],
    "context": ["planned_slack_min", "hour_sin", "hour_cos", "weekend", "n_stations_before_a",
                "same_platform", "segment_a", "segment_b", "hub"],
}
CATEGORICAL = {"segment_a", "segment_b", "hub"}
CLIP = {"db_slack_min": (-60, 60), "db_delay_a_min": (-10, 120), "db_delay_b_min": (-10, 120),
        "trend_a_15": (-30, 60), "trend_a_30": (-30, 60), "trend_b_15": (-30, 60),
        "age_a_min": (0, 240), "age_b_min": (0, 240), "hub_mean_delay": (-5, 60),
        "line_recent_delay_a": (-5, 90)}


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def say(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def h(self, title):
        self.say("")
        self.say(f"== {title} ==")

    def table(self, df):
        self.say(df.to_string(index=False) if len(df) else "(empty)")


# ------------------------------------------------------------------ features

def _naive(s: pd.Series) -> pd.Series:
    return to_naive_utc(s)


def _delay_at(keys, times, planned, obs) -> pd.Series:
    """DB's predicted delay (minutes) of each event as known at each time."""
    st = state_at(keys, times, obs)
    return (st.ct.fillna(planned) - planned).dt.total_seconds() / 60


def load_messages(parsed: Path, t_from, t_to) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First time each message type:code was seen on an event, and on a stop."""
    con, _ = _connect(parsed)
    con.execute("SET TimeZone = 'UTC'")
    rows = con.execute(f"""
        SELECT stop_id, event, CAST(collected_at AS TIMESTAMP) AS obs, event_msgs, stop_msgs
        FROM parsed
        WHERE CAST(date AS DATE) BETWEEN DATE '{t_from.date()}' AND DATE '{t_to.date()}'
          AND collected_at BETWEEN TIMESTAMPTZ '{t_from.isoformat()}+00:00' AND TIMESTAMPTZ '{t_to.isoformat()}+00:00'
          AND source IN ('fchg', 'rchg') AND (event_msgs IS NOT NULL OR stop_msgs IS NOT NULL)""").df()
    con.close()
    rows["obs"] = rows.obs.astype("datetime64[ns]")
    ev = rows.dropna(subset=["event_msgs"]).assign(tok=lambda d: d.event_msgs.str.split("|")).explode("tok")
    ev = ev[ev.event.isin(["ar", "dp"])]
    ev["key"] = ev.stop_id + "|" + ev.event
    ev_first = ev.groupby(["key", "tok"], as_index=False).obs.min()
    ev_first["type"] = ev_first.tok.str.split(":").str[0]
    st = rows.dropna(subset=["stop_msgs"]).assign(tok=lambda d: d.stop_msgs.str.split("|")).explode("tok")
    st["type"] = st.tok.str.split(":").str[0]
    st_first = st.groupby(["stop_id", "type"], as_index=False).obs.min()
    return ev_first, st_first


def _count_before(q_keys: pd.Series, q_times: pd.Series, first: pd.DataFrame, key_col: str) -> pd.Series:
    """Number of rows in `first` with the query's key and first-seen time at or before the query time."""
    q = pd.DataFrame({key_col: q_keys.values, "t": q_times.values, "_row": np.arange(len(q_keys))})
    m = q.merge(first[[key_col, "obs"]], on=key_col, how="inner")
    n = m[m.obs <= m.t].groupby("_row").size()
    return pd.Series(n.reindex(np.arange(len(q_keys)), fill_value=0).values, index=q_keys.index)


def hub_state(plan: pd.DataFrame, obs: pd.DataFrame, g_from, g_to) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hub and line state on a 5-minute grid; every value uses only data visible at the grid time."""
    grid = pd.date_range(g_from.floor(GRID), g_to, freq=GRID).values
    ev = plan.assign(key=plan.stop_id + "|" + plan.event).sort_values("pt").reset_index(drop=True)
    qs = []
    for eva, E in ev.groupby("eva"):
        pts = E.pt.values
        lo = np.searchsorted(pts, grid - np.timedelta64(60, "m"), "left")
        hi = np.searchsorted(pts, grid + np.timedelta64(30, "m"), "right")
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        q = E.iloc[idx].copy()
        q["g"] = np.repeat(grid, hi - lo)
        qs.append(q)
    q = pd.concat(qs, ignore_index=True)
    q = q[q.first_seen <= q.g]                                   # only trains known at that time
    st = state_at(q.key, q.g, obs)
    q["cancel"] = st.cs.eq("c").values
    q["delay"] = ((st.ct.fillna(q.pt) - q.pt).dt.total_seconds() / 60).values
    q["delay_run"] = q.delay.where(~q.cancel)                    # delays of trains that run
    q["late5"] = (q.delay >= 5).astype(float).where(~q.cancel)
    near = q[(q.pt - q.g).abs() <= pd.Timedelta(minutes=30)]
    hub = near.groupby(["eva", "g"]).agg(
        hub_share_cancel=("cancel", "mean"),
        hub_mean_delay=("delay_run", "mean"),                    # NaN ignored: cancelled trains
        hub_share_late5=("late5", "mean"),
    ).reset_index()
    rec = q[(q.event == "ar") & (q.pt <= q.g) & ~q.cancel]
    line = rec.groupby(["eva", "g", "line"], as_index=False).delay.mean().rename(columns={"delay": "line_recent_delay_a"})
    return hub, line


def features_for_day(day: date, dataset_root: Path, parsed: Path) -> pd.DataFrame:
    d = pd.read_parquet(dataset_root / f"service_day={day}" / "part-0.parquet")
    d = d[d.eligible].reset_index(drop=True)
    for c in ("t_cut", "pt_a", "pt_b", "a_obs_cut", "b_obs_cut"):
        d[c] = _naive(d[c])
    t0, t1 = service_day_bounds(day)
    w = load_window(parsed, t0 - pd.Timedelta(days=1), t1 + pd.Timedelta(hours=1), 2)
    key_a, key_b = d.stop_id_a + "|ar", d.stop_id_b + "|dp"

    # trend: prognosis now vs 15 and 30 minutes earlier
    for L, col, keys, pt, now in ((15, "trend_a_15", key_a, d.pt_a, d.db_delay_a_min),
                                  (30, "trend_a_30", key_a, d.pt_a, d.db_delay_a_min),
                                  (15, "trend_b_15", key_b, d.pt_b, d.db_delay_b_min)):
        d[col] = now - _delay_at(keys, d.t_cut - pd.Timedelta(minutes=L), pt, w.obs)

    # freshness
    d["age_a_min"] = (d.t_cut - d.a_obs_cut).dt.total_seconds() / 60
    d["age_b_min"] = (d.t_cut - d.b_obs_cut).dt.total_seconds() / 60

    # messages first seen at or before the cutoff
    ev_first, st_first = load_messages(parsed, t0 - pd.Timedelta(days=1), t1)
    for col, keys, typ in (("n_delay_codes_a", key_a, "d"), ("n_quality_a", key_a, "q"),
                           ("n_delay_codes_b", key_b, "d")):
        d[col] = _count_before(keys, d.t_cut, ev_first[ev_first.type == typ], "key")
    for col, stops, typ in (("h_notice_a", d.stop_id_a, "h"), ("h_notice_b", d.stop_id_b, "h"),
                            ("c_notice_a", d.stop_id_a, "c")):
        d[col] = (_count_before(stops, d.t_cut, st_first[st_first.type == typ], "stop_id") > 0).astype(int)

    # hub state at the last grid point at or before the cutoff
    hub, line = hub_state(w.plan, w.obs, d.t_cut.min() - GRID, d.t_cut.max())
    d["g"] = d.t_cut.dt.floor(GRID)
    d = d.merge(hub, on=["eva", "g"], how="left")
    d = d.merge(line.rename(columns={"line": "line_a"}), on=["eva", "g", "line_a"], how="left")

    # context
    local = d.pt_a.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour + local.dt.minute / 60
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    d["weekend"] = (local.dt.weekday >= 5).astype(int)
    d["same_platform"] = (d.pp_a.notna() & (d.pp_a == d.pp_b)).astype(int)
    d["b_cancel_known"] = d.b_cancel_known.astype(int)
    return d


# ------------------------------------------------------------------ models

def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c, (lo, hi) in CLIP.items():
        if c in X:
            X[c] = X[c].clip(lo, hi)
    for c in cols:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
        else:
            X[c] = X[c].astype(float)
    return X


def _model(kind: str, cols: list[str]):
    if kind == "lr":
        num = [c for c in cols if c not in CATEGORICAL]
        cat = [c for c in cols if c in CATEGORICAL]
        pre = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
                                   StandardScaler()), num)]
            + ([("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []))
        return make_pipeline(pre, LogisticRegression(max_iter=3000))
    # conservative settings: early stopping on a held-back 15 % and large leaves against overfitting
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          categorical_features="from_dtype", early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=20, random_state=0)


def cross_val(df: pd.DataFrame, cols: list[str], kind: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Leave one service day out; returns pooled out-of-day predictions and per-day log loss."""
    p = np.zeros(len(df))
    per_day = []
    X, y = _prep(df, cols), df.label_fail.astype(int).values
    for day in sorted(df.service_day.unique()):
        test = (df.service_day == day).values
        m = _model(kind, cols).fit(X[~test], y[~test])
        p[test] = np.clip(m.predict_proba(X[test])[:, 1], 1e-6, 1 - 1e-6)
        per_day.append({"day": day, "log_loss": log_loss(y[test], p[test], labels=[0, 1])})
    return p, pd.DataFrame(per_day)


def metrics(y, p) -> dict:
    return {"log_loss": round(log_loss(y, p, labels=[0, 1]), 4), "auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4)}


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset" / "v1")
    ap.add_argument("--parsed", type=Path, default=ROOT / "data" / "restore" / "parsed")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[60, 30, 10])
    args = ap.parse_args()

    days = sorted(date.fromisoformat(p.name.split("=")[1]) for p in args.dataset.glob("service_day=*"))
    used = [d for d in days if d <= TRAIN_END]
    if len(used) < 2:
        sys.exit("Need at least two built training days for leave-one-day-out.")
    rep = Report()
    rep.h("1. Data")
    rep.say(f"training days used: {', '.join(str(d) for d in used)}"
            + (f"  (ignored after {TRAIN_END}: {len(days) - len(used)})" if len(days) > len(used) else ""))
    t = time.time()
    frames = [features_for_day(d, args.dataset, args.parsed) for d in used]
    data = pd.concat(frames, ignore_index=True)
    rep.say(f"eligible rows: {len(data):,}, features built in {time.time() - t:.0f} s")

    all_feats = DB + [c for g in GROUPS.values() for c in g]
    rep.h("2. Feature coverage and univariate signal at the 30-minute cutoff")
    d30 = data[data.cutoff_min == 30]
    y30 = d30.label_fail.astype(int)
    rows = []
    for c in all_feats:
        if c in CATEGORICAL:
            continue
        x = d30[c].astype(float)
        ok = x.notna()
        auc = roc_auc_score(y30[ok], x[ok]) if ok.sum() > 100 and x[ok].nunique() > 1 else float("nan")
        rows.append({"feature": c, "non_missing_pct": round(100 * ok.mean(), 1),
                     "mean_if_fail": round(x[ok & y30.astype(bool)].mean(), 2),
                     "mean_if_holds": round(x[ok & ~y30.astype(bool)].mean(), 2),
                     "auc_alone": round(max(auc, 1 - auc), 3) if auc == auc else None})
    rep.table(pd.DataFrame(rows))
    rep.say("auc_alone is direction-free (max of AUC and 1 - AUC); DB features for comparison.")

    for L in args.cutoffs:
        d = data[data.cutoff_min == L].reset_index(drop=True)
        y = d.label_fail.astype(int).values
        rep.h(f"3. Cutoff {L} min: {len(d):,} rows, failure rate {100 * y.mean():.1f} %")

        p_b0, _ = cross_val(d, ["planned_slack_min"], "lr")
        p_b3, day_b3 = cross_val(d, DB, "lr")
        rows = [{"model": "B0 planned slack (LR)", **metrics(y, p_b0)},
                {"model": "B3 calibrated DB (LR)", **metrics(y, p_b3)}]
        per_day = day_b3.rename(columns={"log_loss": "B3"})
        for g, cols in GROUPS.items():
            p, dd = cross_val(d, DB + cols, "lr")
            rows.append({"model": f"B3 + {g} (LR)", **metrics(y, p)})
            per_day[f"+{g}"] = dd.log_loss.values
        p, dd = cross_val(d, all_feats, "lr")
        rows.append({"model": "LR all features", **metrics(y, p)})
        per_day["LR all"] = dd.log_loss.values
        p_hdb, dd = cross_val(d, DB, "hgb")
        rows.append({"model": "GBM DB features only", **metrics(y, p_hdb)})
        per_day["GBM DB"] = dd.log_loss.values
        p_hall, dd = cross_val(d, all_feats, "hgb")
        rows.append({"model": "GBM all features", **metrics(y, p_hall)})
        per_day["GBM all"] = dd.log_loss.values
        res = pd.DataFrame(rows)
        base = res.loc[res.model.str.startswith("B3 calibrated"), "log_loss"].iat[0]
        res["d_log_loss_vs_B3_pct"] = (100 * (res.log_loss - base) / base).round(2)
        rep.table(res)
        rep.say("Log loss per held-out day (lower is better; a useful group improves on B3 on every day):")
        per_day[per_day.columns[1:]] = per_day[per_day.columns[1:]].round(4)
        rep.table(per_day)

        if L == 30:
            rep.h("4. Cutoff 30 min: drop one group from the GBM with all features")
            full = metrics(y, p_hall)["log_loss"]
            rows = []
            for g, cols in GROUPS.items():
                keep = [c for c in all_feats if c not in cols]
                p, _ = cross_val(d, keep, "hgb")
                ll = metrics(y, p)["log_loss"]
                rows.append({"dropped": g, "log_loss": ll, "increase_pct": round(100 * (ll - full) / full, 2)})
            rep.table(pd.DataFrame(rows))
            rep.say("increase_pct: how much worse the model gets without the group (its unique contribution).")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "e05_features.txt"
    path.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
