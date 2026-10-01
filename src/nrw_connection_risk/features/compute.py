"""Feature computation. Every function takes query rows with a cutoff time `t_cut`
(naive UTC) and data that may extend beyond it, and uses only data collected at or
before each row's cutoff. Training passes stored data; live prediction passes the
current window. Nothing here reads files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset.state import state_at
from .columns import ALL_FEATURES, GROUPS
from .config import FeatureConfig

MIN = pd.Timedelta(minutes=1)


def _minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def delay_at(keys: pd.Series, times: pd.Series, planned: pd.Series, obs: pd.DataFrame) -> pd.Series:
    """DB's predicted delay (minutes) of each event as known at each time; 0 if no change was known."""
    st = state_at(keys, times, obs)
    return _minutes(st.ct.fillna(planned) - planned)


# ---------------------------------------------------------------- groups

def trend(rows: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Change of DB's predicted delay over the last 15 and 30 minutes."""
    key_a, key_b = rows.stop_id_a + "|ar", rows.stop_id_b + "|dp"
    out = pd.DataFrame(index=rows.index)
    out["trend_a_15"] = rows.db_delay_a_min - delay_at(key_a, rows.t_cut - 15 * MIN, rows.pt_a, obs)
    out["trend_a_30"] = rows.db_delay_a_min - delay_at(key_a, rows.t_cut - 30 * MIN, rows.pt_a, obs)
    out["trend_b_15"] = rows.db_delay_b_min - delay_at(key_b, rows.t_cut - 15 * MIN, rows.pt_b, obs)
    return out


def freshness(rows: pd.DataFrame) -> pd.DataFrame:
    """Minutes since the last observation of A and B before the cutoff (NaN if never observed)."""
    return pd.DataFrame({"age_a_min": _minutes(rows.t_cut - rows.a_obs_cut),
                         "age_b_min": _minutes(rows.t_cut - rows.b_obs_cut)}, index=rows.index)


def _count_seen(keys: pd.Series, times: pd.Series, first_seen: pd.DataFrame, key_col: str) -> pd.Series:
    """How many rows of `first_seen` (key_col, obs) have the row's key and obs <= the row's time."""
    q = pd.DataFrame({key_col: keys.values, "t": times.values.astype("datetime64[ns]"),
                      "_row": np.arange(len(keys))})
    m = q.merge(first_seen[[key_col, "obs"]], on=key_col, how="inner")
    n = m[m.obs <= m.t].groupby("_row").size()
    return pd.Series(n.reindex(np.arange(len(keys)), fill_value=0).values, index=keys.index)


def messages(rows: pd.DataFrame, event_msgs: pd.DataFrame, stop_msgs: pd.DataFrame) -> pd.DataFrame:
    """Message counts seen up to the cutoff.
    event_msgs: key ('stop_id|event'), type, code, obs (first time seen);
    stop_msgs: stop_id, type, obs (first time seen)."""
    key_a, key_b = rows.stop_id_a + "|ar", rows.stop_id_b + "|dp"
    out = pd.DataFrame(index=rows.index)
    for col, keys, typ in (("n_delay_codes_a", key_a, "d"), ("n_quality_a", key_a, "q"),
                           ("n_delay_codes_b", key_b, "d")):
        out[col] = _count_seen(keys, rows.t_cut, event_msgs[event_msgs.type == typ], "key")
    for col, stops, typ in (("h_notice_a", rows.stop_id_a, "h"), ("h_notice_b", rows.stop_id_b, "h"),
                            ("c_notice_a", rows.stop_id_a, "c")):
        out[col] = (_count_seen(stops, rows.t_cut, stop_msgs[stop_msgs.type == typ], "stop_id") > 0).astype(int)
    return out


def hub_state(plan: pd.DataFrame, obs: pd.DataFrame, g_from: pd.Timestamp, g_to: pd.Timestamp,
              cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hub and line state on a grid. The value at grid time g uses only trains whose
    timetable was known at g and observations collected at or before g."""
    step = pd.Timedelta(minutes=cfg.grid_min)
    grid = pd.date_range(g_from.floor(step), g_to, freq=step).values.astype("datetime64[ns]")
    empty = (pd.DataFrame(columns=["eva", "g", "hub_share_cancel", "hub_mean_delay", "hub_share_late5"]),
             pd.DataFrame(columns=["eva", "g", "line", "line_recent_delay_a"]))
    if len(grid) == 0 or plan.empty:
        return empty
    ev = plan.assign(key=plan.stop_id + "|" + plan.event).sort_values("pt", kind="stable").reset_index(drop=True)
    back, ahead = np.timedelta64(max(cfg.line_lookback_min, cfg.window_min), "m"), np.timedelta64(cfg.window_min, "m")
    qs = []
    for _, E in ev.groupby("eva"):
        pts = E.pt.values.astype("datetime64[ns]")
        lo, hi = np.searchsorted(pts, grid - back, "left"), np.searchsorted(pts, grid + ahead, "right")
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if (hi > lo).any() else np.array([], int)
        q = E.iloc[idx].copy()
        q["g"] = np.repeat(grid, hi - lo)
        qs.append(q)
    q = pd.concat(qs, ignore_index=True)
    q = q[q.first_seen <= q.g]
    if q.empty:
        return empty
    st = state_at(q.key, q.g, obs)
    q["cancel"] = st.cs.eq("c").values
    q["delay"] = _minutes(st.ct.fillna(q.pt) - q.pt).values
    q["delay_run"] = q.delay.where(~q.cancel)
    q["late"] = (q.delay >= cfg.late_threshold_min).astype(float).where(~q.cancel)
    near = q[(q.pt - q.g).abs() <= pd.Timedelta(minutes=cfg.window_min)]
    hub = near.groupby(["eva", "g"]).agg(hub_share_cancel=("cancel", "mean"), hub_mean_delay=("delay_run", "mean"),
                                         hub_share_late5=("late", "mean")).reset_index()
    rec = q[(q.event == "ar") & (q.pt <= q.g) & (q.pt > q.g - pd.Timedelta(minutes=cfg.line_lookback_min))
            & ~q.cancel]
    line = rec.groupby(["eva", "g", "line"], as_index=False).delay.mean().rename(columns={"delay": "line_recent_delay_a"})
    return hub, line


def hub(rows: pd.DataFrame, plan: pd.DataFrame, obs: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Hub state at the last grid point at or before each row's cutoff."""
    step = pd.Timedelta(minutes=cfg.grid_min)
    g = rows.t_cut.dt.floor(step)
    hub_t, line_t = hub_state(plan[plan.eva.isin(rows.eva.unique())], obs, g.min(), g.max(), cfg)
    q = pd.DataFrame({"eva": rows.eva.values, "g": g.values, "line": rows.line_a.values, "_row": np.arange(len(rows))})
    q = q.merge(hub_t, on=["eva", "g"], how="left").merge(line_t, on=["eva", "g", "line"], how="left")
    q = q.sort_values("_row")
    out = q[["hub_mean_delay", "hub_share_late5", "hub_share_cancel", "line_recent_delay_a"]].astype(float)
    out.index = rows.index
    return out


def _feeders_on(paths: pd.Series, name_to_eva: dict[str, str]) -> list[list[str]]:
    """Feeder stations on each planned path (station names separated by |)."""
    return [[name_to_eva[n] for n in p.split("|") if n in name_to_eva] if isinstance(p, str) else [] for p in paths]


def feeder(rows: pd.DataFrame, plan: pd.DataFrame, obs: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Corridor state: the hub-state quantities at the feeder stations on A's (and B's) planned
    path before the hub, averaged over those feeders, at the last grid point at or before the
    cutoff. Same point-in-time rules as the hub state (hub_state does the work)."""
    cols = GROUPS["feeder"]
    out = pd.DataFrame(np.nan, index=rows.index, columns=cols)
    names = cfg.feeder_names
    fplan = plan[plan.eva.isin(list(names))]
    if rows.empty or fplan.empty:
        return out
    name_to_eva = {n: e for e, n in names.items()}
    # planned path into the hub = the path of the arrival event at the hub, used only if that
    # timetable entry had been seen by the cutoff
    arr = plan[plan.eva.isin(rows.eva.unique()) & (plan.event == "ar")].drop_duplicates(["eva", "stop_id"])
    arr = arr.set_index(["eva", "stop_id"])
    step = pd.Timedelta(minutes=cfg.grid_min)
    g = rows.t_cut.dt.floor(step)
    fed_t, line_t = hub_state(fplan, obs, g.min(), g.max(), cfg)
    if fed_t.empty:
        return out
    fed_t = fed_t[["eva", "g", "hub_mean_delay"]]
    for side in ("a", "b"):
        idx = pd.MultiIndex.from_arrays([rows.eva, rows[f"stop_id_{side}"]])
        known = (arr.first_seen.reindex(idx).to_numpy() <= rows.t_cut.to_numpy())
        paths = np.where(known, arr.path.reindex(idx).to_numpy(), None)
        on = _feeders_on(paths, name_to_eva)
        q = pd.DataFrame({"_row": np.arange(len(rows)), "g": g.values, "line": rows.line_a.values, "eva": on})
        q = q.explode("eva").dropna(subset=["eva"])
        if q.empty:
            continue
        q = q.merge(fed_t, on=["eva", "g"], how="left")
        if side == "a":
            q = q.merge(line_t, on=["eva", "g", "line"], how="left")
            agg = q.groupby("_row").agg(d=("hub_mean_delay", "mean"), ld=("line_recent_delay_a", "mean"))
            out.iloc[agg.index, out.columns.get_loc("corridor_line_delay_a")] = agg.ld.to_numpy()
            out.iloc[agg.index, out.columns.get_loc("corridor_delay_a")] = agg.d.to_numpy()
        else:
            agg = q.groupby("_row").hub_mean_delay.mean()
            out.iloc[agg.index, out.columns.get_loc("corridor_delay_b")] = agg.to_numpy()
    return out.astype(float)


def context(rows: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Timetable context, known in advance."""
    local = rows.pt_a.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour + local.dt.minute / 60
    holiday = local.dt.date.isin(cfg.holidays)
    day_type = np.select([holiday | (local.dt.weekday == 6), local.dt.weekday == 5],
                         ["sunday_holiday", "saturday"], "weekday")
    return pd.DataFrame({
        "planned_slack_min": rows.planned_slack_min,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_type": day_type,
        "n_stations_before_a": rows.n_stations_before_a,
        "same_platform": (rows.pp_a.notna() & (rows.pp_a == rows.pp_b)).astype(int),
        "segment_a": rows.segment_a, "segment_b": rows.segment_b, "hub": rows.hub,
    }, index=rows.index)


def db(rows: pd.DataFrame) -> pd.DataFrame:
    """DB's own prognosis at the cutoff (from the dataset)."""
    out = rows[["db_slack_min", "db_delay_a_min", "db_delay_b_min"]].copy()
    out["b_cancel_known"] = rows.b_cancel_known.astype(int)
    return out


def compute_all(rows: pd.DataFrame, plan: pd.DataFrame, obs: pd.DataFrame, event_msgs: pd.DataFrame,
                stop_msgs: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Every feature group for the given rows (dataset rows or live query rows)."""
    if rows.empty:
        return pd.DataFrame(columns=ALL_FEATURES, index=rows.index)
    parts = [db(rows), hub(rows, plan, obs, cfg), freshness(rows), context(rows, cfg),
             trend(rows, obs), messages(rows, event_msgs, stop_msgs), feeder(rows, plan, obs, cfg)]
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out[ALL_FEATURES]
