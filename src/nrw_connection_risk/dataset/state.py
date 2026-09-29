"""Point-in-time joins: what the collector knew about an event at a given moment.

The only way features and baselines may read realtime data. An observation is
visible at time t if it was collected at or before t; anything later is invisible.
"""
from __future__ import annotations

import pandas as pd


def state_at(keys: pd.Series, times: pd.Series, obs: pd.DataFrame) -> pd.DataFrame:
    """Latest observation of each event (key 'stop_id|event') collected at or before
    the matching time. Returns ct, ct_raw, cs, obs aligned to the input index;
    all missing if the event had no observation yet (no change known: planned)."""
    q = pd.DataFrame({"key": keys.values, "t": times.values.astype("datetime64[ns]"),
                      "_row": range(len(keys))}).sort_values("t", kind="stable")
    h = obs[["key", "obs", "ct", "ct_raw", "cs"]].copy()
    h["obs"] = h["obs"].astype("datetime64[ns]")
    h["ct"] = h["ct"].astype("datetime64[ns]")
    h["key"] = h["key"].astype(q["key"].dtype)      # an empty table comes back with object dtype
    h = h.sort_values("obs", kind="stable")
    m = pd.merge_asof(q, h, left_on="t", right_on="obs", by="key", direction="backward",
                      allow_exact_matches=True)
    m = m.sort_values("_row")
    m.index = keys.index
    return m[["ct", "ct_raw", "cs", "obs"]]


def final_state(keys: pd.Series, obs: pd.DataFrame) -> pd.DataFrame:
    """Last observation of each event in the whole window (used for labels only)."""
    last = obs.sort_values("obs", kind="stable").groupby("key", sort=False).tail(1).set_index("key")
    out = last.reindex(keys.values)[["ct", "ct_raw", "cs", "obs"]]
    out.index = keys.index
    return out


def last_poll_at(evas: pd.Series, times: pd.Series, polls: pd.DataFrame) -> pd.Series:
    """Time of the collector's latest observation at the hub at or before each time."""
    q = pd.DataFrame({"eva": evas.values, "t": times.values.astype("datetime64[ns]"),
                      "_row": range(len(evas))}).sort_values("t", kind="stable")
    h = polls.rename(columns={"t": "poll"}).copy()
    h["poll"] = h["poll"].astype("datetime64[ns]")
    h["eva"] = h["eva"].astype(q["eva"].dtype)
    m = pd.merge_asof(q, h.sort_values("poll", kind="stable"), left_on="t", right_on="poll",
                      by="eva", direction="backward").sort_values("_row")
    return pd.Series(m["poll"].values, index=evas.index)
