"""Read message first-seen times from the parsed layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from ..dataset.load import _connect, window_dates


def load_messages(parsed_root: Path, t_from: pd.Timestamp, t_to: pd.Timestamp,
                  stations: list[str], extra: pa.Table | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First time each message type:code was seen on an event (key, type, code, obs)
    and each message type on a stop (stop_id, type, obs), in naive UTC."""
    con, _ = _connect(parsed_root, extra, window_dates(t_from, t_to))
    evas = ", ".join(f"'{e}'" for e in stations)
    rows = con.execute(f"""
        SELECT stop_id, event, CAST(collected_at AS TIMESTAMP) AS obs, event_msgs, stop_msgs
        FROM parsed
        WHERE CAST(date AS DATE) BETWEEN DATE '{t_from.date()}' AND DATE '{t_to.date()}'
          AND collected_at BETWEEN TIMESTAMPTZ '{t_from.isoformat()}+00:00' AND TIMESTAMPTZ '{t_to.isoformat()}+00:00'
          AND eva IN ({evas}) AND source IN ('fchg', 'rchg')
          AND (event_msgs IS NOT NULL OR stop_msgs IS NOT NULL)""").df()
    con.close()
    return first_seen_messages(rows)


def plan_changes(parsed_root: Path, t_from: pd.Timestamp, t_to: pd.Timestamp, stations: list[str]) -> int:
    """Number of hub events whose planned time differs between plan responses. The hub
    features take planned times from the latest plan version, which is only point-in-time
    safe if planned times never change after first publication (checked here, reported in _meta.json)."""
    con, _ = _connect(parsed_root, dates=window_dates(t_from, t_to))
    evas = ", ".join(f"'{e}'" for e in stations)
    n = con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT stop_id, event FROM parsed
          WHERE CAST(date AS DATE) BETWEEN DATE '{t_from.date()}' AND DATE '{t_to.date()}'
            AND collected_at BETWEEN TIMESTAMPTZ '{t_from.isoformat()}+00:00' AND TIMESTAMPTZ '{t_to.isoformat()}+00:00'
            AND eva IN ({evas}) AND source = 'plan' AND pt IS NOT NULL
          GROUP BY stop_id, event HAVING COUNT(DISTINCT pt) > 1)""").fetchone()[0]
    con.close()
    return int(n)


def first_seen_messages(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rows: stop_id, event, obs, event_msgs ('t:c|t:c'), stop_msgs. Pure, reused for live data."""
    ev_cols, st_cols = ["key", "type", "code", "obs"], ["stop_id", "type", "obs"]
    if rows.empty:
        return pd.DataFrame(columns=ev_cols), pd.DataFrame(columns=st_cols)
    rows = rows.assign(obs=rows.obs.astype("datetime64[ns]"))
    ev = rows[rows.event.isin(["ar", "dp"])].dropna(subset=["event_msgs"])
    ev = ev.assign(tok=ev.event_msgs.str.split("|")).explode("tok")
    ev = ev.assign(key=ev.stop_id + "|" + ev.event, type=ev.tok.str.split(":").str[0],
                   code=ev.tok.str.split(":").str[1])
    ev_first = ev.groupby(["key", "type", "code"], as_index=False, dropna=False).obs.min()
    st = rows.dropna(subset=["stop_msgs"])
    st = st.assign(type=st.stop_msgs.str.split("|")).explode("type")
    st = st.assign(type=st.type.str.split(":").str[0])
    st_first = st.groupby(["stop_id", "type"], as_index=False).obs.min()
    return ev_first[ev_cols], st_first[st_cols]
