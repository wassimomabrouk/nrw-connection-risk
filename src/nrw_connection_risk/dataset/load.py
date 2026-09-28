"""Read the parsed layer for a time window (DuckDB over hive-partitioned Parquet)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .timeutil import to_naive_utc


class ParsedLayerError(RuntimeError):
    """The parsed layer is missing, empty or older than the builder requires."""


@dataclass
class Window:
    plan: pd.DataFrame          # one row per (stop_id, event): latest planned version, first_seen
    obs: pd.DataFrame           # every realtime observation of an event: key, obs, ct, ct_raw, cs
    polls: pd.DataFrame         # (eva, t): times at which the collector saw anything at a hub
    data_start: pd.Timestamp
    data_end: pd.Timestamp


def _connect(parsed_root: Path) -> tuple[duckdb.DuckDBPyConnection, str]:
    files = list(parsed_root.rglob("*.parquet")) if parsed_root.exists() else []
    if not files:
        raise ParsedLayerError(f"No parsed Parquet files under {parsed_root}")
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    glob = (parsed_root.as_posix().rstrip("/") + "/**/*.parquet").replace("'", "''")
    con.execute(f"""
        CREATE VIEW parsed AS
        SELECT * FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)""")
    return con, glob


def load_window(parsed_root: Path, t_from: pd.Timestamp, t_to: pd.Timestamp,
                min_parser_version: int) -> Window:
    """All parsed rows collected in [t_from, t_to] (naive UTC bounds)."""
    con, _ = _connect(parsed_root)
    d_from, d_to = t_from.date().isoformat(), t_to.date().isoformat()
    where = (f"CAST(date AS DATE) BETWEEN DATE '{d_from}' AND DATE '{d_to}' "
             f"AND collected_at BETWEEN TIMESTAMPTZ '{t_from.isoformat()}+00:00' "
             f"AND TIMESTAMPTZ '{t_to.isoformat()}+00:00'")
    try:
        old = con.execute(f"""SELECT COUNT(*) FROM parsed WHERE {where}
                              AND (parser_version IS NULL OR parser_version < {min_parser_version})""").fetchone()[0]
    except duckdb.BinderException:
        old = -1
    if old != 0:
        raise ParsedLayerError(
            f"Parsed files in this window were written by a parser older than version "
            f"{min_parser_version}. Rebuild them: python tools/rebuild_parsed.py --raw <raw> --out <root> --replace")

    plan = con.execute(f"""
        SELECT stop_id, event, eva, trip_key AS trip,
               arg_max(pt, collected_at)          AS pt,
               arg_max(pt_raw, collected_at)      AS pt_raw,
               arg_max(pp, collected_at)          AS pp,
               arg_max(COALESCE(line, fb), collected_at) AS line,
               arg_max(ppth, collected_at)        AS path,
               arg_max(wings, collected_at)       AS wings,
               arg_max(tra, collected_at)         AS tra,
               arg_max(tl_category, collected_at) AS cat,
               arg_max(tl_number, collected_at)   AS num,
               min(collected_at)                  AS first_seen
        FROM parsed
        WHERE {where} AND source = 'plan' AND event IN ('ar', 'dp') AND pt IS NOT NULL
        GROUP BY stop_id, event, eva, trip_key""").df()

    obs = con.execute(f"""
        SELECT stop_id || '|' || event AS key, eva, collected_at AS obs, ct, ct_raw, cs
        FROM parsed
        WHERE {where} AND source IN ('fchg', 'rchg') AND event IN ('ar', 'dp')
          AND (ct IS NOT NULL OR cs IS NOT NULL)""").df()

    polls = con.execute(f"""
        SELECT DISTINCT eva, collected_at AS t FROM parsed WHERE {where}""").df()
    # session time zone is UTC, so the cast yields naive UTC (no pytz needed for fetchone)
    span = con.execute(f"""SELECT CAST(min(collected_at) AS TIMESTAMP), CAST(max(collected_at) AS TIMESTAMP)
                           FROM parsed WHERE {where}""").fetchone()
    con.close()
    if span[0] is None:
        raise ParsedLayerError(f"No parsed rows collected between {t_from} and {t_to} (UTC)")

    for df, cols in ((plan, ("pt", "first_seen")), (obs, ("obs", "ct")), (polls, ("t",))):
        for c in cols:
            df[c] = to_naive_utc(df[c]) if len(df) else pd.Series(dtype="datetime64[ns]")
    obs = obs.sort_values("obs", kind="stable").reset_index(drop=True)
    polls = polls.sort_values("t", kind="stable").reset_index(drop=True)
    return Window(plan=plan, obs=obs, polls=polls, data_start=pd.Timestamp(span[0]), data_end=pd.Timestamp(span[1]))
