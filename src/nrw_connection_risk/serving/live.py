"""The collector's data as it stands now: the parsed layer plus responses that are
already in the raw layer but not flushed to Parquet yet (the collector flushes every
10 minutes; without the raw tail, live features would be up to 10 minutes old).
Both go through the production parser, so the rows are identical to what the parsed
layer will hold after the next flush.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from ..collector.parse import SCHEMA, parse_timetable
from ..dataset.load import _partition_globs, load_window
from ..features.load import load_messages

SOURCES = ("plan", "fchg", "rchg")


@dataclass
class LiveSnapshot:
    plan: pd.DataFrame
    obs: pd.DataFrame
    event_msgs: pd.DataFrame
    stop_msgs: pd.DataFrame
    polls: pd.DataFrame            # (eva, t): when the collector saw anything at each hub
    now: pd.Timestamp              # naive UTC
    data_as_of: pd.Timestamp | None  # latest collected_at seen (naive UTC)
    unflushed_rows: int


def _utc(t: pd.Timestamp) -> datetime:
    return t.tz_localize("UTC").to_pydatetime() if t.tzinfo is None else t.to_pydatetime()


def parsed_until(parsed_root: Path, now: datetime) -> datetime | None:
    """Latest collected_at in the parsed layer (today's and yesterday's partitions only)."""
    days = [(now - timedelta(days=1)).date().isoformat(), now.date().isoformat()]
    files = _partition_globs(parsed_root, days)
    if not files:
        return None
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    file_list = ", ".join(f"'{f}'" for f in files)
    t = con.execute(f"SELECT CAST(max(collected_at) AS TIMESTAMP) FROM read_parquet([{file_list}])").fetchone()[0]
    con.close()
    return None if t is None else t.replace(tzinfo=timezone.utc)


def _gzip_members(data: bytes):
    """Decompressed members of a multi-member gzip file. A damaged member (collector
    killed mid-append) is skipped up to the next member header; an incomplete last
    member (being written right now) ends the file."""
    pos = 0
    while pos < len(data):
        d = zlib.decompressobj(wbits=31)
        try:
            out = d.decompress(data[pos:])
        except zlib.error:
            out, d = None, None
        if d is not None and d.eof:
            yield out
            pos = len(data) - len(d.unused_data)
            continue
        nxt = data.find(b"\x1f\x8b\x08", pos + 1)
        if nxt < 0:
            return
        pos = nxt


def _read_records(path: Path):
    """Records of a raw file that may still be appended to; unreadable lines are skipped."""
    try:
        data = path.read_bytes()
    except OSError:
        return
    for member in _gzip_members(data):
        for line in member.decode("utf-8", errors="replace").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def raw_tail(raw_root: Path, after: datetime, until: datetime, stations: set[str] | None = None) -> pa.Table:
    """Parsed rows of every raw response collected in (after, until]. Responses that the
    parser rejects are skipped, as the collector skips them for the parsed layer."""
    rows: list[dict] = []
    hour = after.replace(minute=0, second=0, microsecond=0)
    while hour <= until:
        for source in SOURCES:
            path = raw_root / f"source={source}" / f"date={hour:%Y-%m-%d}" / f"hour={hour:%H}.jsonl.gz"
            if not path.exists():
                continue
            for rec in _read_records(path):
                try:
                    t = datetime.fromisoformat(rec["collected_at"])
                    if not (after < t <= until) or (stations is not None and rec["eva"] not in stations):
                        continue
                    rows += parse_timetable(rec["body"], rec["source"], rec["eva"], t)
                except Exception:          # malformed record or body: never stop the live service
                    continue
        hour += timedelta(hours=1)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    dates = [r["collected_at"].astimezone(timezone.utc).date() for r in rows]
    return table.append_column("date", pa.array(dates, type=pa.date32()))


def load_live(parsed_root: Path, raw_root: Path, now: pd.Timestamp, lookback_h: float, stations: list[str],
              min_parser_version: int, raw_overlap_min: float = 20) -> LiveSnapshot:
    """Everything collected at the hubs in the last `lookback_h` hours up to `now` (naive UTC).

    The raw tail starts `raw_overlap_min` before the parsed layer's latest row, not at it:
    rows the parsed layer lacks (a flush in progress, rows lost when the collector was
    restarted before flushing) are then still read from raw. Rows present in both are
    harmless: every query keeps the latest state, the first sighting or distinct times."""
    now_utc = _utc(now)
    t_from = now - pd.Timedelta(hours=lookback_h)
    flushed = parsed_until(parsed_root, now_utc)
    after = min(flushed - timedelta(minutes=raw_overlap_min), now_utc) if flushed else _utc(t_from)
    after = max(after, _utc(t_from))
    tail = raw_tail(raw_root, after, now_utc, set(stations))
    w = load_window(parsed_root, t_from, now, min_parser_version, stations=stations, extra=tail)
    ev, st = load_messages(parsed_root, t_from, now, stations, extra=tail)
    obs = w.obs.drop_duplicates(["key", "obs", "ct", "cs"]).reset_index(drop=True)   # rows read twice (overlap)
    return LiveSnapshot(plan=w.plan, obs=obs, event_msgs=ev, stop_msgs=st, polls=w.polls, now=now,
                        data_as_of=w.polls.t.max() if len(w.polls) else None, unflushed_rows=tail.num_rows)
