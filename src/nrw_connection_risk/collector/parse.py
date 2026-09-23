"""Parse IRIS timetable XML (plan, fchg, rchg) into flat observation rows.

One row per stop and event (ar = arrival, dp = departure). Stops that carry only
messages (no ar/dp in a change response) get one row with event = "s".
DB times are local German time as yyMMddHHmm; they are kept raw and converted to UTC.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pyarrow as pa

BERLIN = ZoneInfo("Europe/Berlin")
_TRIP_RE = re.compile(r"-\d+$")

SCHEMA = pa.schema([
    ("collected_at", pa.timestamp("us", tz="UTC")),
    ("source", pa.string()),            # plan | fchg | rchg
    ("eva", pa.string()),               # station that was queried
    ("station_name", pa.string()),
    ("stop_id", pa.string()),           # one train at one station on one run
    ("trip_key", pa.string()),          # stop_id without its stop index: one run
    ("event", pa.string()),             # ar | dp | s
    ("tl_category", pa.string()),       # ICE, RE, S, ...
    ("tl_number", pa.string()),
    ("tl_owner", pa.string()),
    ("tl_type", pa.string()),
    ("tl_filter", pa.string()),
    ("line", pa.string()),
    ("pt_raw", pa.string()),            # planned time, DB local format
    ("ct_raw", pa.string()),            # changed (prognosed or actual) time
    ("clt_raw", pa.string()),           # cancellation time
    ("pt", pa.timestamp("us", tz="UTC")),
    ("ct", pa.timestamp("us", tz="UTC")),
    ("clt", pa.timestamp("us", tz="UTC")),
    ("pp", pa.string()),                # planned platform
    ("cp", pa.string()),                # changed platform
    ("ps", pa.string()),                # planned status
    ("cs", pa.string()),                # changed status: p planned, a added, c cancelled
    ("hidden", pa.string()),
    ("ppth", pa.string()),              # planned path (stations separated by |)
    ("cpth", pa.string()),              # changed path
    ("event_msgs", pa.string()),        # messages on ar/dp as type:code, separated by |
    ("stop_msgs", pa.string()),         # messages on the stop
])


def iris_time_to_utc(raw: str | None) -> datetime | None:
    """yyMMddHHmm in Europe/Berlin -> UTC. In the repeated autumn hour the first
    occurrence (summer time) is assumed; the raw string is always kept alongside."""
    if not raw:
        return None
    try:
        local = datetime.strptime(raw, "%y%m%d%H%M").replace(tzinfo=BERLIN)
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def trip_key(stop_id: str | None) -> str | None:
    return _TRIP_RE.sub("", stop_id) if stop_id else None


def _msgs(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    parts = [f"{m.get('t', '')}:{m.get('c', '')}" for m in el.findall("m")]
    return "|".join(parts) if parts else None


def parse_timetable(xml_text: str, source: str, eva: str, collected_at: datetime) -> list[dict]:
    root = ET.fromstring(xml_text)
    station_name = root.get("station")
    rows: list[dict] = []
    for s in root.findall("s"):
        sid = s.get("id")
        tl = s.find("tl")
        base = {
            "collected_at": collected_at,
            "source": source,
            "eva": eva,
            "station_name": station_name,
            "stop_id": sid,
            "trip_key": trip_key(sid),
            "tl_category": tl.get("c") if tl is not None else None,
            "tl_number": tl.get("n") if tl is not None else None,
            "tl_owner": tl.get("o") if tl is not None else None,
            "tl_type": tl.get("t") if tl is not None else None,
            "tl_filter": tl.get("f") if tl is not None else None,
            "stop_msgs": _msgs(s),
        }
        events = [(name, s.find(name)) for name in ("ar", "dp")]
        events = [(n, e) for n, e in events if e is not None]
        if not events:
            rows.append({**base, "event": "s"})
            continue
        for name, e in events:
            rows.append({
                **base,
                "event": name,
                "line": e.get("l"),
                "pt_raw": e.get("pt"),
                "ct_raw": e.get("ct"),
                "clt_raw": e.get("clt"),
                "pt": iris_time_to_utc(e.get("pt")),
                "ct": iris_time_to_utc(e.get("ct")),
                "clt": iris_time_to_utc(e.get("clt")),
                "pp": e.get("pp"),
                "cp": e.get("cp"),
                "ps": e.get("ps"),
                "cs": e.get("cs"),
                "hidden": e.get("hi"),
                "ppth": e.get("ppth"),
                "cpth": e.get("cpth"),
                "event_msgs": _msgs(e),
            })
    return rows
