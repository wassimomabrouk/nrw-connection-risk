"""Time handling. Internally all times are naive UTC (datetime64[ns]); DB's local
times are converted once, with DST-ambiguous values flagged instead of guessed."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from ..collector.parse import BERLIN


def service_day_bounds(day: date, start_hour: int = 4) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of a service day in naive UTC. 23 or 25 hours long on DST days."""
    t0 = datetime.combine(day, time(start_hour), tzinfo=BERLIN)
    t1 = datetime.combine(day + timedelta(days=1), time(start_hour), tzinfo=BERLIN)
    return (pd.Timestamp(t0.astimezone(timezone.utc)).tz_localize(None),
            pd.Timestamp(t1.astimezone(timezone.utc)).tz_localize(None))


def is_ambiguous_local(raw) -> bool:
    """True if a yyMMddHHmm local time falls into the repeated autumn hour or the
    skipped spring hour, where it cannot be converted to UTC unambiguously."""
    if not isinstance(raw, str) or len(raw) != 10:
        return False
    try:
        t = datetime.strptime(raw, "%y%m%d%H%M")
    except ValueError:
        return False
    return (t.replace(tzinfo=BERLIN, fold=0).utcoffset()
            != t.replace(tzinfo=BERLIN, fold=1).utcoffset())


def to_naive_utc(s: pd.Series) -> pd.Series:
    """Any datetime series (tz-aware or naive UTC) -> naive UTC datetime64[ns]."""
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")
