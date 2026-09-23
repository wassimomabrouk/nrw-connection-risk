"""Section 0, step 3: can DB's own prognosis be reconstructed at fixed lead times?

Reads one day of raw IRIS API responses, measures how often the hubs were
polled (fchg), parses the prognosed times (ct) per stop and snapshot, joins
them to the processed month, and checks lead-time coverage plus a first look
at DB prognosis error.

Run:  py section0/s0_raw.py --day 2026-08-12
      (expects s0_processed.py to have downloaded the matching month;
       otherwise it downloads it)
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from common import HUBS, Report, download, hub_sql_list, list_raw_day_files

CUTOFFS_MIN = (60, 30, 10)
MAX_STALENESS_MIN = 15  # a poll must exist within this many minutes before the cutoff


def parse_ct(v: str | None):
    if not v:
        return None
    try:
        return datetime.strptime(v, "%y%m%d%H%M")
    except ValueError:
        return None


def to_berlin_naive(s: pd.Series, assume_tz: str) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        s = s.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT")
    # one resolution everywhere: processed data is ns, raw timestamps are us
    return s.dt.tz_convert("Europe/Berlin").dt.tz_localize(None).astype("datetime64[ns]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-08-12")
    ap.add_argument("--files", nargs="*", type=Path, default=None)
    ap.add_argument("--processed-file", type=Path, default=None)
    ap.add_argument("--req-ts-tz", default="UTC",
                    help="zone of naive request timestamps (checked below)")
    ap.add_argument("--processed-tz", default="Europe/Berlin",
                    help="zone of naive processed timestamps (checked below)")
    args = ap.parse_args()

    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    if args.files:
        files = args.files
    else:
        names = list_raw_day_files(day.year, day.month, day.day)
        if not names:
            raise SystemExit(f"No raw files found for {day}")
        print(f"Raw files for {day}: {len(names)}")
        files = [download(n) for n in names]
    proc = args.processed_file or download(
        f"monthly_processed_data/data-{day.year:04d}-{day.month:02d}.parquet")

    rep = Report(f"s0_raw_{args.day}")
    con = duckdb.connect()
    flist = ", ".join(f"'{f.as_posix()}'" for f in files)
    con.execute(f"""
        CREATE VIEW r AS SELECT *,
          LTRIM(regexp_extract(url, '/(plan|fchg|rchg)/(\\d+)', 2), '0') AS eva
        FROM read_parquet([{flist}], union_by_name = true)""")

    # 1. Overview
    rep.h("1. Raw overview")
    rep.say(f"Files: {len(files)}, total size: {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
    rep.table(con.execute("DESCRIBE r").df()[["column_name", "column_type"]])
    rep.table(con.execute("""
        SELECT api_name, status_code, COUNT(*) AS requests,
               COUNT(DISTINCT eva) AS stations,
               SUM((error IS NOT NULL AND error <> '')::INT) AS with_error,
               MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
        FROM r GROUP BY 1, 2 ORDER BY requests DESC""").df())

    # 2. Poll cadence at hubs
    rep.h("2. fchg poll cadence at hubs (minutes between requests)")
    cad = con.execute(f"""
        WITH t AS (
          SELECT eva, timestamp,
                 (epoch(timestamp) - epoch(LAG(timestamp) OVER (PARTITION BY eva ORDER BY timestamp))) / 60.0 AS gap
          FROM r WHERE api_name LIKE '%fchg%' AND eva IN ({hub_sql_list()}))
        SELECT eva, COUNT(*) AS polls, MEDIAN(gap) AS median_gap,
               quantile_cont(gap, 0.9) AS p90_gap, MAX(gap) AS max_gap
        FROM t GROUP BY 1 ORDER BY 1""").df()
    cad.insert(1, "hub", cad.eva.map(HUBS))
    rep.table(cad)
    missing_hubs = sorted(set(HUBS) - set(cad.eva))
    rep.say(f"Hubs never polled via fchg: {[HUBS[e] for e in missing_hubs] if missing_hubs else 'none'}")

    # 3. Parse snapshots
    rep.h("3. Parsed prognosis snapshots (hubs)")
    cur = con.execute(f"""
        SELECT eva, timestamp, response_data FROM r
        WHERE api_name LIKE '%fchg%' AND eva IN ({hub_sql_list()})
          AND response_data IS NOT NULL ORDER BY eva, timestamp""")
    recs, bad = [], 0
    while True:
        batch = cur.fetchmany(200)
        if not batch:
            break
        for eva, ts, body in batch:
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                bad += 1
                continue
            for s in root.iter("s"):
                ar, dp = s.find("ar"), s.find("dp")
                recs.append({
                    "eva": eva, "req_ts": ts, "stop_id": s.get("id"),
                    "ar_ct": parse_ct(ar.get("ct")) if ar is not None else None,
                    "dp_ct": parse_ct(dp.get("ct")) if dp is not None else None,
                    "ar_cs": ar.get("cs") if ar is not None else None,
                    "dp_cs": dp.get("cs") if dp is not None else None,
                })
    snaps = pd.DataFrame(recs)
    for c in ("ar_ct", "dp_ct"):
        if c in snaps:
            snaps[c] = pd.to_datetime(snaps[c]).astype("datetime64[ns]")
    rep.say(f"Unparseable responses: {bad}")
    if snaps.empty:
        rep.say("No snapshot rows parsed. Prognosis reconstruction NOT possible from this day.")
        rep.save()
        return
    per_stop = snaps.groupby("stop_id").agg(
        snapshots=("req_ts", "size"), distinct_ar_ct=("ar_ct", "nunique"))
    rep.say(f"Snapshot rows: {len(snaps):,}, distinct stops seen: {len(per_stop):,}")
    rep.say(f"Snapshots per stop, median: {per_stop.snapshots.median():.0f}, "
            f"p90: {per_stop.snapshots.quantile(0.9):.0f}")
    rep.say(f"Distinct arrival prognoses per stop, median: {per_stop.distinct_ar_ct.median():.0f}, "
            f"p90: {per_stop.distinct_ar_ct.quantile(0.9):.0f}")
    rep.say(f"Share of snapshot rows with a cancellation flag (cs='c'): "
            f"{100 * ((snaps.ar_cs == 'c') | (snaps.dp_cs == 'c')).mean():.2f} %")

    # 4. Join to processed data
    rep.h("4. Join to processed data by stop id")
    ids = pd.DataFrame({"stop_id": per_stop.index})
    con.register("ids", ids)
    n_match = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{proc.as_posix()}')
        WHERE id IN (SELECT stop_id FROM ids)""").fetchone()[0]
    rep.say(f"Snapshot stops matched in processed month: {n_match:,} of {len(ids):,} "
            f"({100 * n_match / max(len(ids), 1):.1f} %)")
    if n_match == 0:
        rep.say("No id match. Linking prognoses to planned times needs another key (design issue).")
        rep.save()
        return

    # all hub arrivals in the processed month, not only those that appear in snapshots
    pdf = con.execute(f"""
        SELECT id AS stop_id, LTRIM(eva, '0') AS eva, arrival_planned_time AS pa,
               arrival_change_time AS a_change, arrival_is_canceled AS ac
        FROM read_parquet('{proc.as_posix()}')
        WHERE LTRIM(eva, '0') IN ({hub_sql_list()}) AND arrival_planned_time IS NOT NULL""").df()
    snaps["req_ts_raw"] = pd.to_datetime(snaps.req_ts).astype("datetime64[ns]")
    snaps["req_ts"] = to_berlin_naive(snaps.req_ts, args.req_ts_tz)
    pdf["pa"] = to_berlin_naive(pdf.pa, args.processed_tz)
    pdf["a_change"] = to_berlin_naive(pdf.a_change, args.processed_tz)
    pdf["final_arr"] = pdf.a_change.fillna(pdf.pa)
    pdf["final_delay"] = (pdf.final_arr - pdf.pa).dt.total_seconds() / 60

    # Time zone sanity check
    last_seen = snaps.dropna(subset=["ar_ct"]).sort_values("req_ts").groupby("stop_id").agg(
        last_req=("req_ts", "max"), last_ct=("ar_ct", "last"))
    chk = last_seen.join(pdf.set_index("stop_id")[["final_arr"]], how="inner")
    rep.say("Time zone sanity (should be small, near 0 to +10 min; a +-60/120 shift means a wrong zone):")
    rep.say(f"  median(last request time - final arrival): "
            f"{((chk.last_req - chk.final_arr).dt.total_seconds() / 60).median():.1f} min")
    rep.say(f"  median(last ct in snapshots - processed final arrival): "
            f"{((chk.last_ct - chk.final_arr).dt.total_seconds() / 60).median():.1f} min")

    # 5. Lead-time coverage and first look at DB prognosis error
    rep.h(f"5. Lead-time coverage (poll within {MAX_STALENESS_MIN} min before cutoff) and DB prognosis error")
    # every fchg request counts as a poll, including responses that listed no changes
    polls = con.execute(f"""
        SELECT DISTINCT eva, timestamp AS poll_ts FROM r
        WHERE api_name LIKE '%fchg%' AND eva IN ({hub_sql_list()})""").df()
    polls["poll_ts"] = to_berlin_naive(polls.poll_ts, args.req_ts_tz)
    polls = polls.sort_values("poll_ts")
    first_poll, last_poll = polls.poll_ts.min(), polls.poll_ts.max()
    base = pdf.loc[~pdf.ac.fillna(False).astype(bool)].copy()
    sn = snaps[["stop_id", "req_ts", "ar_ct"]].rename(columns={"req_ts": "poll_ts"})
    rows = []
    for L in CUTOFFS_MIN:
        t = base.copy()
        t["cutoff"] = t.pa - pd.Timedelta(minutes=L)
        t = t[(t.cutoff >= first_poll + pd.Timedelta(minutes=MAX_STALENESS_MIN))
              & (t.cutoff <= last_poll)]
        if t.empty:
            continue
        m = pd.merge_asof(t.sort_values("cutoff"), polls, left_on="cutoff",
                          right_on="poll_ts", by="eva", direction="backward")
        fresh = (m.cutoff - m.poll_ts) <= pd.Timedelta(minutes=MAX_STALENESS_MIN)
        covered = m[fresh].merge(sn, on=["stop_id", "poll_ts"], how="left")
        # prognosis at the poll: ct if the stop was listed, else planned (no change known)
        prog = covered.ar_ct.fillna(covered.pa)
        prog_delay = (prog - covered.pa).dt.total_seconds() / 60
        rows.append({
            "lead_min": L,
            "stops": len(t),
            "covered_pct": round(100 * fresh.mean(), 1),
            "mae_timetable": round(covered.final_delay.abs().mean(), 2),
            "mae_db_prognosis": round((covered.final_delay - prog_delay).abs().mean(), 2),
            "pct_final_ge6": round(100 * (covered.final_delay >= 6).mean(), 1),
        })
    rep.table(pd.DataFrame(rows))
    rep.say("Caveat: a stop missing from a snapshot is treated as 'no change known' (planned time).")


    # 6. Zone of raw request timestamps, from plan URLs (the requested hour is local time)
    rep.h("6. Zone of raw request timestamps (plan URL hour vs request time)")
    plan = con.execute(r"""
        SELECT timestamp AS ts, regexp_extract(url, '/plan/[0-9]+/([0-9]{6})/([0-9]{2})', 1) AS d,
               regexp_extract(url, '/plan/[0-9]+/([0-9]{6})/([0-9]{2})', 2) AS h
        FROM r WHERE api_name LIKE '%plan%' AND status_code = '200' USING SAMPLE 20000 ROWS""").df()
    plan = plan[(plan.d != "") & (plan.h != "")]
    if plan.empty:
        rep.say("Could not parse plan URLs.")
    else:
        req_hour = pd.to_datetime(plan.d + plan.h, format="%y%m%d%H")
        ts_hour = pd.to_datetime(plan.ts).dt.floor("h")
        off = ((req_hour - ts_hour).dt.total_seconds() / 3600).round().astype(int)
        vc = off.value_counts().sort_index()
        rep.table(pd.DataFrame({"requested_hour_minus_request_hour": vc.index, "requests": vc.values}))
        rep.say("If the collector asks for the current local hour first, the smallest common offset")
        rep.say("is 0 when timestamps are local time and +2 (summer) when they are UTC.")

    # 7. Label validity: was each stop observed after its event, or only before?
    rep.h("7. Label validity: last observation of a stop vs its final event time")
    last = snaps.groupby("stop_id").agg(last_raw=("req_ts_raw", "max"))
    lab = last.join(pdf.set_index("stop_id")[["pa", "final_arr"]], how="inner").dropna(subset=["final_arr"])
    rows7 = []
    for zone in ("UTC", "Europe/Berlin"):
        lr = to_berlin_naive(lab.last_raw, zone)
        gap = (lr - lab.final_arr).dt.total_seconds() / 60
        rows7.append({"assumed_request_zone": zone, "stops": len(lab),
                      "pct_last_seen_before_event": round(100 * (gap < 0).mean(), 1),
                      "median_gap_min": round(gap.median(), 1)})
    rep.table(pd.DataFrame(rows7))
    lr = to_berlin_naive(lab.last_raw, args.req_ts_tz)
    lab["stale"] = (lr - lab.final_arr) < pd.Timedelta(0)
    lab["hour"] = lab.pa.dt.hour
    by_h = lab.groupby("hour").stale.agg(["size", "mean"]).reset_index()
    by_h["mean"] = (100 * by_h["mean"]).round(1)
    rep.say(f"Share last seen before event, by planned hour (zone {args.req_ts_tz}):")
    rep.table(by_h.rename(columns={"size": "stops", "mean": "pct_stale"}))
    rep.say("A stop last seen before its event carries a prognosis, not an observed time.")
    n_hub_stops = len(pdf[(pdf.pa.dt.date == day)])
    rep.say(f"Hub arrivals planned on {day}: {n_hub_stops:,}, of which seen in any snapshot: "
            f"{lab[lab.pa.dt.date == day].shape[0]:,}")

    rep.h("Key answers")
    rep.say(f"Hubs polled: {len(cad)} of {len(HUBS)}, median poll gap: {cad.median_gap.median():.1f} min")
    rep.say(f"Stop id join rate: {100 * n_match / len(ids):.1f} %")
    for r_ in rows:
        rep.say(f"Lead {r_['lead_min']} min: coverage {r_['covered_pct']} %, "
                f"MAE timetable {r_['mae_timetable']} vs DB prognosis {r_['mae_db_prognosis']}")
    rep.save()


if __name__ == "__main__":
    main()
