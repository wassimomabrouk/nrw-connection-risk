"""Section 0, step 2: profile one month of the processed stop-level data.

Answers: semantics of delay/time columns, null patterns, trip linkage,
station coverage, cancellation encoding, and connection volume and miss
rates at the five hubs.

Run:  py section0/s0_processed.py --month 2026-08
      py section0/s0_processed.py --file path/to/data-2026-08.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from common import HUBS, Report, download, hub_sql_list


def q(con, sql: str):
    return con.execute(sql).df()


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-08")
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--min-transfer", type=int, default=4,
                    help="assumed minimum transfer time at hubs, minutes")
    ap.add_argument("--window-min", type=int, default=5)
    ap.add_argument("--window-max", type=int, default=30)
    args = ap.parse_args()

    path = args.file or download(f"monthly_processed_data/data-{args.month}.parquet")
    rep = Report(f"s0_processed_{args.month}")
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW d AS
        SELECT * REPLACE (LTRIM(eva, '0') AS eva),
               regexp_replace(id, '-[0-9]+$', '') AS run_key_id,
               train_line_ride_id || '|' || CAST(CAST(COALESCE(departure_planned_time, arrival_planned_time) AS DATE) AS VARCHAR)
                 AS run_key_day
        FROM read_parquet('{path.as_posix()}')""")
    hubs = hub_sql_list()
    summary: dict[str, str] = {}

    # 1. File and schema
    rep.h("1. File and schema")
    n_rows = one(con, "SELECT COUNT(*) FROM d")
    rep.say(f"File: {path.name}, size: {path.stat().st_size / 1e6:.1f} MB, rows: {n_rows:,}")
    schema = q(con, f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')")[["column_name", "column_type"]]
    rep.table(schema)
    tz_cols = schema[schema.column_type.str.contains("TIME ZONE")].column_name.tolist()
    rep.say(f"Time zone aware columns: {tz_cols if tz_cols else 'none (naive timestamps, zone must be established)'}")
    summary["rows_month"] = f"{n_rows:,}"

    # 2. Time coverage
    rep.h("2. Time coverage")
    rep.table(q(con, """
        SELECT MIN(arrival_planned_time) AS min_arr_planned,
               MAX(arrival_planned_time) AS max_arr_planned,
               MIN(departure_planned_time) AS min_dep_planned,
               MAX(departure_planned_time) AS max_dep_planned,
               MIN(time) AS min_time, MAX(time) AS max_time
        FROM d"""))
    rep.table(q(con, """
        SELECT CAST(COALESCE(arrival_planned_time, departure_planned_time) AS DATE) AS day,
               COUNT(*) AS rows
        FROM d GROUP BY 1 ORDER BY 1"""))

    rep.say(f"Rows without any planned time: {one(con, 'SELECT COUNT(*) FROM d WHERE arrival_planned_time IS NULL AND departure_planned_time IS NULL'):,}")

    # 3. Nulls
    rep.h("3. Null share per column (%)")
    cols = schema.column_name.tolist()
    exprs = ", ".join(
        f"ROUND(100.0 * AVG(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END), 2) AS \"{c}\""
        for c in cols
    )
    rep.table(q(con, f"SELECT {exprs} FROM d").T.reset_index().rename(
        columns={"index": "column", 0: "null_pct"}))

    # 4. Uniqueness
    rep.h("4. Uniqueness")
    rep.table(q(con, """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT id) AS distinct_id,
               COUNT(DISTINCT (train_line_ride_id, train_line_station_num)) AS distinct_ride_stationnum,
               COUNT(DISTINCT (train_line_ride_id, eva)) AS distinct_ride_eva
        FROM d"""))

    # 5. Semantics of delay_in_min and time
    rep.h("5. Semantics of delay_in_min and time")
    con.execute("""
        CREATE VIEW s AS SELECT *,
          (epoch(arrival_change_time) - epoch(arrival_planned_time)) / 60.0 AS arr_delay,
          (epoch(departure_change_time) - epoch(departure_planned_time)) / 60.0 AS dep_delay
        FROM d""")
    rep.table(q(con, """
        SELECT
          ROUND(100.0 * AVG(CASE WHEN arr_delay IS NOT NULL THEN (delay_in_min = arr_delay)::INT END), 2)
            AS pct_delay_eq_arr_delay,
          ROUND(100.0 * AVG(CASE WHEN dep_delay IS NOT NULL THEN (delay_in_min = dep_delay)::INT END), 2)
            AS pct_delay_eq_dep_delay,
          ROUND(100.0 * AVG((time = arrival_change_time)::INT), 2) AS pct_time_eq_arr_change,
          ROUND(100.0 * AVG((time = departure_change_time)::INT), 2) AS pct_time_eq_dep_change,
          ROUND(100.0 * AVG((time = arrival_planned_time)::INT), 2) AS pct_time_eq_arr_planned,
          ROUND(100.0 * AVG((time = departure_planned_time)::INT), 2) AS pct_time_eq_dep_planned
        FROM s"""))
    rep.say("Change vs planned consistency:")
    rep.table(q(con, """
        SELECT SUM((arrival_change_time IS NOT NULL AND arrival_planned_time IS NULL)::INT) AS arr_change_without_planned,
               SUM((arrival_change_time IS NULL AND arrival_planned_time IS NOT NULL)::INT) AS arr_planned_without_change,
               SUM((arrival_planned_time IS NULL AND departure_planned_time IS NULL)::INT) AS rows_without_any_planned,
               ROUND(100.0 * AVG(CASE WHEN arr_delay IS NOT NULL THEN (arr_delay = 0)::INT END), 2) AS pct_arr_delay_exactly_0
        FROM s"""))
    rep.say("Sample rows (id structure):")
    rep.table(q(con, """
        SELECT id, train_line_ride_id, train_line_station_num, train_type, train_number,
               arrival_planned_time, departure_planned_time
        FROM d WHERE arrival_planned_time IS NOT NULL USING SAMPLE 8 ROWS"""))
    rep.say("Arrival delay distribution where change time exists (minutes):")
    rep.table(q(con, """
        SELECT COUNT(*) AS n,
               quantile_cont(arr_delay, 0.01) AS p01, quantile_cont(arr_delay, 0.25) AS p25,
               quantile_cont(arr_delay, 0.5) AS p50, quantile_cont(arr_delay, 0.75) AS p75,
               quantile_cont(arr_delay, 0.95) AS p95, quantile_cont(arr_delay, 0.99) AS p99,
               ROUND(100.0 * AVG((arr_delay < 0)::INT), 2) AS pct_early,
               ROUND(100.0 * AVG((arr_delay >= 6)::INT), 2) AS pct_ge6
        FROM s WHERE arr_delay IS NOT NULL"""))

    # 6. Cancellations
    rep.h("6. Cancellations")
    rep.table(q(con, """
        SELECT ROUND(100.0 * AVG(arrival_is_canceled::INT), 3) AS pct_arr_canceled,
               ROUND(100.0 * AVG(departure_is_canceled::INT), 3) AS pct_dep_canceled,
               ROUND(100.0 * AVG(CASE WHEN arrival_is_canceled THEN (arrival_change_time IS NOT NULL)::INT END), 2)
                 AS pct_canceled_arr_with_change_time,
               ROUND(100.0 * AVG(CASE WHEN arrival_is_canceled THEN (delay_in_min IS NOT NULL)::INT END), 2)
                 AS pct_canceled_arr_with_delay
        FROM d"""))

    rep.say("Replacement trains and additional stops (columns not in the dataset README):")
    rep.table(q(con, """
        SELECT ROUND(100.0 * AVG(is_replacement_train::INT), 3) AS pct_replacement_train,
               ROUND(100.0 * AVG((replaced_train_number IS NOT NULL)::INT), 3) AS pct_with_replaced_number,
               ROUND(100.0 * AVG(is_additional_stop::INT), 3) AS pct_additional_stop
        FROM d"""))

    # 7. Trip linkage
    rep.h("7a. Candidate keys for one train run (one day)")
    rows_ = []
    for key in ["train_line_ride_id", "run_key_id", "run_key_day"]:
        r = con.execute(f"""
            WITH o AS (
              SELECT {key} k, train_line_station_num n, arrival_planned_time pa,
                     LAG(departure_planned_time) OVER (PARTITION BY {key} ORDER BY train_line_station_num) prev_pd
              FROM d WHERE {key} IS NOT NULL),
            g AS (SELECT k, COUNT(*) c, COUNT(DISTINCT n) dn FROM o GROUP BY 1)
            SELECT (SELECT COUNT(*) FROM g),
                   (SELECT ROUND(100.0 * AVG((c > dn)::INT), 2) FROM g),
                   (SELECT MEDIAN(dn) FROM g),
                   (SELECT ROUND(100.0 * AVG((prev_pd > pa)::INT), 3) FROM o WHERE pa IS NOT NULL AND prev_pd IS NOT NULL)
        """).fetchone()
        rows_.append({"key": key, "runs": r[0], "pct_runs_with_repeated_station_num": r[1],
                      "median_stops": r[2], "pct_order_violations": r[3]})
    import pandas as pd
    rep.table(pd.DataFrame(rows_))
    rep.say("A good run key has ~0 % repeated station numbers and ~0 % order violations.")

    rep.h("7b. Ride-level stats via train_line_ride_id (as provided)")
    con.execute("""
        CREATE TABLE rides AS
        SELECT train_line_ride_id AS rid,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT train_line_station_num) AS n_stops,
               MIN(train_line_station_num) AS min_num,
               MAX(train_line_station_num) AS max_num,
               (epoch(MAX(COALESCE(arrival_planned_time, departure_planned_time)))
                - epoch(MIN(COALESCE(departure_planned_time, arrival_planned_time)))) / 3600.0 AS span_h
        FROM d WHERE train_line_ride_id IS NOT NULL GROUP BY 1""")
    rep.table(q(con, """
        SELECT COUNT(*) AS rides,
               MEDIAN(n_stops) AS median_stops,
               ROUND(100.0 * AVG((n_stops >= 2)::INT), 2) AS pct_ge2_stops,
               ROUND(100.0 * AVG((max_num - min_num + 1 = n_stops)::INT), 2) AS pct_contiguous_numbering,
               ROUND(100.0 * AVG((min_num <= 1)::INT), 2) AS pct_starts_at_0_or_1,
               ROUND(100.0 * AVG((span_h > 24)::INT), 3) AS pct_span_over_24h,
               ROUND(100.0 * AVG((n_rows > n_stops)::INT), 3) AS pct_with_duplicate_stops
        FROM rides"""))
    rep.say("Coverage within rides (observed stops / numbered range):")
    rep.table(q(con, """
        SELECT quantile_cont(n_stops * 1.0 / (max_num - min_num + 1), 0.1) AS p10,
               quantile_cont(n_stops * 1.0 / (max_num - min_num + 1), 0.5) AS p50,
               quantile_cont(n_stops * 1.0 / (max_num - min_num + 1), 0.9) AS p90
        FROM rides WHERE n_stops >= 2"""))
    viol = one(con, """
        WITH o AS (
          SELECT train_line_ride_id rid, train_line_station_num k,
                 arrival_planned_time pa,
                 LAG(departure_planned_time) OVER (PARTITION BY train_line_ride_id
                                                   ORDER BY train_line_station_num) prev_pd
          FROM d WHERE train_line_ride_id IS NOT NULL)
        SELECT ROUND(100.0 * AVG((prev_pd > pa)::INT), 3) FROM o
        WHERE pa IS NOT NULL AND prev_pd IS NOT NULL""")
    rep.say(f"Planned-time order violations along station_num: {viol} %")

    # 8. Station coverage and hub check
    rep.h("8. Station coverage and hub EVA check")
    rep.say(f"Distinct EVA: {one(con, 'SELECT COUNT(DISTINCT eva) FROM d'):,}")
    rep.table(q(con, f"""
        SELECT eva, ANY_VALUE(station_name) AS station_name,
               COUNT(*) AS rows,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT CAST(COALESCE(arrival_planned_time, departure_planned_time) AS DATE)), 0)
                 AS rows_per_day
        FROM d WHERE eva IN ({hubs}) GROUP BY 1 ORDER BY rows DESC"""))
    for eva, name in HUBS.items():
        base = name.split()[0]
        hits = q(con, f"""
            SELECT eva, station_name, COUNT(*) AS rows FROM d
            WHERE station_name ILIKE '%{base}%' AND station_name ILIKE '%Hbf%'
            GROUP BY 1, 2 ORDER BY rows DESC LIMIT 3""")
        rep.say(f"Name check for {name} (expected EVA {eva}):")
        rep.table(hits)
    rep.say("Train types at hubs:")
    rep.table(q(con, f"""
        SELECT train_type, COUNT(*) AS rows,
               ROUND(AVG((epoch(arrival_change_time) - epoch(arrival_planned_time)) / 60.0), 2)
                 AS mean_arr_delay_where_changed
        FROM d WHERE eva IN ({hubs}) GROUP BY 1 ORDER BY rows DESC LIMIT 20"""))

    # 9. Connections at hubs
    rep.h(f"9. Connections at hubs (window {args.window_min} to {args.window_max} min, "
          f"min transfer {args.min_transfer} min, null change time treated as on time)")
    con.execute(f"""
        CREATE TABLE pairs AS
        WITH arr AS (
          SELECT eva, run_key_id rid, arrival_planned_time pa,
                 COALESCE(arrival_change_time, arrival_planned_time) aa,
                 COALESCE(arrival_is_canceled, FALSE) ac
          FROM d WHERE eva IN ({hubs}) AND arrival_planned_time IS NOT NULL),
        dep AS (
          SELECT eva, run_key_id rid, departure_planned_time pd,
                 COALESCE(departure_change_time, departure_planned_time) ad,
                 COALESCE(departure_is_canceled, FALSE) dc
          FROM d WHERE eva IN ({hubs}) AND departure_planned_time IS NOT NULL)
        SELECT a.eva, (epoch(p.pd) - epoch(a.pa)) / 60.0 AS slack, a.ac, p.dc,
               (epoch(a.aa) + {args.min_transfer} * 60 > epoch(p.ad)) AS late_miss,
               CAST(a.pa AS DATE) AS day
        FROM arr a JOIN dep p
          ON a.eva = p.eva AND a.rid IS DISTINCT FROM p.rid
         AND p.pd >= a.pa + INTERVAL {args.window_min} MINUTE
         AND p.pd <= a.pa + INTERVAL {args.window_max} MINUTE""")
    by_hub = q(con, """
        SELECT eva, COUNT(*) AS pairs,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT day), 0) AS pairs_per_day,
               ROUND(100.0 * AVG((late_miss AND NOT ac AND NOT dc)::INT), 2) AS pct_miss_delay,
               ROUND(100.0 * AVG(ac::INT), 2) AS pct_A_canceled,
               ROUND(100.0 * AVG((dc AND NOT ac)::INT), 2) AS pct_B_canceled
        FROM pairs GROUP BY 1 ORDER BY pairs DESC""")
    by_hub.insert(1, "hub", by_hub.eva.map(HUBS))
    rep.table(by_hub)
    rep.say("By planned slack bucket:")
    rep.table(q(con, """
        SELECT CASE WHEN slack < 10 THEN '05-09' WHEN slack < 15 THEN '10-14'
                    WHEN slack < 20 THEN '15-19' ELSE '20-30' END AS slack_bucket,
               COUNT(*) AS pairs,
               ROUND(100.0 * AVG((late_miss AND NOT ac AND NOT dc)::INT), 2) AS pct_miss_delay
        FROM pairs GROUP BY 1 ORDER BY 1"""))
    rep.say("Note: these are ALL arrival/departure pairs in the window, not real passenger")
    rep.say("transfers. Filtering to plausible connections is a design decision for DESIGN.md.")

    # Key answers
    total_pairs = one(con, "SELECT COUNT(*) FROM pairs")
    miss = one(con, "SELECT ROUND(100.0 * AVG((late_miss AND NOT ac AND NOT dc)::INT), 2) FROM pairs")
    link = one(con, "SELECT ROUND(100.0 * AVG((n_stops >= 2)::INT), 1) FROM rides")
    rep.h("Key answers")
    rep.say(f"Rows this month: {summary['rows_month']}")
    rep.say(f"Rides with 2+ linked stops: {link} %")
    rep.say(f"Hub pairs this month: {total_pairs:,}, miss rate by delay: {miss} %")
    rep.say("Prognosis snapshots: not in processed data (one final change time per stop), see s0_raw.py")
    rep.save()


if __name__ == "__main__":
    main()
