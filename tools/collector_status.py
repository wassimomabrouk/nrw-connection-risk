"""Quick look at what the collector has stored so far.

Run from the repo root:  py tools\\collector_status.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

DATA = Path(__file__).resolve().parents[1] / "data" / "collector"


def main() -> None:
    hb = DATA / "heartbeat.json"
    if hb.exists():
        h = json.loads(hb.read_text(encoding="utf-8"))
        print(f"heartbeat updated: {h['updated_at']}")
        print(f"  successes: {h['success_count']}  failures: {h['failure_count']}")
        print(f"  API calls: {h['api_calls_total']}  pending rows: {h['parsed_rows_pending']}\n")
    files = list((DATA / "parsed").rglob("*.parquet"))
    if not files:
        print("No parsed files yet (rows are flushed every 10 min and on stop).")
        return
    con = duckdb.connect()
    glob = (DATA / "parsed").as_posix() + "/**/*.parquet"
    print(con.execute(f"""
        SELECT source, COUNT(*) AS rows, COUNT(DISTINCT stop_id) AS stops,
               COUNT(DISTINCT eva) AS stations,
               MIN(collected_at) AS first_obs, MAX(collected_at) AS last_obs
        FROM read_parquet('{glob}', hive_partitioning = true)
        GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    print()
    print(con.execute(f"""
        SELECT station_name, source, COUNT(*) AS rows,
               SUM((ct IS NOT NULL)::INT) AS with_ct, SUM((cs = 'c')::INT) AS cancelled
        FROM read_parquet('{glob}', hive_partitioning = true)
        GROUP BY 1, 2 ORDER BY 1, 2""").df().to_string(index=False))
    raw = list((DATA / "raw").rglob("*.jsonl.gz"))
    print(f"\nraw files: {len(raw)}, total {sum(f.stat().st_size for f in raw) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
