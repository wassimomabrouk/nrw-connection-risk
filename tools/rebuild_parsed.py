"""Rebuild the parsed Parquet layer from the raw layer.

The raw layer (gzip JSON lines with untouched API responses) is the source of
truth; the parsed layer can always be regenerated from it, for example after a
restore from backup or after a change to the parser.

Run from the repo root:
    python tools/rebuild_parsed.py                                   (all raw data)
    python tools/rebuild_parsed.py --raw data/restore/raw --out data/restore --replace

<out>/parsed must be empty or absent; --replace deletes it first. Appending to an
existing parsed layer would duplicate every row.
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime
import shutil
from pathlib import Path

from nrw_connection_risk.collector.parse import parse_timetable
from nrw_connection_risk.collector.storage import ParsedStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "collector" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rebuilt",
                    help="output root; parsed files go to <out>/parsed/")
    ap.add_argument("--replace", action="store_true",
                    help="delete an existing <out>/parsed before rebuilding")
    args = ap.parse_args()

    target = args.out / "parsed"
    if target.exists() and any(target.rglob("*.parquet")):
        if not args.replace:
            raise SystemExit(f"{target} already contains Parquet files. Use --replace to rebuild it.")
        shutil.rmtree(target)

    files = sorted(args.raw.rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"No raw files under {args.raw}")
    store = ParsedStore(args.out)
    responses = rows = bad_lines = parse_errors = 0
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1       # e.g. a file cut off mid-write
                        continue
                    try:
                        parsed = parse_timetable(rec["body"], rec["source"], rec["eva"],
                                                 datetime.fromisoformat(rec["collected_at"]))
                    except Exception:
                        parse_errors += 1
                        continue
                    store.add(parsed)
                    responses += 1
                    rows += len(parsed)
        except (EOFError, OSError):          # truncated gzip of the hour being written
            bad_lines += 1
        store.flush()
    print(f"raw files: {len(files)}, responses: {responses:,}, rows: {rows:,}, "
          f"unreadable lines: {bad_lines}, parse errors: {parse_errors}")
    print(f"output: {args.out / 'parsed'}")


if __name__ == "__main__":
    main()
