"""Section 0, step 1: inventory of the dataset without downloading it.

Answers: history depth, gaps, rows and size per month, schema stability.
Run:  py section0/s0_inventory.py            (full, reads parquet footers remotely)
      py section0/s0_inventory.py --skip-schema   (file listing only)
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from common import REPO_ID, Report


def month_range(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-schema", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    rep = Report("s0_inventory")

    # ---------- monthly processed files ----------
    rep.h("Monthly processed files")
    monthly = []
    for entry in api.list_repo_tree(
        REPO_ID, path_in_repo="monthly_processed_data", repo_type="dataset"
    ):
        m = re.search(r"data-(\d{4}-\d{2})\.parquet$", entry.path)
        if m:
            monthly.append((m.group(1), entry.path, getattr(entry, "size", None)))
    monthly.sort()
    if not monthly:
        rep.say("No monthly files found.")
    else:
        months = [m for m, _, _ in monthly]
        rep.say(f"Files: {len(monthly)}, first: {months[0]}, last: {months[-1]}")
        missing = sorted(set(month_range(months[0], months[-1])) - set(months))
        rep.say(f"Missing months inside range: {missing if missing else 'none'}")

    # ---------- raw days ----------
    rep.h("Raw data days")
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    days = defaultdict(int)
    for f in files:
        m = re.match(r"raw_data/year=(\d+)/month=(\d+)/day=(\d+)/.+\.parquet$", f)
        if m:
            days[date(int(m.group(1)), int(m.group(2)), int(m.group(3)))] += 1
    if not days:
        rep.say("No raw files found.")
    else:
        ds = sorted(days)
        rep.say(f"Days: {len(ds)}, first: {ds[0]}, last: {ds[-1]}")
        full = [ds[0] + timedelta(n) for n in range((ds[-1] - ds[0]).days + 1)]
        gaps = [d for d in full if d not in days]
        rep.say(f"Missing days inside range: {len(gaps)}")
        if gaps:
            rep.say("First missing days:", ", ".join(str(g) for g in gaps[:20]))
        per_month = pd.DataFrame({"day": ds, "files": [days[d] for d in ds]})
        per_month["month"] = per_month.day.map(lambda d: f"{d.year}-{d.month:02d}")
        rep.say("Raw files per day, by month:")
        rep.table(per_month.groupby("month").files.agg(["min", "median", "max"]).reset_index())

    # ---------- rows, size, schema per month ----------
    if monthly and not args.skip_schema:
        import duckdb

        rep.h("Rows, size and schema per month (parquet footers only)")
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        rows = []
        schemas: dict[tuple, list[str]] = defaultdict(list)
        for month, path, size in monthly:
            url = f"hf://datasets/{REPO_ID}/{path}"
            try:
                n = con.execute(
                    f"SELECT SUM(num_rows) FROM parquet_file_metadata('{url}')"
                ).fetchone()[0]
                cols = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{url}')"
                ).fetchall()
                sig = tuple((c[0], c[1]) for c in cols)
                schemas[sig].append(month)
                rows.append({"month": month, "rows": int(n),
                             "size_mb": round(size / 1e6, 1) if size else None})
            except Exception as exc:  # keep going, report the failure
                rows.append({"month": month, "rows": None, "size_mb": None})
                rep.say(f"{month}: failed to read footer ({exc.__class__.__name__}: {exc})")
        rep.table(pd.DataFrame(rows))

        rep.h("Schema versions")
        rep.say(f"Distinct schemas: {len(schemas)}")
        for i, (sig, ms) in enumerate(schemas.items(), 1):
            rep.say(f"Schema {i}: {len(ms)} months, {ms[0]} to {ms[-1]}")
            for name, typ in sig:
                rep.say(f"    {name}: {typ}")

    rep.save()


if __name__ == "__main__":
    main()
