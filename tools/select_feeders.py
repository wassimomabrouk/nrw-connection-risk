"""Choose feeder stations from the data: stations 2 to 8 stops before a hub on the
paths of arriving trains, picked greedily so that they cover as many transfer
candidates as possible. Optionally looks up their EVA numbers via the DB API.

Run from the repo root:
    python tools/select_feeders.py                     # ranking only, no API calls
    python tools/select_feeders.py --resolve           # plus EVA lookup (one API call per station)
"""
from __future__ import annotations

import argparse
import os
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def hubs_from_config() -> dict[str, str]:
    with open(ROOT / "config" / "collector.toml", "rb") as f:
        st = tomllib.load(f)["stations"]
    return {str(s["eva"]): s["name"] for s in st if s.get("role", "hub") == "hub"}


def load_arrivals(parsed: Path, hubs: dict[str, str]) -> pd.DataFrame:
    """Latest planned path of every arrival at a hub."""
    con = duckdb.connect()
    glob = (parsed.as_posix().rstrip("/") + "/**/*.parquet").replace("'", "''")
    evas = ", ".join(f"'{e}'" for e in hubs)
    df = con.execute(f"""
        SELECT stop_id, eva, arg_max(ppth, collected_at) AS path
        FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
        WHERE source = 'plan' AND event = 'ar' AND eva IN ({evas}) AND ppth IS NOT NULL
        GROUP BY stop_id, eva""").df()
    con.close()
    return df


def candidate_weights(dataset: Path) -> pd.Series:
    """Transfer candidates per arrival (30-minute cutoff rows), from the built dataset."""
    files = list(dataset.glob("service_day=*/part-0.parquet"))
    if not files:
        return pd.Series(dtype=float)
    d = pd.concat(pd.read_parquet(f, columns=["stop_id_a", "cutoff_min"]) for f in files)
    return d[d.cutoff_min == 30].groupby("stop_id_a").size()


def resolve(names: list[str]) -> dict[str, str | None]:
    """Station name -> EVA via the Timetables API station search."""
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    with open(ROOT / "config" / "collector.toml", "rb") as f:
        base = tomllib.load(f)["api"]["base_url"].rstrip("/")
    headers = {"DB-Client-Id": os.environ["DB_CLIENT_ID"], "DB-Api-Key": os.environ["DB_API_KEY"],
               "accept": "application/xml"}
    out = {}
    for name in names:
        r = requests.get(f"{base}/station/{quote(name, safe='')}", headers=headers, timeout=30)
        found = [(s.get("name"), s.get("eva")) for s in ET.fromstring(r.content).iter("station")] if r.ok else []
        exact = [eva for n, eva in found if n == name]
        out[name] = exact[0] if exact else (found[0][1] if len(found) == 1 else None)
        if not exact:
            print(f"  {name}: no exact match, API returned {found[:3]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, default=ROOT / "data" / "restore" / "parsed")
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset" / "v1")
    ap.add_argument("--n", type=int, default=12, help="number of feeder stations to pick")
    ap.add_argument("--min-pos", type=int, default=2, help="closest position before the hub (1 = previous stop)")
    ap.add_argument("--max-pos", type=int, default=8)
    ap.add_argument("--resolve", action="store_true", help="look up EVA numbers via the API")
    args = ap.parse_args()

    hubs = hubs_from_config()
    arr = load_arrivals(args.parsed, hubs)
    w = candidate_weights(args.dataset)
    arr["weight"] = arr.stop_id.map(w).fillna(0) if len(w) else 1.0
    arr = arr[arr.weight > 0]
    total = arr.weight.sum()
    hub_names = set(hubs.values())

    rows = []
    for sid, eva, path, weight in arr[["stop_id", "eva", "path", "weight"]].itertuples(index=False):
        stops = [s for s in path.split("|") if s]
        for k, name in enumerate(reversed(stops), start=1):     # k = 1 is the previous stop
            if args.min_pos <= k <= args.max_pos and name not in hub_names:
                rows.append((sid, eva, name, k, weight))
    cand = pd.DataFrame(rows, columns=["stop_id", "eva", "station", "pos", "weight"])
    print(f"arrivals with candidates: {len(arr):,}, candidates: {total:,.0f}, "
          f"station positions {args.min_pos} to {args.max_pos} before the hub")

    chosen, covered = [], set()
    for _ in range(args.n):
        free = cand[~cand.stop_id.isin(covered)].drop_duplicates(["stop_id", "station"])
        if free.empty:
            break
        gain = free.groupby("station").weight.sum().sort_values(ascending=False)
        best = gain.index[0]
        rows_best = free[free.station == best]
        covered |= set(rows_best.stop_id)
        cum = arr[arr.stop_id.isin(covered)].weight.sum()
        feeds = rows_best.groupby("eva").weight.sum().sort_values(ascending=False)
        chosen.append({"station": best, "adds_pct": round(100 * gain.iloc[0] / total, 1),
                       "cumulative_pct": round(100 * cum / total, 1),
                       "median_pos": int(cand[cand.station == best].pos.median()),
                       "feeds": ", ".join(hubs[e] for e in feeds.index[:2])})
    res = pd.DataFrame(chosen)
    print(res.to_string(index=False))
    print("adds_pct: share of all transfer candidates newly covered; median_pos: stops before the hub.")

    if args.resolve:
        evas = resolve(list(res.station))
        print("\n# paste into config/collector.toml")
        for name in res.station:
            eva = evas.get(name)
            print(f'\n[[stations]]\neva = "{eva or "RESOLVE_MANUALLY"}"\nname = "{name}"\nrole = "feeder"')


if __name__ == "__main__":
    main()
