"""Time the live service's work per scoring run on this machine's data, without the API:
reading the data (the 6-hour window, optionally plus the carried last state of older
events) and computing the features (hub stations only, or hubs and feeder stations).
Read-only. Run on the server from the repo root, next to the running service:

    .venv/bin/python tools/bench_live.py --runs 3

The service scores once a minute, so a run must stay well below 60 seconds.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from nrw_connection_risk.dataset.config import load_config
from nrw_connection_risk.features.compute import compute_all
from nrw_connection_risk.features.config import load_feature_config
from nrw_connection_risk.serving.config import load_serving_config
from nrw_connection_risk.serving.live import carry_start, load_live
from nrw_connection_risk.serving.score import candidate_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    scfg = load_serving_config(ROOT / "config" / "serving.toml", ROOT)
    dcfg = load_config(ROOT / "config" / "dataset.toml")
    fcfg = load_feature_config(ROOT / "config" / "features.toml")
    hubs, both = list(dcfg.hubs), fcfg.stations(dcfg.hubs)
    cases = [("6 h window, hubs (as deployed until now)", hubs, False),
             ("6 h + carried state, hubs", hubs, True),
             (f"6 h + carried state, hubs + {len(both) - len(hubs)} feeders", both, True)]
    for name, stations, carry in cases:
        print(name)
        for _ in range(args.runs):
            now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
            t0 = time.perf_counter()
            snap = load_live(scfg.parsed, scfg.raw, now, scfg.lookback_h, stations, dcfg.min_parser_version,
                             carry_from=carry_start(now, dcfg.start_hour_local) if carry else None)
            t1 = time.perf_counter()
            rows = candidate_rows(snap, dcfg, scfg.min_minutes_ahead, scfg.max_minutes_ahead,
                                  scfg.max_collector_gap_min, [10, 30, 60])
            if len(rows):
                compute_all(rows, snap.plan, snap.obs, snap.event_msgs, snap.stop_msgs, fcfg)
            t2 = time.perf_counter()
            print(f"   read {t1 - t0:5.1f} s   features {t2 - t1:5.1f} s   total {t2 - t0:5.1f} s"
                  f"   ({len(snap.obs):,} observations, {len(rows):,} candidates)")


if __name__ == "__main__":
    main()
