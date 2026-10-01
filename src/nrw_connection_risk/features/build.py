"""Materialise features for built dataset days: one Parquet table per service day with
keys, target and every feature group (training selects groups via config/features.toml).

Run from the repo root:
    python -m nrw_connection_risk.features.build --from 2026-09-24 --to 2026-09-30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..dataset.build import _git_commit, write_day
from ..dataset.config import load_config
from ..dataset.load import load_window
from ..dataset.timeutil import service_day_bounds, to_naive_utc
from .columns import ALL_FEATURES, KEYS, TARGET
from .compute import compute_all
from .config import FeatureConfig, load_feature_config
from .load import load_messages, plan_changes

ROOT = Path(__file__).resolve().parents[3]
TIME_COLS = ("t_cut", "pt_a", "pt_b", "a_obs_cut", "b_obs_cut")


def build_day(day: date, dataset_day: Path, parsed_root: Path, hubs: dict[str, str],
              cfg: FeatureConfig, min_parser_version: int = 2) -> tuple[pd.DataFrame, dict]:
    """Features for the eligible rows of one built dataset day, plus build facts."""
    rows = pd.read_parquet(dataset_day / "part-0.parquet")
    rows = rows[rows.eligible].reset_index(drop=True)
    for c in TIME_COLS:
        rows[c] = to_naive_utc(rows[c])
    t0, t1 = service_day_bounds(day)
    # Data is read only up to the last cutoff of the day: anything later may not be used
    # anyway, so this is a second, structural guard against leakage.
    t_from = t0 - pd.Timedelta(days=1)
    t_to = rows.t_cut.max() if len(rows) else t1
    stations = list(hubs) + [e for e, _ in cfg.feeders]      # the table holds every group, feeder included
    w = load_window(parsed_root, t_from, t_to, min_parser_version, stations=stations)
    ev_msgs, st_msgs = load_messages(parsed_root, t_from, t_to, list(hubs))
    feats = compute_all(rows, w.plan, w.obs, ev_msgs, st_msgs, cfg)
    keys = rows[[c for c in KEYS + TARGET if c not in feats.columns]]
    df = pd.concat([keys, feats[ALL_FEATURES]], axis=1)
    facts = {"read_from": t_from.isoformat(), "read_to": t_to.isoformat(),
             "plan_changes": plan_changes(parsed_root, t_from, t_to, stations)}
    return df, facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build features per service day")
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset" / "v1")
    ap.add_argument("--parsed", type=Path, default=ROOT / "data" / "restore" / "parsed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "features")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "features.toml")
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "config" / "dataset.toml")
    ap.add_argument("--from", dest="d_from", type=date.fromisoformat, required=True)
    ap.add_argument("--to", dest="d_to", type=date.fromisoformat, required=True)
    args = ap.parse_args(argv)

    cfg = load_feature_config(args.config)
    dcfg = load_config(args.dataset_config)
    out_root = args.out / f"v{cfg.feature_version}"
    built, day = 0, args.d_from
    while day <= args.d_to:
        src = args.dataset / f"service_day={day}"
        if not (src / "part-0.parquet").exists():
            print(f"{day}: no dataset day, skipped")
        else:
            df, facts = build_day(day, src, args.parsed, dcfg.hubs, cfg, dcfg.min_parser_version)
            meta = {"service_day": day.isoformat(), "rows": len(df), "features": ALL_FEATURES,
                    "non_missing_pct": {c: round(100 * float(df[c].notna().mean()), 1) if len(df) else None
                                        for c in ALL_FEATURES},
                    **facts, "config": cfg.as_dict(),
                    "dataset_built_at": json.loads((src / "_meta.json").read_text(encoding="utf-8")).get("built_at"),
                    "git_commit": _git_commit(),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            target = write_day(df, meta, out_root)
            built += 1
            low = [c for c, v in meta["non_missing_pct"].items() if v is not None and v < 50]
            print(f"{day}: {len(df):,} rows, {len(ALL_FEATURES)} features -> {target}"
                  + (f"  (mostly missing: {low})" if low else ""))
            if facts["plan_changes"]:
                print(f"  WARNING: {facts['plan_changes']} hub or feeder events changed their planned time "
                      f"between plan versions; hub and feeder features assume planned times never change")
        day += timedelta(days=1)
    print(f"built {built} day(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
