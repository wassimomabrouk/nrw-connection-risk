"""Fit a deployable bundle (the chosen model + B3) and save it to models/.

Run from the repo root:
    python -m nrw_connection_risk.training.fit_bundle --model gbm --note "stand-in"
    python -m nrw_connection_risk.training.fit_bundle --model lr --period train+validation   # final refit

The model choice itself is made on validation (DESIGN.md section 4); this command only
fits and packages it. Test and robustness days can never be used.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..dataset.build import _git_commit
from ..dataset.config import load_config
from ..features.config import load_feature_config
from ..monitoring.profile import build_profile
from .bundle import COMPANION, Bundle, new_model_id, save_bundle, versions
from .config import MODELS, load_training_config
from .evaluate import available_days, load_rows
from .models import make_model
from .splits import days_in

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit and save a model bundle")
    ap.add_argument("--model", choices=MODELS, default="gbm")
    ap.add_argument("--period", choices=("train", "train+validation"), default="train")
    ap.add_argument("--groups", nargs="+", help="override the feature groups (default: config/features.toml)")
    ap.add_argument("--features", type=Path, default=None, help="default: data/features/v{feature_version}")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "training.toml")
    ap.add_argument("--features-config", type=Path, default=ROOT / "config" / "features.toml")
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "config" / "dataset.toml")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    ap.add_argument("--note", default="", help="free text stored in the model card")
    args = ap.parse_args(argv)

    cfg = load_training_config(args.config)
    fcfg = load_feature_config(args.features_config)
    if args.groups:
        fcfg = replace(fcfg, groups=tuple(args.groups))
    features_root = args.features or ROOT / "data" / "features" / f"v{fcfg.feature_version}"
    avail = available_days(features_root)
    days = days_in("train", avail, cfg.splits)
    if args.period == "train+validation":
        days += days_in("validation", avail, cfg.splits)
    if not days:
        raise SystemExit(f"no built feature days for period {args.period} under {features_root}")
    rows = load_rows(features_root, days, cfg.splits)

    cols = fcfg.columns()
    print(f"fitting {args.model} and {COMPANION} on {len(days)} day(s) ({days[0]} to {days[-1]}), {len(rows):,} rows")
    models = {"model": make_model(args.model, cfg, cols).fit(rows),
              COMPANION: make_model(COMPANION, cfg, cols).fit(rows)}
    meta = {
        "model_id": new_model_id(args.model), "model": args.model, "companion": COMPANION,
        "note": args.note, "period": args.period,
        "fit_days": {"first": str(days[0]), "last": str(days[-1]), "count": len(days)}, "rows": len(rows),
        "cutoffs": sorted(int(c) for c in rows.cutoff_min.unique()),
        "feature_version": fcfg.feature_version, "feature_groups": list(fcfg.groups), "feature_columns": cols,
        "cutoff_mode": cfg.cutoff_mode, "calibration": cfg.calibration,
        "training_config": cfg.as_dict(), "feature_config": fcfg.as_dict(),
        "dataset_config": load_config(args.dataset_config).as_dict(),
        "git_commit": _git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": versions(),
        # training distribution of every input, for drift monitoring (monitoring/profile.py)
        "reference_profile": build_profile(rows, cols),
        "reference_fail_rate": {str(int(k)): float(v) for k, v in rows.groupby("cutoff_min").label_fail.mean().items()},
    }
    out = save_bundle(Bundle(models=models, meta=meta), args.out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
