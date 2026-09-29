"""Build the training table for service days: one row per transfer candidate and cutoff.

Run from the repo root:
    python -m nrw_connection_risk.dataset.build --parsed data/restore/parsed --day 2026-09-24
    python -m nrw_connection_risk.dataset.build --parsed data/restore/parsed --from 2026-09-24 --to 2026-09-30

Output: <out>/service_day=YYYY-MM-DD/part-0.parquet and _meta.json (counts, funnel,
settings, versions). Rebuilding a day replaces it. Days whose data does not yet
reach label_horizon_h hours past their end are skipped, not built partially.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import columns
from .candidates import build_candidates
from .config import DatasetConfig, load_config
from .load import ParsedLayerError, load_window
from .metrics import roc_auc
from .state import final_state, last_poll_at, state_at
from .timeutil import is_ambiguous_local, service_day_bounds

ROOT = Path(__file__).resolve().parents[3]

EXCLUSION_ORDER = ["not_known_at_cutoff", "collector_gap", "a_cancelled_at_cutoff",
                   "unobserved", "stale_label", "dst_ambiguous"]


class IncompleteDay(RuntimeError):
    """The data does not yet cover the day plus the label horizon."""


def _minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def build_day(day: date, parsed_root: Path, cfg: DatasetConfig) -> tuple[pd.DataFrame, dict]:
    t0, t1 = service_day_bounds(day, cfg.start_hour_local)
    need_until = t1 + pd.Timedelta(hours=cfg.label_horizon_h)
    w = load_window(parsed_root, t0 - pd.Timedelta(days=1), need_until + pd.Timedelta(hours=6),
                    cfg.min_parser_version, stations=list(cfg.hubs))
    if w.data_end < need_until:
        raise IncompleteDay(f"data ends {w.data_end} UTC, labels need data until {need_until} UTC")

    cands, funnel = build_candidates(w.plan, t0, t1, cfg.hubs, cfg.min_transfer_min, cfg.max_slack_min)
    key_a = cands.stop_id_a + "|ar"
    key_b = cands.stop_id_b + "|dp"

    # ---- label: final state after the event (DESIGN.md section 2) ----
    fa, fb = final_state(key_a, w.obs), final_state(key_b, w.obs)
    lab = pd.DataFrame(index=cands.index)
    lab["a_ct_final"], lab["a_cs_final"], lab["a_last_obs"] = fa.ct, fa.cs, fa.obs
    lab["b_ct_final"], lab["b_cs_final"], lab["b_last_obs"] = fb.ct, fb.cs, fb.obs
    a_cancel, b_cancel = fa.cs.eq("c"), fb.cs.eq("c")
    lab["delay_a_final_min"] = _minutes(fa.ct - cands.pt_a)
    lab["delay_b_final_min"] = _minutes(fb.ct - cands.pt_b)
    lab["real_slack_min"] = _minutes(fb.ct - fa.ct)
    delay_fail = lab.real_slack_min < cfg.min_transfer_min
    lab["label_fail"] = a_cancel | b_cancel | delay_fail
    lab["fail_reason"] = np.select([a_cancel, b_cancel, delay_fail], ["a_cancelled", "b_cancelled", "delay"], "none")
    observed = (fa.ct.notna() | a_cancel) & (fb.ct.notna() | b_cancel)
    stale = ((fa.ct.notna() & ~a_cancel & (fa.obs < fa.ct))
             | (fb.ct.notna() & ~b_cancel & (fb.obs < fb.ct)))
    ambiguous = pd.Series([any(is_ambiguous_local(v) for v in vals) for vals in
                           zip(cands.pt_raw_a, cands.pt_raw_b, fa.ct_raw, fb.ct_raw)], index=cands.index)

    # ---- point-in-time state at each cutoff ----
    parts = []
    for L in cfg.cutoffs_min:
        t_cut = cands.pt_a - pd.Timedelta(minutes=L)
        sa, sb = state_at(key_a, t_cut, w.obs), state_at(key_b, t_cut, w.obs)
        poll = last_poll_at(cands.eva, t_cut, w.polls)
        d = cands.copy()
        d.insert(0, "cutoff_min", L)
        d.insert(1, "t_cut", t_cut)
        d["a_ct_cut"], d["a_cs_cut"], d["a_obs_cut"] = sa.ct, sa.cs, sa.obs
        d["b_ct_cut"], d["b_cs_cut"], d["b_obs_cut"] = sb.ct, sb.cs, sb.obs
        a_pred = sa.ct.fillna(cands.pt_a)
        b_pred = sb.ct.fillna(cands.pt_b)
        d["db_delay_a_min"] = _minutes(a_pred - cands.pt_a)
        d["db_delay_b_min"] = _minutes(b_pred - cands.pt_b)
        d["db_slack_min"] = _minutes(b_pred - a_pred)
        d["b_cancel_known"] = sb.cs.eq("c")
        d["collector_age_min"] = _minutes(t_cut - poll)
        d = d.join(lab)

        reasons = {
            "not_known_at_cutoff": (cands.first_seen_a > t_cut) | (cands.first_seen_b > t_cut),
            "collector_gap": poll.isna() | (d.collector_age_min > cfg.max_collector_gap_min),
            "a_cancelled_at_cutoff": sa.cs.eq("c"),
            "unobserved": ~observed,
            "stale_label": stale,
            "dst_ambiguous": ambiguous,
        }
        reason = pd.Series(None, index=d.index, dtype=object)
        for name in reversed(EXCLUSION_ORDER):            # first reason in the order wins
            reason = reason.mask(reasons[name].fillna(False).astype(bool), name)
        d["exclusion_reason"] = reason
        d["eligible"] = reason.isna()
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "service_day", day.isoformat())
    out = out[columns.ALL]                               # fixed column order, see columns.py
    return out, _summary(day, out, funnel, cfg, w, t0, t1)


def _summary(day, out, funnel, cfg, w, t0, t1) -> dict:
    per_cutoff = []
    for L, g in out.groupby("cutoff_min", sort=False):
        e = g[g.eligible]
        score_db = (-e.db_slack_min).where(~e.b_cancel_known, 1e9)
        per_cutoff.append({
            "cutoff_min": int(L), "rows": len(g), "eligible": len(e),
            "fail_pct": round(100 * e.label_fail.mean(), 2) if len(e) else None,
            "auc_planned_slack": round(roc_auc(-e.planned_slack_min, e.label_fail), 3),
            "auc_db_slack": round(roc_auc(score_db, e.label_fail), 3),
            "excluded": {k: int(v) for k, v in g.exclusion_reason.value_counts().items()},
        })
    first = out[out.cutoff_min == cfg.cutoffs_min[0]]
    obs_ok = first.exclusion_reason.ne("unobserved") & first.exclusion_reason.ne("stale_label")
    return {
        "service_day": day.isoformat(),
        "window_utc": [t0.isoformat(), t1.isoformat()],
        "candidates": int(len(first)),
        "fail_pct_all_observed": round(100 * first[obs_ok].label_fail.mean(), 2) if len(first) else None,
        "fail_reasons": {k: int(v) for k, v in first[obs_ok].fail_reason.value_counts().items()},
        "per_cutoff": per_cutoff,
        "funnel": funnel,
        "config": cfg.as_dict(),
        "data_span_utc": [w.data_start.isoformat(), w.data_end.isoformat()],
        "git_commit": _git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def write_day(df: pd.DataFrame, meta: dict, out_root: Path) -> Path:
    """Write atomically: build in a temp folder, then replace the day's folder."""
    target = out_root / f"service_day={meta['service_day']}"
    tmp = out_root / f".tmp-{meta['service_day']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    df = df.copy()
    for c in df.columns:                      # store every timestamp explicitly as UTC
        if pd.api.types.is_datetime64_dtype(df[c]) and getattr(df[c].dt, "tz", None) is None:
            df[c] = df[c].dt.tz_localize("UTC")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp / "part-0.parquet", compression="zstd")
    (tmp / "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


def _days(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the transfer dataset per service day")
    ap.add_argument("--parsed", type=Path, default=ROOT / "data" / "restore" / "parsed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "dataset")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "dataset.toml")
    ap.add_argument("--day", type=date.fromisoformat)
    ap.add_argument("--from", dest="d_from", type=date.fromisoformat)
    ap.add_argument("--to", dest="d_to", type=date.fromisoformat)
    args = ap.parse_args(argv)
    if args.day:
        args.d_from = args.d_to = args.day
    if not args.d_from or not args.d_to:
        ap.error("give --day or both --from and --to")

    cfg = load_config(args.config)
    out_root = args.out / f"v{cfg.builder_version}"
    built = 0
    for day in _days(args.d_from, args.d_to):
        try:
            df, meta = build_day(day, args.parsed, cfg)
        except IncompleteDay as exc:
            print(f"{day}: skipped, not complete yet ({exc})")
            continue
        except ParsedLayerError as exc:
            print(f"{day}: {exc}", file=sys.stderr)
            return 2
        target = write_day(df, meta, out_root)
        built += 1
        print(f"{day}: {meta['candidates']:,} candidates, failure rate {meta['fail_pct_all_observed']} % "
              f"-> {target}")
        for c in meta["per_cutoff"]:
            print(f"    cutoff {c['cutoff_min']:>2} min: eligible {c['eligible']:>6,}  fail {c['fail_pct']} %  "
                  f"AUC planned {c['auc_planned_slack']}  AUC DB {c['auc_db_slack']}  excluded {c['excluded']}")
    print(f"built {built} day(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
