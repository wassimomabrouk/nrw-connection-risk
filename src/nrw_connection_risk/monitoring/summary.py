"""Summary over all monitored days: a table per day and the pooled live comparison of
model and B3 per cutoff, with 95% intervals from resampling whole days.

Writes data/monitoring/summary.json (for the dashboard) and summary.md (readable).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..training.metrics import bootstrap_vs, scores


def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def _fmt(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def _diff(x):
    return "n/a" if x is None else f"{x:+.3f}"


def _share(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def write_summary(cfg) -> dict:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((cfg.out / "daily").glob("service_day=*.json"))]
    L = str(cfg.primary_cutoff)
    days = []
    for r in reports:
        row = {"service_day": r["service_day"], "status": r["status"]}
        if r["status"] == "ok":
            p = r["performance"].get(L, {})
            ops = r["operations"]
            drifts = [d for d in r["drift"].values() if d.get("available")]
            worst = max(((k, v) for d in drifts for k, v in d["psi"].items()), key=lambda kv: kv[1], default=(None, None))
            skews = [d for d in r.get("skew", {}).values() if isinstance(d, dict) and d.get("available") and d.get("rows")]
            skew_rows = sum(d["rows"] for d in skews)
            skew_same = sum(d["rows"] * d["identical_share"] for d in skews) / skew_rows if skew_rows else None
            row.update(logged=ops["logged"], rows=p.get("rows"), fail_rate=p.get("fail_rate"),
                       log_loss_model=p.get("model", {}).get("log_loss"), log_loss_B3=p.get("B3", {}).get("log_loss"),
                       gain_vs_B3=p.get("log_loss_gain_vs_B3"), auc_model=p.get("model", {}).get("auc"),
                       auc_B3=p.get("B3", {}).get("auc"), coverage=ops["coverage"].get(L),
                       data_age_p95_s=ops["data_age_s"]["p95"], max_psi_feature=worst[0], max_psi=worst[1],
                       skew_identical=_r(skew_same),
                       model_ids=r["model_ids"])
        else:
            row["reason"] = r.get("reason")
        days.append(row)

    files = sorted((cfg.out / "joined").glob("service_day=*.parquet"))
    pooled = {}
    if files:
        j = pd.concat([pd.read_parquet(f).assign(service_day=f.stem.split("=", 1)[1]) for f in files], ignore_index=True)
        ev = j[j.outcome.eq("evaluated")]
        for cut, g in ev.groupby("cutoff_min"):
            y = g.label_fail.astype(int).to_numpy()
            pm, pb = g.p_model.to_numpy(float), g.p_B3.to_numpy(float)
            b = bootstrap_vs(g.service_day.to_numpy(), y, pm, pb, cfg.bootstrap_n, cfg.bootstrap_n_auc, cfg.seed)
            pooled[str(int(cut))] = {
                "days": b["days"], "rows": int(len(g)), "fail_rate": _r(y.mean()),
                "log_loss_model": _r(scores(y, pm)["log_loss"]), "log_loss_B3": _r(scores(y, pb)["log_loss"]),
                "gain_vs_B3": _r(b["log_loss_gain"]), "gain_ci": [_r(v) for v in b["log_loss_gain_ci"]],
                "auc_diff": _r(b["auc_diff"]), "auc_diff_ci": [_r(v) for v in b["auc_diff_ci"]]}

    summary = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "primary_cutoff": cfg.primary_cutoff, "days": sum(d["status"] == "ok" for d in days),
               "per_day": days, "pooled": pooled,
               "latest": next((r for r in reversed(reports) if r["status"] == "ok"), None)}
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (cfg.out / "summary.md").write_text(_markdown(summary, cfg), encoding="utf-8")
    return summary


def _markdown(s: dict, cfg) -> str:
    L = s["primary_cutoff"]
    out = ["# Live monitoring", "", f"Created {s['created_at']}; {s['days']} evaluated day(s). Live predictions are "
           "scored against the outcomes the dataset builder derives from the collected data (same labels as "
           "training). Gain = relative log loss reduction of the model vs B3 (positive = better).", ""]
    if s["pooled"]:
        out += ["## All days pooled", "", "| cutoff | days | rows | fail rate | log loss model | log loss B3 | "
                "gain vs B3 [95% CI] | AUC diff [95% CI] |", "|---|---|---|---|---|---|---|---|"]
        for cut in sorted(s["pooled"], key=int, reverse=True):
            p = s["pooled"][cut]
            out.append(f"| {cut} min | {p['days']} | {p['rows']:,} | {_fmt(p['fail_rate'], 3)} | "
                       f"{_fmt(p['log_loss_model'])} | {_fmt(p['log_loss_B3'])} | {_pct(p['gain_vs_B3'])} "
                       f"[{_pct(p['gain_ci'][0])}, {_pct(p['gain_ci'][1])}] | "
                       f"{_diff(p['auc_diff'])} [{_diff(p['auc_diff_ci'][0])}, {_diff(p['auc_diff_ci'][1])}] |")
        if min(p["days"] for p in s["pooled"].values()) < 10:
            out += ["", "> Fewer than 10 days: intervals from resampling days are not reliable yet."]
        out.append("")
    out += [f"## Per day ({L} min)", "", "| day | status | logged | rows | fail rate | log loss model | "
            "log loss B3 | gain | coverage | data age p95 | largest drift (PSI) | live = offline |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in s["per_day"]:
        if d["status"] != "ok":
            out.append(f"| {d['service_day']} | {d['status']} | | | | | | | | | | {d.get('reason') or ''} |")
            continue
        out.append(f"| {d['service_day']} | ok | {d['logged']:,} | {d['rows'] or 0:,} | {_fmt(d['fail_rate'], 3)} | "
                   f"{_fmt(d['log_loss_model'])} | {_fmt(d['log_loss_B3'])} | {_pct(d['gain_vs_B3'])} | "
                   f"{_share(d['coverage'])} | "
                   f"{d['data_age_p95_s']} s | {d['max_psi_feature'] or 'n/a'} {_fmt(d['max_psi'], 3)} | "
                   f"{_share(d['skew_identical'])} |")
    latest = s["latest"]
    if latest:
        out += ["", f"## Drift on {latest['service_day']}", ""]
        for model_id, d in latest["drift"].items():
            if not d.get("available"):
                out.append(f"- {model_id}: {d['reason']}")
                continue
            out += [f"Model {model_id}, {d['rows']:,} logged rows. PSI below 0.1 stable, 0.1 to 0.25 moderate, "
                    "above 0.25 large.", "", "| input | PSI | level |", "|---|---|---|"]
            out += [f"| {k} | {v:.3f} | {d['level'][k]} |" for k, v in d["psi"].items()]
            out += ["", "| cutoff | fail rate live | fail rate training |", "|---|---|---|"]
            out += [f"| {c} min | {_fmt(v['live'], 3)} | {_fmt(v['training'], 3)} |"
                    for c, v in sorted(d["fail_rate"].items(), key=lambda kv: -int(kv[0]))]
        out += ["", f"## Training/serving skew on {latest['service_day']}", "",
                "Features recomputed offline at each prediction's scoring time vs the features the API logged."]
        for model_id, d in latest.get("skew", {}).items():
            if not isinstance(d, dict) or not d.get("available") or not d.get("rows"):
                out.append(f"- {model_id}: {d.get('reason', 'no rows') if isinstance(d, dict) else d}")
                continue
            least = ", ".join(f"{k} {_share(d['per_feature'][k]['equal_share'])}" for k in d["least_equal"])
            out.append(f"- {model_id}: {d['rows']:,} rows, {_share(d['identical_share'])} identical in every feature"
                       + (f"; least equal: {least}" if least else ""))
        ops = latest["operations"]
        out += ["", f"## Operations on {latest['service_day']}", "",
                f"- Logged predictions: {ops['logged']:,} ({', '.join(f'{k} {v:,}' for k, v in ops['by_outcome'].items())})",
                "- Coverage of eligible connections: " + ", ".join(f"{c} min {100 * v:.1f}%" for c, v in
                                                                   sorted(ops['coverage'].items(), key=lambda kv: -int(kv[0]))),
                f"- Logging lag after the cutoff: median {ops['lag_s']['p50']} s, p95 {ops['lag_s']['p95']} s",
                f"- Data age at scoring: median {ops['data_age_s']['p50']} s, p95 {ops['data_age_s']['p95']} s, "
                f"max {ops['data_age_s']['max']} s"]
    return "\n".join(out) + "\n"
