"""e08a: where did the live gain over B3 go after the Essen-Duisburg works ended?

Late October (26-30 Oct, Mon-Fri: autumn holidays, construction at Essen) against early
November (2-5 Nov, Mon-Thu), per hub, at the 30-minute cutoff, on the live predictions the
monitoring joined to their outcomes (training-period days only). Is the model worse, or B3
better? Did the network state at the hubs change? Motivates tuning the training window
(DESIGN.md section 4). Output: exploration/out/e08a_post_construction_drift.txt

Reads data/monitoring/joined, which exists on the server only. Run there from the repo root:
    .venv/bin/python exploration/e08a_post_construction_drift.py
"""
import numpy as np
import pandas as pd

J = "data/monitoring/joined/service_day={}.parquet"
DAYS = [d.isoformat() for d in pd.date_range("2026-10-26", "2026-11-05").date]
LATE_OCT = {"2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"}
EARLY_NOV = {"2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"}
COLS = ["outcome", "cutoff_min", "label_fail", "p_model", "p_B3", "hub", "model_id",
        "hub_share_cancel", "hub_share_late5", "hub_mean_delay"]
pd.set_option("display.width", 220)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def summary(g):
    y = g.label_fail.astype(int).to_numpy()
    m, b = ll(y, g.p_model), ll(y, g.p_B3)
    return pd.Series({"rows": len(g), "fail": round(y.mean(), 3), "ll_model": round(m, 4), "ll_B3": round(b, 4),
                      "gain_%": round(100 * (1 - m / b), 1),
                      "cancel": round(g.hub_share_cancel.mean(), 3), "late5": round(g.hub_share_late5.mean(), 3),
                      "mean_delay": round(g.hub_mean_delay.mean(), 2)})


parts = []
for d in DAYS:
    try:
        j = pd.read_parquet(J.format(d), columns=COLS)
    except FileNotFoundError:
        continue
    j = j[j.outcome.eq("evaluated") & j.cutoff_min.eq(30)]
    parts.append(j.assign(day=d))
e = pd.concat(parts, ignore_index=True)
e["period"] = np.where(e.day.isin(LATE_OCT), "late Oct", np.where(e.day.isin(EARLY_NOV), "early Nov", "-"))

print("1. Per day, all hubs (30 min)")
t = e.groupby("day").apply(summary, include_groups=False)
t["models"] = e.groupby("day").model_id.agg(lambda s: ", ".join(sorted(set(map(str, s)))))
print(t.to_string())

print("\n2. Per hub: late October (26-30 Oct) vs early November (2-5 Nov)")
p = e[e.period.ne("-")].groupby(["hub", "period"]).apply(summary, include_groups=False)
print(p.to_string())

print("\n3. Gain over B3 per hub and day (%)")
g = e.groupby(["day", "hub"]).apply(lambda x: summary(x)["gain_%"], include_groups=False).unstack("hub")
print(g.to_string())
