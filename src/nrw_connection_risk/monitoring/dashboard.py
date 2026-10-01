"""The monitoring dashboard: one self-contained HTML page (inline SVG charts, no scripts,
no external resources) built from summary.json and the service's live health.

Colors: model = categorical slot 1 (blue), B3 = slot 2 (orange), validated pair from the
reference palette; status colors only with an icon and a label; light and dark themes.
"""
from __future__ import annotations

import math
from html import escape

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--warn:#fab219;
--crit:#d03b3b;--goodtext:#006300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--goodtext:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goodtext:#0ca30c}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 20px}.note{color:var(--muted);font-size:13px;margin:6px 0 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:24px;font-weight:600;margin-top:2px}
.tile .detail{color:var(--muted);font-size:13px}.hero{grid-column:span 2}.hero .value{font-size:48px;line-height:1.1}
.chart{overflow-x:auto}.chart svg{min-width:640px}.wide text{font-size:13px}
.bars{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;align-items:center;font-size:12px;font-variant-numeric:tabular-nums;margin-top:8px}
.bars .name{color:var(--ink2);text-align:right;white-space:nowrap}
.bars .track{position:relative;height:14px}.bars .bar{position:absolute;left:0;top:2px;height:10px;background:var(--s1);border-radius:0 3px 3px 0}
.bars .val{position:absolute;top:-1px;color:var(--ink2);white-space:nowrap}
.bars .rule{position:absolute;top:-3px;bottom:-3px;border-left:1px solid var(--axis)}
.bars .head{position:relative;height:16px;color:var(--muted);font-size:11px}.bars .head span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:12px;margin-bottom:16px}
.card{margin-bottom:12px}svg{width:100%;height:auto;display:block}
svg text{fill:var(--muted);font-size:11px;font-family:inherit}
.legend{display:flex;gap:16px;color:var(--ink2);font-size:13px;margin:4px 0 8px}
.key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600}th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto}details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:8px}
.status{display:inline-flex;align-items:center;gap:6px;font-weight:600}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.empty{color:var(--ink2);padding:24px 0}a{color:var(--s1)}
"""

STATUS = {"ok": ("var(--good)", "✓", "OK"), "degraded": ("var(--warn)", "!", "Degraded"),
          "starting": ("var(--warn)", "…", "Starting"), "error": ("var(--crit)", "✕", "Error"),
          "no_model": ("var(--crit)", "✕", "No model")}
ARROW = " \u2192"
LEVEL = {"stable": ("var(--good)", "✓"), "moderate": ("var(--warn)", "!"), "large": ("var(--crit)", "✕")}


def _pct(x, signed=True) -> str:
    return "n/a" if x is None else (f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%")


def _num(x, d=4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= step * m:
            step *= m
            break
    start = math.floor(lo / step) * step
    return [round(start + i * step, 10) for i in range(int(math.ceil((hi - start) / step)) + 1)]


def _status_badge(status: str) -> str:
    color, icon, label = STATUS.get(status, ("var(--muted)", "?", status))
    return f'<span class="status"><span style="color:{color}">{icon}</span> {escape(label)}</span>'


def _legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="key" style="background:{c}"></span>{escape(t)}</span>' for t, c in items) + "</div>"


# ---------------------------------------------------------------- charts

def _tick_decimals(ticks: list[float]) -> int:
    """Fewest decimals that keep neighbouring tick labels distinct (0.462 / 0.464, not 0.46 / 0.46)."""
    if len(ticks) < 2:
        return 2
    step = ticks[1] - ticks[0]
    for d in range(0, 7):
        if abs(round(step, d) - step) < step * 1e-6:
            return d
    return 6


def line_chart(days: list[str], series: list[tuple[str, str, list]], fmt=lambda v: f"{v:.4f}",
               axis_fmt=None) -> str:
    """Values per day, one line per series (None = gap), with hover titles."""
    W, H, L, R, T, B = 940, 260, 48, 70, 12, 28
    vals = [v for _, _, ys in series for v in ys if v is not None]
    if not vals:
        return '<p class="empty">No values yet.</p>'
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.1 or 0.1
    ticks = _ticks(lo - pad, hi + pad)
    y0, y1 = ticks[0], ticks[-1]
    if axis_fmt is None:
        dec = _tick_decimals(ticks)
        axis_fmt = lambda v: f"{v:.{dec}f}"   # noqa: E731
    xs = lambda i: L + (W - L - R) * (i / max(len(days) - 1, 1) if len(days) > 1 else 0.5)   # noqa: E731
    ys = lambda v: T + (H - T - B) * (1 - (v - y0) / (y1 - y0))   # noqa: E731
    out = [f'<svg class="wide" viewBox="0 0 {W} {H}" role="img">']
    for t in ticks:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(t):.1f}" y2="{ys(t):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                   f'<text x="{L - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">{axis_fmt(t)}</text>')
    step = max(1, math.ceil(len(days) / 8))
    for i, d in enumerate(days):
        if i % step == 0 or i == len(days) - 1:
            out.append(f'<text x="{xs(i):.1f}" y="{H - 8}" text-anchor="middle">{escape(d[5:])}</text>')
    for name, color, yv in series:
        pts = [(xs(i), ys(v)) for i, v in enumerate(yv) if v is not None]
        if len(pts) > 1:
            out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(yv):
            if v is not None:
                out.append(f'<g><title>{escape(days[i])}: {escape(name)} {fmt(v)}</title>'
                           f'<circle cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="10" fill="transparent"/>'
                           f'<circle cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="4" fill="{color}" stroke="var(--surface)" '
                           f'stroke-width="2"/></g>')
        last = next(((i, v) for i, v in reversed(list(enumerate(yv))) if v is not None), None)
        if last and len(series) <= 4:
            out.append(f'<text x="{xs(last[0]) + 10:.1f}" y="{ys(last[1]) + 4:.1f}" style="fill:var(--ink2)">'
                       f'{escape(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def calibration_chart(cal: dict) -> str:
    """Observed failure rate vs mean predicted probability per bin, model and B3."""
    W, H, L, R, T, B = 440, 330, 44, 16, 12, 34
    xs = lambda v: L + (W - L - R) * v   # noqa: E731
    ys = lambda v: T + (H - T - B) * (1 - v)   # noqa: E731
    out = [f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" style="min-width:340px;max-width:460px">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(t):.1f}" y2="{ys(t):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 6}" y="{ys(t) + 4:.1f}" text-anchor="end">{t:g}</text>'
                   f'<text x="{xs(t):.1f}" y="{H - 16}" text-anchor="middle">{t:g}</text>')
    out.append(f'<line x1="{xs(0)}" y1="{ys(0)}" x2="{xs(1)}" y2="{ys(1)}" stroke="var(--axis)" stroke-width="1"/>'
               f'<text x="{W - R}" y="{H - 2}" text-anchor="end">mean predicted</text>')
    for name, color in (("model", "var(--s1)"), ("B3", "var(--s2)")):
        bins = [b for b in cal.get(name, []) if b.get("n")]
        pts = [(xs(b["mean_p"]), ys(b["observed"])) for b in bins]
        if len(pts) > 1:
            out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for b, (x, y) in zip(bins, pts):
            out.append(f'<g><title>{escape(name)} {escape(str(b["bin"]))}: predicted {b["mean_p"]:.3f}, observed '
                       f'{b["observed"]:.3f}, {b["n"]:,} rows</title><circle cx="{x:.1f}" cy="{y:.1f}" r="10" '
                       f'fill="transparent"/><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                       f'stroke="var(--surface)" stroke-width="2"/></g>')
    out.append("</svg></div>")
    return "".join(out)


def psi_chart(psi: dict, levels: dict, moderate=0.1, large=0.25) -> str:
    """PSI per input as HTML bars (text stays readable at any width), with the two thresholds;
    values beyond the axis end are marked with an arrow."""
    items = sorted(psi.items(), key=lambda kv: -kv[1])
    top = max(0.3, min(1.0, max((v for _, v in items), default=0) * 1.1))
    pos = lambda v: 75 * min(v, top) / top   # bar area: 75% of the track, the rest for the value  # noqa: E731
    rules = "".join(f'<div class="rule" style="left:{pos(t):.1f}%"></div>' for t in (moderate, large))
    out = ['<div class="bars">']
    for k, v in items:
        color, icon = LEVEL.get(levels.get(k, "stable"), ("var(--muted)", ""))
        w = pos(v)
        out.append(f'<div class="name">{escape(k)}</div><div class="track" title="{escape(k)}: PSI {v:.3f} '
                   f'({escape(levels.get(k, ""))})">{rules}<div class="bar" style="width:{max(w, 0.4):.1f}%"></div>'
                   f'<span class="val" style="left:calc({w:.1f}% + 6px)">{v:.3f}{ARROW if v > top else ""} '
                   f'<span style="color:{color}">{icon}</span></span></div>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- page

def render(summary: dict | None, health: dict, model: dict | None) -> str:
    L = str((summary or {}).get("primary_cutoff", 30))
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>NRW connection risk monitoring</title>'
            f'<style>{CSS}</style></head><body><main>')
    parts = [head, "<h1>NRW connection risk: live monitoring</h1>",
             '<p class="sub">How the deployed model performs on live traffic at five Rhine-Ruhr hubs, compared with '
             "DB's own prognosis turned into a probability (B3). Outcomes come from the collected data, labelled like "
             "the training data.</p>"]
    age = health.get("data_age_s")
    tiles = [("Service", _status_badge(health.get("status", "?")),
              f"data age {age} s" if age is not None else "no data yet")]
    latest_day = None
    if summary and summary.get("days"):
        pooled = summary.get("pooled", {}).get(L)
        per_day = [d for d in summary["per_day"] if d["status"] == "ok"]
        latest_day = per_day[-1] if per_day else None
        if pooled:
            ci = pooled["gain_ci"]
            tiles.insert(0, (f"Log loss vs B3, {L} min, all days", _pct(pooled["gain_vs_B3"]),
                             f"95% CI {_pct(ci[0])} to {_pct(ci[1])}, {pooled['days']} day(s), "
                             f"{pooled['rows']:,} predictions", "hero"))
        if latest_day:
            tiles += [("Coverage, latest day", _pct(latest_day["coverage"], signed=False),
                       f"of eligible connections, {latest_day['service_day']}"),
                      ("Live = offline features", _pct(latest_day.get("skew_identical"), signed=False),
                       "rows identical in every input"),
                      ("Largest drift (PSI)", _num(latest_day["max_psi"], 3), escape(latest_day["max_psi_feature"] or "n/a"))]
    parts.append('<div class="tiles">' + "".join(
        f'<div class="tile{" hero" if len(t) > 3 else ""}"><div class="label">{escape(t[0])}</div>'
        f'<div class="value">{t[1]}</div><div class="detail">{t[2]}</div></div>' for t in tiles) + "</div>")

    if not summary or not summary.get("days"):
        parts.append('<div class="card"><p class="empty">No evaluated day yet. The daily evaluation runs every morning '
                     "at 09:15 UTC, once a service day's outcomes are known.</p></div>")
    else:
        per_day = [d for d in summary["per_day"] if d["status"] == "ok"]
        days = [d["service_day"] for d in per_day]
        legend = _legend([("model", "var(--s1)"), ("B3 (DB prognosis)", "var(--s2)")])
        parts.append(f'<div class="card"><h2>Log loss per day, {L} minutes before arrival</h2>'
                     '<p class="note">Lower is better. Each point is one service day of live predictions.</p>'
                     + legend + '<div class="chart">' + line_chart(days, [("model", "var(--s1)", [d["log_loss_model"] for d in per_day]),
                                                  ("B3", "var(--s2)", [d["log_loss_B3"] for d in per_day])]) + "</div></div>")
        latest = summary.get("latest") or {}
        cal = latest.get("performance", {}).get(L, {}).get("calibration")
        drift = next((d for d in latest.get("drift", {}).values() if d.get("available")), None)
        cards = []
        if cal:
            cards.append(f'<div class="card"><h2>Calibration, {escape(latest["service_day"])}</h2>'
                         '<p class="note">Observed failure rate per band of predicted probability; on the diagonal = '
                         "well calibrated.</p>" + legend + calibration_chart(cal) + "</div>")
        if drift:
            cards.append(f'<div class="card"><h2>Input drift, {escape(latest["service_day"])}</h2>'
                         '<p class="note">Population stability index of each input against the training data. '
                         "Lines at 0.1 (moderate) and 0.25 (large). ✓ stable, ! moderate, ✕ large.</p>" + psi_chart(drift["psi"], drift["level"]) + "</div>")
        if cards:
            parts.append('<div class="cards">' + "".join(cards) + "</div>")

        rows = "".join(
            f"<tr><td>{escape(d['service_day'])}</td><td>{d['logged']:,}</td><td>{(d['rows'] or 0):,}</td>"
            f"<td>{_num(d['fail_rate'], 3)}</td><td>{_num(d['log_loss_model'])}</td><td>{_num(d['log_loss_B3'])}</td>"
            f"<td>{_pct(d['gain_vs_B3'])}</td><td>{_pct(d['coverage'], False)}</td><td>{d['data_age_p95_s']} s</td>"
            f"<td>{_pct(d.get('skew_identical'), False)}</td></tr>" for d in reversed(per_day))
        parts.append(f'<div class="card"><h2>Per day ({L} min)</h2><div class="scroll"><table><tr><th>day</th>'
                     "<th>logged</th><th>evaluated</th><th>fail rate</th><th>log loss model</th><th>log loss B3</th>"
                     "<th>gain</th><th>coverage</th><th>data age p95</th><th>live = offline</th></tr>"
                     f"{rows}</table></div></div>")
        prow = "".join(
            f"<tr><td>{c} min</td><td>{p['days']}</td><td>{p['rows']:,}</td><td>{_num(p['fail_rate'], 3)}</td>"
            f"<td>{_num(p['log_loss_model'])}</td><td>{_num(p['log_loss_B3'])}</td><td>{_pct(p['gain_vs_B3'])}</td>"
            f"<td>{_pct(p['gain_ci'][0])} to {_pct(p['gain_ci'][1])}</td></tr>"
            for c, p in sorted(summary.get("pooled", {}).items(), key=lambda kv: -int(kv[0])))
        parts.append('<div class="card"><h2>All days pooled</h2><div class="scroll"><table><tr><th>cutoff</th>'
                     "<th>days</th><th>rows</th><th>fail rate</th><th>log loss model</th><th>log loss B3</th>"
                     f"<th>gain vs B3</th><th>95% CI</th></tr>{prow}</table></div>"
                     '<p class="note">Intervals from resampling whole service days; below 10 days they are not '
                     "reliable.</p></div>")

    note = escape(model.get("note", "")) if model else ""
    parts.append(f'<p class="note">Model {escape(str(health.get("model_id")))}{" (" + note + ")" if note else ""}. '
                 "Live monitoring does not replace the locked test evaluation. "
                 'Data: <a href="/v1/monitoring">/v1/monitoring</a> · API: <a href="/docs">/docs</a></p>')
    parts.append("</main></body></html>")
    return "".join(parts)
