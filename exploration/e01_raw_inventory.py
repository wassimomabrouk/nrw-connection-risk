"""Exploration 01: what is actually inside the raw API responses?

The parser (collector/parse.py) extracts only the fields needed so far. The raw
layer keeps full responses, so this script inventories every XML element and
attribute that occurs, with a focus on elements relevant for defining transfers:
  - <conn>: connections DB itself manages at a station (with a status such as
    waiting / cannot wait / alternative), if the API delivers them
  - <m>:    messages, by type and position (stop, arrival, departure)
  - <ref>, <rtr>: references to related trips (replacements, splits, joins)

Run from the repo root:
    python exploration/e01_raw_inventory.py --raw data/restore/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
FOCUS = {"conn", "ref", "rtr", "m"}


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def say(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def h(self, title):
        self.say("")
        self.say(f"== {title} ==")


def walk(el, path, stats, attrs, examples, source):
    p = f"{path}/{el.tag}"
    stats[(source, p)] += 1
    for k, v in el.attrib.items():
        a = attrs[(p, k)]
        a["n"] += 1
        if len(a["values"]) < 50:
            a["values"][v] += 1
        elif v in a["values"]:
            a["values"][v] += 1
    for child in el:
        walk(child, p, stats, attrs, examples, source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "restore" / "raw")
    ap.add_argument("--max-examples", type=int, default=3)
    args = ap.parse_args()

    files = sorted(args.raw.rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"No raw files under {args.raw}")

    stats: Counter = Counter()
    attrs: dict = defaultdict(lambda: {"n": 0, "values": Counter()})
    examples: dict[str, list[str]] = defaultdict(list)
    msg_types: Counter = Counter()
    msg_codes: Counter = Counter()
    conn_status: Counter = Counter()
    responses: Counter = Counter()
    stops_with: Counter = Counter()
    first_ts, last_ts = None, None

    for f in files:
        try:
            fh = gzip.open(f, "rt", encoding="utf-8")
            lines = list(fh)
        except (EOFError, OSError):
            continue
        for line in lines:
            try:
                rec = json.loads(line)
                root = ET.fromstring(rec["body"])
            except Exception:
                continue
            src = rec["source"]
            responses[src] += 1
            ts = rec["collected_at"]
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
            walk(root, "", stats, attrs, examples, src)
            for s in root.findall("s"):
                tags = {c.tag for c in s.iter()} & FOCUS
                for t in tags:
                    stops_with[(src, t)] += 1
                for t in tags - {"m"}:
                    if len(examples[t]) < args.max_examples:
                        examples[t].append(ET.tostring(s, encoding="unicode")[:1500])
                for parent_tag, parent in [("s", s), ("ar", s.find("ar")), ("dp", s.find("dp"))]:
                    if parent is None:
                        continue
                    for m in parent.findall("m"):
                        msg_types[(src, parent_tag, m.get("t"))] += 1
                        if m.get("t") in ("d", "f") and m.get("c"):
                            msg_codes[(m.get("t"), m.get("c"))] += 1
                for c in s.iter("conn"):
                    conn_status[(src, c.get("cs"))] += 1

    rep = Report()
    rep.h("1. Coverage")
    rep.say(f"raw files: {len(files)}, first response: {first_ts}, last response: {last_ts}")
    for src, n in sorted(responses.items()):
        rep.say(f"  {src}: {n:,} responses")

    rep.h("2. Element paths (count per source)")
    for (src, p), n in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1])):
        rep.say(f"  {src:5s} {p:40s} {n:>10,}")

    rep.h("3. Attributes per element (count, up to 8 most common values)")
    for (p, k), a in sorted(attrs.items()):
        top = ", ".join(f"{v!r}:{c}" for v, c in a["values"].most_common(8))
        if len(top) > 160:
            top = top[:160] + " ..."
        rep.say(f"  {p}@{k}  n={a['n']:,}  {top}")

    rep.h("4. Stops containing focus elements")
    for (src, t), n in sorted(stops_with.items()):
        rep.say(f"  {src:5s} {t:5s} {n:>10,}")

    rep.h("5. Connection elements <conn>: status values (cs)")
    if conn_status:
        for (src, cs), n in sorted(conn_status.items()):
            rep.say(f"  {src:5s} cs={cs!s:6s} {n:>10,}")
    else:
        rep.say("  none found")

    rep.h("6. Messages <m> by source, position and type")
    for (src, pos, t), n in sorted(msg_types.items()):
        rep.say(f"  {src:5s} {pos:3s} t={t!s:3s} {n:>10,}")
    rep.say("Most common delay/free-text codes (type, code): count")
    for (t, c), n in msg_codes.most_common(20):
        rep.say(f"  {t}:{c:>5s}  {n:>8,}")

    rep.h("7. Examples of stops with <conn>, <ref> or <rtr> (truncated)")
    for t, exs in examples.items():
        for i, x in enumerate(exs, 1):
            rep.say(f"--- {t} example {i}")
            rep.say(x)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "e01_raw_inventory.txt"
    path.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
