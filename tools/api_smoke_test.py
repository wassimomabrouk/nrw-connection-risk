"""Smoke test for the DB Timetables API: keys, endpoints, rate-limit headers,
and how far back past events stay in the fchg response.

Needs a .env file in the repo root with DB_CLIENT_ID and DB_API_KEY.
Run:  py tools\\api_smoke_test.py            (Köln Hbf)
      py tools\\api_smoke_test.py 8000085    (another EVA)
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

BASE = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
BERLIN = ZoneInfo("Europe/Berlin")


def get(path: str, headers: dict) -> requests.Response:
    r = requests.get(f"{BASE}{path}", headers=headers, timeout=30)
    print(f"GET {path} -> HTTP {r.status_code}, {len(r.content):,} bytes")
    limit_headers = {k: v for k, v in r.headers.items() if "limit" in k.lower() or "retry" in k.lower()}
    if limit_headers:
        print("  rate-limit headers:", limit_headers)
    if r.status_code != 200:
        print("  body:", r.text[:300])
    return r


def parse_ct(v: str | None):
    return datetime.strptime(v, "%y%m%d%H%M").replace(tzinfo=BERLIN) if v else None


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cid, key = os.getenv("DB_CLIENT_ID"), os.getenv("DB_API_KEY")
    if not cid or not key:
        raise SystemExit("DB_CLIENT_ID / DB_API_KEY missing. Create .env in the repo root (see .env.example).")
    eva = sys.argv[1] if len(sys.argv) > 1 else "8000207"
    headers = {"DB-Client-Id": cid, "DB-Api-Key": key, "accept": "application/xml"}
    now = datetime.now(BERLIN)
    print(f"Now (Berlin): {now:%Y-%m-%d %H:%M}, station EVA {eva}\n")

    # 1. Plan for the current hour
    r = get(f"/plan/{eva}/{now:%y%m%d}/{now:%H}", headers)
    if r.status_code == 200:
        root = ET.fromstring(r.content)
        stops = root.findall("s")
        print(f"  station: {root.get('station')}, planned stops this hour: {len(stops)}")
        if stops:
            s = stops[0]
            print(f"  example id: {s.get('id')}")
    print()

    # 2. Full changes
    r = get(f"/fchg/{eva}", headers)
    if r.status_code != 200:
        raise SystemExit("fchg failed, check the subscription to the Timetables API.")
    root = ET.fromstring(r.content)
    stops = root.findall("s")
    rel_ar, n_ar, n_dp, n_cancel, n_msg_only = [], 0, 0, 0, 0
    for s in stops:
        ar, dp = s.find("ar"), s.find("dp")
        if ar is None and dp is None:
            n_msg_only += 1
        if ar is not None and ar.get("ct"):
            n_ar += 1
            rel_ar.append((parse_ct(ar.get("ct")) - now).total_seconds() / 60)
        if dp is not None and dp.get("ct"):
            n_dp += 1
        if any(e is not None and e.get("cs") == "c" for e in (ar, dp)):
            n_cancel += 1
    print(f"  stops in fchg: {len(stops)}, with arrival ct: {n_ar}, with departure ct: {n_dp}, "
          f"cancelled: {n_cancel}, messages only: {n_msg_only}")
    if rel_ar:
        rel_ar.sort()
        past = [x for x in rel_ar if x < 0]
        print(f"  arrival ct relative to now (min): earliest {rel_ar[0]:.0f}, latest {rel_ar[-1]:.0f}")
        print(f"  arrivals already in the past: {len(past)} "
              f"(oldest {-past[0]:.0f} min ago)" if past else "  arrivals already in the past: 0")
    print()

    # 3. Recent changes (last ~2 minutes)
    r = get(f"/rchg/{eva}", headers)
    if r.status_code == 200:
        print(f"  stops in rchg: {len(ET.fromstring(r.content).findall('s'))}")
    print("\nDone. Requests used: 3")


if __name__ == "__main__":
    main()
