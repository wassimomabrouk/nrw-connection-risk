import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from nrw_connection_risk.collector.client import ApiResponse
from nrw_connection_risk.collector.config import Settings, Station
from nrw_connection_risk.collector.main import Collector
from nrw_connection_risk.collector.parse import SCHEMA, parse_timetable
from nrw_connection_risk.collector.storage import ParsedStore, RawStore

T = datetime(2026, 9, 23, 16, 5, tzinfo=timezone.utc)


def test_raw_store_appends_readable_multimember_gzip(tmp_path, fchg_xml):
    store = RawStore(tmp_path)
    r = ApiResponse(url="u", status=200, body=fchg_xml, collected_at=T, duration_ms=12.3)
    p1 = store.write("fchg", "8000207", r)
    p2 = store.write("fchg", "8000207", r)
    assert p1 == p2 and p1.name == "hour=16.jsonl.gz"
    with gzip.open(p1, "rt", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 2 and lines[0]["body"] == fchg_xml


def test_parsed_store_writes_schema_with_missing_fields(tmp_path, fchg_xml):
    store = ParsedStore(tmp_path)
    store.add(parse_timetable(fchg_xml, "fchg", "8000207", T))
    assert store.flush() == 4 and store.pending() == 0
    files = list((tmp_path / "parsed" / "source=fchg" / "date=2026-09-23").glob("*.parquet"))
    table = pq.read_table(files[0])
    assert table.schema.equals(SCHEMA) and table.num_rows == 4


class StubClient:
    def __init__(self, plan_xml, fchg_xml):
        self.plan_xml, self.fchg_xml = plan_xml, fchg_xml
        self.calls = []
        self.calls_total = 0
        self.errors_total = 0

    def _r(self, body):
        self.calls_total += 1
        return ApiResponse("u", 200, body, datetime.now(timezone.utc), 1.0)

    def plan(self, eva, d, h):
        self.calls.append(("plan", eva, d, h))
        return self._r(self.plan_xml)

    def fchg(self, eva):
        self.calls.append(("fchg", eva))
        return self._r(self.fchg_xml)

    def rchg(self, eva):
        self.calls.append(("rchg", eva))
        return None                    # simulate a failed request


def settings(tmp_path: Path) -> Settings:
    return Settings(
        base_url="https://x", max_calls_per_minute=50, timeout_s=5, max_retries=1,
        rchg_interval_s=60, fchg_interval_s=1800, plan_interval_s=3600,
        plan_hours_behind=2, plan_hours_ahead=3, data_dir=tmp_path / "data",
        flush_interval_s=600, heartbeat_file=tmp_path / "data" / "heartbeat.json",
        ping_interval_s=300, stations=(Station("8000207", "Köln Hbf"), Station("8000001", "Aachen Hbf")),
        client_id="id", api_key="key", healthcheck_url=None)


def test_collector_once_runs_every_job_and_persists(tmp_path, plan_xml, fchg_xml):
    stub = StubClient(plan_xml, fchg_xml)
    c = Collector(settings(tmp_path), client=stub)
    c.run(once=True)
    kinds = [x[0] for x in stub.calls]
    assert kinds.count("fchg") == 2 and kinds.count("rchg") == 2
    assert kinds.count("plan") == 2 * 6            # hours -2..+3 per station
    hb = json.loads((tmp_path / "data" / "heartbeat.json").read_text())
    assert hb["failure_count"]["rchg"] == 2 and hb["success_count"]["fchg"] == 2
    assert list((tmp_path / "data" / "parsed" / "source=plan").rglob("*.parquet"))
    assert list((tmp_path / "data" / "raw" / "source=fchg").rglob("*.jsonl.gz"))


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_collector_loop_respects_intervals(tmp_path, plan_xml, fchg_xml):
    stub = StubClient(plan_xml, fchg_xml)
    clock = FakeClock()
    c = Collector(settings(tmp_path), client=stub, clock=clock, sleep=clock.sleep)
    c.run(duration_s=10 * 60)                       # 10 simulated minutes
    kinds = [x[0] for x in stub.calls]
    assert kinds.count("fchg") == 2                 # once per station (interval 30 min)
    assert 18 <= kinds.count("rchg") <= 22          # ~10 per station
    assert kinds.count("plan") == 2 * 6
