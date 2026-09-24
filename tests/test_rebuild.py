import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from nrw_connection_risk.collector.client import ApiResponse
from nrw_connection_risk.collector.storage import RawStore

ROOT = Path(__file__).resolve().parents[1]


def test_rebuild_parsed_from_raw(tmp_path, fchg_xml, plan_xml):
    raw = RawStore(tmp_path / "src")
    t = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
    raw.write("fchg", "8000207", ApiResponse("u", 200, fchg_xml, t, 1.0))
    raw.write("plan", "8000207", ApiResponse("u", 200, plan_xml, t, 1.0))
    out = tmp_path / "rebuilt"
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "rebuild_parsed.py"),
                          "--raw", str(tmp_path / "src" / "raw"), "--out", str(out)],
                         capture_output=True, text=True, check=True)
    assert "responses: 2" in res.stdout and "rows: 7" in res.stdout
    files = list((out / "parsed").rglob("*.parquet"))
    assert sum(pq.read_metadata(f).num_rows for f in files) == 7
