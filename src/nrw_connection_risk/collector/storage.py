"""Two storage layers.

raw:    every API response, untouched, as gzip JSON lines, one file per UTC hour and source.
        Appending creates multi-member gzip files, which standard readers handle.
parsed: observation rows buffered in memory and flushed to Parquet, partitioned by
        source and UTC date.
"""
from __future__ import annotations

import gzip
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .client import ApiResponse
from .parse import SCHEMA

log = logging.getLogger(__name__)


class RawStore:
    def __init__(self, root: Path):
        self.root = root / "raw"

    def write(self, source: str, eva: str, resp: ApiResponse) -> Path:
        t = resp.collected_at
        folder = self.root / f"source={source}" / f"date={t:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"hour={t:%H}.jsonl.gz"
        record = {
            "collected_at": t.isoformat(),
            "source": source,
            "eva": eva,
            "url": resp.url,
            "status": resp.status,
            "duration_ms": round(resp.duration_ms, 1),
            "body": resp.body,
        }
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


class ParsedStore:
    def __init__(self, root: Path):
        self.root = root / "parsed"
        self.buffer: dict[tuple[str, str], list[dict]] = {}

    def add(self, rows: list[dict]) -> None:
        for r in rows:
            key = (r["source"], r["collected_at"].astimezone(timezone.utc).strftime("%Y-%m-%d"))
            self.buffer.setdefault(key, []).append(r)

    def pending(self) -> int:
        return sum(len(v) for v in self.buffer.values())

    def flush(self) -> int:
        written = 0
        for (source, date), rows in list(self.buffer.items()):
            if not rows:
                continue
            folder = self.root / f"source={source}" / f"date={date}"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%H%M%S")
            path = folder / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            pq.write_table(table, path, compression="zstd")
            written += len(rows)
            del self.buffer[(source, date)]
        if written:
            log.info("flushed %d parsed rows", written)
        return written
