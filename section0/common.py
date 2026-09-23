"""Shared constants and helpers for the section 0 feasibility scripts."""
from __future__ import annotations

from pathlib import Path

REPO_ID = "piebro/deutsche-bahn-data"

# EVA numbers of the five candidate hubs. Verified by name in s0_processed.py.
HUBS = {
    "8000207": "Köln Hbf",
    "8000085": "Düsseldorf Hbf",
    "8000086": "Duisburg Hbf",
    "8000098": "Essen Hbf",
    "8000001": "Aachen Hbf",
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

def hub_sql_list() -> str:
    return ", ".join(f"'{e}'" for e in HUBS)


class Report:
    """Prints to console and collects lines into out/<name>.txt."""

    def __init__(self, name: str):
        self.name = name
        self.lines: list[str] = []

    def say(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def h(self, title: str) -> None:
        self.say("")
        self.say(f"== {title} ==")

    def table(self, df) -> None:
        if df is None or len(df) == 0:
            self.say("(empty)")
        else:
            self.say(df.to_string(index=False))

    def save(self) -> Path:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{self.name}.txt"
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {path}")
        return path


def list_raw_day_files(year: int, month: int, day: int) -> list[str]:
    """Paths of all raw parquet files stored for one day (names differ over time)."""
    from huggingface_hub import HfApi

    folder = f"raw_data/year={year}/month={month}/day={day}"
    return sorted(
        e.path for e in HfApi().list_repo_tree(REPO_ID, path_in_repo=folder, repo_type="dataset")
        if e.path.endswith(".parquet")
    )


def download(filename: str) -> Path:
    """Download one file from the HF dataset into section0/data (cached)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    return Path(local)
