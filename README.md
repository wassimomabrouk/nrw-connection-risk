# NRW Connection Risk

Predicting missed train connections at five NRW rail hubs (Köln Hbf, Düsseldorf Hbf, Duisburg Hbf, Essen Hbf, Aachen Hbf) from live Deutsche Bahn timetable data, and comparing the model against DB's own prognosis at fixed lead times.

**Status: work in progress.** The data collector has been running in production since 2026-09-24 (Oracle Cloud, systemd, external monitoring, daily off-site backup; see [docs/OPERATIONS.md](docs/OPERATIONS.md)). Modelling starts once enough data has been collected.

## Why a custom collector

A feasibility study on the public historical dataset showed that about 35 to 40% of its delay labels are prognoses recorded before the train arrived, not observed times, and that DB's prognosis cannot be reconstructed at fixed lead times because stations were polled only about five times per day. Details: [FEASIBILITY.md](FEASIBILITY.md).

The project therefore collects its own data from the DB Timetables API: recent changes every minute, the full change state every 30 minutes, and hourly timetable slices, within the free plan's limit of 60 requests per minute.

## Repository structure

```
config/collector.toml            stations, polling intervals, storage settings
src/nrw_connection_risk/
    collector/                   API client, XML parsing, storage, scheduler, health checks
tests/                           unit and integration tests (pytest)
tools/                           API smoke test, collection status, raw-to-parsed rebuild
deploy/                          systemd units for the collector and the daily backup
docs/                            operations runbook
section0/                        feasibility scripts on the historical dataset
```

## Setup

Requires Python 3.11 or newer and a free DB API Marketplace application subscribed to the Timetables API.

```
python -m venv .venv
.venv\Scripts\activate          (Linux: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          (Linux: cp .env.example .env), then add your keys
pytest
```

## Running the collector

```
python -m nrw_connection_risk.collector.main                  run until stopped
python -m nrw_connection_risk.collector.main --duration-min 15
python tools/collector_status.py                              summary of collected data
```

Data is written to `data/collector/`: raw API responses as gzip JSON lines (`raw/`), parsed observations as Parquet (`parsed/`), a heartbeat file and logs.

## Data source

Deutsche Bahn Timetables API via the DB API Marketplace. The historical analysis in `section0/` uses [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0).
