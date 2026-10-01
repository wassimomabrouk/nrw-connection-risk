# NRW Connection Risk

[![CI](https://github.com/wassimomabrouk/nrw-connection-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/wassimomabrouk/nrw-connection-risk/actions/workflows/ci.yml)

Predicting missed train connections at five NRW rail hubs (Köln Hbf, Düsseldorf Hbf, Duisburg Hbf, Essen Hbf, Aachen Hbf) from live Deutsche Bahn timetable data, and comparing the model against DB's own prognosis at fixed lead times.

**Status: work in progress.** The data collector has been running in production since 2026-09-24 (Oracle Cloud, systemd, external monitoring, daily off-site backup; see [docs/OPERATIONS.md](docs/OPERATIONS.md)). The pipeline from raw data to a live prediction API is built and tested; the model is selected on validation data in November and evaluated once on a locked test period in December. Until then the API serves a stand-in model.

## Why a custom collector

A feasibility study on the public historical dataset showed that about 35 to 40% of its delay labels are prognoses recorded before the train arrived, not observed times, and that DB's prognosis cannot be reconstructed at fixed lead times because stations were polled only about five times per day. Details: [FEASIBILITY.md](FEASIBILITY.md).

The project therefore collects its own data from the DB Timetables API: recent changes every minute, the full change state every 30 minutes, and hourly timetable slices, within the free plan's limit of 60 requests per minute.

## Repository structure

```
config/collector.toml            stations, polling intervals, storage settings
config/dataset.toml              candidate rules, cutoffs and data-quality thresholds for the dataset
config/features.toml             feature groups in use, hub-state settings, holidays
config/training.toml             splits, models, calibration, evaluation settings
config/serving.toml              live service: data paths, scoring interval, prediction log
config/monitoring.toml           daily evaluation of the live predictions, drift thresholds
src/nrw_connection_risk/
    collector/                   API client, XML parsing, storage, scheduler, health checks
    dataset/                     transfer candidates, point-in-time state, labels (training table)
    features/                    point-in-time features, shared by training and live prediction
    training/                    baselines B0-B3, models, calibration, time-based evaluation, test lock, model bundle
    serving/                     live scoring every minute and the FastAPI service
    monitoring/                  daily live evaluation against observed outcomes, drift (PSI), operations
tests/                           unit and integration tests (pytest)
tools/                           API smoke test, collection status, raw-to-parsed rebuild, feeder selection
deploy/                          systemd units for the collector, the daily backup and the daily monitoring
Dockerfile, compose.yml          container for the API
.github/workflows/ci.yml         tests on Python 3.11 and 3.14, image build and smoke test
docs/                            operations runbook, dataset card, feature card, training and evaluation, serving, monitoring
exploration/                     exploration scripts e01 to e05 and their reports
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

## Building the training dataset

```
python tools/rebuild_parsed.py --raw data/restore/raw --out data/restore --replace
python -m nrw_connection_risk.dataset.build --parsed data/restore/parsed --from 2026-09-24 --to 2026-09-30
```

One Parquet table per service day: every transfer candidate at the five hubs, DB's prognosis as known 60, 30 and 10 minutes before arrival, and the observed outcome. Details: [docs/DATASET.md](docs/DATASET.md).

## Building the features

```
python -m nrw_connection_risk.features.build --from 2026-09-24 --to 2026-09-30
```

Six feature groups (DB prognosis, hub state, freshness, context, trend, messages) for every eligible row. The functions are pure and will also serve live predictions; tests rewrite all data after a point in time and check that no earlier feature changes. Details: [docs/FEATURES.md](docs/FEATURES.md).

## Training and evaluation

```
python -m nrw_connection_risk.training.evaluate --mode cv
```

Compares the timetable, a historical rate, DB's own rule and DB's prognosis turned into a probability (B3, the headline baseline, gradient boosting on DB's four numbers) against the models, with confidence intervals from resampling whole days. The test period is locked in code and every evaluation of it is logged. Details: [docs/TRAINING.md](docs/TRAINING.md).

## Live predictions

```
python -m nrw_connection_risk.training.fit_bundle --model gbm
uvicorn nrw_connection_risk.serving.api:create_app_from_env --factory --port 8000
```

Every minute, all upcoming connections at the five hubs get a failure probability from the model and from the DB baseline (B3), computed with the same feature code as training; a test checks that live and offline features are identical. Predictions at the 60, 30 and 10-minute marks are logged for monitoring. Runs in Docker next to the collector. Details: [docs/SERVING.md](docs/SERVING.md).

## Monitoring

```
python -m nrw_connection_risk.monitoring.daily
```

Every morning the server labels the previous day's logged predictions with the dataset builder (the same labels as training) and scores the model, B3 and the DB rule on live traffic. It also compares every input with the training data (population stability index), recomputes each logged prediction's features offline to check that live and training features are identical, and reports coverage and data age. A dashboard in the API shows it all. Details: [docs/MONITORING.md](docs/MONITORING.md).

## Data source

Deutsche Bahn Timetables API via the DB API Marketplace. The historical analysis in `section0/` uses [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0).
