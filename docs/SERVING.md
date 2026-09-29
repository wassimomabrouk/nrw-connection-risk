# Serving

A FastAPI service that scores every upcoming transfer candidate at the five hubs once a minute, with the project's model and with B3 (DB's prognosis as a probability) side by side. Code: `src/nrw_connection_risk/serving/`. Settings: `config/serving.toml`.

## How it works

```
collector (systemd)                          API container
  raw/     every response, written at once ──┐
  parsed/  Parquet, flushed every 10 min ────┼─> live window (last 6 h, up to now)
                                             │      candidates arriving in 2-65 min
                                             │      point-in-time state + features   (same code as training)
                                             │      P(fail): model and B3            (model bundle)
                                             │   -> in memory: served by the API
                                             └   -> prediction log at the 60/30/10-min marks (monitoring)
```

- **Fresh data.** The collector writes each API response to the raw layer immediately but flushes parsed Parquet only every 10 minutes. The service reads the parsed layer plus the raw responses from 20 minutes before its last flush onwards, parsed with the production parser, so predictions use data that is at most a minute old. The overlap also covers rows the parsed layer lacks (a flush in progress, a collector restart before flushing); rows read twice change nothing. Damaged raw records are skipped, never fatal. Only the partitions of the needed days are opened, so a scoring run does not slow down as the collection grows.
- **No training/serving skew.** A live row is built exactly like a dataset row with cutoff = now: the same candidate rules (T1–T7), the same point-in-time state, the same feature functions with the feature settings stored in the model bundle. `tests/test_serving.py` checks that the features computed live at each cutoff moment equal those of the offline dataset and feature builders.
- **Which model.** The bundle holds one model per trained cutoff (60, 30, 10 minutes). A connection is scored with the model of the cutoff nearest to the time left until arrival.
- **Not scored** (as in the dataset): A already cancelled, no data at the hub for more than 45 minutes, times in the DST transition hour. Listed with `include_unscored=true`.

## Model bundle

```
python -m nrw_connection_risk.training.fit_bundle --model gbm --note "stand-in"
```

Fits the chosen model and B3 on all built training days (never test or robustness days) and writes `models/<id>/bundle.joblib` and `models/<id>/model.json` (model card: fit days, features, settings, code commit, library versions). The API loads the newest bundle, or the one named in `config/serving.toml`. A bundle only loads with the scikit-learn minor version it was saved with, and only if the service's dataset settings (hubs, minimum transfer time, maximum slack, builder version) equal those in its model card. Otherwise the service keeps running without predictions and reports the reason in `/health` (HTTP 503) instead of crashing.

Until validation (November) the deployed bundle is a **stand-in** fitted on the days available; it shows the system working, not the final model.

## Endpoints

Interactive documentation: `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /health` | status (`ok`, `degraded` if data is older than 5 min, `error` with HTTP 503, `starting`, `no_model`), last run, data age, scoring time, prediction-log errors |
| `GET /v1/connections` | upcoming connections with P(fail); filters `hub` (name or EVA), `within_min`, `min_risk`, `include_unscored`, `sort=time\|risk`, `limit` |
| `GET /v1/connections/lookup?stop_id_a=…&stop_id_b=…` | one connection |
| `GET /v1/hubs` | hubs and how many connections are scored at each |
| `GET /v1/model` | model card of the loaded bundle |

Example connection:

```json
{
  "hub": "Köln Hbf",
  "arrival":   {"train": "RE 1", "station": "Aachen Hbf", "planned": "2026-10-02T10:12:00+02:00", "expected": "2026-10-02T10:18:00+02:00", "delay_min": 6.0},
  "departure": {"train": "ICE 2", "station": "Frankfurt(Main)Hbf", "planned": "2026-10-02T10:25:00+02:00", "expected": null, "delay_min": 0.0},
  "minutes_to_arrival": 30.0, "planned_transfer_min": 13.0, "expected_transfer_min": 7.0, "horizon_min": 30,
  "risk": {"model": 0.41, "db_baseline": 0.33, "db_rule_says_missed": false},
  "status": "ok"
}
```

## Prediction log

Each connection is logged once per cutoff, at the first scoring run at or after that moment (lag in `lag_s`, at most one scoring interval), with both probabilities, DB's rule and all features: `data/serving/predictions/date=YYYY-MM-DD/`. A restarted service reads the day's log first and does not log the same connection and cutoff twice; rows that fail to flush stay buffered and the error appears in `/health`. This mirrors the training table, so live predictions can be joined to the observed outcomes and evaluated like the offline evaluation (monitoring, phase 7).

## Deployment (Oracle server, next to the collector)

On your PC:
```
py -m nrw_connection_risk.training.fit_bundle --model gbm --note "stand-in"
py -m pip show scikit-learn                           (note the version)
scp -i KEY -r models\<model_id> ubuntu@HOST:~/nrw-connection-risk/models/
```

On the server (once: `sudo apt-get install -y docker.io docker-compose-v2 && sudo usermod -aG docker ubuntu`, then log in again):
```
cd ~/nrw-connection-risk && git pull && mkdir -p data/serving models
.venv/bin/pip install -e '.[dev]' && .venv/bin/python -m pytest -q && sudo systemctl restart nrw-collector
docker compose up -d --build              # set SKLEARN_VERSION=... first if your PC has another version than 1.9.1
curl -s localhost:8000/health
```

The port is bound to localhost on the server. From your PC: `ssh -i KEY -L 8000:localhost:8000 ubuntu@HOST`, then open http://localhost:8000/docs.

## Known limitations

- The candidate set uses the timetable known now; a train added to the timetable later can change which departures count as first reach (T7) in the offline data.
- Scoring runs every minute, so a prediction may be up to a minute older than the moment it is served.
- The live window covers the last 6 hours, the offline builder reads from the day before. They differ only for an event whose last update is more than 6 hours old, which the full change list (fetched every 30 minutes) makes rare.
- The service runs on the same small server as the collector (1 OCPU); `/health` reports the scoring time.
