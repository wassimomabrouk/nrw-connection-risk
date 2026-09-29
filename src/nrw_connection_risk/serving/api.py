"""HTTP API for live connection risk.

A background thread scores all upcoming connections every minute (serving/score.py)
and keeps the latest result in memory; requests only read that result, so they are
fast and never touch the data files.

Run (repo root):
    uvicorn nrw_connection_risk.serving.api:create_app_from_env --factory --port 8000
Environment: NRW_ROOT (repo root, default: current directory), NRW_SERVING_CONFIG.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from ..dataset.config import DatasetConfig, load_config
from ..training.bundle import COMPANION, Bundle, dataset_mismatch, latest_bundle, load_bundle
from .config import ServingConfig, load_serving_config
from .live import load_live
from .predlog import PredictionLog
from .score import score

log = logging.getLogger(__name__)
BERLIN = "Europe/Berlin"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")


# ---------------------------------------------------------------- scoring service

class Service:
    """Holds the model, the latest scores and the prediction log; runs the scoring loop."""

    def __init__(self, scfg: ServingConfig, dcfg: DatasetConfig, bundle: Bundle | None, clock=utc_now,
                 startup_error: str | None = None):
        self.scfg, self.dcfg, self.bundle, self.clock = scfg, dcfg, bundle, clock
        self.startup_error = startup_error
        self.scores: pd.DataFrame | None = None
        self.data_as_of: pd.Timestamp | None = None
        self.log_error: str | None = None
        self.last_run: pd.Timestamp | None = None
        self.last_success: pd.Timestamp | None = None
        self.last_error: str | None = None
        self.last_duration_s: float | None = None
        self.runs = 0
        self.lock = threading.Lock()
        self.predlog = PredictionLog(scfg.log_dir, bundle.cutoffs, scfg.max_late_min) if bundle else None
        if self.predlog:
            try:
                self.predlog.restore(self.clock())
            except Exception as e:
                self.log_error = f"restore: {type(e).__name__}: {e}"
        self._last_flush = time.monotonic()

    def run_once(self, now: pd.Timestamp | None = None) -> None:
        if self.bundle is None:
            return
        now = now if now is not None else self.clock()
        t0 = time.monotonic()
        try:
            snap = load_live(self.scfg.parsed, self.scfg.raw, now, self.scfg.lookback_h, list(self.dcfg.hubs),
                             self.dcfg.min_parser_version)
            scores = score(snap, self.bundle, self.dcfg, self.scfg.min_minutes_ahead, self.scfg.max_minutes_ahead,
                           self.scfg.max_collector_gap_min)
            logged = self.predlog.add(scores, now) if len(scores) else 0
            with self.lock:
                self.scores, self.last_success, self.last_error = scores, now, None
                self.data_as_of = snap.data_as_of
            log.info("scored %d connections (%d logged) in %.1fs", len(scores), logged, time.monotonic() - t0)
        except Exception as e:                       # keep serving the last good result
            with self.lock:
                self.last_error = f"{type(e).__name__}: {e}"
            log.exception("scoring failed")
        finally:
            with self.lock:
                self.last_run, self.runs = now, self.runs + 1
                self.last_duration_s = round(time.monotonic() - t0, 2)
        if self.predlog and time.monotonic() - self._last_flush >= self.scfg.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if self.predlog:
            try:
                self.predlog.flush()
                self.log_error = None
            except Exception as e:                   # rows stay buffered and are retried
                self.log_error = f"{type(e).__name__}: {e}"
                log.exception("prediction log flush failed")
        self._last_flush = time.monotonic()

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            started = time.monotonic()
            self.run_once()
            stop.wait(max(1.0, self.scfg.interval_s - (time.monotonic() - started)))
        self.flush()

    def health(self) -> dict:
        with self.lock:
            scores, last_success, data_as_of = self.scores, self.last_success, self.data_as_of
            err = self.startup_error or self.last_error
        now = self.clock()
        age = (now - data_as_of).total_seconds() if data_as_of is not None and pd.notna(data_as_of) else None
        if self.startup_error:
            status = "error"
        elif self.bundle is None:
            status = "no_model"
        elif last_success is None:
            status = "error" if err else "starting"
        elif err or (now - last_success).total_seconds() > 3 * self.scfg.interval_s:
            status = "error"
        elif age is not None and age > self.scfg.max_data_age_s:
            status = "degraded"
        else:
            status = "ok"
        return {"status": status, "model_id": self.bundle.model_id if self.bundle else None,
                "last_run": _iso(self.last_run), "last_success": _iso(last_success),
                "last_duration_s": self.last_duration_s, "runs": self.runs,
                "data_as_of": _iso(data_as_of), "data_age_s": None if age is None else round(age),
                "connections": 0 if scores is None else int(len(scores)),
                "predictions_pending_log": self.predlog.pending() if self.predlog else 0, "last_error": err,
                "log_error": self.log_error}


def _iso(t, tz: str | None = None) -> str | None:
    if t is None or (not isinstance(t, str) and pd.isna(t)):
        return None
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return (t.tz_convert(tz) if tz else t).isoformat()


def _num(x, digits=4):
    return None if x is None or pd.isna(x) else round(float(x), digits)


# ---------------------------------------------------------------- response models

class Health(BaseModel):
    status: Literal["ok", "degraded", "error", "starting", "no_model"]
    model_id: str | None
    last_run: str | None
    last_success: str | None
    last_duration_s: float | None
    runs: int
    data_as_of: str | None
    data_age_s: int | None
    connections: int
    predictions_pending_log: int
    last_error: str | None
    log_error: str | None


class Stop(BaseModel):
    stop_id: str
    train: str
    planned: str
    expected: str | None
    delay_min: float
    station: str | None      # origin of the arriving train / destination of the departing one


class Risk(BaseModel):
    model: float | None              # P(connection fails), this project's model
    db_baseline: float | None        # P(fail) from B3: DB's own prognosis turned into a probability
    db_rule_says_missed: bool        # DB's numbers imply less than the minimum transfer time


class Connection(BaseModel):
    hub: str
    arrival: Stop
    departure: Stop
    departure_cancelled: bool
    minutes_to_arrival: float
    planned_transfer_min: float
    expected_transfer_min: float
    horizon_min: int                 # which trained cutoff (60/30/10 min) produced the prediction
    risk: Risk
    status: str


class Connections(BaseModel):
    scored_at: str | None
    data_as_of: str | None
    model_id: str | None
    count: int
    connections: list[Connection]


def _connection(r) -> Connection:
    return Connection(
        hub=r.hub,
        arrival=Stop(stop_id=r.stop_id_a, train=r.train_a, planned=_iso(r.pt_a, BERLIN), expected=_iso(r.a_ct_cut, BERLIN),
                     delay_min=_num(r.db_delay_a_min, 1), station=r.origin_a if isinstance(r.origin_a, str) else None),
        departure=Stop(stop_id=r.stop_id_b, train=r.train_b, planned=_iso(r.pt_b, BERLIN),
                       expected=_iso(r.b_ct_cut, BERLIN), delay_min=_num(r.db_delay_b_min, 1),
                       station=r.destination_b if isinstance(r.destination_b, str) else None),
        departure_cancelled=bool(r.b_cancel_known), minutes_to_arrival=_num(r.minutes_to_arrival, 1),
        planned_transfer_min=_num(r.planned_slack_min, 1), expected_transfer_min=_num(r.db_slack_min, 1),
        horizon_min=int(r.cutoff_min),
        risk=Risk(model=_num(r.p_model), db_baseline=_num(getattr(r, f"p_{COMPANION}")),
                  db_rule_says_missed=bool(r.db_rule_fail)),
        status=r.status)


# ---------------------------------------------------------------- app

def create_app(service: Service, start_scorer: bool = True) -> FastAPI:
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        thread = None
        if start_scorer and service.bundle is not None:
            thread = threading.Thread(target=service.loop, args=(stop,), name="scorer", daemon=True)
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=30)

    app = FastAPI(title="NRW connection risk", version="1.0",
                  description="Probability that a train connection at a Rhine-Ruhr hub is missed, "
                              "next to DB's own prognosis. Data: DB Timetables API.", lifespan=lifespan)
    app.state.service = service

    def hub_eva(hub: str | None) -> str | None:
        if hub is None:
            return None
        for eva, name in service.dcfg.hubs.items():
            if hub == eva or hub.casefold() == name.casefold():
                return eva
        raise HTTPException(404, f"unknown hub {hub!r}; known: {', '.join(service.dcfg.hubs.values())}")

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "nrw-connection-risk", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=Health, responses={503: {"description": "the service is failing"}})
    def health(response: Response):
        h = service.health()
        if h["status"] == "error":             # lets Docker's healthcheck mark the container unhealthy
            response.status_code = 503
        return h

    @app.get("/v1/model")
    def model():
        if service.bundle is None:
            raise HTTPException(503, "no model loaded")
        m = service.bundle.meta
        return {k: m[k] for k in ("model_id", "model", "companion", "note", "period", "fit_days", "rows",
                                  "cutoffs", "feature_version", "feature_groups", "cutoff_mode", "calibration",
                                  "git_commit", "created_at", "versions") if k in m}

    @app.get("/v1/hubs")
    def hubs():
        with service.lock:
            s = service.scores
        counts = s.eva.value_counts() if s is not None and len(s) else pd.Series(dtype=int)
        return [{"eva": e, "name": n, "connections": int(counts.get(e, 0))} for e, n in service.dcfg.hubs.items()]

    @app.get("/v1/connections", response_model=Connections)
    def connections(hub: str | None = Query(None, description="hub name or EVA number"),
                    within_min: float = Query(65, gt=0, description="arrival at most this many minutes ahead"),
                    min_risk: float = Query(0, ge=0, le=1, description="only connections with at least this P(fail)"),
                    include_unscored: bool = Query(False, description="also list connections that cannot be scored"),
                    sort: Literal["time", "risk"] = "time", limit: int = Query(200, ge=1, le=2000)):
        eva = hub_eva(hub)
        with service.lock:
            s, scored_at = service.scores, service.last_success
        if s is None or not len(s):
            return Connections(scored_at=_iso(scored_at), data_as_of=None, model_id=_model_id(service), count=0,
                               connections=[])
        sel = s[s.minutes_to_arrival <= within_min]
        if eva:
            sel = sel[sel.eva == eva]
        if not include_unscored:
            sel = sel[sel.status.eq("ok")]
        if min_risk > 0:
            sel = sel[sel.p_model >= min_risk]
        sel = sel.sort_values(["p_model", "pt_a"], ascending=[False, True]) if sort == "risk" \
            else sel.sort_values(["pt_a", "pt_b"])
        items = [_connection(r) for r in sel.head(limit).itertuples(index=False)]
        return Connections(scored_at=_iso(scored_at), data_as_of=_iso(s.data_as_of.iloc[0]),
                           model_id=_model_id(service), count=len(sel), connections=items)

    @app.get("/v1/connections/lookup", response_model=Connection)
    def lookup(stop_id_a: str, stop_id_b: str):
        with service.lock:
            s = service.scores
        hit = s[(s.stop_id_a == stop_id_a) & (s.stop_id_b == stop_id_b)] if s is not None and len(s) else []
        if not len(hit):
            raise HTTPException(404, "connection not among the currently scored connections")
        return _connection(next(hit.itertuples(index=False)))

    return app


def _model_id(service: Service) -> str | None:
    return service.bundle.model_id if service.bundle else None


def build_service(root: Path, config_path: Path | None = None) -> Service:
    scfg = load_serving_config(config_path or root / "config" / "serving.toml", root)
    dcfg = load_config(root / "config" / "dataset.toml")
    path = scfg.models_dir / scfg.bundle if scfg.bundle else latest_bundle(scfg.models_dir)
    if path is None or not (path / "model.json").exists():
        log.warning("no model bundle under %s: serving without predictions", scfg.models_dir)
        return Service(scfg, dcfg, None)
    try:                                         # a bad bundle must not crash-loop the container
        bundle = load_bundle(path)
    except Exception as e:
        log.exception("could not load %s", path)
        return Service(scfg, dcfg, None, startup_error=f"model {path.name}: {type(e).__name__}: {e}")
    wrong = dataset_mismatch(bundle.meta, dcfg.as_dict())
    if wrong:
        return Service(scfg, dcfg, None, startup_error=f"model {bundle.model_id} was trained with other "
                                                        f"dataset settings: {'; '.join(wrong)}")
    log.info("loaded model %s", bundle.model_id)
    return Service(scfg, dcfg, bundle)


def create_app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(os.environ.get("NRW_ROOT", ".")).resolve()
    cfg = os.environ.get("NRW_SERVING_CONFIG")
    return create_app(build_service(root, Path(cfg) if cfg else None))
