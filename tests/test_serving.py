"""Live serving: model bundle, live data (parsed layer + raw tail), scoring, prediction
log and the HTTP API. The central test: features computed live equal the features the
offline pipeline computes for the same moment."""
import gzip
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import write_parsed
from synthetic_features import write_days
from test_dataset_build import CFG, DAY, KOELN, rchg, scenario, utc
from test_features_build import AACHEN, FEEDER_DATA, FEEDERS, MESSAGE
from nrw_connection_risk.collector.client import ApiResponse
from nrw_connection_risk.collector.storage import RawStore
from nrw_connection_risk.dataset.build import build_day as build_dataset_day
from nrw_connection_risk.dataset.build import write_day
from nrw_connection_risk.dataset.load import load_window
from nrw_connection_risk.features.build import build_day as build_feature_day
from nrw_connection_risk.features.columns import ALL_FEATURES
from nrw_connection_risk.features.config import FeatureConfig
from nrw_connection_risk.serving.api import Service, build_service, create_app
from nrw_connection_risk.serving.config import ServingConfig
from nrw_connection_risk.serving.live import load_live, raw_tail
from nrw_connection_risk.serving.predlog import PredictionLog
from nrw_connection_risk.serving.score import score
from nrw_connection_risk.training import bundle as bundle_mod
from nrw_connection_risk.training.bundle import BundleVersionError, latest_bundle, load_bundle
from nrw_connection_risk.training.fit_bundle import main as fit_bundle

T = pd.Timestamp


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    write_days(tmp / "features", date(2026, 9, 24), 3, 1500)
    fit_bundle(["--features", str(tmp / "features"), "--out", str(tmp / "models"), "--note", "test"])
    return latest_bundle(tmp / "models")


@pytest.fixture(scope="module")
def bundle(bundle_dir):
    return load_bundle(bundle_dir)


# ---------------------------------------------------------------- bundle

def test_bundle_round_trip(bundle_dir, bundle, tmp_path):
    meta = json.loads((bundle_dir / "model.json").read_text())
    assert meta["model"] == "gbm" and meta["companion"] == "B3" and meta["note"] == "test"
    assert meta["cutoffs"] == [10, 30, 60] and meta["fit_days"]["count"] == 3
    assert meta["feature_config"]["groups"] == ["db", "hub", "freshness", "context"]
    rows = pd.read_parquet(bundle_dir.parents[1] / "features" / "service_day=2026-09-24" / "part-0.parquet")
    p = bundle.predict(rows)
    assert set(p) == {"model", "B3"} and all(((v > 0) & (v < 1)).all() for v in p.values())
    np.testing.assert_array_equal(load_bundle(bundle_dir).predict(rows)["model"], p["model"])


def test_bundle_refuses_other_sklearn_version(bundle_dir, monkeypatch):
    monkeypatch.setattr(bundle_mod, "versions", lambda: {"python": "3.12.0", "scikit-learn": "0.99.0",
                                                         "pandas": "3.0.0", "numpy": "2.0.0"})
    with pytest.raises(BundleVersionError, match="scikit-learn"):
        load_bundle(bundle_dir)


def test_fit_bundle_never_reads_locked_days(tmp_path):
    write_days(tmp_path / "f", date(2026, 11, 23), 2, 100)        # test period only
    with pytest.raises(SystemExit, match="no built feature days"):
        fit_bundle(["--features", str(tmp_path / "f"), "--out", str(tmp_path / "m")])


# ---------------------------------------------------------------- live data

def raw_write(root, responses):
    store = RawStore(root)
    for source, eva, t, xml in responses:
        store.write(source, eva, ApiResponse(url="test", status=200, body=xml, collected_at=t, duration_ms=1.0))


def test_live_snapshot_includes_unflushed_responses(tmp_path):
    """Parsed layer up to 07:00 plus raw responses after it == everything flushed."""
    responses = scenario()
    flushed = [r for r in responses if r[2] <= utc("2026-09-24 07:00")]
    write_parsed(tmp_path / "live", flushed)
    raw_write(tmp_path / "live", [r for r in responses if r not in flushed])
    now = T("2026-09-24 07:55")
    snap = load_live(tmp_path / "live" / "parsed", tmp_path / "live" / "raw", now, 6, [KOELN], 2)
    w = load_window(write_parsed(tmp_path / "all", responses), now - pd.Timedelta(hours=6), now, 2, stations=[KOELN])
    assert snap.unflushed_rows == 1 and snap.data_as_of == T("2026-09-24 07:50")
    sort = lambda df, cols: df.sort_values(cols).reset_index(drop=True)   # noqa: E731
    pd.testing.assert_frame_equal(sort(snap.obs, ["key", "obs"]), sort(w.obs, ["key", "obs"]))
    pd.testing.assert_frame_equal(sort(snap.plan, ["stop_id", "event"]), sort(w.plan, ["stop_id", "event"]))


def test_overlap_between_parsed_and_raw_changes_nothing(tmp_path, bundle):
    """In production raw holds every response, also the flushed ones: the overlap window
    reads some rows twice, and the features must not change."""
    responses = scenario() + [MESSAGE]
    now = T("2026-09-24 08:02")
    only_parsed = write_parsed(tmp_path / "a", responses)
    write_parsed(tmp_path / "b", [r for r in responses if r[2] <= utc("2026-09-24 07:55")])
    raw_write(tmp_path / "b", responses)
    a = score(load_live(only_parsed, tmp_path / "a" / "raw", now, 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
    snap = load_live(tmp_path / "b" / "parsed", tmp_path / "b" / "raw", now, 6, [KOELN], 2)
    b = score(snap, bundle, CFG, 2, 65, 45)
    assert snap.unflushed_rows > 0
    pd.testing.assert_frame_equal(a[ALL_FEATURES + ["p_model", "p_B3"]], b[ALL_FEATURES + ["p_model", "p_B3"]])


def test_raw_tail_skips_damaged_members_and_bad_bodies(tmp_path):
    good = ("rchg", KOELN, utc("2026-09-24 07:50"), rchg('<s id="1-2609240900-5"><ar ct="2609241020"/></s>'))
    later = ("rchg", KOELN, utc("2026-09-24 07:52"), rchg('<s id="1-2609240900-5"><ar ct="2609241022"/></s>'))
    raw_write(tmp_path, [good])
    path = next(tmp_path.rglob("*.jsonl.gz"))
    with open(path, "ab") as f:                        # collector killed mid-append ...
        f.write(gzip.compress(b'{"collected_at": "2026-09-24T07:51:00+00:00"}\n')[:20])
    raw_write(tmp_path, [("rchg", KOELN, utc("2026-09-24 07:51"), "<timetable><s id='x'>"), later])  # ... then bad XML
    t = raw_tail(tmp_path / "raw", datetime(2026, 9, 24, 7, tzinfo=timezone.utc),
                 datetime(2026, 9, 24, 8, tzinfo=timezone.utc))
    assert [v.as_py() for v in t.column("ct_raw")] == ["2609241020", "2609241022"]


def test_raw_tail_survives_a_file_being_written(tmp_path):
    raw_write(tmp_path, [("rchg", KOELN, utc("2026-09-24 07:50"),
                          rchg('<s id="1-2609240900-5"><ar ct="2609241020"/></s>'))])
    path = next(tmp_path.rglob("*.jsonl.gz"))
    with open(path, "ab") as f:                          # a second record, cut off mid-write
        f.write(gzip.compress(b'{"collected_at": "2026-09-24T07:51:00+00:00", "body": "<time')[:25])
    t = raw_tail(tmp_path / "raw", datetime(2026, 9, 24, 7, tzinfo=timezone.utc),
                 datetime(2026, 9, 24, 8, tzinfo=timezone.utc))
    assert t.num_rows == 1 and t.column("ct_raw")[0].as_py() == "2609241020"


# ---------------------------------------------------------------- training/serving consistency

def test_live_features_equal_offline_features(tmp_path, bundle):
    """At each cutoff moment, the live scorer computes exactly the features that the
    dataset + feature builders compute for that cutoff: no training/serving skew."""
    parsed = write_parsed(tmp_path, scenario() + [MESSAGE])
    df, meta = build_dataset_day(DAY, parsed, CFG)
    offline, _ = build_feature_day(DAY, write_day(df, meta, tmp_path / "ds"), parsed, CFG.hubs, FeatureConfig())
    for L in (60, 30, 10):
        off = offline[offline.cutoff_min == L]
        now = off.t_cut.iloc[0]                  # build_day returns naive UTC in memory
        live = score(load_live(parsed, tmp_path / "raw", now, 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
        assert (live.cutoff_min == L).all() and live.status.eq("ok").all()
        key = ["stop_id_a", "stop_id_b"]
        pd.testing.assert_frame_equal(off.set_index(key)[ALL_FEATURES].sort_index(),
                                      live.set_index(key)[ALL_FEATURES].sort_index(), check_dtype=False)
        assert live.p_model.between(0, 1).all() and live.p_B3.between(0, 1).all()


def with_feeder_group(bundle):
    """The same models, with a model card whose feature settings use the feeder group."""
    meta = json.loads(json.dumps(bundle.meta))
    meta["feature_config"] = {**FEEDERS.as_dict(), "groups": ["db", "hub", "feeder"]}
    return bundle_mod.Bundle(models=bundle.models, meta=meta)


def test_live_feeder_features_equal_offline_features(tmp_path, bundle):
    """The feeder group, too, is computed live exactly as offline, and the service reads the
    feeder stations only for a model that uses them."""
    parsed = write_parsed(tmp_path, scenario() + FEEDER_DATA)
    df, meta = build_dataset_day(DAY, parsed, CFG)
    offline, _ = build_feature_day(DAY, write_day(df, meta, tmp_path / "ds"), parsed, CFG.hubs, FEEDERS)
    fb = with_feeder_group(bundle)
    svc = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log"), CFG, fb)
    assert svc.stations == [KOELN, AACHEN]
    plain = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log2"), CFG, bundle)
    assert plain.stations == [KOELN]
    for L in (60, 30, 10):
        off = offline[offline.cutoff_min == L]
        live = score(load_live(parsed, tmp_path / "raw", off.t_cut.iloc[0], 6, svc.stations, 2), fb, CFG, 2, 65, 45)
        key = ["stop_id_a", "stop_id_b"]
        pd.testing.assert_frame_equal(off.set_index(key)[ALL_FEATURES].sort_index(),
                                      live.set_index(key)[ALL_FEATURES].sort_index(), check_dtype=False)
        assert live.corridor_delay_a.notna().any()


def test_rows_that_cannot_be_scored(tmp_path, bundle):
    # A is cancelled before the cutoff: no prediction needed (as in the dataset)
    cancel = ("rchg", KOELN, utc("2026-09-24 07:30"), rchg('<s id="1-2609240900-5"><ar cs="c" clt="2609240930"/></s>'))
    parsed = write_parsed(tmp_path / "a", scenario() + [cancel])
    s = score(load_live(parsed, tmp_path / "raw", T("2026-09-24 07:42"), 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
    assert s.status.eq("a_cancelled").all() and s.p_model.isna().all()
    # nothing seen at the hub for 72 minutes: collector gap
    parsed = write_parsed(tmp_path / "b", scenario(with_first_obs=False))
    s = score(load_live(parsed, tmp_path / "raw", T("2026-09-24 07:12"), 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
    assert s.status.eq("collector_gap").all() and s.p_model.isna().all()


# ---------------------------------------------------------------- prediction log

def fake_scores(minutes, status="ok"):
    n = len(minutes)
    return pd.DataFrame({"stop_id_a": ["A"] * n, "stop_id_b": [f"B{i}" for i in range(n)], "status": status,
                         "minutes_to_arrival": minutes, "pt_a": T("2026-09-24 08:12"), "p_model": 0.3})


def test_prediction_log_once_per_cutoff(tmp_path):
    plog = PredictionLog(tmp_path, [10, 30, 60], max_late_min=5)
    now = T("2026-09-24 07:10")
    assert plog.add(fake_scores([61.0, 54.0]), now) == 0       # B1 missed the 60 mark by 6 min
    assert plog.add(fake_scores([60.0, 53.0]), now) == 1       # B0 logged at 60
    assert plog.add(fake_scores([59.0, 52.0]), now) == 0       # not twice
    assert plog.add(fake_scores([30.0, 29.5]), now) == 2       # both at 30
    assert plog.add(fake_scores([9.9], status="a_cancelled"), now) == 0
    rows = pd.concat(plog.buffer)
    assert sorted(zip(rows.stop_id_b, rows.cutoff_min)) == [("B0", 30), ("B0", 60), ("B1", 30)]
    assert list(rows.lag_s) == [0.0, 0.0, 30.0]
    plog.add(fake_scores([]), T("2026-09-24 10:30"))            # arrivals long past are forgotten
    assert plog.logged == {}


def test_prediction_log_flush(tmp_path, bundle):
    parsed = write_parsed(tmp_path, scenario())
    s = score(load_live(parsed, tmp_path / "raw", T("2026-09-24 07:42"), 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
    plog = PredictionLog(tmp_path / "log", bundle.cutoffs, 5)
    assert plog.add(s, T("2026-09-24 07:42")) == 2 and plog.flush() == 2 and plog.flush() == 0
    df = pd.read_parquet(tmp_path / "log")
    assert len(df) == 2 and (df.cutoff_min == 30).all() and set(ALL_FEATURES) <= set(df.columns) | {"hub"}
    assert str(df.scored_at.dt.tz) == "UTC" and df.p_B3.notna().all()


# ---------------------------------------------------------------- API

@pytest.fixture
def api(tmp_path, bundle):
    parsed = write_parsed(tmp_path, scenario())
    now = T("2026-09-24 07:42")
    svc = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log"), CFG, bundle,
                  clock=lambda: now)
    return svc, TestClient(create_app(svc, start_scorer=False))


def test_api_before_and_after_scoring(api):
    svc, client = api
    assert client.get("/health").json()["status"] == "starting"
    assert client.get("/v1/connections").json()["count"] == 0
    svc.run_once()
    h = client.get("/health").json()
    assert h["connections"] == 2 and h["last_error"] is None
    assert h["status"] == "degraded" and h["data_age_s"] == 42 * 60    # last data at 07:00
    body = client.get("/v1/connections", params={"hub": "köln hbf", "sort": "risk"}).json()
    assert body["count"] == 2
    c = body["connections"][0]
    assert c["arrival"]["train"] == "RE 1" and c["arrival"]["planned"] == "2026-09-24T10:12:00+02:00"
    assert c["arrival"]["delay_min"] == 2.0 and c["horizon_min"] == 30 and c["planned_transfer_min"] in (13.0, 26.0)
    risks = [x["risk"]["model"] for x in body["connections"]]
    assert risks == sorted(risks, reverse=True)
    assert client.get("/v1/connections", params={"min_risk": 0.999}).json()["count"] == 0
    assert client.get("/v1/connections", params={"hub": "8000207", "within_min": 10}).json()["count"] == 0
    one = client.get("/v1/connections/lookup", params={"stop_id_a": "1-2609240900-5", "stop_id_b": "2-2609241025-1"})
    assert one.status_code == 200 and one.json()["departure"]["train"] == "ICE 2"
    assert client.get("/v1/connections/lookup", params={"stop_id_a": "x", "stop_id_b": "y"}).status_code == 404
    assert client.get("/v1/connections", params={"hub": "Paris Nord"}).status_code == 404
    assert client.get("/v1/hubs").json() == [{"eva": KOELN, "name": "Köln Hbf", "connections": 2}]
    assert client.get("/v1/model").json()["model"] == "gbm"


def test_api_reports_scoring_errors(tmp_path, bundle):
    svc = Service(ServingConfig(parsed=tmp_path / "missing", raw=tmp_path / "raw", log_dir=tmp_path / "log"),
                  CFG, bundle, clock=lambda: T("2026-09-24 07:42"))
    svc.run_once()
    r = TestClient(create_app(svc, start_scorer=False)).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error" and "ParsedLayerError" in r.json()["last_error"]


def test_health_reports_data_age_when_nothing_is_scored(tmp_path, bundle):
    """At 09:00 no arrival is ahead, but the data (last seen 08:30) still has an age."""
    parsed = write_parsed(tmp_path, scenario())
    svc = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log"), CFG, bundle,
                  clock=lambda: T("2026-09-24 09:00"))
    svc.run_once()
    h = TestClient(create_app(svc, start_scorer=False)).get("/health").json()
    assert h["connections"] == 0 and h["data_age_s"] == 30 * 60 and h["status"] == "degraded"


def test_flush_errors_are_reported_and_rows_kept(tmp_path, bundle):
    parsed = write_parsed(tmp_path, scenario())
    (tmp_path / "not_a_dir").write_text("x")
    svc = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "not_a_dir" / "log"),
                  CFG, bundle, clock=lambda: T("2026-09-24 07:42"))
    svc.run_once()
    svc.flush()
    assert svc.log_error is not None and svc.predlog.pending() == 2
    assert svc.health()["log_error"] == svc.log_error


def test_restart_does_not_log_twice(tmp_path, bundle):
    parsed = write_parsed(tmp_path, scenario())
    now = T("2026-09-24 07:42")
    s = score(load_live(parsed, tmp_path / "raw", now, 6, [KOELN], 2), bundle, CFG, 2, 65, 45)
    first = PredictionLog(tmp_path / "log", bundle.cutoffs, 5)
    first.add(s, now)
    first.flush()
    second = PredictionLog(tmp_path / "log", bundle.cutoffs, 5)       # the service restarted
    assert second.restore(now) == 2 and second.add(s, now) == 0


def test_bad_or_mismatched_bundle_does_not_crash_the_service(tmp_path, bundle_dir):
    import shutil
    from pathlib import Path
    root = tmp_path / "repo"
    shutil.copytree(Path(__file__).resolve().parents[1] / "config", root / "config")
    shutil.copytree(bundle_dir, root / "models" / bundle_dir.name)
    # the test bundle was fitted with the repo's dataset.toml, so it loads ...
    assert build_service(root).bundle is not None
    # ... but not if the service's dataset settings differ
    toml = root / "config" / "dataset.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace("min_transfer_min = 4", "min_transfer_min = 5"),
                    encoding="utf-8")
    r = TestClient(create_app(build_service(root))).get("/health")
    assert r.status_code == 503 and "min_transfer_min" in r.json()["last_error"]
    # a corrupt bundle file
    toml.write_text(toml.read_text(encoding="utf-8").replace("min_transfer_min = 5", "min_transfer_min = 4"),
                    encoding="utf-8")
    (root / "models" / bundle_dir.name / "bundle.joblib").write_bytes(b"not a model")
    r = TestClient(create_app(build_service(root))).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error"


def test_service_without_model(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "config"
    for name in ("serving.toml", "dataset.toml", "collector.toml"):
        (root / "config" / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    client = TestClient(create_app(build_service(root)))
    assert client.get("/health").json()["status"] == "no_model"
    assert client.get("/v1/model").status_code == 503
    assert client.get("/v1/connections").json()["count"] == 0


def test_background_loop_scores_and_flushes_on_stop(tmp_path, bundle):
    import threading
    import time
    parsed = write_parsed(tmp_path, scenario())
    svc = Service(ServingConfig(parsed=parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log", interval_s=0.05),
                  CFG, bundle, clock=lambda: T("2026-09-24 07:42"))
    stop = threading.Event()
    thread = threading.Thread(target=svc.loop, args=(stop,))
    thread.start()
    deadline = time.monotonic() + 20
    while svc.runs < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=20)
    assert svc.runs >= 2 and svc.last_error is None and not thread.is_alive()
    assert len(pd.read_parquet(tmp_path / "log")) == 2           # logged once, flushed at shutdown


def test_unreadable_models_folder_does_not_crash_the_service(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from nrw_connection_risk.serving import api as api_mod
    root = tmp_path / "repo"
    shutil.copytree(Path(__file__).resolve().parents[1] / "config", root / "config")

    def denied(_):
        raise PermissionError(13, "Permission denied", "models/x/model.json")
    monkeypatch.setattr(api_mod, "latest_bundle", denied)
    r = TestClient(create_app(build_service(root))).get("/health")
    assert r.status_code == 503 and "PermissionError" in r.json()["last_error"]
