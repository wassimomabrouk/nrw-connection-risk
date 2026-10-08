"""Monitoring: reference profile and PSI, reading the prediction log, labelling live
predictions with the dataset builder, the daily report and the summary."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_parsed
from synthetic_features import write_days
from test_dataset_build import CFG, DAY, scenario
from nrw_connection_risk.monitoring.daily import (MonitoringConfig, evaluate_day, finished_days, join,
                                                  load_predictions, main, outcomes)
from nrw_connection_risk.monitoring.profile import build_profile, drift, level, psi
from nrw_connection_risk.monitoring.dashboard import render as render_dashboard
from nrw_connection_risk.monitoring.summary import write_summary
from nrw_connection_risk.serving.api import Service
from nrw_connection_risk.serving.config import ServingConfig
from nrw_connection_risk.training.bundle import latest_bundle, load_bundle
from nrw_connection_risk.training.fit_bundle import main as fit_bundle

T = pd.Timestamp


# ---------------------------------------------------------------- profile and PSI

def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"db_slack_min": rng.normal(10, 5, 20000), "hub": rng.choice(["K", "D"], 20000)})
    prof = build_profile(train, ["db_slack_min", "hub"])
    same = pd.DataFrame({"db_slack_min": rng.normal(10, 5, 5000), "hub": rng.choice(["K", "D"], 5000)})
    shifted = pd.DataFrame({"db_slack_min": rng.normal(16, 5, 5000), "hub": rng.choice(["K", "D", "E"], 5000)})
    d_same, d_shift = drift(prof, same), drift(prof, shifted)
    assert d_same["db_slack_min"] < 0.01 and d_same["hub"] < 0.01
    assert d_shift["db_slack_min"] > 0.25 and d_shift["hub"] > 0.25          # a hub never seen in training
    assert level(0.05) == "stable" and level(0.2) == "moderate" and level(0.3) == "large"


def test_profile_counts_missing_values_as_their_own_bin():
    train = pd.DataFrame({"age_b_min": [np.nan] * 50 + list(range(50))})
    prof = build_profile(train, ["age_b_min"])
    assert prof["age_b_min"]["shares"][-1] == pytest.approx(0.5) and sum(prof["age_b_min"]["shares"]) == pytest.approx(1)
    all_missing = pd.DataFrame({"age_b_min": [np.nan] * 100})
    assert drift(prof, all_missing)["age_b_min"] > 0.25
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0


# ---------------------------------------------------------------- live day end to end

@pytest.fixture(scope="module")
def monitored(tmp_path_factory):
    """The scenario day served live at its three cutoffs, then monitored."""
    tmp = tmp_path_factory.mktemp("mon")
    write_days(tmp / "features", date(2026, 9, 24), 3, 1500)
    fit_bundle(["--features", str(tmp / "features"), "--out", str(tmp / "models")])
    bundle = load_bundle(latest_bundle(tmp / "models"))
    parsed = write_parsed(tmp, scenario())
    for t in ("07:12", "07:42", "08:02"):
        now = T(f"2026-09-24 {t}")
        svc = Service(ServingConfig(parsed=parsed, raw=tmp / "raw", log_dir=tmp / "predictions"), CFG, bundle,
                      clock=lambda now=now: now)
        svc.run_once(now)
        svc.flush()
    cfg = MonitoringConfig(parsed=parsed, predictions=tmp / "predictions", models=tmp / "models", out=tmp / "mon",
                           bootstrap_n=50, bootstrap_n_auc=10, min_drift_rows=1)
    return cfg, bundle, evaluate_day(DAY, cfg, CFG)


def test_bundle_carries_the_training_profile(monitored):
    _, bundle, _ = monitored
    assert set(bundle.meta["reference_profile"]) == set(bundle.meta["feature_columns"])
    assert set(bundle.meta["reference_fail_rate"]) == {"10", "30", "60"}


def test_live_predictions_get_the_training_labels(monitored):
    cfg, _, rep = monitored
    assert rep["status"] == "ok" and rep["operations"]["logged"] == 6
    assert rep["operations"]["by_outcome"] == {"evaluated": 6}
    assert rep["operations"]["coverage"] == {"10": 1.0, "30": 1.0, "60": 1.0}
    j = pd.read_parquet(cfg.out / "joined" / f"service_day={DAY}.parquet").set_index(["stop_id_b", "cutoff_min"])
    assert not j.loc[("2-2609241025-1", 30)].label_fail            # B1 held (4 minutes are enough)
    assert j.loc[("3-2609241038-1", 30)].fail_reason == "b_cancelled"
    # the labels are exactly the dataset builder's
    ds = outcomes(DAY, cfg.parsed, CFG).set_index(["stop_id_a", "stop_id_b", "cutoff_min"]).label_fail
    for (a, b, c), y in zip(zip(j.stop_id_a, j.index.get_level_values(0), j.index.get_level_values(1)), j.label_fail):
        assert ds[(a, b, c)] == y


def test_report_contents(monitored):
    _, _, rep = monitored
    p = rep["performance"]["30"]
    assert p["rows"] == 2 and p["fail_rate"] == 0.5 and "calibration" in p
    assert set(p) >= {"model", "B3", "db_rule", "log_loss_gain_vs_B3"}
    d = next(iter(rep["drift"].values()))
    assert d["available"] and "db_slack_min" in d["psi"] and d["fail_rate"]["30"]["live"] == 0.5
    assert rep["operations"]["lag_s"]["max"] == 0.0


def test_unmatched_and_excluded_rows_are_not_evaluated():
    pred = pd.DataFrame({"stop_id_a": ["a", "b"], "stop_id_b": ["x", "y"], "cutoff_min": [30, 30]})
    out = pd.DataFrame({"stop_id_a": ["b"], "stop_id_b": ["y"], "cutoff_min": [30], "label_fail": [True],
                        "fail_reason": ["delay"], "eligible": [False], "exclusion_reason": ["collector_gap"]})
    assert list(join(pred, out).outcome) == ["not_a_candidate", "excluded"]


def test_day_without_outcomes_yet(tmp_path, monitored):
    cfg, _, _ = monitored
    parsed = write_parsed(tmp_path, scenario(with_tail=False))           # labels need 6 more hours
    from dataclasses import replace
    rep = evaluate_day(DAY, replace(cfg, parsed=parsed, out=tmp_path / "mon"), CFG)
    assert rep["status"] == "no_outcomes" and "IncompleteDay" in rep["reason"]


def test_prediction_log_reading_filters_the_day_and_dedupes(tmp_path):
    rows = pd.DataFrame({"stop_id_a": ["a", "a", "b"], "stop_id_b": ["x", "x", "y"], "cutoff_min": [30, 30, 30],
                         "scored_at": pd.to_datetime(["2026-09-24 07:42", "2026-09-24 07:43", "2026-09-25 03:00"], utc=True),
                         "data_as_of": pd.to_datetime(["2026-09-24 07:41"] * 3, utc=True),
                         "pt_a": pd.to_datetime(["2026-09-24 08:12", "2026-09-24 08:12", "2026-09-25 03:30"], utc=True),
                         "p_model": [0.1, 0.9, 0.5]})
    (tmp_path / "date=2026-09-24").mkdir()
    (tmp_path / "date=2026-09-25").mkdir()
    rows.iloc[:2].to_parquet(tmp_path / "date=2026-09-24" / "part-1.parquet")
    rows.iloc[2:].to_parquet(tmp_path / "date=2026-09-25" / "part-1.parquet")
    got = load_predictions(tmp_path, DAY)
    assert len(got) == 1 and got.p_model[0] == 0.1            # first logged kept; 03:30 UTC belongs to the next day


def test_summary_and_cli(monitored, tmp_path):
    cfg, _, rep = monitored
    (cfg.out / "daily").mkdir(parents=True, exist_ok=True)
    (cfg.out / "daily" / f"service_day={DAY}.json").write_text(json.dumps(rep))
    s = write_summary(cfg)
    assert s["days"] == 1 and set(s["pooled"]) == {"10", "30", "60"}
    md = (cfg.out / "summary.md").read_text(encoding="utf-8")
    assert "## All days pooled" in md and "## Drift on 2026-09-24" in md and "Fewer than 10 days" in md
    # the CLI with a config file finds finished days by itself
    toml = tmp_path / "monitoring.toml"
    toml.write_text(f'''[data]
parsed = "{cfg.parsed.as_posix()}"
predictions = "{cfg.predictions.as_posix()}"
models = "{cfg.models.as_posix()}"
out = "{(tmp_path / 'out').as_posix()}"
[evaluation]
primary_cutoff = 30
calibration_bins = 10
bootstrap_n = 20
bootstrap_n_auc = 5
seed = 1
[drift]
moderate = 0.1
large = 0.25
min_rows = 1
''')
    from pathlib import Path
    repo_config = Path(__file__).resolve().parents[1] / "config"
    ds = tmp_path / "dataset.toml"
    ds.write_text((repo_config / "dataset.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "collector.toml").write_text('[[stations]]\neva = "8000207"\nname = "Köln Hbf"\n', encoding="utf-8")
    assert DAY in finished_days(cfg, CFG, pd.Timestamp("2026-10-01", tz="UTC"))
    assert main(["--config", str(toml), "--dataset-config", str(ds)]) == 0
    assert (tmp_path / "out" / "daily" / f"service_day={DAY}.json").exists()
    assert (tmp_path / "out" / "summary.md").exists()


# ---------------------------------------------------------------- skew, dashboard, API routes

def test_day_constant_inputs_are_not_ranked_as_drift(monitored):
    """day_type is the same for every row of a day, so its daily PSI against a training mix of day
    types is always large: reported separately, never the 'largest drift'."""
    from nrw_connection_risk.monitoring.daily import load_monitoring_config
    from nrw_connection_risk.monitoring.summary import judged
    cfg, _, rep = monitored
    d = next(iter(rep["drift"].values()))
    assert "day_type" in d["psi"] and d["psi"]["day_type"] > 0.25        # the raw report keeps it
    j = judged(d, cfg.drift_skip)
    assert "day_type" not in j["psi"] and "day_type" not in j["level"] and "day_type" in j["not_judged"]
    (cfg.out / "daily").mkdir(parents=True, exist_ok=True)
    (cfg.out / "daily" / f"service_day={DAY}.json").write_text(json.dumps(rep))
    s = write_summary(cfg)
    assert s["per_day"][0]["max_psi_feature"] != "day_type"
    assert "day_type" not in next(iter(s["latest"]["drift"].values()))["psi"]
    assert "Not judged per day" in (cfg.out / "summary.md").read_text(encoding="utf-8")
    assert "Not judged per day: day_type" in render_dashboard(s, {"status": "ok"}, None)
    repo = load_monitoring_config(Path(__file__).resolve().parents[1] / "config" / "monitoring.toml",
                                  Path(__file__).resolve().parents[1])
    assert repo.drift_skip == ("day_type",)


def test_skew_check_finds_live_equal_to_offline(monitored):
    _, _, rep = monitored
    s = next(iter(rep["skew"].values()))
    assert s["available"] and s["rows"] == 6 and s["identical_share"] == 1.0 and s["least_equal"] == []


def test_skew_check_catches_a_difference(monitored):
    from nrw_connection_risk.monitoring.skew import compare
    cfg, _, _ = monitored
    live = pd.read_parquet(cfg.out / "joined" / f"service_day={DAY}.parquet")
    offline = live.copy()
    offline.loc[0, "db_slack_min"] += 3
    offline.loc[1, "hub"] = "Essen Hbf"
    r = compare(live, offline, ["db_slack_min", "hub", "age_a_min"])
    assert r["identical_share"] == pytest.approx(4 / 6, abs=1e-4)
    assert r["per_feature"]["db_slack_min"]["mean_abs_diff_where_different"] == 3.0
    assert set(r["least_equal"]) == {"db_slack_min", "hub"}


def test_dashboard_renders_with_and_without_data(monitored):
    from nrw_connection_risk.monitoring.dashboard import render
    cfg, bundle, rep = monitored
    empty = render(None, {"status": "starting", "data_age_s": None, "model_id": "m"}, None)
    assert "No evaluated day yet" in empty and "<svg" not in empty
    (cfg.out / "daily").mkdir(parents=True, exist_ok=True)
    (cfg.out / "daily" / f"service_day={DAY}.json").write_text(json.dumps(rep))
    page = render(write_summary(cfg), {"status": "ok", "data_age_s": 40, "model_id": bundle.model_id}, bundle.meta)
    for text in ("Log loss per day", "Calibration", "Input drift", "Per day (30 min)", "All days pooled",
                 "Live = offline features", "100.0%"):
        assert text in page
    assert page.count("<svg") == 2 and "<script" not in page


def test_api_serves_dashboard_and_summary(tmp_path, monitored):
    from fastapi.testclient import TestClient
    from nrw_connection_risk.serving.api import create_app
    cfg, bundle, _ = monitored
    svc = Service(ServingConfig(parsed=cfg.parsed, raw=tmp_path / "raw", log_dir=tmp_path / "log",
                                monitoring_dir=tmp_path / "mon"), CFG, bundle, clock=lambda: T("2026-09-24 07:42"))
    client = TestClient(create_app(svc, start_scorer=False))
    assert client.get("/v1/monitoring").status_code == 404
    r = client.get("/dashboard")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "No evaluated day" in r.text
    (tmp_path / "mon").mkdir()
    (tmp_path / "mon" / "summary.json").write_text(json.dumps({"days": 0, "per_day": [], "pooled": {}}))
    assert client.get("/v1/monitoring").json()["days"] == 0


def test_line_chart_axis_labels_stay_distinct_for_close_values():
    import re
    from nrw_connection_risk.monitoring.dashboard import line_chart
    svg = line_chart(["2026-09-30"], [("model", "red", [0.4688]), ("B3", "blue", [0.4620])])
    labels = re.findall(r'text-anchor="end">([^<]+)<', svg)
    assert len(labels) >= 3 and len(set(labels)) == len(labels)
    assert all(len(x.split(".")[1]) >= 3 for x in labels)
    wide = re.findall(r'text-anchor="end">([^<]+)<', line_chart(["a", "b"], [("m", "red", [0.3, 0.6])]))
    assert len(set(wide)) == len(wide) and all(len(x.split(".")[1]) <= 2 for x in wide)
