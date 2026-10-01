"""e06 feature selection: the pre-registered rule, its guards, and an end-to-end run on
synthetic data whose truth is known (failure depends on DB's slack and the hub state only)."""
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_features import write_days
from nrw_connection_risk.training.config import TrainingConfig
from nrw_connection_risk.training.feature_selection import (Rule, SelectionConfig, compare, load_selection_config,
                                                            run, threshold, variants_and_tests, verdict)

ROOT = Path(__file__).resolve().parents[1]
D = date.fromisoformat
RULE = Rule()


def res(g60=0.01, g30=0.02, g10=0.01, ci=(0.01, 0.03), share=0.8):
    one = lambda g: {"gain": g, "ci": ci, "day_share": share}   # noqa: E731
    return {60: one(g60), 30: one(g30), 10: one(g10)}


# ---------------------------------------------------------------- the rule

def test_repo_config_is_the_pre_registered_design():
    sc = load_selection_config(ROOT / "config" / "feature_selection.toml")
    assert (sc.first_day, sc.last_day, sc.min_train_days) == (D("2026-09-24"), D("2026-10-14"), 7)
    assert sc.reference == ("db", "hub", "freshness") and set(sc.drop) == {"hub", "freshness"}
    assert set(sc.add) == {"context", "trend", "messages"}
    assert sc.rule == Rule(30, 0.005, 2.0, 0.6, 0.002)


def test_threshold_is_the_larger_of_minimum_and_noise():
    assert threshold(RULE, {30: {"gain": 0.001}}) == 0.005
    assert threshold(RULE, {30: {"gain": -0.004}}) == pytest.approx(0.008)


def test_verdict_needs_every_condition():
    assert verdict(res(), RULE, 0.005) == (True, [])
    for bad, word in [(res(g30=0.004), "gain at 30"), (res(ci=(-0.001, 0.03)), "interval"),
                      (res(share=0.5), "days"), (res(g60=-0.003), "at 60 min"), (res(g10=-0.0021), "at 10 min")]:
        ok, failed = verdict(bad, RULE, 0.005)
        assert not ok and len(failed) == 1 and word in failed[0]
    assert verdict(res(g60=-0.0019, g10=-0.002), RULE, 0.005)[0]         # small losses elsewhere are tolerated
    assert not verdict(res(g30=0.007), RULE, 0.008)[0]                   # a noisy reference raises the bar
    noisy10 = {60: {"gain": 0.0005}, 30: {"gain": 0.001}, 10: {"gain": -0.005}}
    assert verdict(res(g10=-0.008), RULE, 0.005, noisy10)[0]             # within 2 x the 10-min noise floor
    assert not verdict(res(g10=-0.011), RULE, 0.005, noisy10)[0]
    assert not verdict(res(g60=-0.003), RULE, 0.005, noisy10)[0]         # 60 min is quiet: 0.2% applies


def test_compare_gain_sign_interval_and_day_share():
    rng = np.random.default_rng(0)
    n = 4000
    rows = pd.DataFrame({"service_day": np.repeat([f"2026-10-0{i}" for i in range(1, 5)], n // 4),
                         "cutoff_min": 30, "label_fail": rng.random(n) < 0.3})
    y = rows.label_fail.to_numpy()
    good, bad = np.where(y, 0.6, 0.2), np.full(n, 0.3)
    r = compare(rows, good, bad, 200, 0)[30]
    assert r["gain"] > 0 and r["ci"][0] > 0 and r["day_share"] == 1.0 and r["days"] == 4
    assert compare(rows, bad, good, 200, 0)[30]["gain"] < 0
    assert compare(rows, good, good, 200, 0)[30]["gain"] == 0


def test_config_validation():
    base = dict(first_day=D("2026-09-24"), last_day=D("2026-10-14"), min_train_days=7)
    with pytest.raises(ValueError, match="reference"):
        SelectionConfig(**base, reference=("db",), drop=("hub",), add=())
    with pytest.raises(ValueError, match="never dropped"):
        SelectionConfig(**base, reference=("db", "hub"), drop=("db",), add=())
    with pytest.raises(ValueError, match="unknown"):
        SelectionConfig(**base, reference=("db",), drop=(), add=("weather",))


def test_every_group_is_tested_against_the_variant_without_it():
    sc = SelectionConfig(D("2026-09-24"), D("2026-10-14"), 7, ("db", "hub", "freshness"), ("hub",), ("trend",),
                         {"messages": D("2026-09-30")})
    v, tests = variants_and_tests(sc)
    for t in tests:
        assert t.group in v[t.with_group].groups and t.group not in v[t.without].groups
        assert set(v[t.with_group].groups) - set(v[t.without].groups) == {t.group}
    restricted = next(t for t in tests if t.kind == "restricted")
    assert v[restricted.with_group].start == v[restricted.without].start == D("2026-09-30")


# ---------------------------------------------------------------- guards

def small_cfg():
    return replace(TrainingConfig(), bootstrap_n=200,
                   gbm={**TrainingConfig().gbm, "max_iter": 150, "min_samples_leaf": 40})


def test_window_must_be_training_days_and_complete(tmp_path):
    write_days(tmp_path / "f", D("2026-09-24"), 6, 100)
    sc = SelectionConfig(D("2026-09-24"), D("2026-09-30"), 3, ("db", "hub"), ("hub",), ())
    with pytest.raises(SystemExit, match="incomplete"):                    # 30 Sep not built
        run(sc, small_cfg(), tmp_path / "f", tmp_path / "runs", None, log=lambda *_: None)
    late = replace(sc, last_day=D("2026-11-10"))
    with pytest.raises(SystemExit, match="training period"):
        run(late, small_cfg(), tmp_path / "f", tmp_path / "runs", None, log=lambda *_: None)


# ---------------------------------------------------------------- end to end

def test_end_to_end_finds_the_known_truth(tmp_path):
    """Truth: DB's slack and the hub state. Starting from db + freshness, e06 must drop
    freshness (noise), add hub, reject trend and messages, and apply both changes together."""
    first = D("2026-09-24")
    write_days(tmp_path / "f", first, 10, 1200, seed=3)
    sc = SelectionConfig(first, first + timedelta(days=9), 4, ("db", "freshness"), ("freshness",), ("hub", "trend"),
                         {"messages": first + timedelta(days=2)})
    pub = tmp_path / "e06.md"
    info = run(sc, small_cfg(), tmp_path / "f", tmp_path / "runs", pub, config_text=b"x", log=lambda *_: None)

    assert info["final_groups"] == ["db", "hub"]
    kept = {t["group"]: t["keep"] for t in info["tests"]}
    assert kept == {"freshness": False, "hub": True, "trend": False, "messages": False}
    hub = next(t for t in info["tests"] if t["group"] == "hub")["result"]["30"]
    assert hub["gain"] > 0.02 and hub["ci"][0] > 0
    assert info["combination"]["outcome"].startswith("The combination is at least as good")
    restricted = next(t for t in info["tests"] if t["group"] == "messages")["result"]["30"]
    assert restricted["days"] == 4 and info["eval_days"] == 6           # 8 days from the start, 4 to fit first

    run_dir = tmp_path / "runs" / info["run_id"]
    saved = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert saved["final_groups"] == ["db", "hub"] and len(saved["config_sha256"]) == 64
    text = pub.read_text(encoding="utf-8")
    assert text == (run_dir / "report.md").read_text(encoding="utf-8")
    assert "**Feature groups: db, hub**" in text and "Noise floor" in text and "against B3" in text
