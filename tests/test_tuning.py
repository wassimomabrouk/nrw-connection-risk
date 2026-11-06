"""Phase 9 tuning: the pre-registered configuration, candidates, the rule, fit windows, and an
end-to-end run on synthetic data whose truth changes half-way (as the network did when the
Essen-Duisburg works ended), so that a short training window must win."""
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_features import write_days
from nrw_connection_risk.features.columns import GROUPS
from nrw_connection_risk.training.config import TrainingConfig, load_training_config
from nrw_connection_risk.training.evaluate import load_rows, predict_folds
from nrw_connection_risk.training.models import make_model
from nrw_connection_risk.training.splits import fit_days_for
from nrw_connection_risk.training.tuning import (DEFAULT, NOISE, Candidate, TuningConfig, TuningRule, candidates,
                                                 decide, load_tuning_config, passes, run, thresholds,
                                                 with_candidate)

ROOT = Path(__file__).resolve().parents[1]
D = date.fromisoformat
FEATS = GROUPS["db"] + GROUPS["hub"]


# ---------------------------------------------------------------- configuration

def test_repo_config_is_the_pre_registered_plan():
    tc = load_tuning_config(ROOT / "config" / "tuning.toml")
    assert (tc.first_day, tc.last_day, tc.min_train_days, tc.fold_days) == (D("2026-09-30"), D("2026-11-08"), 14, 7)
    assert tc.windows == (0, 21, 14)
    assert tc.gbm_grid == {"max_leaf_nodes": (15, 31, 63), "min_samples_leaf": (100, 1000)}
    assert tc.lr_grid == {"C": (0.1, 1.0, 10.0)}
    assert tc.rule == TuningRule(30, 0.002, 2.0, 0.002)
    cfg = load_training_config(ROOT / "config" / "training.toml")
    assert cfg.first_fit_day == tc.first_day and cfg.b3 == {} and all(cfg.window(m) == 0 for m in ("gbm", "B3", "lr"))
    # the same budget for the model and its benchmark: 17 alternatives each, plus the noise run
    assert len(candidates(tc, cfg, "gbm")) == len(candidates(tc, cfg, "B3")) == 19
    assert len(candidates(tc, cfg, "lr")) == 9


def test_candidates_start_with_the_current_settings_and_never_repeat_them():
    tc = TuningConfig(D("2026-09-30"), D("2026-11-08"), windows=(0, 14), gbm_grid={"max_leaf_nodes": (15, 31)},
                      lr_grid={"C": (1.0,)})
    cfg = TrainingConfig()
    c = candidates(tc, cfg, "gbm")
    assert [x.name for x in c[:2]] == [DEFAULT, NOISE] and c[1].seed_offset == 1
    assert {(x.window, x.params) for x in c[2:]} == {(0, (("max_leaf_nodes", 15),)), (14, (("max_leaf_nodes", 15),)),
                                                     (14, (("max_leaf_nodes", 31),))}
    lr = candidates(tc, cfg, "lr")
    assert [x.name for x in lr] == [DEFAULT, "window 14 d, C 1.0"]          # deterministic: no noise run
    with pytest.raises(ValueError, match="windows"):
        TuningConfig(D("2026-09-30"), D("2026-11-08"), windows=(14,))
    with pytest.raises(ValueError, match="unknown"):
        TuningConfig(D("2026-09-30"), D("2026-11-08"), gbm_grid={"depth": (3,)})


def test_candidate_settings_reach_only_their_model():
    cfg = TrainingConfig(cutoff_mode="feature")              # one learner, so its settings are visible
    c = Candidate("x", 0, (("max_leaf_nodes", 7),))
    b3 = with_candidate(cfg, "B3", c)
    assert make_model("B3", b3, FEATS).params["max_leaf_nodes"] == 7
    gbm = with_candidate(cfg, "gbm", c)
    assert make_model("gbm", gbm, FEATS).params["max_leaf_nodes"] == 7
    assert make_model("B3", gbm, FEATS).params["max_leaf_nodes"] == 7     # B3 follows [gbm] unless [b3]
    own = replace(cfg, b3={"max_leaf_nodes": 15})
    assert make_model("B3", own, FEATS).params["max_leaf_nodes"] == 15
    assert make_model("gbm", own, FEATS).params["max_leaf_nodes"] == 31
    assert with_candidate(cfg, "lr", Candidate("x", 0, (("C", 0.1),))).lr["C"] == 0.1
    assert with_candidate(cfg, "gbm", Candidate("x", seed_offset=1)).seed == cfg.seed + 1


# ---------------------------------------------------------------- fit windows

def test_fit_window_keeps_the_last_calendar_days():
    days = [D("2026-10-01") + timedelta(days=i) for i in range(30)]
    assert fit_days_for(days) == days
    assert fit_days_for(days, D("2026-10-10")) == days[9:]
    assert fit_days_for(days, None, 7) == days[-7:]
    assert fit_days_for(days, D("2026-10-28"), 7) == days[27:]                  # fewer days than the window
    gap = days[:20] + days[25:]                                                # 5 days not built
    assert fit_days_for(gap, None, 7) == days[25:]


def test_training_honours_first_fit_day_and_each_models_window(tmp_path):
    days = write_days(tmp_path, D("2026-09-28"), 6, 600)
    r = load_rows(tmp_path, days, TrainingConfig().splits)
    cfg = TrainingConfig(run=("B3", "gbm"), gbm={"max_iter": 20, "min_samples_leaf": 20},
                         first_fit_day=D("2026-09-29"), fit_window={"gbm": 2})
    pred = predict_folds(r, [(days[:5], [days[5]])], cfg, FEATS, log=lambda *_: None)
    ev = r[r.service_day == days[5]].reset_index(drop=True)
    own = lambda ds, name: make_model(name, cfg, FEATS).fit(r[r.service_day.isin(ds)]).predict(ev)   # noqa: E731
    np.testing.assert_allclose(pred.p_gbm, own(days[3:5], "gbm"))             # last 2 days
    np.testing.assert_allclose(pred.p_B3, own(days[1:5], "B3"))               # from first_fit_day


# ---------------------------------------------------------------- the rule

def res(g60=0.0, g30=0.01, g10=0.0, ci=(0.002, 0.02)):
    one = lambda g: {"gain": g, "ci": ci, "day_share": 0.7}   # noqa: E731
    return {60: one(g60), 30: one(g30), 10: one(g10)}


def test_thresholds_follow_the_noise():
    rule = TuningRule()
    assert thresholds(rule, None) == (0.002, {})
    thr, tol = thresholds(rule, {60: {"gain": -0.003}, 30: {"gain": 0.0005}, 10: {"gain": 0.0}})
    assert thr == 0.002 and tol == {60: pytest.approx(0.006), 30: 0.002, 10: 0.002}
    assert thresholds(rule, {30: {"gain": -0.004}})[0] == pytest.approx(0.008)


def test_every_condition_is_needed():
    rule, tol = TuningRule(), {60: 0.002, 10: 0.002}
    assert passes(res(), rule, 0.005, tol) == (True, [])
    for bad, word in [(res(g30=0.004), "gain at 30"), (res(ci=(-0.001, 0.02)), "interval"),
                      (res(g60=-0.003), "at 60 min"), (res(g10=-0.0021), "at 10 min")]:
        ok, failed = passes(bad, rule, 0.005, tol)
        assert not ok and len(failed) == 1 and word in failed[0]
    assert passes(res(g60=-0.003), rule, 0.005, {60: 0.004, 10: 0.002})[0]     # noisy cutoff: wider tolerance


def test_decide_takes_the_best_passing_candidate_or_keeps_the_current_settings():
    rng = np.random.default_rng(0)
    n = 6000
    rows = pd.DataFrame({"service_day": np.repeat([f"2026-10-{i:02d}" for i in range(10, 16)], n // 6),
                         "cutoff_min": np.tile([60, 30, 10], n // 3), "label_fail": rng.random(n) < 0.3})
    y = rows.label_fail.to_numpy()
    base = np.full(n, 0.3)
    p = {DEFAULT: base, NOISE: base + rng.normal(0, 1e-4, n),
         "good": np.where(y, 0.4, 0.25), "better": np.where(y, 0.5, 0.2), "same": base + 1e-5}
    cands = [Candidate(k) for k in p]
    d = decide(rows, p, cands, TuningRule(), 200, 0)
    assert d["winner"] == "better" and {r["candidate"]: r["passes"] for r in d["results"]} == \
        {"good": True, "better": True, "same": False}
    d = decide(rows, {DEFAULT: base, "same": base + 1e-5}, [Candidate(DEFAULT), Candidate("same")], TuningRule(), 200, 0)
    assert d["winner"] == DEFAULT and d["noise"] is None


# ---------------------------------------------------------------- end to end

def test_end_to_end_a_short_window_wins_after_the_world_changed(tmp_path):
    """The hub effect flips sign after 12 days: models fit on the recent days predict the new
    regime, models fit on every day mix both. Truth for gbm and lr: the short window."""
    first = D("2026-09-30")
    write_days(tmp_path / "f", first, 12, 2000, seed=1, hub_effect=-2.0)
    write_days(tmp_path / "f", first + timedelta(days=12), 12, 2000, seed=2, hub_effect=2.0)
    tc = TuningConfig(first, first + timedelta(days=23), min_train_days=9, fold_days=3, windows=(0, 4),
                      gbm_grid={"max_leaf_nodes": (15,)}, lr_grid={"C": (1.0,)})
    cfg = replace(TrainingConfig(), bootstrap_n=200,
                  gbm={**TrainingConfig().gbm, "max_iter": 60, "min_samples_leaf": 40, "max_leaf_nodes": 15})
    pub = tmp_path / "e08.md"
    info = run(tc, cfg, tmp_path / "f", tmp_path / "runs", pub, FEATS, config_text=b"x", log=lambda *_: None)

    assert info["models"]["gbm"]["winner"] == "window 4 d, max_leaf_nodes 15"
    assert info["models"]["lr"]["winner"] == "window 4 d, C 1.0"
    assert info["eval_days"] == 15 and len(info["folds"]) == 5
    assert all(max(f["fit"]) < min(f["evaluate"]) for f in info["folds"])
    gbm = next(r for r in info["models"]["gbm"]["results"] if r["candidate"] == "window 4 d, max_leaf_nodes 15")
    assert gbm["result"]["30"]["gain"] > 0.02 and gbm["result"]["30"]["ci"][0] > 0
    run_dir = tmp_path / "runs" / info["run_id"]
    saved = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert saved["models"]["gbm"]["config_lines"][0] == "[fit_window] gbm = 4" and len(saved["config_sha256"]) == 64
    text = pub.read_text(encoding="utf-8")
    assert text == (run_dir / "report.md").read_text(encoding="utf-8")
    for part in ("## Decisions", "## gbm: every candidate", "## B3: every candidate", "## lr: every candidate",
                 "per fold and per hub", "## Against B3"):
        assert part in text
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    assert {"p_gbm_current", "p_gbm_winner", "p_B3_current", "p_lr_winner"} <= set(pred.columns)


def test_refuses_an_incomplete_window_and_non_training_days(tmp_path):
    write_days(tmp_path / "f", D("2026-09-30"), 5, 200)
    tc = TuningConfig(D("2026-09-30"), D("2026-10-05"), min_train_days=3, fold_days=1)
    with pytest.raises(SystemExit, match="incomplete"):
        run(tc, TrainingConfig(), tmp_path / "f", tmp_path / "runs", None, FEATS, log=lambda *_: None)
    with pytest.raises(SystemExit, match="training period"):
        run(replace(tc, last_day=D("2026-11-12")), TrainingConfig(), tmp_path / "f", tmp_path / "runs", None, FEATS,
            log=lambda *_: None)
