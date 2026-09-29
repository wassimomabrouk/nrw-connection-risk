"""Training pipeline: splits and the test lock, metrics, baselines, calibration, and
end-to-end runs on synthetic data with a known truth."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from synthetic_features import make_day, write_days
from nrw_connection_risk.features.columns import GROUPS
from nrw_connection_risk.training.config import Splits, TrainingConfig
from nrw_connection_risk.training.evaluate import load_rows, main, predict_folds, select_model
from nrw_connection_risk.training.metrics import (auc, bootstrap_vs, pointwise_log_loss, recall_at_precision,
                                                  scores)
from nrw_connection_risk.training.models import (GBM, Calibrated, DBRule, HistoryRate, LogReg, PerCutoff,
                                                 make_model)
from nrw_connection_risk.training.splits import LockedPeriodError, guard, period_of, rolling_origin

S = Splits()
D = date.fromisoformat


def rows(n=3000, seed=0, day="2026-09-24", **kw):
    return make_day(D(day), n, np.random.default_rng(seed), **kw)


# ---------------------------------------------------------------- splits and lock

def test_periods_follow_design_section_6():
    assert period_of(D("2026-11-08"), S) == "train" and period_of(D("2026-11-09"), S) == "validation"
    assert period_of(D("2026-11-23"), S) == "test" and period_of(D("2026-12-12"), S) == "test"
    assert period_of(D("2026-12-13"), S) == "robustness" and period_of(D("2026-09-23"), S) == "outside"


def test_guard_blocks_locked_periods():
    guard([D("2026-09-24"), D("2026-11-20")], S)
    with pytest.raises(LockedPeriodError):
        guard([D("2026-11-23")], S)
    with pytest.raises(LockedPeriodError):
        guard([D("2027-01-10")], S, frozenset({"test"}))      # unlocking the test does not unlock robustness
    guard([D("2026-11-23")], S, frozenset({"test"}))


def test_loader_refuses_test_days(tmp_path):
    write_days(tmp_path, D("2026-11-22"), 2, 50)                  # 22 Nov validation, 23 Nov test
    assert len(load_rows(tmp_path, [D("2026-11-22")], S)) == 50
    with pytest.raises(LockedPeriodError):
        load_rows(tmp_path, [D("2026-11-22"), D("2026-11-23")], S)


def test_rolling_origin_only_fits_on_the_past():
    days = [D("2026-09-24") + pd.Timedelta(days=i) for i in range(10)]
    days = [d.date() if hasattr(d, "date") else d for d in days]
    folds = rolling_origin(days, 4, 3)
    assert [(len(f), len(e)) for f, e in folds] == [(4, 3), (7, 3)]
    for fit, ev in folds:
        assert max(fit) < min(ev)


# ---------------------------------------------------------------- metrics

def test_fast_auc_equals_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.round(rng.random(5000) + 0.3 * y, 1)            # many ties
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_recall_at_precision():
    y = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    assert recall_at_precision(y, p, 1.0) == (pytest.approx(2 / 3), 0.8)
    assert recall_at_precision(y, p, 0.75)[0] == 1.0
    assert np.isnan(recall_at_precision(y, np.full(6, 0.5), 0.9)[0])


def test_scores_leave_log_loss_undefined_for_a_binary_rule():
    y = np.array([1, 0, 1, 0])
    s = scores(y, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.isnan(s["log_loss"]) and np.isnan(s["auc"]) and s["brier"] == 0.25


def test_bootstrap_by_day():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(20), 500).astype(str)
    y = rng.integers(0, 2, len(days))
    good = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.1, len(y)), 0.01, 0.99)
    weak = np.clip(0.5 + 0.1 * (2 * y - 1) + rng.normal(0, 0.1, len(y)), 0.01, 0.99)
    same = bootstrap_vs(days, y, weak, weak, 200, 50, 0)
    assert same["log_loss_gain"] == 0 and same["log_loss_gain_ci"] == (0, 0)
    b = bootstrap_vs(days, y, good, weak, 200, 50, 0)
    expected = 1 - pointwise_log_loss(y, good).mean() / pointwise_log_loss(y, weak).mean()
    assert b["log_loss_gain"] == pytest.approx(expected) and b["days"] == 20
    assert 0 < b["log_loss_gain_ci"][0] < b["log_loss_gain"] < b["log_loss_gain_ci"][1]
    assert b["auc_diff_ci"][0] > 0


# ---------------------------------------------------------------- baselines and models

def test_history_rate_shrinks_small_cells():
    df = pd.DataFrame({"planned_slack_min": [5.0] * 100 + [5.0] * 2, "hub": ["K"] * 100 + ["D"] * 2,
                       "segment_a": "regional", "segment_b": "regional",
                       "label_fail": [True] * 50 + [False] * 50 + [True, True]})
    m = HistoryRate((4, 6, 31), smoothing=20).fit(df)
    p = m.predict(pd.DataFrame({"planned_slack_min": [5.0, 5.0, 25.0], "hub": ["K", "D", "K"],
                                "segment_a": "regional", "segment_b": "regional"}))
    bucket = (52 + 20 * 52 / 102) / (102 + 20)
    assert p[1] > p[0] and p[1] < 1                           # 2 of 2 failed, shrunk towards the bucket
    hub_d = (2 + 20 * bucket) / (2 + 20)
    assert p[1] == pytest.approx((2 + 20 * hub_d) / (2 + 20))
    assert p[2] == pytest.approx(52 / 102)                    # unseen bucket: overall rate


def test_db_rule():
    df = pd.DataFrame({"db_slack_min": [3.9, 4.0, 10.0], "b_cancel_known": [0, 0, 1]})
    assert list(DBRule().fit(df).predict(df)) == [1.0, 0.0, 1.0]


def test_models_handle_missing_values_and_unseen_categories():
    train, test = rows(2000, 0), rows(500, 1)
    test.loc[:10, "hub"] = "Aachen Hbf"                      # not in the training data
    test.loc[:10, "hub_mean_delay"] = np.nan
    cols = GROUPS["db"] + GROUPS["hub"] + ["hub", "segment_a"]
    for m in (LogReg(cols), GBM(cols, max_iter=50)):
        p = m.fit(train).predict(test)
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_per_cutoff_fits_one_model_each():
    m = PerCutoff(lambda: LogReg(["db_slack_min"])).fit(rows(1500))
    assert set(m.models_) == {10, 30, 60}
    # the truth is sharper closer to the event
    assert abs(m.models_[10].pipe_[-1].coef_[0][0]) > abs(m.models_[60].pipe_[-1].coef_[0][0])


class Overconfident:
    """A deliberately miscalibrated model: logistic regression with its logit tripled."""

    def fit(self, r):
        self.m = LogReg(GROUPS["db"]).fit(r)
        return self

    def predict(self, r):
        p = np.clip(self.m.predict(r), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_repairs_an_overconfident_model(method):
    rng = np.random.default_rng(3)
    train = pd.concat([make_day(D(f"2026-09-{24 + i}"), 1500, rng) for i in range(4)], ignore_index=True)
    train["service_day"] = pd.to_datetime(train.service_day).dt.date
    test = make_day(D("2026-09-30"), 3000, rng)
    y = test.label_fail.astype(int).to_numpy()
    raw = pointwise_log_loss(y, Overconfident().fit(train).predict(test)).mean()
    cal = Calibrated(Overconfident, method, folds=4).fit(train)
    assert not cal.skipped_
    assert pointwise_log_loss(y, cal.predict(test)).mean() < 0.9 * raw


def test_calibration_is_skipped_with_one_fit_day():
    r = rows(1000)
    r["service_day"] = pd.to_datetime(r.service_day).dt.date
    assert Calibrated(lambda: LogReg(["db_slack_min"]), "platt").fit(r).skipped_


def test_b3_and_gbm_share_implementation_and_settings():
    cfg = TrainingConfig(cutoff_mode="feature")
    b3, gbm = make_model("B3", cfg, GROUPS["db"] + GROUPS["hub"]), make_model("gbm", cfg, GROUPS["db"] + GROUPS["hub"])
    assert type(b3) is type(gbm) is GBM and b3.params == gbm.params
    assert b3.cols == GROUPS["db"] + ["cutoff_min"]


def test_select_model_prefers_simpler_within_tie():
    cfg = TrainingConfig()
    t = pd.DataFrame({"model": ["B3", "lr", "gbm"], "slice": "all", "cutoff": 30,
                      "log_loss": [0.30, 0.2019, 0.2000]})
    assert select_model(t, cfg) == "lr"                                   # within 1 %
    assert select_model(t.assign(log_loss=[0.30, 0.21, 0.20]), cfg) == "gbm"


# ---------------------------------------------------------------- end to end

def test_future_labels_do_not_reach_the_past(tmp_path):
    """Flipping every label of the evaluation day leaves its predictions unchanged."""
    days = write_days(tmp_path, D("2026-09-24"), 3, 800)
    r = load_rows(tmp_path, days, S)
    cfg = TrainingConfig(run=("B1", "B3-lin", "gbm"), gbm={"max_iter": 30})
    folds = [(days[:2], [days[2]])]
    a = predict_folds(r, folds, cfg, GROUPS["db"] + GROUPS["hub"], log=lambda *_: None)
    flipped = r.assign(label_fail=np.where(r.service_day == days[2], ~r.label_fail, r.label_fail))
    b = predict_folds(flipped, folds, cfg, GROUPS["db"] + GROUPS["hub"], log=lambda *_: None)
    for m in cfg.run:
        np.testing.assert_array_equal(a[f"p_{m}"], b[f"p_{m}"])


@pytest.fixture(scope="module")
def cv_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cv")
    write_days(tmp / "features", D("2026-09-24"), 12, 2500)
    main(["--features", str(tmp / "features"), "--out", str(tmp / "runs"), "--min-train-days", "7",
          "--fold-days", "1"])
    return next((tmp / "runs").iterdir())


def test_cv_recovers_the_known_truth(cv_run):
    comp = pd.read_csv(cv_run / "comparisons.csv").set_index(["model", "cutoff"])
    # hub state carries information DB's prognosis lacks: both models beat B3 at every cutoff
    for m in ("lr", "gbm"):
        for L in (60, 30, 10):
            lo = float(comp.log_loss_gain_ci[(m, L)].strip("()").split(",")[0])
            assert comp.log_loss_gain[(m, L)] > 0.05 and lo > 0
    assert (comp.loc["B0"].log_loss_gain < -0.2).all()          # the timetable alone is far worse than DB


def test_cv_run_outputs(cv_run):
    info = json.loads((cv_run / "run.json").read_text())
    assert info["mode"] == "cv" and len(info["folds"]) == 5
    assert all(max(f["fit"]) < min(f["evaluate"]) for f in info["folds"])
    assert info["selected_model"] in ("lr", "gbm") and info["decision"]["model_at_threshold"]["recall"] > \
        info["decision"]["db_rule"]["recall"]
    assert any("Only 5 evaluation day" in w for w in info["warnings"])
    pred = pd.read_parquet(cv_run / "predictions.parquet")
    assert len(pred) == 5 * 2500 and {"p_B3", "p_gbm", "fold"} <= set(pred.columns)
    report = (cv_run / "report.md").read_text(encoding="utf-8")
    for heading in ("## Headline", "## All cutoffs", "## Decision layer", "## Slices", "## Calibration",
                    "## Log loss per evaluation day"):
        assert heading in report


def test_test_mode_is_locked_and_logged(tmp_path):
    feats = tmp_path / "features"
    write_days(feats, D("2026-11-06"), 2, 600)          # training
    write_days(feats, D("2026-11-20"), 2, 600, seed=1)  # validation
    write_days(feats, D("2026-11-23"), 1, 600, seed=2)  # test
    log = tmp_path / "test_log.jsonl"
    base = ["--features", str(feats), "--out", str(tmp_path / "runs"), "--test-log", str(log),
            "--models", "B2", "B3-lin"]
    with pytest.raises(SystemExit, match="locked"):
        main(base + ["--mode", "test"])
    assert main(base + ["--mode", "validate"]) == 0
    assert main(base + ["--mode", "test", "--final"]) == 0
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["previous_runs"] == 0
    with pytest.raises(SystemExit, match="already evaluated"):
        main(base + ["--mode", "test", "--final"])
    assert main(base + ["--mode", "test", "--final", "--rerun-reason", "bug in metric fixed"]) == 0
    assert json.loads(log.read_text().splitlines()[1])["rerun_reason"] == "bug in metric fixed"
    runs = {json.loads((d / "run.json").read_text())["mode"]: d for d in (tmp_path / "runs").iterdir()}
    folds = json.loads((runs["test"] / "run.json").read_text())["folds"][0]
    assert folds["fit"] == ["2026-11-06", "2026-11-07", "2026-11-20", "2026-11-21"]
    assert folds["evaluate"] == ["2026-11-23"]
