# Training and evaluation

Fits every baseline and model on time-based splits and writes one self-contained run folder per evaluation. Settings: `config/training.toml` (model feature groups: `config/features.toml`). Code: `src/nrw_connection_risk/training/`. Design: [DESIGN.md](../DESIGN.md) sections 4 to 6.

## Running

```
python -m nrw_connection_risk.training.evaluate --mode cv          # rolling origin over the training period
python -m nrw_connection_risk.training.evaluate --mode validate    # fit on training, evaluate on validation
python -m nrw_connection_risk.training.evaluate --mode test --final   # once, at the end (see below)
```

Useful overrides for experiments: `--groups db hub` (feature groups of lr and gbm), `--models B3 gbm`, `--calibration platt|isotonic`, `--cutoff-mode feature`, `--min-train-days N --fold-days N`.

## What is compared

| Name | What | Implementation |
|---|---|---|
| B0 | timetable | logistic regression on planned slack |
| B1 | history | failure rate per hub x segment pair x slack bucket, shrunk towards coarser rates |
| B2 | DB rule | fails if DB's predicted slack < 4 min or B's cancellation is known (binary) |
| B3-lin | DB, linear | logistic regression on DB's 4 inputs |
| **B3** | **DB, headline** | gradient boosting on DB's 4 inputs |
| lr | model | logistic regression on the configured feature groups |
| gbm | model | gradient boosting on the configured feature groups, **same class and settings as B3** |

Logistic regression: numeric inputs clipped at the training 0.5%/99.5% quantiles, median-imputed with missing indicators, standardised; categorical inputs one-hot. Gradient boosting: `HistGradientBoostingClassifier` with native missing values and categories, early stopping, large leaves (100 rows) against overfitting. By default each cutoff (60, 30, 10 min) gets its own model; `cutoff_mode = "feature"` fits one model with the cutoff as an input. Calibration (optional): out-of-fold predictions over blocks of whole training days, Platt or isotonic calibrator on them, base model refit on all training days.

## Splits and the test lock

- **cv**: rolling origin within the training period. The first fold fits on the first 14 days and evaluates the next 7; every later fold adds the previous block to the fit. No fold ever fits on a day after the days it evaluates.
- **validate**: fit on all training days, evaluate on the validation period. Used for model, calibration and threshold choices.
- **test**: fit on training + validation, evaluate on the test period. The loader refuses test and robustness days unless the mode unlocks them; `--mode test` additionally requires `--final`, and every evaluation is appended to `reports/test_log.jsonl` (kept in git). A second evaluation needs `--rerun-reason "..."`, which is logged next to the first.

## Feature selection (e06)

```
python -m nrw_connection_risk.training.feature_selection
```

Decides the model's feature groups once, with the rule pre-registered in DESIGN.md section 3 and `config/feature_selection.toml`: gradient boosting (one model per cutoff, no calibration) on expanding windows of whole days with daily folds; every group is compared with the variant without it (leave out for the reference groups, add for the others), against a noise floor from refitting the reference with another seed. Refuses days outside the training period and an incomplete window (`--allow-missing` overrides and is recorded). Writes `runs/<timestamp>-e06/` (`report.md`, `decision.json`, `predictions.parquet`) and a copy of the report to `exploration/out/e06_feature_selection.md`. Fits several variants on every fold, so it takes a while; it prints progress.

e07 (feeder group) uses the same code with its own configuration, on top of the groups e06 wrote into `config/features.toml`, and writes `exploration/out/e07_feeder_selection.md`. A reference read `from` that file leaves out the groups the run tests, so e07 still runs on the e06 result after its own result (feeder kept) was written there:

```
python -m nrw_connection_risk.training.feature_selection --config config/feature_selection_e07.toml
```

## Tuning (e08)

```
python -m nrw_connection_risk.training.tuning
```

Tunes the model (gbm), its benchmark B3 and the logistic regression (lr), each on its own and with the same budget, with the rule pre-registered in DESIGN.md section 4 and `config/tuning.toml`. A candidate is a training window (every day since `first_day`, or only the last 21 or 14 days before each fold) plus learner settings (gradient boosting: `max_leaf_nodes`, `min_samples_leaf`; lr: `C`). Every candidate runs on the rolling origin of DESIGN.md section 6 (14 days, then weekly blocks) and is compared with the current settings against a noise floor from refitting them with another seed; it replaces them only if it beats them clearly at 30 minutes and is not worse at 60 and 10. Refuses days outside the training period and an incomplete window (`--allow-missing` overrides and is recorded). Writes `runs/<timestamp>-e08/` (`report.md`, `decision.json`, the predictions of the current and the winning settings) and a copy of the report to `exploration/out/e08_tuning.md`. The report lists the lines to set in `config/training.toml`.

How models are fit, set in `config/training.toml`: `first_fit_day` (models only use days from then, here the first day with feeder data), `[fit_window]` (days of history per model, 0 = all) and `[b3]` (B3's own gradient-boosting settings, empty = the same as `[gbm]`). Evaluation, the fit command and the live bundle all follow them; the model card records the days each model was fit on.

## Output: `runs/<timestamp>-<mode>/`

| File | Content |
|---|---|
| `report.md` | headline table at 30 min vs B3 with 95% intervals, all cutoffs, decision layer, slices by segment and hub, calibration tables, log loss per day |
| `run.json` | folds, config, feature groups and columns, code commit, selected model, decision threshold, warnings |
| `metrics.csv` | every metric per model, cutoff and slice (long format) |
| `comparisons.csv` | every model vs B3 per cutoff: log loss gain, Brier gain, AUC difference, with intervals |
| `predictions.parquet` | out-of-sample probability of every model for every evaluated row |

Metrics: log loss (primary, at 30 min), Brier score, ROC AUC, calibration table, and recall at the precision of the DB rule. Intervals come from resampling whole service days (1,000 resamples; 200 for AUC), because connections on one day share disruptions. Below 10 evaluation days the report warns that the intervals are unreliable.

Model selection (reported in every run, binding only on validation): best log loss at 30 minutes among lr and gbm; within 1%, the simpler (lr) wins.

## Tests

`tests/test_training.py` runs on synthetic feature tables whose truth is known (failure depends on DB's slack and on the hub state, `tests/synthetic_features.py`): the pipeline must find that lr and gbm beat B3 at every cutoff with intervals above zero, and that B0 is far worse. Further tests: the lock and its log, fit days always before evaluation days, flipping the labels of an evaluation day does not change its predictions, the fast AUC equals scikit-learn's, calibration repairs a deliberately overconfident model, B1's shrinkage by hand. `tests/test_feature_selection.py` checks the rule condition by condition and runs e06 end to end on synthetic data: it must keep the hub group (the truth), reject noise groups, and apply two changes together.
