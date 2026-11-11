# Explorations

Each exploration answers one design question and ends in a decision recorded in [DESIGN.md](../DESIGN.md). The early ones are standalone scripts; from e06 on, explorations run on the project's own pipeline code, so that what is decided is exactly what is deployed.

| | Question | Code | Report | Decided |
|---|---|---|---|---|
| e01 | What does the API return, and how complete is it? | `e01_raw_inventory.py` | `out/e01_raw_inventory.txt` | data inventory |
| e02 | Which arrival/departure pairs are plausible transfers? | `e02_transfer_candidates.py` | `out/e02_transfer_candidates.txt` | section 1 (rules T1 to T7) |
| e03 | Is the label observable, and how often do connections fail? | `e03_label_feasibility.py` | `out/e03_label_feasibility.txt` | section 2 |
| e04 | How good is DB's own prognosis at fixed lead times? | `e04_db_prognosis_baseline.py` | `out/e04_db_prognosis_baseline.txt` | section 4 (baselines) |
| e05 | Which feature groups add information to DB's prognosis? (first look, three days) | `e05_features.py` | `out/e05_features.txt` | section 3 draft, B3 as gradient boosting |
| e06 | Which feature groups does the model use? (pre-registered rule, three weeks) | `src/.../training/feature_selection.py`, `config/feature_selection.toml` | `out/e06_feature_selection.md` | section 3: DB, hub, freshness, context |
| e07a | What can the upstream feeder stations tell us? (inspection, one day) | `e07a_feeder_inspection.py` | `out/e07a_feeder_inspection.txt` | feeder group definition |
| e07 | Is the feeder group worth adding? (same rule as e06) | same code, `config/feature_selection_e07.toml` | `out/e07_feeder_selection.md` | section 3: feeder kept (+0.69% at 30 min; expected: small or none) |
| e08a | Why did the live gain over B3 drop in early November? (live predictions, per hub) | `e08a_post_construction_drift.py` | `out/e08a_post_construction_drift.txt` | section 4: the training window is tuned |
| e08 | Which training window and learner settings? (tuning, pre-registered rule) | `src/.../training/tuning.py`, `config/tuning.toml` | `out/e08_tuning.md` | section 4: no setting changed (shorter windows were worse; expected: a small gain or none) |
| e08b | Where do the shorter training windows lose? (per fold and per day, from the gains e08 stored) | `e08b_window_by_fold.py` | `out/e08b_window_by_fold.txt` | section 4: they lose most after the change, where a gain was expected |

Scripts e01 to e05 read the restored data (`data/restore/`); e07a reads the collector's parsed layer on the server; e08a reads the live predictions joined to their outcomes by the daily monitoring (server, training-period days). e08b reads the decision file of the e08 run (`runs/`, not in the repository). None of them looks at the validation or test periods.
