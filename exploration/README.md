# Explorations

Each exploration answers one design question and ends in a decision recorded in [DESIGN.md](../DESIGN.md). The early ones are standalone scripts; from e06 on, explorations run on the project's own pipeline code, so that what is decided is exactly what is deployed.

| | Question | Code | Report | Decided |
|---|---|---|---|---|
| e01 | What does the API return, and how complete is it? | `e01_raw_inventory.py` | `out/e01_raw_inventory.txt` | data inventory |
| e02 | Which arrival/departure pairs are plausible transfers? | `e02_transfer_candidates.py` | `out/e02_transfer_candidates.txt` | section 1 (rules T1 to T7) |
| e03 | Is the label observable, and how often do connections fail? | `e03_label_feasibility.py` | `out/e03_label_feasibility.txt` | section 2 |
| e04 | How good is DB's own prognosis at fixed lead times? | `e04_db_prognosis_baseline.py` | `out/e04_db_prognosis_baseline.txt` | section 4 (baselines) |
| e05 | Which feature groups add information to DB's prognosis? (first look, three days) | `e05_features.py` | `out/e05_features.txt` | section 3 draft, B3 as gradient boosting |
| e06 | Which feature groups does the model use? (pre-registered rule, three weeks) | `src/.../training/feature_selection.py`, `config/feature_selection.toml` | `out/e06_feature_selection.md` (after the run, from 16 October) | section 3 |
| e07a | What can the upstream feeder stations tell us? (inspection, one day) | `e07a_feeder_inspection.py` | `out/e07a_feeder_inspection.txt` | feeder group definition |
| e07 | Is the feeder group worth adding? (same rule as e06) | same code, `config/feature_selection_e07.toml` | `out/e07_feeder_selection.md` (after the run, from 3 November) | section 3 |

Scripts e01 to e05 read the restored data (`data/restore/`); e07a reads the collector's parsed layer on the server. None of them looks at the validation or test periods.
