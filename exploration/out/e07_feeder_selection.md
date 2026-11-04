# e07: feeder group

Service days 2026-09-30 to 2026-11-01 (33 days, 2,922,178 rows), expanding window with daily folds: 26 evaluation days (2026-10-07 to 2026-11-01). Model: gradient boosting, one model per cutoff, no calibration, settings from config/training.toml. Rule: config/feature_selection_e07.toml (sha256 134290377953), DESIGN.md section 3. Code d4b8fbc.

Reference groups: db, hub, freshness, context (from config/features.toml).

## Noise floor

Refitting the reference with another seed changes log loss by -0.00% at 10 min, +0.11% at 30 min, -0.04% at 60 min. Threshold at 30 min: max(0.5%, 2 x 0.11%) = **0.50%**. Largest loss allowed at the other cutoffs: 0.20% at 60 min, 0.20% at 10 min.

## Decisions

Gain = relative reduction of log loss by having the group (positive = the group helps).

| Group | Test | Gain 60 min | Gain 30 min | 95% interval (30) | Gain 10 min | Days better (30) | AUC diff (30) | Decision |
|---|---|---|---|---|---|---|---|---|
| feeder | add | +0.90% | +0.69% | +0.43% to +0.97% | +0.62% | 85% | +0.0017 | **keep** |


Gain at 30 min per hub (for information):

| Group | Aachen Hbf | Duisburg Hbf | Düsseldorf Hbf | Essen Hbf | Köln Hbf |
|---|---|---|---|---|---|
| feeder | +0.09% | +0.98% | +0.96% | -0.04% | +0.73% |

## Result

**Feature groups: db, hub, freshness, context, feeder**

## Every variant against B3

Log loss and gain over B3 (DB's prognosis as a probability, same model class) fitted on the same days and evaluated on the same rows. For information only; the decision is made against the reference above.

| variant | groups | log loss 10 | vs B3 10 | log loss 30 | vs B3 30 | log loss 60 | vs B3 60 |
|---|---|---|---|---|---|---|---|
| reference | db, hub, freshness, context | 0.24786 | +5.23% | 0.31178 | +6.07% | 0.37582 | +6.32% |
| reference, other seed | db, hub, freshness, context | 0.24787 | +5.22% | 0.31143 | +6.17% | 0.37596 | +6.29% |
| with feeder | db, hub, freshness, context, feeder | 0.24633 | +5.81% | 0.30962 | +6.72% | 0.37243 | +7.17% |

## Gain per evaluation day at 30 min

| day | other seed | feeder |
|---|---|---|
| 2026-10-07 | -0.05% | +0.11% |
| 2026-10-08 | +0.14% | -0.06% |
| 2026-10-09 | -0.10% | +0.10% |
| 2026-10-10 | -0.13% | -0.37% |
| 2026-10-11 | +0.59% | +1.24% |
| 2026-10-12 | +0.49% | +0.57% |
| 2026-10-13 | +0.35% | -0.03% |
| 2026-10-14 | -0.49% | +0.59% |
| 2026-10-15 | +0.24% | +0.92% |
| 2026-10-16 | -0.00% | +0.28% |
| 2026-10-17 | -0.24% | +1.59% |
| 2026-10-18 | +0.58% | +0.72% |
| 2026-10-19 | -0.21% | +0.64% |
| 2026-10-20 | +0.14% | +1.65% |
| 2026-10-21 | -0.19% | +0.59% |
| 2026-10-22 | +0.21% | +2.58% |
| 2026-10-23 | -0.02% | +1.63% |
| 2026-10-24 | +0.78% | +1.91% |
| 2026-10-25 | -0.15% | -0.07% |
| 2026-10-26 | -0.07% | +0.98% |
| 2026-10-27 | -0.02% | +0.04% |
| 2026-10-28 | -0.05% | +0.15% |
| 2026-10-29 | +0.16% | +0.46% |
| 2026-10-30 | +0.46% | +1.17% |
| 2026-10-31 | +0.46% | +0.46% |
| 2026-11-01 | +0.35% | +0.99% |
