# e06: second feature exploration

Service days 2026-09-24 to 2026-10-14 (21 days, 2,000,697 rows), expanding window with daily folds: 14 evaluation days (2026-10-01 to 2026-10-14). Model: gradient boosting, one model per cutoff, no calibration, settings from config/training.toml. Rule: config/feature_selection.toml (sha256 2bfc3d0b34fb), DESIGN.md section 3. Code 087a0c6.

Reference groups: db, hub, freshness.

## Noise floor

Refitting the reference with another seed changes log loss by -0.09% at 10 min, +0.00% at 30 min, -0.08% at 60 min. Threshold at 30 min: max(0.5%, 2 x 0.00%) = **0.50%**. Largest loss allowed at the other cutoffs: 0.20% at 60 min, 0.20% at 10 min.

## Decisions

Gain = relative reduction of log loss by having the group (positive = the group helps).

| Group | Test | Gain 60 min | Gain 30 min | 95% interval (30) | Gain 10 min | Days better (30) | AUC diff (30) | Decision |
|---|---|---|---|---|---|---|---|---|
| hub | leave out | +2.19% | +1.41% | +0.90% to +1.99% | +0.99% | 86% | +0.0051 | **keep** |
| freshness | leave out | +0.67% | +0.59% | +0.35% to +0.86% | +0.81% | 100% | +0.0014 | **keep** |
| context | add | +3.04% | +3.07% | +2.30% to +3.71% | +3.28% | 93% | +0.0073 | **keep** |
| trend | add | +0.01% | +0.40% | +0.20% to +0.62% | +0.97% | 79% | +0.0009 | **drop** |
| messages | add | +0.13% | +0.16% | +0.03% to +0.32% | +0.16% | 71% | +0.0002 | **drop** |

- trend: gain at 30 min +0.40% < 0.50%
- messages: gain at 30 min +0.16% < 0.50%

Gain at 30 min per hub (for information):

| Group | Aachen Hbf | Duisburg Hbf | Düsseldorf Hbf | Essen Hbf | Köln Hbf |
|---|---|---|---|---|---|
| hub | -1.86% | +0.47% | +1.52% | +4.72% | +0.90% |
| freshness | +1.89% | +0.92% | +0.32% | +1.52% | +0.39% |
| context | +4.62% | +5.39% | +3.03% | +4.54% | +2.20% |
| trend | +0.41% | +0.07% | +0.43% | +0.80% | +0.32% |
| messages | +0.86% | +0.31% | +0.05% | +0.48% | +0.08% |

## Result

**Feature groups: db, hub, freshness, context**

## Every variant against B3

Log loss and gain over B3 (DB's prognosis as a probability, same model class) fitted on the same days and evaluated on the same rows. For information only; the decision is made against the reference above.

| variant | groups | log loss 10 | vs B3 10 | log loss 30 | vs B3 30 | log loss 60 | vs B3 60 |
|---|---|---|---|---|---|---|---|
| reference | db, hub, freshness | 0.23588 | +1.69% | 0.30246 | +2.15% | 0.36598 | +2.88% |
| reference, other seed | db, hub, freshness | 0.23609 | +1.60% | 0.30246 | +2.15% | 0.36627 | +2.80% |
| without hub | db, freshness | 0.23824 | +0.71% | 0.30678 | +0.75% | 0.37417 | +0.70% |
| without freshness | db, hub | 0.23780 | +0.89% | 0.30425 | +1.57% | 0.36846 | +2.22% |
| with context | db, hub, freshness, context | 0.22816 | +4.91% | 0.29317 | +5.15% | 0.35487 | +5.83% |
| with trend | db, hub, freshness, trend | 0.23358 | +2.65% | 0.30126 | +2.53% | 0.36595 | +2.89% |
| with messages | db, hub, freshness, messages | 0.23550 | +1.85% | 0.30197 | +2.30% | 0.36550 | +3.01% |

## Gain per evaluation day at 30 min

| day | other seed | hub (leave out) | freshness (leave out) | context | trend | messages |
|---|---|---|---|---|---|---|
| 2026-10-01 | -0.15% | -0.03% | +0.25% | +4.50% | -0.06% | +0.03% |
| 2026-10-02 | +0.02% | +0.86% | +0.31% | +3.81% | -0.03% | +0.22% |
| 2026-10-03 | -0.10% | +1.95% | +0.11% | +4.43% | +0.92% | +0.37% |
| 2026-10-04 | +0.44% | +1.79% | +0.08% | +2.62% | +0.86% | +0.11% |
| 2026-10-05 | -0.51% | -0.23% | +0.33% | +3.10% | +0.90% | +0.09% |
| 2026-10-06 | -0.04% | +1.33% | +0.75% | +4.32% | +0.23% | +0.21% |
| 2026-10-07 | -0.11% | +1.80% | +0.32% | +1.37% | +0.44% | +0.73% |
| 2026-10-08 | -0.18% | +2.03% | +0.60% | +2.76% | +0.15% | -0.02% |
| 2026-10-09 | +0.24% | +2.29% | +1.10% | +4.05% | +0.99% | -0.15% |
| 2026-10-10 | +0.10% | +0.78% | +0.72% | -0.52% | +0.42% | -0.15% |
| 2026-10-11 | -0.14% | +3.01% | +0.42% | +0.83% | -0.52% | -0.07% |
| 2026-10-12 | +0.50% | +3.24% | +0.09% | +3.54% | +0.31% | +0.05% |
| 2026-10-13 | +0.10% | +0.53% | +1.67% | +3.17% | +0.15% | +0.09% |
| 2026-10-14 | -0.12% | +0.80% | +1.17% | +3.85% | +0.94% | +0.73% |
