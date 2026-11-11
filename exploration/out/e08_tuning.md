# e08: tuning

Service days 2026-09-30 to 2026-11-08 (40 days, 3,644,905 rows), rolling origin with weekly blocks (DESIGN.md section 6): 4 folds, 26 evaluation days (2026-10-14 to 2026-11-08). One model per cutoff, no calibration (chosen on validation). Rule: config/tuning.toml (sha256 829bd6c3a368), DESIGN.md section 4. Code aef42ed.


Folds:

- fit 2026-09-30 to 2026-10-13 (14 days), evaluate 2026-10-14 to 2026-10-20
- fit 2026-09-30 to 2026-10-20 (21 days), evaluate 2026-10-21 to 2026-10-27
- fit 2026-09-30 to 2026-10-27 (28 days), evaluate 2026-10-28 to 2026-11-03
- fit 2026-09-30 to 2026-11-03 (35 days), evaluate 2026-11-04 to 2026-11-08

## Decisions

Gain = relative reduction of log loss against the current settings (positive = better). A candidate replaces them only if its gain at 30 min reaches the threshold, its 95% interval (resampling whole days) lies above 0, and it is not worse at the other cutoffs by more than the tolerance.

| Model | Noise (other seed) 60 / 30 / 10 | Threshold at 30 | Winner | Gain 30 min | Settings in config/training.toml |
|---|---|---|---|---|---|
| gbm | -0.03% / +0.00% / -0.14% | 0.20% | **current settings** |  | [fit_window] gbm = 0; [gbm]: unchanged |
| B3 | -0.05% / +0.01% / -0.05% | 0.20% | **current settings** |  | [fit_window] B3 = 0; [b3]: unchanged |
| lr | none (deterministic) | 0.20% | **current settings** |  | [fit_window] lr = 0; [lr]: unchanged |

## gbm: every candidate against the current settings

| Candidate | Gain 60 | Gain 30 | 95% interval (30) | Gain 10 | Days better (30) | Passes |
|---|---|---|---|---|---|---|
| all days, max_leaf_nodes 15, min_samples_leaf 100 | +0.08% | -0.17% | -0.31% to -0.03% | -0.17% | 27% | no |
| all days, max_leaf_nodes 15, min_samples_leaf 1000 | +0.11% | -0.20% | -0.36% to -0.04% | -0.18% | 27% | no |
| all days, max_leaf_nodes 31, min_samples_leaf 1000 | -0.02% | -0.22% | -0.38% to -0.05% | -0.11% | 19% | no |
| all days, max_leaf_nodes 63, min_samples_leaf 100 | -0.18% | -0.22% | -0.43% to -0.00% | -0.25% | 38% | no |
| window 21 d, max_leaf_nodes 31, min_samples_leaf 100 | -0.12% | -0.23% | -0.40% to -0.08% | -0.36% | 4% | no |
| window 21 d, max_leaf_nodes 15, min_samples_leaf 100 | -0.05% | -0.36% | -0.53% to -0.19% | -0.35% | 23% | no |
| window 21 d, max_leaf_nodes 15, min_samples_leaf 1000 | +0.02% | -0.37% | -0.52% to -0.21% | -0.38% | 15% | no |
| all days, max_leaf_nodes 63, min_samples_leaf 1000 | -0.39% | -0.50% | -0.77% to -0.24% | -0.35% | 27% | no |
| window 21 d, max_leaf_nodes 31, min_samples_leaf 1000 | -0.19% | -0.51% | -0.69% to -0.32% | -0.44% | 8% | no |
| window 21 d, max_leaf_nodes 63, min_samples_leaf 100 | -0.48% | -0.51% | -0.76% to -0.27% | -0.70% | 23% | no |
| window 14 d, max_leaf_nodes 31, min_samples_leaf 100 | -0.83% | -0.85% | -1.31% to -0.47% | -0.79% | 8% | no |
| window 14 d, max_leaf_nodes 15, min_samples_leaf 100 | -0.66% | -0.86% | -1.30% to -0.52% | -0.77% | 8% | no |
| window 21 d, max_leaf_nodes 63, min_samples_leaf 1000 | -0.67% | -0.89% | -1.16% to -0.59% | -0.75% | 23% | no |
| window 14 d, max_leaf_nodes 15, min_samples_leaf 1000 | -0.53% | -0.97% | -1.47% to -0.54% | -0.80% | 12% | no |
| window 14 d, max_leaf_nodes 31, min_samples_leaf 1000 | -0.86% | -1.09% | -1.60% to -0.64% | -0.77% | 8% | no |
| window 14 d, max_leaf_nodes 63, min_samples_leaf 100 | -1.33% | -1.21% | -1.76% to -0.75% | -0.94% | 12% | no |
| window 14 d, max_leaf_nodes 63, min_samples_leaf 1000 | -1.66% | -1.63% | -2.20% to -1.11% | -1.33% | 12% | no |

## B3: every candidate against the current settings

| Candidate | Gain 60 | Gain 30 | 95% interval (30) | Gain 10 | Days better (30) | Passes |
|---|---|---|---|---|---|---|
| window 14 d, max_leaf_nodes 15, min_samples_leaf 1000 | +0.23% | +0.14% | -0.06% to +0.33% | -0.01% | 69% | no |
| all days, max_leaf_nodes 15, min_samples_leaf 1000 | +0.16% | +0.13% | +0.05% to +0.23% | +0.11% | 73% | no |
| all days, max_leaf_nodes 63, min_samples_leaf 1000 | +0.12% | +0.12% | +0.03% to +0.25% | +0.07% | 69% | no |
| all days, max_leaf_nodes 31, min_samples_leaf 1000 | +0.13% | +0.12% | +0.03% to +0.23% | +0.07% | 69% | no |
| window 14 d, max_leaf_nodes 31, min_samples_leaf 1000 | +0.22% | +0.11% | -0.11% to +0.32% | -0.02% | 50% | no |
| window 21 d, max_leaf_nodes 15, min_samples_leaf 1000 | +0.16% | +0.09% | -0.06% to +0.24% | +0.01% | 73% | no |
| window 21 d, max_leaf_nodes 31, min_samples_leaf 1000 | +0.12% | +0.07% | -0.08% to +0.24% | -0.03% | 62% | no |
| window 14 d, max_leaf_nodes 63, min_samples_leaf 1000 | +0.21% | +0.06% | -0.15% to +0.28% | -0.07% | 50% | no |
| all days, max_leaf_nodes 15, min_samples_leaf 100 | +0.07% | +0.05% | +0.03% to +0.06% | +0.05% | 88% | no |
| window 21 d, max_leaf_nodes 63, min_samples_leaf 1000 | +0.13% | +0.05% | -0.11% to +0.22% | -0.04% | 58% | no |
| window 14 d, max_leaf_nodes 15, min_samples_leaf 100 | +0.10% | +0.01% | -0.18% to +0.18% | -0.11% | 58% | no |
| window 21 d, max_leaf_nodes 15, min_samples_leaf 100 | +0.06% | -0.00% | -0.13% to +0.10% | -0.05% | 65% | no |
| all days, max_leaf_nodes 63, min_samples_leaf 100 | -0.05% | -0.04% | -0.06% to -0.02% | -0.06% | 15% | no |
| window 21 d, max_leaf_nodes 31, min_samples_leaf 100 | -0.00% | -0.08% | -0.20% to +0.02% | -0.12% | 15% | no |
| window 14 d, max_leaf_nodes 63, min_samples_leaf 100 | -0.08% | -0.09% | -0.28% to +0.10% | -0.24% | 35% | no |
| window 14 d, max_leaf_nodes 31, min_samples_leaf 100 | -0.00% | -0.09% | -0.27% to +0.08% | -0.21% | 35% | no |
| window 21 d, max_leaf_nodes 63, min_samples_leaf 100 | -0.04% | -0.13% | -0.26% to -0.03% | -0.15% | 19% | no |

## lr: every candidate against the current settings

| Candidate | Gain 60 | Gain 30 | 95% interval (30) | Gain 10 | Days better (30) | Passes |
|---|---|---|---|---|---|---|
| all days, C 0.1 | +0.00% | +0.00% | -0.00% to +0.00% | +0.00% | 50% | no |
| all days, C 10.0 | -0.00% | -0.00% | -0.00% to +0.00% | -0.00% | 50% | no |
| window 21 d, C 0.1 | -0.18% | -0.18% | -0.32% to -0.06% | -0.13% | 35% | no |
| window 21 d, C 1.0 | -0.18% | -0.18% | -0.32% to -0.06% | -0.13% | 8% | no |
| window 21 d, C 10.0 | -0.18% | -0.18% | -0.32% to -0.06% | -0.14% | 35% | no |
| window 14 d, C 0.1 | -0.28% | -0.41% | -0.89% to -0.04% | -0.36% | 35% | no |
| window 14 d, C 1.0 | -0.29% | -0.41% | -0.90% to -0.05% | -0.37% | 27% | no |
| window 14 d, C 10.0 | -0.29% | -0.42% | -0.90% to -0.05% | -0.37% | 50% | no |

## Winners per fold and per hub at 30 min (for information)

Gain of each model's winner over its current settings. The winner was chosen as the best of many candidates on these days, so its gain is optimistic; validation is the first test on new days.

| model |
|---|
| (every model keeps its current settings) |

| model |
|---|
| (every model keeps its current settings) |

## Against B3 (for information)

Log loss of each setting and its gain over B3 with B3's settings of the same row, on the same rows.

| model | settings | B3 settings | log loss 10 | vs B3 10 | log loss 30 | vs B3 30 | log loss 60 | vs B3 60 |
|---|---|---|---|---|---|---|---|---|
| gbm | current settings | current settings | 0.24790 | +5.36% | 0.31325 | +5.86% | 0.37698 | +6.50% |
| lr | current settings | current settings | 0.28248 | -7.84% | 0.33597 | -0.96% | 0.39348 | +2.41% |
