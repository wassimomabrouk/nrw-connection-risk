# Design

Status: **draft v0.2 (2026-09-25)**. Sections 1, 2, 4, 5 and 6 are decided; section 3 (features) is written after the feature exploration. The complete document is locked before any model is trained, and later changes are recorded in the change log with their reason.

## 1. Unit of prediction: the transfer candidate

A *transfer candidate* is a pair (A, B) at one of the five hubs, where A is a planned arrival and B a planned departure, such that a passenger could reasonably plan to change from A to B. DB's API does not expose the connections DB itself manages (no `<conn>` elements in 24 hours of data, see `exploration/out/e01_raw_inventory.txt`), so candidates are derived from the timetable with the following rules. Evidence for every rule, with counts and examples: `exploration/out/e02_transfer_candidates.txt` (service day 2026-09-24).

| Rule | Definition | Pairs per day after the rule |
|---|---|---|
| Window | planned slack `pt(B) - pt(A)` between 4 and 30 minutes | 67,825 |
| T1 bus | neither A nor B is a bus | 61,822 |
| T2 same trip | B is not the continuation of A's own run | 61,498 |
| T3 transition | B is not the run A turns into (`tra`) | 61,498 |
| T4 wings | A and B are not coupled parts of one train (`wings`) | 61,498 |
| T5 backtrack | B does not call next at the station A came from directly before the hub | 46,553 |
| T6 redundant | B serves at least one station that A does not serve after the hub | 43,644 |
| T7 first reach | B is the earliest departure (with at least 4 minutes of slack) to at least one station that A has neither passed nor will serve itself | **35,870** |

Result: about 36,000 candidates per day across the five hubs (Köln 16,083, Düsseldorf 11,859, Essen 3,469, Duisburg 3,150, Aachen 1,309), covering 3,356 of 3,624 arrivals, with a median of 11 candidates per arrival.

**Rationale.**
- The 4-minute minimum reflects that DB does not plan shorter connections at hubs of this size. Station-specific minimum transfer times are not available through the API. Sensitivity: 3, 5 and 7 minutes.
- T5 to T7 encode what a journey planner does: a passenger changes trains only to reach somewhere the current train does not go, and takes the first train that gets there. Staying seated counts as an option, which makes T6 a special case of T7.
- Buses are excluded because their outcome is unobservable: only 9.5% of planned bus events ever receive a realtime prognosis, against 99.0% of train events. This matters in the current period, since the RE1 is replaced by buses between Duisburg and Mülheim; at Duisburg, T1 removes half of all pairs.

**Known limitations.**
- No passenger volumes exist. Every candidate carries equal weight, so the candidate set is dominated by S-Bahn transfers (S to S is the largest group at 9.9%). Evaluation is therefore reported per segment as well as overall (see section 5).
- T7 uses the order of departures, not of arrivals downstream, which are not in the data: a later ICE that overtakes an earlier regional train on the same corridor is removed.
- Stop-level messages of type `c` (likely connection notices) carry no content in the API and are not used.

## 2. Label

For each candidate whose arrival A is not already cancelled at the prediction cutoff (such a case needs no prediction):

- **Connection fails** (positive class) if B is cancelled, if A is cancelled after the prediction cutoff, or if the realised slack `actual_dep(B) - actual_arr(A)` is below 4 minutes.
- Otherwise the connection **holds**. If DB holds B for a late A, B's actual departure is later and the connection holds; this is observed, not modelled separately.

Actual times are the last prognosis (`ct`) observed after the event. Candidates where A or B never received a realtime value are excluded from training and evaluation. Delay-caused and cancellation-caused failures are also reported separately.

**Evidence** (service day 2026-09-24, `exploration/out/e03_label_feasibility.txt`):
- The label is observable for 99.9% of candidates; 0.00% of final values come from an observation made before the event (median 40 observations per event). The stale-label problem of the historical data does not occur.
- 22.1% of connections fail: 16.6% by delay, 2.9% because B is cancelled, 2.6% because A is cancelled. The positive class is not rare, so no resampling is planned. With minimum transfer times of 3, 5 and 7 minutes the rate is 20.9%, 24.0% and 28.9%.
- The failure rate falls from 46% at 4 to 5 minutes of planned slack to 13% at 20 to 30 minutes, and is highest for connections from long-distance trains (up to 50%) and in the evening (up to 39% at 21:00).
- In 37% of the cases where A's delay alone would break the connection, it held because B was late as well. Delays of A and B are correlated, so a model must predict both, not A's delay alone.
- One day is not representative (weather, disruptions, construction). These figures are re-estimated on the full training period.

## 3. Features (to be written after the feature exploration)

Point-in-time rule, fixed now: a feature may only use information available at the prediction cutoff (60, 30 and 10 minutes before A's planned arrival), as observed by the collector at that time.

## 4. Baselines and models

**How good DB already is** (service day 2026-09-24, `exploration/out/e04_db_prognosis_baseline.txt`). For every candidate, DB's prognosis for A and B was reconstructed as it was known at the cutoff, using only observations up to that moment:

| Cutoff before A | AUC planned slack | AUC DB predicted slack | DB rule precision | DB rule recall | MAE of A's arrival prognosis |
|---|---|---|---|---|---|
| 60 min | 0.665 | 0.812 | 0.841 | 0.394 | 6.8 min |
| 30 min | 0.669 | 0.873 | 0.840 | 0.574 | 5.4 min |
| 10 min | 0.672 | 0.914 | 0.850 | 0.723 | 4.0 min |

DB's prognosis ranks connections far better than the timetable, and when DB's numbers imply a failure it is right 84% of the time. But it misses 61% of failures at 60 minutes and 43% at 30 minutes: DB's point prognosis rarely anticipates delays growing. It is weakest for S-Bahn to S-Bahn transfers (AUC 0.755) and at Essen Hbf (0.792). This is the headroom the project targets.

**Baselines**, all evaluated on exactly the same candidates and cutoffs as the models:

| | Baseline | Definition |
|---|---|---|
| B0 | Timetable | Logistic regression on planned slack only |
| B1 | History | Failure rate per hub, segment and planned-slack bucket, estimated on the training period |
| B2 | DB rule | Fails if DB's predicted slack is below 4 minutes or a cancellation is known (binary) |
| B3 | Calibrated DB | Logistic regression on DB's predicted slack, predicted delays of A and B and known cancellations. **This is the headline benchmark.** |

B3 turns DB's prognosis into a probability, so the model is compared against the best version of DB's own information, not against a binary rule.

**Models:** logistic regression with the full feature set, then LightGBM, each with probability calibration (isotonic or Platt, chosen on validation). One model per cutoff or one model with the cutoff as a feature, chosen on validation. The model with the best validation log loss at the 30-minute cutoff is selected; ties within 1% go to the simpler model.

**Pre-registered expectation:** a modest AUC gain over B3 (in the order of 0.01 to 0.03 at 30 minutes), a larger gain in recall at matched precision and in calibration, and the largest gains at the 60-minute cutoff and for S-Bahn transfers, where DB is weakest. A result below these expectations is reported as it is.

## 5. Evaluation

- **Primary metric:** log loss at the 30-minute cutoff, model vs B3. **Secondary:** ROC AUC, Brier score, calibration curve, and recall at the precision of the DB rule (B2).
- **All metrics at all three cutoffs** (60, 30, 10 minutes) and **per segment** (long-distance, regional, S-Bahn, for A and B) and per hub, because candidates carry equal weight and S-Bahn transfers dominate the total.
- **Uncertainty:** 95% confidence intervals by block bootstrap over service days, since candidates on the same day share disruptions and are not independent.
- **Decision layer:** at a threshold chosen on validation, the share of failing connections flagged vs the share of false alarms, reported as a table. Kept deliberately simple.

## 6. Data splits

Strictly by time, whole service days only, no random splits (connections on the same day share disruptions).

| Period | Dates (service days) | Use |
|---|---|---|
| Training | 2026-09-24 to 2026-11-08 | fitting, cross-validation by rolling origin |
| Validation | 2026-11-09 to 2026-11-22 | model and threshold selection, calibration |
| **Test (locked)** | **2026-11-23 to 2026-12-12** | evaluated once, after all choices are fixed |
| Robustness | from 2026-12-13 | reported separately |

The locked test period ends before the annual timetable change on 2026-12-13, which changes lines and schedules. The period after it serves as a separate robustness check of how a model trained on the old timetable holds up, and as the start of the live phase. For the final model, training and validation are merged and refit before the test evaluation.

## Change log

- 2026-09-25: v0.1, sections 1 and 2; label evidence from e03.
- 2026-09-25: v0.2, sections 4 to 6 after e04 (DB prognosis baseline).
