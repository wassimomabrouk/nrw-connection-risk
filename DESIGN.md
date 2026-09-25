# Design

Status: **draft v0.1 (2026-09-25)**. Sections 1 and 2 are decided; sections 3 to 6 are written after the remaining explorations. The complete document is locked before any model is trained, and later changes are recorded in the change log with their reason.

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

## 3. Features (to be written after exploration e04)

Point-in-time rule, fixed now: a feature may only use information available at the prediction cutoff (60, 30 and 10 minutes before A's planned arrival), as observed by the collector at that time.

## 4. Baselines and models (to be written)

## 5. Evaluation (to be written)

## 6. Data splits (to be written)

## Change log

- 2026-09-25: v0.1, sections 1 and 2.
