# Feature card

Features for every eligible row of the dataset (one transfer candidate at one cutoff). Settings: `config/features.toml`. Code: `src/nrw_connection_risk/features/`. Design decisions: [DESIGN.md](../DESIGN.md) section 3.

## Lineage

```
parsed layer (hubs and feeders) ───┐
                                   ├──> features.build ──> data/features/v{feature_version}/service_day=YYYY-MM-DD/
dataset day (eligible rows) ───────┘                        part-0.parquet, _meta.json
```

`features/compute.py` holds pure functions (no file access). The build CLI feeds them stored data; the prediction API will feed them the collector's current window. Training and serving therefore share one implementation, and a test checks that both give identical values (see Leakage guarantees).

## Building

```
python -m nrw_connection_risk.features.build --from 2026-09-24 --to 2026-09-30
```

Defaults: dataset `data/dataset/v1`, parsed layer `data/restore/parsed`, output `data/features`. Days without a built dataset day are skipped. Each day is written atomically (temporary folder, then rename). The table holds **all** groups; `groups` in `config/features.toml` selects which ones a model uses, so group experiments need no rebuild.

## Columns

Carried along, never model inputs: `service_day, cutoff_min, t_cut (UTC), eva, stop_id_a, stop_id_b, label_fail, fail_reason`.

| Group | Column | Meaning | Missing when |
|---|---|---|---|
| db | `db_slack_min` | DB's predicted transfer time at the cutoff | never |
| | `db_delay_a_min`, `db_delay_b_min` | DB's predicted delay of A's arrival and B's departure (0 if no change known) | never |
| | `b_cancel_known` | B's cancellation known at the cutoff (0/1) | never |
| hub | `hub_mean_delay` | mean predicted delay of non-cancelled trains planned within ±30 min of the grid time at the hub | no train in the window |
| | `hub_share_late5` | share of them predicted 5+ min late | same |
| | `hub_share_cancel` | share cancelled | same |
| | `line_recent_delay_a` | mean delay of arrivals of A's line at the hub in the last 60 min | no such arrival |
| freshness | `age_a_min`, `age_b_min` | whole minutes since the last observation of A / B: minute marks between the last observation up to the cutoff minute and the cutoff minute | event never observed before the cutoff minute |
| context | `planned_slack_min` | planned transfer time | never |
| | `hour_sin`, `hour_cos` | local hour of A's planned arrival (Europe/Berlin, DST-aware) | never |
| | `day_type` | weekday / saturday / sunday_holiday (NRW holidays from the config) | never |
| | `n_stations_before_a`, `same_platform`, `segment_a`, `segment_b`, `hub` | timetable context | never |
| trend | `trend_a_15`, `trend_a_30`, `trend_b_15` | change of DB's predicted delay over the last 15 / 30 minutes | never |
| messages | `n_delay_codes_a`, `n_quality_a`, `n_delay_codes_b` | distinct delay-cause / quality codes seen on A's arrival / B's departure | never (0) |
| | `h_notice_a`, `h_notice_b`, `c_notice_a` | disruption (`h`) or connection (`c`) notice seen on the stop (0/1) | never (0) |
| feeder | `corridor_delay_a` | mean predicted delay of non-cancelled trains planned within ±30 min at each feeder station on A's planned path before the hub, averaged over those feeders | A passes no feeder, or A's arrival was not yet in the timetable at the cutoff |
| | `corridor_line_delay_a` | mean delay of arrivals of A's line in the last 60 min at those feeders, averaged | no such arrival |
| | `corridor_delay_b` | as `corridor_delay_a`, for B's path into the hub | B starts at the hub or passes no feeder |

Categorical: `segment_a, segment_b, hub, day_type`. Hub state is computed on a 5-minute grid and joined at the last grid point at or before the cutoff, so it is up to 5 minutes older than the cutoff. The feeder group applies the same hub-state code to the 12 feeder stations (`[feeder.stations]` in `config/features.toml`, names as they appear in planned paths), so it follows the same point-in-time rules; a train's path is used only if its arrival at the hub was already in the timetable at the cutoff. Freshness is counted on the minute grid, so it does not depend on the second at which the collector polls or the service scores (both change with every restart; with fractional minutes the live distribution shifted at every restart, found by the daily drift check). Feature version 2 added the feeder group and whole-minute freshness (`data/features/v2`); whether models use it is decided by e07 (DESIGN.md section 3).

## Leakage guarantees

1. Every function uses only data collected at or before the row's cutoff: realtime state via `state_at` (as-of join), trains in the hub state only once their timetable had been seen, messages by the time they were **first** seen.
2. The build reads data only up to the last cutoff of the day, so later data cannot enter by construction.
3. Tests (`tests/test_features_compute.py`, `tests/test_features_build.py`):
   - *Future rewrite:* everything collected after a time T is replaced by different data (other delays, cancellations, messages, new trains); features of rows with a cutoff at or before T must be identical. Run on random data and end to end from XML through the parsed layer and the dataset.
   - *Truncation (training = serving):* each row's features must equal the features computed from the data exactly as it stood at that row's cutoff. This catches even a one-second look-ahead.
   - Both tests were checked against five deliberately injected leaks (ignoring when a timetable was published, counting messages seen later, reading the hub state 5 minutes late, looking 1 minute ahead in the trend, rounding the grid up); each one makes them fail.

## Known limitations

- The hub state takes planned times from the latest plan version. This is point-in-time safe only if planned times never change after first publication. The build checks this and reports `plan_changes` in `_meta.json` (with a warning if not 0).
- Connection notices (`c`) appear only after the event in practice (e05); the column exists for re-testing but is not expected to help.
- Missing values are left as NaN. Gradient boosting uses them directly; linear models need imputation plus indicator columns (training pipeline).
- The feeder group takes paths and planned times from the latest timetable version, like the hub state (`plan_changes` in `_meta.json` counts hub and feeder events whose planned time changed).
- A's own train is among the trains averaged at a feeder; its own upstream delay adds nothing beyond DB's prognosis (e07a), and it is one of about 9 trains per feeder and half hour.
