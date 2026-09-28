# Dataset card

The training table behind every model and baseline in this project: one row per transfer candidate and prediction cutoff, with the realtime state known at the cutoff and the observed outcome. Definitions come from [DESIGN.md](../DESIGN.md) sections 1 and 2; this document describes how they are implemented.

## Lineage

```
raw layer         every API response, untouched (gzip JSON lines)          source of truth, backed up daily
   │ collector/parse.py  (parser version 2)
parsed layer      one row per stop, event and observation (Parquet)        rebuildable: tools/rebuild_parsed.py
   │ dataset/build.py    (builder version 1, config/dataset.toml)
dataset           one row per candidate and cutoff (Parquet, per service day)
```

The collector and the dataset builder use the same parser, so training data and live data cannot drift apart through two parser implementations. The builder refuses parsed files written by an older parser version and says how to rebuild them.

## Building

From the repo root, with the raw layer restored from the backup:

```
rclone copy gdrive:nrw-connection-risk-backup/raw data/restore/raw            # new files only
python tools/rebuild_parsed.py --raw data/restore/raw --out data/restore --replace
python -m nrw_connection_risk.dataset.build --parsed data/restore/parsed --from 2026-09-24 --to 2026-09-30
```

Output: `data/dataset/v1/service_day=YYYY-MM-DD/part-0.parquet` plus `_meta.json` (candidate funnel, counts per cutoff and exclusion reason, failure rate, AUC of the planned-slack and DB-prognosis scores, settings, git commit, data span). Rebuilding a day replaces it atomically. A day is built only once the data reaches 6 hours past its end (`label_horizon_h`); earlier it is skipped, never built partially.

## Grain and time

- **Service day:** arrivals at the hub from 04:00 to 04:00 local time (23 or 25 hours on DST days).
- **Row:** transfer candidate (A, B) × cutoff (60, 30, 10 minutes before A's planned arrival). About 36,000 candidates and 108,000 rows per day.
- **Times:** every timestamp is UTC. DB's local times are also kept raw (`pt_raw_a`, ...).

## Columns

The classification is code, not only documentation: [`dataset/columns.py`](../src/nrw_connection_risk/dataset/columns.py), enforced by a test.

| Group | Columns | Use |
|---|---|---|
| Keys | `service_day`, `cutoff_min`, `t_cut`, `eva`, `hub`, `stop_id_a/b`, `trip_a/b` | identification, grouping, splits |
| Planned | planned times and slack, category, number, line, segment, platform, origin, previous and next station, destination, path lengths | **features** (timetable, known in advance) |
| At cutoff | DB's prognosis for A and B at `t_cut` (`a_ct_cut`, `b_ct_cut`, cancellation status, observation time), `db_delay_a_min`, `db_delay_b_min`, `db_slack_min`, `b_cancel_known`, `collector_age_min` | **features** and baselines B2 and B3 |
| Label | final times and status of A and B, final delays, `real_slack_min`, `label_fail`, `fail_reason` | **targets and evaluation only, never features** |
| Quality | `first_seen_a/b`, `exclusion_reason`, `eligible` | filtering |

**Point-in-time guarantee:** every "at cutoff" value comes from the latest observation collected at or before `t_cut`. Tests check that an observation collected one minute after the cutoff is invisible and one collected exactly at the cutoff is visible.

## Label

`label_fail` is true if A is cancelled, B is cancelled, or `real_slack_min = actual departure of B - actual arrival of A` is below 4 minutes. `fail_reason` records which, in that order of priority. Actual times are the last prognosis observed for the event.

## Exclusions

Rows are kept with `eligible = false` and a reason, so every exclusion can be counted and audited. The first matching reason wins.

| Reason | Meaning |
|---|---|
| `not_known_at_cutoff` | A or B was not yet in any timetable response at the cutoff (also covers the time before the collector started) |
| `collector_gap` | no observation at the hub for more than 45 minutes before the cutoff (outage); DB's state at that time is unknown |
| `a_cancelled_at_cutoff` | A was already known to be cancelled: no prediction needed (DESIGN.md section 2) |
| `unobserved` | A or B never received a realtime value, so the outcome is unknown |
| `stale_label` | the last observation of A or B was made before the event, so the "actual" time is still a prognosis |
| `dst_ambiguous` | a time falls into the repeated autumn hour or the skipped spring hour and cannot be converted to UTC unambiguously |

## Known limitations

- Candidates carry equal weight; there are no passenger volumes (DESIGN.md section 1).
- The candidate set uses the latest timetable version. A timetable change between the cutoff and the event can alter which departures count as candidates; trains added after the cutoff are excluded by `not_known_at_cutoff`.
- Rule T7 judges departures at the same minute together: each is kept if it reaches a station that no earlier departure reaches. The exploration script processed them in row order, which could drop one of two equally early trains depending on file order. The builder's rule is order-independent and keeps a superset.

## Reproducing the exploration

On two synthetic days (5 hubs, delays, cancellations, the collector's polling pattern) the builder reproduces exploration scripts e02 to e04 exactly on the first (candidates, failure rate, eligible rows and AUCs at every cutoff) and, on the second, keeps one additional candidate, a departure sharing its minute with another, which the exploration dropped because of row order. For the real service day 2026-09-24 it must reproduce the exploration results within a few candidates: 35,870 candidates (e02), a failure rate of 22.13% (e03), and at the 60, 30 and 10 minute cutoffs 33,040, 33,766 and 34,186 eligible rows with AUCs of 0.665 / 0.669 / 0.672 for planned slack and 0.812 / 0.873 / 0.914 for DB's prognosis (e04).
