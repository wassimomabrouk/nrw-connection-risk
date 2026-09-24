# Section 0: Feasibility

Status: closed on 2026-09-23. Decision: **GO with a changed data strategy** (own collector, see D1).

## 1. Question

Can public Deutsche Bahn data support a supervised model that predicts missed connections at five NRW hubs (Köln Hbf, Düsseldorf Hbf, Duisburg Hbf, Essen Hbf, Aachen Hbf) and a fair comparison with DB's own prognosis at fixed lead times (60, 30, 10 minutes)?

## 2. Data examined

`piebro/deutsche-bahn-data` on Hugging Face (CC BY 4.0), collected from the DB Timetables API (IRIS).

- Monthly processed files (one row per train stop), 2024-07 to 2026-08, no missing months, one stable schema. The schema contains three columns not documented in the dataset card: `is_additional_stop`, `is_replacement_train`, `replaced_train_number`.
- Raw API responses (`plan`, `fchg`) with request timestamps, 2024-07-01 to present, no missing days.

Profiled in detail: 2026-08 (14.5M rows), 2025-10 (2.0M rows) and the raw responses of 2026-08-12. Scripts: `section0/`. Reports: `section0/out/`.

## 3. Findings

**F1. Structural break in November 2025.** Coverage grew from 131 stations (2025-10) to 5,260 stations (2026-08), and monthly rows from about 2M to about 15M. Before the break, trips are fragmentary (median 2 observed stops, median coverage 29% of each trip); after it, trips are nearly complete (median 8 stops, median coverage 100%). The five hubs are present in both periods.

**F2. Run identification.** `train_line_ride_id` identifies a recurring route, not one run: 80% of its values repeat station numbers and 49% show planned-time order violations. The stop `id` without its final segment (`<trip>-<YYMMddHHmm>`) identifies one run: 0.00% repeated station numbers, 0.004% order violations, 1.32M runs in 2026-08.

**F3. Column semantics.** `delay_in_min` is the departure delay (100% agreement); arrival delay must be derived from `arrival_change_time - arrival_planned_time`. Change times are always populated when a planned time exists, so "no change" equals "on time" by construction. Timestamps are naive and in local German time (DB prognosis values from the raw XML equal the processed change times, median difference 0 min). About 36,600 rows per month carry no planned time and are unusable.

**F4. Hubs.** EVA numbers verified by name (stored with a leading zero, e.g. `08000207`). Rows per day in 2026-08: Köln 1,149, Düsseldorf 1,161, Essen 870, Duisburg 686, Aachen 486. Train types include buses and many operators (S, NX, ICE, RB, RE, VIA, Bus, RRB, ...).

**F5. Signal and balance.** Arrival delays (2026-08): median 1 min, p95 15 min, 15.6% at 6 min or more. Among all arrival/departure pairs at the hubs with 5 to 9 minutes of planned slack, 26% are missed by delay. These pairs are unfiltered and overstate real transfers, but the minority class is clearly not rare.

**F6. Cancellations.** 3.7% of stops overall, 6 to 7% of arriving trains at the Rhine-Ruhr hubs and 12.5% at Aachen in 2026-08. Cancelled stops still carry change times and delays, so they require a separate label.

**F7. Polling cadence (decisive).** Each station was polled via `fchg` about 5 times per day, roughly every 5.5 hours. A snapshot exists within 15 minutes before the 30-minute cutoff for only 3.8% of hub arrivals. **DB's prognosis at fixed lead times cannot be reconstructed from the historical data.**

**F8. Label validity (decisive).** 34 to 40% of hub arrivals (depending on the zone assumed for request timestamps) were last observed *before* their event, so their final change time is a prognosis made up to several hours earlier, not an observed time. The share is about 41% for arrivals between 05:00 and 17:00 and 2 to 5% after 19:00, so the error is systematic, not random. **The historical data cannot provide trustworthy delay labels.**

## 4. Decisions

**D1. Own collector.** The project collects its own data from the DB Timetables API (free key, 60 requests per minute on the free plan): recent changes (`rchg`, the last two minutes) for the five hubs every minute, the full change state (`fchg`) every 30 minutes to repair any gap, and hourly timetable slices (`plan`). The main NRW feeder stations can be added within the rate limit. This yields observed labels, DB's prognosis at every lead time, and point-in-time features. The collector is the first production component of the project.

**D2. Role of the historical data.** Used only for what depends on planned times: network structure, the definition of plausible transfers, timetable EDA, and sizing. Not used for labels.

**D3. Hosting.** The collector runs 24/7 on an Oracle Cloud Always Free instance (VM.Standard.A1.Flex, 1 OCPU, 6 GB, Frankfurt), managed by systemd with automatic restart. The account stays on the free tier. Oracle may stop idle Always Free instances; this risk is accepted and mitigated by an external heartbeat check (healthchecks.io, alert after 15 minutes of silence), automatic restart after reboots, and a daily off-provider backup of the raw layer to Google Drive (rclone, `copy` semantics so deletions never propagate).

**D4. Run key.** One train run is identified by the stop `id` without its final segment.

**D5. Scope filters.** Buses are excluded. Cancellations are labelled separately and never treated as on time.

**D6. Timestamps.** The collector stores all timestamps as timezone-aware UTC and keeps DB's local-time values alongside them.

**D7. Timeline.** Collection starts as soon as the collector is stable. Exploration, connection definition, feature design and the locked `DESIGN.md` happen during the first 8 to 12 weeks of collection. Training and the locked test follow on the collected data, with the test period after the training period.

## 5. Open items

- Choose the feeder station set within the rate budget (design phase).
- Aachen's elevated cancellation rate in 2026-08 (possible construction work) is worth a closer look during EDA.

Resolved after closing: the free plan allows 60 requests per minute (confirmed at subscription); `fchg` responses retain past arrivals for about 12 hours (700 minutes observed on 2026-09-23), so every arrival is observed after the event.

## 6. Implementation status

- Collector v1 (five hubs) deployed on 2026-09-24; continuous collection since **2026-09-24 03:38 UTC**. First 11 hours: 3,799 API calls, 0 errors.
- Operations (service, monitoring, backup, restore): [docs/OPERATIONS.md](docs/OPERATIONS.md).
