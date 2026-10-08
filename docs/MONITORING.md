# Monitoring

Every morning the server evaluates the previous service day of the live API: how good the logged predictions were, whether the inputs still look like the training data, and whether the service covered all connections. Code: `src/nrw_connection_risk/monitoring/`. Settings: `config/monitoring.toml`. Schedule: `deploy/nrw-monitor.timer` (09:15 UTC).

## How a day is evaluated

```
prediction log (API)          collector data
one row per connection        the day plus 6 hours
and cutoff (60/30/10)         |
        |                     dataset builder: candidates, labels, exclusions (as in training)
        +------ join on (arrival, departure, cutoff) ------+
                               |
            evaluated rows: model vs B3 vs DB rule        -> daily/service_day=D.json
            all logged rows: drift against training        -> joined/service_day=D.parquet
            coverage, logging lag, data age                -> summary.json, summary.md
```

- **Labels.** Outcomes come from the dataset builder, so a live prediction is labelled exactly like a training row: same failure rule, same exclusions (collector gaps, stale labels, DST hour).
- **Row types.** A logged prediction is `evaluated` if the dataset builder has it as an eligible row, `excluded` if it is a candidate with an exclusion reason, and `not_a_candidate` if the full-day timetable does not make it a transfer candidate (the live service only knew the timetable up to the moment of scoring).
- **Performance** per cutoff: log loss, Brier score, AUC and calibration of the model and of B3, DB rule precision and recall, recall of each at the DB rule's precision. The summary pools all days and gives 95% intervals from resampling whole days.
- **Drift.** Every model card holds a profile of the training data: decile bins for numeric inputs, value shares for categorical ones, and the share of missing values. The population stability index (PSI) compares each input's live distribution with it: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large. The live failure rate is compared with the training failure rate per cutoff. Days with fewer than 500 logged rows get no PSI. Inputs that are constant within a service day by design (`day_type`: one day is one day type) are reported but not ranked: against a training mix of day types their daily PSI is always large and would hide real drift in other inputs (`skip` in `config/monitoring.toml`).
- **Training/serving skew on production traffic.** For every logged prediction, the features are recomputed offline with the training pipeline's own code from the collected data as it stood at the moment of scoring, and compared with the features the API logged. The report gives the share of rows identical in every input and, per input, the share of equal values and the mean difference where they differ. Expected small sources of difference: the live service reads the last 6 hours while the offline builder reads from the day before (an event without an update for 6 hours), and a response arriving within the second of a scoring run.
- **Operations.** Coverage (share of the day's eligible connections that the live service logged and that could be evaluated), logging lag after each cutoff, and data age at the moment of scoring.

## Dashboard

The API serves the results at `/dashboard` (through the SSH tunnel: http://localhost:8000/dashboard) and as JSON at `/v1/monitoring`: service status, the pooled log loss gain over B3 with its interval, daily log loss of model and B3, calibration and input drift of the latest day, coverage and skew. The page is plain HTML with inline SVG, no scripts or external resources, in light and dark mode.

## Running it

```
python -m nrw_connection_risk.monitoring.daily                    # all finished days without a report
python -m nrw_connection_risk.monitoring.daily --day 2026-10-01   # one day (again)
```

A day is finished once its data reaches 6 hours past its end (04:00 local the next morning). Reports are written to `data/monitoring/` on the server. Days without predictions are skipped; days whose outcomes cannot be built yet are recorded with the reason (`no_outcomes`).

## Reading the results honestly

- Until November the API serves a stand-in model fitted on a few days, so live figures describe the system, not the final model.
- Live monitoring does not replace the locked test (DESIGN.md section 6). It checks that the deployed model behaves as evaluated offline and warns early when the data changes, for example at the timetable change on 13 December.
- The first days after a deployment can be incomplete: a day is only fully covered if the API ran for the whole day.
