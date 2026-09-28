"""Every column of the training table, by what it may be used for.

Training code must build features only from PLANNED and AT_CUTOFF. LABEL columns
describe what happened after the cutoff; using them as inputs is leakage.
A test checks that the builder's output matches this classification exactly.
"""
KEYS = ["service_day", "cutoff_min", "t_cut", "eva", "hub", "stop_id_a", "stop_id_b", "trip_a", "trip_b"]

# timetable information, known before the cutoff
PLANNED = ["pt_a", "pt_b", "pt_raw_a", "pt_raw_b", "planned_slack_min", "cat_a", "cat_b", "num_a", "num_b",
           "line_a", "line_b", "segment_a", "segment_b", "pp_a", "pp_b", "origin_a", "prev_station_a",
           "n_stations_before_a", "next_station_b", "destination_b", "n_stations_after_b"]

# realtime state as observed by the collector at or before t_cut
AT_CUTOFF = ["a_ct_cut", "a_cs_cut", "a_obs_cut", "b_ct_cut", "b_cs_cut", "b_obs_cut", "db_delay_a_min",
             "db_delay_b_min", "db_slack_min", "b_cancel_known", "collector_age_min"]

# outcome after the cutoff: targets and evaluation only, never features
LABEL = ["a_ct_final", "a_cs_final", "a_last_obs", "b_ct_final", "b_cs_final", "b_last_obs",
         "delay_a_final_min", "delay_b_final_min", "real_slack_min", "label_fail", "fail_reason"]

# data quality and filtering
QUALITY = ["first_seen_a", "first_seen_b", "exclusion_reason", "eligible"]

FEATURE_SAFE = PLANNED + AT_CUTOFF
ALL = KEYS + PLANNED + AT_CUTOFF + LABEL + QUALITY
