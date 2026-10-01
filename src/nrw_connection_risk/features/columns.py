"""Feature groups and their columns (DESIGN.md section 3)."""
GROUPS: dict[str, list[str]] = {
    "db": ["db_slack_min", "db_delay_a_min", "db_delay_b_min", "b_cancel_known"],
    "hub": ["hub_mean_delay", "hub_share_late5", "hub_share_cancel", "line_recent_delay_a"],
    "freshness": ["age_a_min", "age_b_min"],
    "context": ["planned_slack_min", "hour_sin", "hour_cos", "day_type", "n_stations_before_a",
                "same_platform", "segment_a", "segment_b", "hub"],
    "trend": ["trend_a_15", "trend_a_30", "trend_b_15"],
    "messages": ["n_delay_codes_a", "n_quality_a", "n_delay_codes_b", "h_notice_a", "h_notice_b", "c_notice_a"],
    "feeder": ["corridor_delay_a", "corridor_line_delay_a", "corridor_delay_b"],
}
CATEGORICAL = {"segment_a", "segment_b", "hub", "day_type"}

# carried along for evaluation and slicing, never used as model inputs
KEYS = ["service_day", "cutoff_min", "t_cut", "eva", "stop_id_a", "stop_id_b", "segment_a", "segment_b"]
TARGET = ["label_fail", "fail_reason"]


def feature_columns(groups: list[str]) -> list[str]:
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    cols: list[str] = []
    for g in groups:
        cols += [c for c in GROUPS[g] if c not in cols]
    return cols


ALL_FEATURES = feature_columns(list(GROUPS))
