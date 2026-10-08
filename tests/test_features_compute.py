"""Feature functions on small hand-made inputs, plus the leakage property test.

All times are naive UTC, as inside the pipeline."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nrw_connection_risk.dataset.state import state_at
from nrw_connection_risk.features.columns import ALL_FEATURES, GROUPS, feature_columns
from nrw_connection_risk.features.compute import (compute_all, context, feeder, freshness, hub, hub_state,
                                                  messages, trend)
from nrw_connection_risk.features.config import FeatureConfig, load_feature_config
from nrw_connection_risk.features.load import first_seen_messages

T = pd.Timestamp
NEUSS, DORTMUND = "8000274", "8000080"
CFG = FeatureConfig(feeders=((NEUSS, "Neuss Hbf"), (DORTMUND, "Dortmund Hbf")))
KOELN = "8000207"


def obs_frame(rows):
    """(key, obs, ct, cs) tuples -> observation table as load_window returns it."""
    o = pd.DataFrame(rows, columns=["key", "obs", "ct", "cs"])
    o["obs"] = pd.to_datetime(o.obs).astype("datetime64[ns]")
    o["ct"] = pd.to_datetime(o.ct).astype("datetime64[ns]")
    o["ct_raw"] = None
    o["eva"] = KOELN
    return o


def plan_frame(rows):
    """(stop_id, event, pt, line, first_seen) tuples -> plan table at one hub."""
    p = pd.DataFrame(rows, columns=["stop_id", "event", "pt", "line", "first_seen"])
    p["pt"] = pd.to_datetime(p.pt).astype("datetime64[ns]")
    p["first_seen"] = pd.to_datetime(p.first_seen).astype("datetime64[ns]")
    p["eva"] = KOELN
    return p


# ---------------------------------------------------------------- columns and config

def test_feature_columns_validate_groups():
    assert feature_columns(["db"]) == GROUPS["db"]
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES)) == sum(len(v) for v in GROUPS.values())
    with pytest.raises(ValueError):
        feature_columns(["db", "weather"])


def test_config_file_loads():
    cfg = load_feature_config(Path(__file__).resolve().parents[1] / "config" / "features.toml")
    assert cfg.groups == ("db", "hub", "freshness", "context")
    assert date(2026, 10, 3) in cfg.holidays and cfg.grid_min == 5


# ---------------------------------------------------------------- groups

def test_trend_is_change_of_db_delay():
    # A (planned 08:12): +2 min known from 07:00, +8 min known from 07:40
    o = obs_frame([("A|ar", "2026-09-24 07:00", "2026-09-24 08:14", None),
                   ("A|ar", "2026-09-24 07:40", "2026-09-24 08:20", None)])
    rows = pd.DataFrame({"stop_id_a": ["A", "A"], "stop_id_b": ["B", "B"],
                         "t_cut": [T("2026-09-24 07:50"), T("2026-09-24 07:25")],
                         "pt_a": [T("2026-09-24 08:12")] * 2, "pt_b": [T("2026-09-24 08:25")] * 2,
                         "db_delay_a_min": [8.0, 2.0], "db_delay_b_min": [0.0, 0.0]})
    out = trend(rows, o)
    assert list(out.trend_a_15) == [6, 0]      # 07:35 -> 2 ; 07:10 -> 2
    assert list(out.trend_a_30) == [6, 2]      # 07:20 -> 2 ; 06:55 -> nothing known yet (0)
    assert list(out.trend_b_15) == [0, 0]      # B never changed


def test_freshness_whole_minutes_since_last_observation():
    o = obs_frame([("A|ar", "2026-09-24 07:48:25", "2026-09-24 08:14", None),
                   ("B2|dp", "2026-09-24 07:20:50", "2026-09-24 08:30", None)])
    rows = pd.DataFrame({"stop_id_a": ["A", "A2"], "stop_id_b": ["B", "B2"],
                         "t_cut": [T("2026-09-24 07:50")] * 2})
    out = freshness(rows, o)
    assert out.age_a_min[0] == 2 and np.isnan(out.age_a_min[1])       # 07:48:25 -> marks 07:49, 07:50
    assert np.isnan(out.age_b_min[0]) and out.age_b_min[1] == 30


def test_freshness_does_not_depend_on_polling_or_scoring_second():
    """The collector's polling second and the service's scoring second change with every
    restart; the age must not."""
    def ages(poll_second, cut):
        o = obs_frame([("A|ar", f"2026-09-24 07:4{m}:{poll_second:02d}", "2026-09-24 08:14", None)
                       for m in (5, 6)])
        rows = pd.DataFrame({"stop_id_a": ["A"], "stop_id_b": ["B"], "t_cut": [T(cut)]})
        return freshness(rows, o).age_a_min[0]
    training = [ages(s, "2026-09-24 07:50:00") for s in (2, 25, 50)]      # cut at the minute mark
    live = [ages(s, "2026-09-24 07:50:37") for s in (2, 25, 50)]          # scored 37 s after it
    assert training == live == [4.0, 4.0, 4.0]                            # last seen 07:46:xx
    # an observation after the cutoff minute's mark is not used, even if the service scores later
    o = obs_frame([("A|ar", "2026-09-24 07:46:10", "2026-09-24 08:14", None),
                   ("A|ar", "2026-09-24 07:50:20", "2026-09-24 08:15", None)])
    rows = pd.DataFrame({"stop_id_a": ["A"], "stop_id_b": ["B"], "t_cut": [T("2026-09-24 07:50:37")]})
    assert freshness(rows, o).age_a_min[0] == 4


def test_first_seen_messages_parses_and_keeps_earliest():
    raw = pd.DataFrame({
        "stop_id": ["A", "A", "A", "B"], "event": ["ar", "ar", "", "dp"],
        "obs": pd.to_datetime(["2026-09-24 07:10", "2026-09-24 07:00", "2026-09-24 06:00", "2026-09-24 07:30"]),
        "event_msgs": ["d:43|q:70", "d:43", None, "d:80"],
        "stop_msgs": [None, None, "h:0|c:0", None]})
    ev, st = first_seen_messages(raw)
    ev = ev.set_index(["key", "type", "code"]).obs
    assert ev[("A|ar", "d", "43")] == T("2026-09-24 07:00")     # earliest of two sightings
    assert ev[("A|ar", "q", "70")] == T("2026-09-24 07:10")
    assert ev[("B|dp", "d", "80")] == T("2026-09-24 07:30")
    assert set(zip(st.stop_id, st.type)) == {("A", "h"), ("A", "c")}


def test_first_seen_messages_empty_input():
    ev, st = first_seen_messages(pd.DataFrame(columns=["stop_id", "event", "obs", "event_msgs", "stop_msgs"]))
    assert ev.empty and st.empty and list(ev.columns) == ["key", "type", "code", "obs"]


def test_messages_count_only_what_was_seen_before_the_cutoff():
    ev = pd.DataFrame({"key": ["A|ar", "A|ar", "A|ar", "B|dp"], "type": ["d", "d", "q", "d"],
                       "code": ["43", "80", "70", "43"],
                       "obs": pd.to_datetime(["2026-09-24 07:00", "2026-09-24 08:00", "2026-09-24 07:10",
                                              "2026-09-24 07:05"])})
    st = pd.DataFrame({"stop_id": ["A", "B"], "type": ["h", "c"],
                       "obs": pd.to_datetime(["2026-09-24 06:00", "2026-09-24 09:00"])})
    rows = pd.DataFrame({"stop_id_a": ["A", "A"], "stop_id_b": ["B", "B"],
                         "t_cut": [T("2026-09-24 07:30"), T("2026-09-24 08:30")]})
    out = messages(rows, ev, st)
    assert list(out.n_delay_codes_a) == [1, 2]      # code 80 appears at 08:00
    assert list(out.n_quality_a) == [1, 1]
    assert list(out.n_delay_codes_b) == [1, 1]
    assert list(out.h_notice_a) == [1, 1] and list(out.c_notice_a) == [0, 0]
    assert list(out.h_notice_b) == [0, 0]            # B only has a "c" notice


def hub_example():
    plan = plan_frame([
        ("A", "ar", "2026-09-24 08:10", "RE1", "2026-09-24 04:00"),
        ("B", "ar", "2026-09-24 08:20", "RE1", "2026-09-24 04:00"),
        ("C", "dp", "2026-09-24 08:15", "S11", "2026-09-24 04:00"),
        ("D", "ar", "2026-09-24 08:05", "RE5", "2026-09-24 08:01"),   # timetable published at 08:01
        ("E", "ar", "2026-09-24 07:30", "RE1", "2026-09-24 04:00"),   # already arrived
    ])
    obs = obs_frame([
        ("A|ar", "2026-09-24 07:50", "2026-09-24 08:16", None),      # +6
        ("B|ar", "2026-09-24 07:55", None, "c"),                      # cancelled
        ("C|dp", "2026-09-24 08:03", "2026-09-24 08:35", None),      # +20, but seen after 08:00
        ("E|ar", "2026-09-24 07:35", "2026-09-24 07:34", None),      # +4
    ])
    return plan, obs


def test_hub_state_uses_only_what_was_known_at_the_grid_time():
    plan, obs = hub_example()
    h, line = hub_state(plan, obs, T("2026-09-24 08:00"), T("2026-09-24 08:00"), CFG)
    h = h.set_index(["eva", "g"]).loc[(KOELN, T("2026-09-24 08:00"))]
    # within +-30 min of 08:00: E (+4), A (+6), B (cancelled), C (0: its +20 is not known yet);
    # D is excluded because its timetable was not known at 08:00
    assert h.hub_share_cancel == pytest.approx(1 / 4)
    assert h.hub_mean_delay == pytest.approx((4 + 6 + 0) / 3)
    assert h.hub_share_late5 == pytest.approx(1 / 3)
    line = line.set_index(["eva", "g", "line"]).line_recent_delay_a
    assert line[(KOELN, T("2026-09-24 08:00"), "RE1")] == 4       # E arrived within the last hour
    assert (KOELN, T("2026-09-24 08:00"), "S11") not in line.index


def test_hub_joins_on_the_last_grid_point_before_the_cutoff():
    plan, obs = hub_example()
    rows = pd.DataFrame({"eva": [KOELN, KOELN], "line_a": ["RE1", "RE5"],
                         "t_cut": [T("2026-09-24 08:04"), T("2026-09-24 08:04")]})
    out = hub(rows, plan, obs, CFG)
    assert out.hub_mean_delay[0] == pytest.approx(10 / 3)         # grid 08:00, not 08:05
    assert out.line_recent_delay_a[0] == 4 and np.isnan(out.line_recent_delay_a[1])


def feeder_example():
    """Hub arrivals A (via Neuss and Dortmund) and B (via Dortmund), departure B at the hub,
    and traffic at the two feeders."""
    hub_plan = plan_frame([
        ("A", "ar", "2026-09-24 08:40", "RE1", "2026-09-24 04:00"),
        ("Bh", "ar", "2026-09-24 08:48", "RE5", "2026-09-24 04:00"),
        ("Bh", "dp", "2026-09-24 08:50", "RE5", "2026-09-24 04:00"),
        ("C", "ar", "2026-09-24 08:45", "S11", "2026-09-24 04:00"),
    ])
    hub_plan["path"] = ["Dortmund Hbf|Neuss Hbf|Köln-Ehrenfeld", "Dortmund Hbf|Essen Hbf", None, "Köln-Mülheim"]
    fed = plan_frame([
        ("n1", "dp", "2026-09-24 08:10", "RE1", "2026-09-24 04:00"),    # Neuss: +4
        ("n2", "dp", "2026-09-24 08:20", "S8", "2026-09-24 04:00"),     # Neuss: cancelled
        ("n3", "ar", "2026-09-24 07:40", "RE1", "2026-09-24 04:00"),    # Neuss: RE1 arrived +2
        ("n4", "dp", "2026-09-24 08:15", "RE6", "2026-09-24 08:03"),    # Neuss: timetable known only at 08:03
        ("d1", "dp", "2026-09-24 08:05", "ICE", "2026-09-24 04:00"),    # Dortmund: +10, known only at 08:02
        ("d2", "ar", "2026-09-24 07:50", "RE1", "2026-09-24 04:00"),    # Dortmund: RE1 arrived +6
    ])
    fed["eva"] = [NEUSS] * 4 + [DORTMUND] * 2
    fed["path"] = None
    obs = obs_frame([
        ("n1|dp", "2026-09-24 07:55", "2026-09-24 08:14", None),
        ("n2|dp", "2026-09-24 07:56", None, "c"),
        ("n3|ar", "2026-09-24 07:45", "2026-09-24 07:42", None),
        ("d1|dp", "2026-09-24 08:02", "2026-09-24 08:15", None),
        ("d2|ar", "2026-09-24 07:57", "2026-09-24 07:56", None),
    ])
    return pd.concat([hub_plan, fed], ignore_index=True), obs


def test_feeder_corridor_state_on_the_path_before_the_hub():
    plan, obs = feeder_example()
    rows = pd.DataFrame({"eva": [KOELN] * 3, "stop_id_a": ["A", "A", "C"], "stop_id_b": ["Bh", "Bh", "Bh"],
                         "line_a": ["RE1", "RE1", "S11"],
                         "t_cut": [T("2026-09-24 08:04"), T("2026-09-24 08:10"), T("2026-09-24 08:04")]})
    out = feeder(rows, plan, obs, CFG)
    # grid 08:00: Neuss has n1 (+4), n3 (+2), n2 cancelled (excluded), n4 not yet published -> mean 3;
    # Dortmund has d1 at 0 (its +10 is seen at 08:02) and d2 (+6) -> mean 3. A passes both: 3.
    assert out.corridor_delay_a[0] == pytest.approx(3.0)
    # A's line RE1 arrivals in the last 60 min: Neuss n3 +2, Dortmund d2 +6 -> mean over feeders 4
    assert out.corridor_line_delay_a[0] == pytest.approx(4.0)
    # B comes in via Dortmund only
    assert out.corridor_delay_b[0] == pytest.approx(3.0)
    # grid 08:10: d1's +10 is known now (Dortmund 8), n4 counts at Neuss with 0 -> Neuss (4+2+0)/3 = 2
    assert out.corridor_delay_a[1] == pytest.approx((2.0 + 8.0) / 2)
    assert out.corridor_delay_b[1] == pytest.approx(8.0)
    # C passes no feeder
    assert np.isnan(out.corridor_delay_a[2]) and np.isnan(out.corridor_line_delay_a[2])


def test_feeder_group_is_missing_without_feeder_stations():
    plan, obs = feeder_example()
    rows = pd.DataFrame({"eva": [KOELN], "stop_id_a": ["A"], "stop_id_b": ["Bh"], "line_a": ["RE1"],
                         "t_cut": [T("2026-09-24 08:04")]})
    assert feeder(rows, plan, obs, FeatureConfig()).isna().all().all()          # model cards before version 2
    assert feeder(rows, plan[plan.eva == KOELN], obs, CFG).isna().all().all()   # feeder data not loaded


def test_feeder_stations_are_read_only_for_models_that_use_them():
    assert CFG.stations([KOELN]) == [KOELN]
    used = FeatureConfig(groups=("db", "feeder"), feeders=CFG.feeders)
    assert used.stations([KOELN]) == [KOELN, DORTMUND, NEUSS]
    assert FeatureConfig.from_dict(used.as_dict()) == used
    old = {k: v for k, v in FeatureConfig().as_dict().items() if k != "feeders"}   # a version-1 model card
    assert FeatureConfig.from_dict(old).feeders == ()


def test_context_day_type_and_local_hour():
    cfg = FeatureConfig(holidays=frozenset({date(2026, 10, 3)}))
    rows = pd.DataFrame({
        # 22:30 UTC Friday 2 Oct = 00:30 local Saturday 3 Oct (a holiday)
        "pt_a": [T("2026-09-24 08:12"), T("2026-09-26 08:12"), T("2026-10-02 22:30"), T("2026-10-25 12:00")],
        "planned_slack_min": [5.0] * 4, "n_stations_before_a": [3] * 4,
        "pp_a": ["5", "5", None, "1"], "pp_b": ["5", "6", None, "1"],
        "segment_a": ["regional"] * 4, "segment_b": ["regional"] * 4, "hub": ["Köln Hbf"] * 4})
    out = context(rows, cfg)
    assert list(out.day_type) == ["weekday", "saturday", "sunday_holiday", "sunday_holiday"]
    assert list(out.same_platform) == [1, 0, 0, 1]
    local_hours = np.arctan2(out.hour_sin, out.hour_cos) % (2 * np.pi) * 24 / (2 * np.pi)
    # 10:12 CEST, 10:12 CEST, 00:30 CEST, 13:00 CET (after the end of DST on 25 Oct)
    assert np.allclose(local_hours, [10.2, 10.2, 0.5, 13.0])


# ---------------------------------------------------------------- leakage property

def random_world(seed: int, n_trains: int = 80):
    """A random hub day: timetable, realtime observations, messages and query rows."""
    rng = np.random.default_rng(seed)
    base = T("2026-09-24 05:00")
    minute = lambda x: pd.to_timedelta(np.asarray(x, dtype=float), unit="m")   # noqa: E731
    evas = rng.choice([KOELN, "8000085"], n_trains)
    pt_ar = base + minute(rng.integers(0, 600, n_trains))
    plan = pd.concat([
        pd.DataFrame({"stop_id": [f"t{i}" for i in range(n_trains)], "event": "ar", "pt": pt_ar}),
        pd.DataFrame({"stop_id": [f"t{i}" for i in range(n_trains)], "event": "dp",
                      "pt": pt_ar + minute(rng.integers(2, 6, n_trains))})], ignore_index=True)
    plan["eva"] = np.tile(evas, 2)
    plan["line"] = np.tile(rng.choice(["RE1", "RE5", "S11", "ICE"], n_trains), 2)
    plan["first_seen"] = plan.pt - minute(rng.integers(5, 300, len(plan)))
    # planned paths into the hub; some pass the feeder stations
    paths = rng.choice(["Neuss Hbf|Köln-Ehrenfeld", "Dortmund Hbf|Neuss Hbf", "Dortmund Hbf|Essen Hbf",
                        "Aachen Hbf|Düren", None], n_trains)
    plan["path"] = np.concatenate([paths, [None] * n_trains])
    # traffic at the two feeder stations
    n_f = 160
    fpt = base + minute(rng.integers(-60, 600, n_f))
    fplan = pd.DataFrame({"stop_id": [f"f{i}" for i in range(n_f)], "event": rng.choice(["ar", "dp"], n_f),
                          "pt": fpt, "eva": rng.choice([NEUSS, DORTMUND], n_f),
                          "line": rng.choice(["RE1", "RE5", "S11", "ICE"], n_f), "path": None})
    fplan["first_seen"] = fplan.pt - minute(rng.integers(5, 300, n_f))
    plan = pd.concat([plan, fplan], ignore_index=True)
    plan["pt"], plan["first_seen"] = plan.pt.astype("datetime64[ns]"), plan.first_seen.astype("datetime64[ns]")

    n_obs = 800
    ev = plan.sample(n_obs, replace=True, random_state=seed).reset_index(drop=True)
    obs = pd.DataFrame({"key": ev.stop_id + "|" + ev.event, "eva": ev.eva,
                        "obs": ev.pt - minute(rng.integers(-40, 200, n_obs)),
                        "ct": ev.pt + minute(rng.integers(0, 25, n_obs)), "ct_raw": None,
                        "cs": np.where(rng.random(n_obs) < 0.05, "c", None)})
    obs["obs"], obs["ct"] = obs.obs.astype("datetime64[ns]"), obs.ct.astype("datetime64[ns]")
    obs = obs.sort_values("obs", kind="stable").reset_index(drop=True)

    n_msg = 150
    em = plan.sample(n_msg, replace=True, random_state=seed + 1).reset_index(drop=True)
    event_msgs = pd.DataFrame({"key": em.stop_id + "|" + em.event,
                               "type": rng.choice(["d", "q"], n_msg), "code": rng.choice(["43", "80", "70"], n_msg),
                               "obs": (em.pt - minute(rng.integers(-30, 200, n_msg))).astype("datetime64[ns]")})
    event_msgs = event_msgs.groupby(["key", "type", "code"], as_index=False).obs.min()
    stop_msgs = pd.DataFrame({"stop_id": em.stop_id, "type": rng.choice(["h", "c"], n_msg),
                              "obs": (em.pt - minute(rng.integers(-30, 300, n_msg))).astype("datetime64[ns]")})
    stop_msgs = stop_msgs.groupby(["stop_id", "type"], as_index=False).obs.min()

    # query rows: arrival of train i, departure of train j at the same hub, cutoffs 60/30/10
    hubs = plan[plan.eva.isin([KOELN, "8000085"])]
    a = hubs[hubs.event == "ar"].reset_index(drop=True)
    d = hubs[hubs.event == "dp"].reset_index(drop=True)
    pairs = [(i, j) for i in range(len(a)) for j in range(len(d))
             if a.eva[i] == d.eva[j] and 4 <= (d.pt[j] - a.pt[i]).total_seconds() / 60 <= 30]
    ai, dj = map(list, zip(*pairs))
    base_rows = pd.DataFrame({
        "stop_id_a": a.stop_id[ai].values, "stop_id_b": d.stop_id[dj].values, "eva": a.eva[ai].values,
        "pt_a": a.pt[ai].values, "pt_b": d.pt[dj].values, "line_a": a.line[ai].values,
        "hub": a.eva[ai].values, "segment_a": "regional", "segment_b": "regional",
        "pp_a": "1", "pp_b": rng.choice(["1", "2"], len(ai)), "n_stations_before_a": 3})
    base_rows["planned_slack_min"] = (base_rows.pt_b - base_rows.pt_a).dt.total_seconds() / 60
    rows = pd.concat([base_rows.assign(cutoff_min=L, t_cut=base_rows.pt_a - pd.Timedelta(minutes=L))
                      for L in (60, 30, 10)], ignore_index=True)
    return plan, obs, event_msgs, stop_msgs, rows


def dataset_columns(rows, obs):
    """The at-cutoff columns the dataset builder provides, computed the same way."""
    rows = rows.copy()
    sa = state_at(rows.stop_id_a + "|ar", rows.t_cut, obs)
    sb = state_at(rows.stop_id_b + "|dp", rows.t_cut, obs)
    a_pred, b_pred = sa.ct.fillna(rows.pt_a), sb.ct.fillna(rows.pt_b)
    rows["a_obs_cut"], rows["b_obs_cut"] = sa.obs, sb.obs
    rows["db_delay_a_min"] = (a_pred - rows.pt_a).dt.total_seconds() / 60
    rows["db_delay_b_min"] = (b_pred - rows.pt_b).dt.total_seconds() / 60
    rows["db_slack_min"] = (b_pred - a_pred).dt.total_seconds() / 60
    rows["b_cancel_known"] = sb.cs.eq("c")
    return rows


def perturb_after(t, plan, obs, event_msgs, stop_msgs, seed):
    """Replace everything collected after t with different random data."""
    rng = np.random.default_rng(seed)
    late = obs.obs > t
    obs = obs.copy()
    obs.loc[late, "ct"] = obs.loc[late, "ct"] + pd.to_timedelta(rng.integers(-5, 40, late.sum()), unit="m")
    obs.loc[late, "cs"] = np.where(rng.random(late.sum()) < 0.3, "c", None)
    extra = obs[late].assign(obs=lambda x: x.obs + pd.Timedelta(minutes=1), cs="c")
    obs = pd.concat([obs, extra], ignore_index=True).sort_values("obs", kind="stable").reset_index(drop=True)
    new_plan = plan.sample(20, random_state=seed).assign(
        stop_id=lambda x: "new" + x.stop_id, first_seen=t + pd.Timedelta(minutes=1))
    new_msgs = event_msgs.sample(30, random_state=seed).assign(code="99", obs=t + pd.Timedelta(seconds=30))
    new_stop = stop_msgs.sample(30, random_state=seed).assign(type="h", obs=t + pd.Timedelta(minutes=2))
    event_msgs = pd.concat([event_msgs.assign(obs=event_msgs.obs.where(event_msgs.obs <= t, t + pd.Timedelta(hours=9))),
                            new_msgs], ignore_index=True)
    stop_msgs = pd.concat([stop_msgs, new_stop], ignore_index=True)
    return pd.concat([plan, new_plan], ignore_index=True), obs, event_msgs, stop_msgs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_feature_changes_when_the_future_changes(seed):
    """Leakage property: rewriting everything collected after T must not change any
    feature of any row whose cutoff is at or before T."""
    plan, obs, event_msgs, stop_msgs, rows = random_world(seed)
    t = T("2026-09-24 10:02")
    before = compute_all(dataset_columns(rows, obs), plan, obs, event_msgs, stop_msgs, CFG)

    plan2, obs2, ev2, st2 = perturb_after(t, plan, obs, event_msgs, stop_msgs, seed)
    after = compute_all(dataset_columns(rows, obs2), plan2, obs2, ev2, st2, CFG)

    known = rows.t_cut <= t
    assert known.sum() > 50 and (~known).sum() > 50
    pd.testing.assert_frame_equal(before[known], after[known])
    # the test has teeth: rows after T do change
    changed = (before[~known].fillna(-999) != after[~known].fillna(-999)).any(axis=1)
    assert changed.mean() > 0.3
    assert set(before.columns) == set(ALL_FEATURES)
    # the feeder group is exercised too: populated, and changed by the future after T
    for c in GROUPS["feeder"]:
        assert before[c].notna().mean() > 0.15
    f = GROUPS["feeder"]
    assert (before.loc[~known, f].fillna(-999) != after.loc[~known, f].fillna(-999)).any(axis=1).mean() > 0.05


def truncate_at(t, plan, obs, event_msgs, stop_msgs):
    """The data as the collector had it at time t."""
    return (plan[plan.first_seen <= t], obs[obs.obs <= t],
            event_msgs[event_msgs.obs <= t], stop_msgs[stop_msgs.obs <= t])


def test_features_equal_when_data_ends_exactly_at_the_cutoff():
    """Stricter than the test above: each row's features must be reproducible from the
    data as it stood at that row's own cutoff, so even a one-second look-ahead fails.
    This is also the training/serving check: the API will compute features from exactly
    that truncated view."""
    plan, obs, event_msgs, stop_msgs, rows = random_world(4)
    obs = obs.assign(obs=obs.obs + pd.to_timedelta(np.random.default_rng(4).integers(0, 60, len(obs)), unit="s"))
    obs = obs.sort_values("obs", kind="stable").reset_index(drop=True)
    full = compute_all(dataset_columns(rows, obs), plan, obs, event_msgs, stop_msgs, CFG)
    assert full[GROUPS["feeder"]].notna().mean().min() > 0.15     # the feeder group is exercised
    groups = list(rows.groupby("t_cut").groups.items())
    picked = np.random.default_rng(0).choice(len(groups), 40, replace=False)
    for t, idx in (groups[i] for i in picked):
        p, o, e, s = truncate_at(t, plan, obs, event_msgs, stop_msgs)
        live = compute_all(dataset_columns(rows.loc[idx], o), p, o, e, s, CFG)
        pd.testing.assert_frame_equal(full.loc[idx], live, obj=f"features at cutoff {t}")


def test_compute_all_on_no_rows():
    plan, obs, event_msgs, stop_msgs, rows = random_world(1)
    out = compute_all(dataset_columns(rows.iloc[:0], obs), plan, obs, event_msgs, stop_msgs, CFG)
    assert out.empty and list(out.columns) == ALL_FEATURES
