"""Point-in-time joins: nothing collected after the cutoff may be visible."""
import pandas as pd

from nrw_connection_risk.dataset.state import final_state, last_poll_at, state_at
from nrw_connection_risk.dataset.timeutil import is_ambiguous_local, service_day_bounds

OBS = pd.DataFrame({
    "key": ["x|ar", "x|ar", "x|ar", "y|dp"],
    "obs": pd.to_datetime(["2026-09-24 07:00", "2026-09-24 07:50", "2026-09-24 08:30", "2026-09-24 07:10"]),
    "ct": pd.to_datetime(["2026-09-24 08:14", "2026-09-24 08:20", "2026-09-24 08:21", pd.NaT]),
    "ct_raw": ["2609241014", "2609241020", "2609241021", None],
    "cs": [None, None, None, "c"],
})


def test_observation_after_cutoff_is_invisible():
    keys = pd.Series(["x|ar"] * 4)
    t = pd.Series(pd.to_datetime(["2026-09-24 06:59", "2026-09-24 07:49", "2026-09-24 07:50", "2026-09-24 09:00"]))
    s = state_at(keys, t, OBS)
    assert s.ct.isna().iloc[0]                                           # nothing known yet
    assert s.ct.iloc[1] == pd.Timestamp("2026-09-24 08:14")              # 07:50 not yet visible
    assert s.ct.iloc[2] == pd.Timestamp("2026-09-24 08:20")              # visible at exactly 07:50
    assert s.ct.iloc[3] == pd.Timestamp("2026-09-24 08:21")


def test_state_keeps_input_order_and_index():
    keys = pd.Series(["y|dp", "x|ar"], index=[10, 3])
    t = pd.Series(pd.to_datetime(["2026-09-24 09:00", "2026-09-24 07:05"]), index=[10, 3])
    s = state_at(keys, t, OBS)
    assert list(s.index) == [10, 3] and s.cs.loc[10] == "c" and s.ct.loc[3] == pd.Timestamp("2026-09-24 08:14")


def test_final_state_is_the_last_observation():
    f = final_state(pd.Series(["x|ar", "y|dp", "z|ar"]), OBS)
    assert f.ct.iloc[0] == pd.Timestamp("2026-09-24 08:21") and f.cs.iloc[1] == "c" and f.obs.isna().iloc[2]


def test_last_poll_at():
    polls = pd.DataFrame({"eva": ["1", "1", "2"], "t": pd.to_datetime(["2026-09-24 07:00", "2026-09-24 07:30",
                                                                        "2026-09-24 07:20"])})
    got = last_poll_at(pd.Series(["1", "2", "1"]), pd.Series(pd.to_datetime(
        ["2026-09-24 07:29", "2026-09-24 07:25", "2026-09-24 06:00"])), polls)
    assert got.iloc[0] == pd.Timestamp("2026-09-24 07:00") and got.iloc[1] == pd.Timestamp("2026-09-24 07:20")
    assert pd.isna(got.iloc[2])


def test_service_day_bounds_across_dst():
    t0, t1 = service_day_bounds(pd.Timestamp("2026-09-24").date())
    assert (t0, t1) == (pd.Timestamp("2026-09-24 02:00"), pd.Timestamp("2026-09-25 02:00"))
    t0, t1 = service_day_bounds(pd.Timestamp("2026-10-24").date())     # clocks go back on 25 Oct
    assert t1 - t0 == pd.Timedelta(hours=25)
    t0, t1 = service_day_bounds(pd.Timestamp("2026-03-28").date())     # clocks go forward on 29 Mar
    assert t1 - t0 == pd.Timedelta(hours=23)


def test_ambiguous_local_times():
    assert is_ambiguous_local("2610250230")        # repeated autumn hour
    assert is_ambiguous_local("2603290230")        # skipped spring hour
    assert not is_ambiguous_local("2610250330") and not is_ambiguous_local(None)
