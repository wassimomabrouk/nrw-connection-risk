"""Candidate rules T1 to T7 (DESIGN.md section 1), one small case each."""
import pandas as pd
import pytest

from nrw_connection_risk.dataset.candidates import build_candidates, segment

HUBS = {"8000207": "Köln Hbf"}
T0, T1 = pd.Timestamp("2026-09-24 02:00"), pd.Timestamp("2026-09-25 02:00")   # naive UTC


def ev(stop_id, event, hhmm, path=None, cat="RE", wings=None, tra=None, eva="8000207", seen="2026-09-24 03:00"):
    pt = pd.Timestamp(f"2026-09-24 {hhmm}")
    return {"stop_id": stop_id, "event": event, "eva": eva, "trip": stop_id.rsplit("-", 1)[0],
            "pt": pt, "pt_raw": (pt + pd.Timedelta(hours=2)).strftime("%y%m%d%H%M"), "pp": "1",
            "line": cat, "path": path, "wings": wings, "tra": tra, "cat": cat, "num": "1",
            "first_seen": pd.Timestamp(seen)}


# A arrives 08:00 UTC from Aachen via Köln-Ehrenfeld and continues to Düsseldorf
A_AR = ev("1-2609240900-5", "ar", "08:00", "Aachen Hbf|Köln-Ehrenfeld")
A_DP = ev("1-2609240900-5", "dp", "08:03", "Köln Messe/Deutz|Düsseldorf Hbf")


def run(*extra, **kw):
    plan = pd.DataFrame([A_AR, A_DP, *extra])
    cands, funnel = build_candidates(plan, T0, T1, HUBS, **kw)
    return set(cands.stop_id_b), funnel


def test_keeps_a_plain_transfer():
    kept, _ = run(ev("2-2609240900-1", "dp", "08:10", "Siegburg/Bonn|Frankfurt(Main)Hbf", cat="ICE"))
    assert kept == {"2-2609240900-1"}


@pytest.mark.parametrize("minute,kept", [("08:03", False), ("08:04", True), ("08:30", True), ("08:31", False)])
def test_slack_window_is_4_to_30_minutes(minute, kept):
    got, _ = run(ev("2-2609240900-1", "dp", minute, "Bonn Hbf"))
    assert (got == {"2-2609240900-1"}) is kept


def test_t1_bus_excluded():
    got, funnel = run(ev("2-2609240900-1", "dp", "08:10", "Bonn Hbf", cat="Bus"))
    assert got == set() and funnel[1]["step"] == "T1 bus" and funnel[1]["total"] == 0


def test_t2_same_trip_excluded():
    # the arrival's own departure moved to 08:05 so it lies inside the window
    plan = pd.DataFrame([A_AR, {**A_DP, "pt": pd.Timestamp("2026-09-24 08:05")}])
    cands, funnel = build_candidates(plan, T0, T1, HUBS)
    assert cands.empty and funnel[2]["step"] == "T2 same trip" and funnel[1]["total"] == 1 and funnel[2]["total"] == 0


def test_t3_transition_excluded():
    a = {**A_AR, "tra": "7-2609240905-1"}
    plan = pd.DataFrame([a, A_DP, ev("7-2609240905-1", "dp", "08:10", "Bonn Hbf")])
    cands, _ = build_candidates(plan, T0, T1, HUBS)
    assert cands.empty


def test_t4_wings_excluded():
    a = {**A_AR, "wings": "8-2609240900"}
    plan = pd.DataFrame([a, A_DP, ev("8-2609240900-4", "dp", "08:10", "Siegen Hbf")])
    cands, _ = build_candidates(plan, T0, T1, HUBS)
    assert cands.empty


def test_t5_backtrack_excluded():
    got, _ = run(ev("2-2609240900-1", "dp", "08:10", "Köln-Ehrenfeld|Düren|Aachen Hbf"))
    assert got == set()


def test_t6_redundant_excluded():
    got, _ = run(ev("2-2609240900-1", "dp", "08:10", "Köln Messe/Deutz|Düsseldorf Hbf"))
    assert got == set()


def test_t7_first_reach_keeps_only_the_earliest_train_to_new_stations():
    early = ev("2-2609240900-1", "dp", "08:10", "Siegburg/Bonn|Frankfurt(Main)Hbf", cat="ICE")
    later_same = ev("3-2609240900-1", "dp", "08:20", "Siegburg/Bonn|Frankfurt(Main)Hbf", cat="ICE")
    later_new = ev("4-2609240900-1", "dp", "08:25", "Siegburg/Bonn|Siegen Hbf")
    got, _ = run(early, later_same, later_new)
    assert got == {"2-2609240900-1", "4-2609240900-1"}


def test_t7_judges_departures_at_the_same_minute_together():
    # both leave at 08:10 and both are the first to reach Bonn: both are kept, in any row order
    narrow = ev("8-2609240900-1", "dp", "08:10", "Bonn Hbf")
    wide = ev("6-2609240900-9", "dp", "08:10", "Siegburg/Bonn|Bonn Hbf|Koblenz Hbf")
    later = ev("5-2609240900-1", "dp", "08:20", "Bonn Hbf|Koblenz Hbf")        # nothing new any more
    for order in ([A_AR, A_DP, narrow, wide, later], [later, wide, narrow, A_DP, A_AR]):
        cands, _ = build_candidates(pd.DataFrame(order), T0, T1, HUBS)
        assert set(cands.stop_id_b) == {"8-2609240900-1", "6-2609240900-9"}


def test_only_arrivals_inside_the_service_day():
    late = {**A_AR, "pt": pd.Timestamp("2026-09-25 02:00")}   # exactly t1: excluded
    plan = pd.DataFrame([late, ev("2-2609240900-1", "dp", "08:10", "Bonn Hbf")])
    plan.loc[1, "pt"] = pd.Timestamp("2026-09-25 02:10")
    cands, _ = build_candidates(plan, T0, T1, HUBS)
    assert cands.empty


def test_output_columns_and_segments():
    cands, _ = build_candidates(pd.DataFrame([A_AR, A_DP, ev("2-2609240900-1", "dp", "08:13",
                                              "Siegburg/Bonn|Frankfurt(Main)Hbf", cat="ICE")]), T0, T1, HUBS)
    r = cands.iloc[0]
    assert r.planned_slack_min == 13 and r.hub == "Köln Hbf"
    assert (r.segment_a, r.segment_b) == ("regional", "long-distance")
    assert (r.prev_station_a, r.next_station_b, r.destination_b) == ("Köln-Ehrenfeld", "Siegburg/Bonn", "Frankfurt(Main)Hbf")
    assert segment("S") == "S-Bahn" and segment(None) == "unknown"
