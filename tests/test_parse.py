from datetime import datetime, timezone

from nrw_connection_risk.collector.parse import iris_time_to_utc, parse_timetable, trip_key

T = datetime(2026, 9, 23, 16, 0, tzinfo=timezone.utc)


def test_trip_key_strips_stop_index_including_negative_ids():
    assert trip_key("4124699725257303251-2609231709-15") == "4124699725257303251-2609231709"
    assert trip_key("-771222-2609231800-1") == "-771222-2609231800"


def test_summer_time_converts_to_utc():
    assert iris_time_to_utc("2609231812") == datetime(2026, 9, 23, 16, 12, tzinfo=timezone.utc)


def test_winter_time_converts_to_utc():
    assert iris_time_to_utc("2612011200") == datetime(2026, 12, 1, 11, 0, tzinfo=timezone.utc)


def test_repeated_autumn_hour_assumes_summer_time():
    # 2026-10-25 02:30 local exists twice; the first occurrence (UTC+2) is assumed
    assert iris_time_to_utc("2610250230") == datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc)


def test_invalid_time_returns_none():
    assert iris_time_to_utc("notatime") is None
    assert iris_time_to_utc(None) is None


def test_plan_rows(plan_xml):
    rows = parse_timetable(plan_xml, "plan", "8000207", T)
    assert len(rows) == 3                          # RE: ar + dp, ICE: dp only
    re_ar = rows[0]
    assert re_ar["station_name"] == "Köln Hbf"
    assert re_ar["event"] == "ar" and re_ar["tl_category"] == "RE" and re_ar["tl_number"] == "10123"
    assert re_ar["pt_raw"] == "2609231812" and re_ar["pp"] == "5" and re_ar["line"] == "1"
    assert re_ar["trip_key"] == "4124699725257303251-2609231709"


def test_change_rows_delays_messages_cancellations(fchg_xml):
    rows = parse_timetable(fchg_xml, "fchg", "8000207", T)
    by = {(r["stop_id"], r["event"]): r for r in rows}
    ar = by[("4124699725257303251-2609231709-15", "ar")]
    assert ar["ct_raw"] == "2609231819" and ar["event_msgs"] == "d:43"
    assert by[("4124699725257303251-2609231709-15", "dp")]["cp"] == "6"
    cancelled = by[("-771222-2609231800-1", "dp")]
    assert cancelled["cs"] == "c" and cancelled["clt"] is not None
    msg_only = by[("999-2609231800-3", "s")]
    assert msg_only["stop_msgs"] == "h:0"
