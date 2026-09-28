"""End to end: XML -> production parser -> parsed layer -> training table for one day."""
import json
from datetime import date, datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import write_parsed
from nrw_connection_risk.dataset.build import IncompleteDay, build_day, main
from nrw_connection_risk.dataset.config import DatasetConfig
from nrw_connection_risk.dataset.load import ParsedLayerError

DAY = date(2026, 9, 24)
CFG = DatasetConfig(hubs={"8000207": "Köln Hbf"})
KOELN = "8000207"

# local times (UTC+2): A arrives 10:12 and continues to Düsseldorf,
# B1 ICE to Frankfurt 10:25 (slack 13), B2 RE to Siegen 10:38 (slack 26)
PLAN = """<timetable station="Köln Hbf">
<s id="1-2609240900-5"><tl c="RE" n="1"/><ar pt="2609241012" ppth="Aachen Hbf|Köln-Ehrenfeld"/>
  <dp pt="2609241015" ppth="Köln Messe/Deutz|Düsseldorf Hbf"/></s>
<s id="2-2609241025-1"><tl c="ICE" n="2"/><dp pt="2609241025" ppth="Siegburg/Bonn|Frankfurt(Main)Hbf"/></s>
<s id="3-2609241038-1"><tl c="RE" n="3"/><dp pt="2609241038" ppth="Köln Messe/Deutz|Siegen Hbf"/></s>
</timetable>"""


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def rchg(body):
    return f'<timetable station="Köln Hbf">{body}</timetable>'


def scenario(plan_at="2026-09-24 06:00", with_first_obs=True, with_tail=True):
    r = [("plan", KOELN, utc(plan_at), PLAN)]
    if with_first_obs:   # 09:00 local: A expected 2 minutes late
        r.append(("rchg", KOELN, utc("2026-09-24 07:00"), rchg('<s id="1-2609240900-5"><ar ct="2609241014"/></s>')))
    r += [
        # 09:50 local: A now 8 minutes late (visible only from the 10-minute cutoff at 10:02)
        ("rchg", KOELN, utc("2026-09-24 07:50"), rchg('<s id="1-2609240900-5"><ar ct="2609241020"/></s>')),
        # 10:30 local, after the events: A arrived 10:21, B1 left on time, B2 cancelled
        ("rchg", KOELN, utc("2026-09-24 08:30"), rchg(
            '<s id="1-2609240900-5"><ar ct="2609241021"/></s>'
            '<s id="2-2609241025-1"><dp ct="2609241025"/></s>'
            '<s id="3-2609241038-1"><dp cs="c" clt="2609241000"/></s>')),
    ]
    if with_tail:        # data must reach 6 h past the end of the service day
        r.append(("fchg", KOELN, utc("2026-09-25 08:30"), rchg('<s id="9-2609251000-1"><ar ct="2609251001"/></s>')))
    return r


@pytest.fixture
def table(tmp_path):
    df, meta = build_day(DAY, write_parsed(tmp_path, scenario()), CFG)
    return df.set_index(["cutoff_min", "stop_id_b"]), meta


def test_candidates_and_labels(table):
    df, meta = table
    assert meta["candidates"] == 2 and len(df) == 6 and df.eligible.all()
    b1, b2 = df.loc[(30, "2-2609241025-1")], df.loc[(30, "3-2609241038-1")]
    assert b1.planned_slack_min == 13 and b1.real_slack_min == 4 and not b1.label_fail   # 4 min is enough
    assert b2.label_fail and b2.fail_reason == "b_cancelled"
    assert b1.delay_a_final_min == 9


def test_point_in_time_state_has_no_leakage(table):
    df, _ = table
    b1 = "2-2609241025-1"
    assert df.loc[(60, b1)].db_delay_a_min == 2      # 07:00 observation visible at 07:12
    assert df.loc[(30, b1)].db_delay_a_min == 2      # 07:50 observation NOT visible at 07:42
    assert df.loc[(10, b1)].db_delay_a_min == 8      # visible at 08:02
    assert df.loc[(10, b1)].db_slack_min == 5
    assert not df.loc[(10, "3-2609241038-1")].b_cancel_known   # cancellation only known at 08:30
    assert (df.t_cut < df.a_obs_cut).sum() == 0 and (df.t_cut < df.b_obs_cut).sum() == 0


def test_collector_gap_excludes_cutoffs(tmp_path):
    df, _ = build_day(DAY, write_parsed(tmp_path, scenario(with_first_obs=False)), CFG)
    reasons = df.groupby("cutoff_min").exclusion_reason.first().to_dict()
    assert reasons[60] == "collector_gap" and reasons[30] == "collector_gap" and reasons[10] is None


def test_candidates_not_known_at_cutoff_are_excluded(tmp_path):
    df, _ = build_day(DAY, write_parsed(tmp_path, scenario(plan_at="2026-09-24 07:30")), CFG)
    reasons = df.groupby("cutoff_min").exclusion_reason.first().to_dict()
    assert reasons[60] == "not_known_at_cutoff" and reasons[30] is None


def test_incomplete_day_is_refused(tmp_path):
    with pytest.raises(IncompleteDay):
        build_day(DAY, write_parsed(tmp_path, scenario(with_tail=False)), CFG)


def test_old_parser_version_is_refused(tmp_path):
    parsed = write_parsed(tmp_path, scenario())
    for f in parsed.rglob("*.parquet"):              # simulate files written by parser v1
        pq.write_table(pq.read_table(f).drop(["parser_version", "fb", "wings", "tra"]), f)
    with pytest.raises(ParsedLayerError, match="rebuild"):
        build_day(DAY, parsed, CFG)


def test_cli_writes_day_atomically_and_replaces_it(tmp_path, capsys):
    parsed = write_parsed(tmp_path, scenario())
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "collector.toml").write_text('[[stations]]\neva = "8000207"\nname = "Köln Hbf"\n', encoding="utf-8")
    (cfg_dir / "dataset.toml").write_text(
        "builder_version = 1\nmin_parser_version = 2\n[candidates]\nmin_transfer_min = 4\nmax_slack_min = 30\n"
        "[service_day]\nstart_hour_local = 4\n[cutoffs]\nminutes_before_arrival = [60, 30, 10]\n"
        "[quality]\nmax_collector_gap_min = 45\nlabel_horizon_h = 6\n", encoding="utf-8")
    args = ["--parsed", str(parsed), "--out", str(tmp_path / "ds"), "--config", str(cfg_dir / "dataset.toml"),
            "--from", "2026-09-24", "--to", "2026-09-25"]
    assert main(args) == 0 and main(args) == 0
    out = capsys.readouterr().out
    assert "2026-09-25: skipped, not complete yet" in out and "built 1 day(s)" in out
    day_dir = tmp_path / "ds" / "v1" / "service_day=2026-09-24"
    assert sorted(p.name for p in day_dir.iterdir()) == ["_meta.json", "part-0.parquet"]
    t = pq.read_table(day_dir / "part-0.parquet")
    assert t.num_rows == 6 and str(t.schema.field("pt_a").type) == "timestamp[ns, tz=UTC]"
    meta = json.loads((day_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["candidates"] == 2 and meta["funnel"][-1]["total"] == 2
    assert not list((tmp_path / "ds" / "v1").glob(".tmp-*"))


def test_every_column_is_classified_once(table):
    from nrw_connection_risk.dataset import columns as C

    df, _ = table
    got = set(df.reset_index().columns)
    assert len(C.ALL) == len(set(C.ALL))                       # no column in two groups
    assert got == set(C.ALL), f"unclassified: {got - set(C.ALL)}, missing: {set(C.ALL) - got}"
    assert not set(C.FEATURE_SAFE) & set(C.LABEL)
