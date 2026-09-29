"""End to end: XML -> parsed layer -> dataset day -> feature table, including the
leakage property through the whole stack (DuckDB reads, plan aggregation, messages)."""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import write_parsed
from test_dataset_build import CFG, DAY, KOELN, rchg, scenario, utc
from nrw_connection_risk.dataset.build import build_day as build_dataset_day
from nrw_connection_risk.dataset.build import write_day
from nrw_connection_risk.features.build import build_day, main
from nrw_connection_risk.features.columns import ALL_FEATURES
from nrw_connection_risk.features.config import FeatureConfig

ROOT = Path(__file__).resolve().parents[1]
ROW = ["cutoff_min", "stop_id_a", "stop_id_b"]

# 09:20 local: DB attaches delay code 43 to A's arrival
MESSAGE = ("rchg", KOELN, utc("2026-09-24 07:20"),
           rchg('<s id="1-2609240900-5"><ar ct="2609241014"><m id="r1" t="d" c="43" ts="2609240920"/></ar></s>'))


def build(tmp_path, responses):
    parsed = write_parsed(tmp_path, responses)
    df, meta = build_dataset_day(DAY, parsed, CFG)
    day_dir = write_day(df, meta, tmp_path / "dataset" / "v1")
    return parsed, day_dir, build_day(DAY, day_dir, parsed, CFG.hubs, FeatureConfig())


def test_feature_table_for_one_day(tmp_path):
    _, _, (f, facts) = build(tmp_path, scenario() + [MESSAGE])
    f = f.set_index(ROW)
    assert len(f) == 6 and set(ALL_FEATURES) <= set(f.columns)
    assert {"label_fail", "fail_reason", "t_cut", "service_day"} <= set(f.columns)
    b1 = "2-2609241025-1"
    r10 = f.loc[(10, "1-2609240900-5", b1)]
    # grid 08:00 UTC; A (+8), A's departure and B1 (both 0) are within 30 minutes
    assert r10.hub_mean_delay == pytest.approx(8 / 3) and r10.hub_share_late5 == pytest.approx(1 / 3)
    assert r10.trend_a_15 == 6 and r10.db_delay_a_min == 8
    assert f.loc[(60, "1-2609240900-5", b1)].n_delay_codes_a == 0      # code seen at 07:20, cutoff 07:12
    assert f.loc[(30, "1-2609240900-5", b1)].n_delay_codes_a == 1
    assert facts["plan_changes"] == 0 and facts["read_to"] == "2026-09-24T08:02:00"


def test_future_data_does_not_change_features(tmp_path):
    """Everything collected after 07:45 UTC is rewritten: new delays, a new message, a
    new train in the timetable. Rows with cutoffs at 07:12 and 07:42 must not change."""
    t = pd.Timestamp("2026-09-24 07:45")
    future = [
        ("rchg", KOELN, utc("2026-09-24 07:46"), rchg(
            '<s id="1-2609240900-5"><ar ct="2609241040"><m id="r9" t="q" c="70" ts="2609240946"/></ar></s>'
            '<s id="2-2609241025-1"><dp ct="2609241031"/></s>')),
        ("plan", KOELN, utc("2026-09-24 07:47"), """<timetable station="Köln Hbf">
            <s id="7-2609240930-3"><tl c="RE" n="7"/><ar pt="2609241000" ppth="Horrem|Köln-Ehrenfeld"/></s>
            </timetable>"""),
    ]
    _, _, (base, _) = build(tmp_path / "a", scenario() + [MESSAGE])
    _, _, (pert, _) = build(tmp_path / "b", scenario() + [MESSAGE] + future)
    known = lambda df: df[df.t_cut <= t].set_index(ROW).sort_index()[ALL_FEATURES]   # noqa: E731
    assert len(known(base)) == 4
    pd.testing.assert_frame_equal(known(base), known(pert))
    assert not known(base).equals(pert[pert.t_cut > t].set_index(ROW).sort_index()[ALL_FEATURES])


def test_cli_writes_parquet_and_meta(tmp_path, capsys):
    parsed = write_parsed(tmp_path, scenario())
    df, meta = build_dataset_day(DAY, parsed, CFG)
    write_day(df, meta, tmp_path / "dataset" / "v1")
    rc = main(["--dataset", str(tmp_path / "dataset" / "v1"), "--parsed", str(parsed),
               "--out", str(tmp_path / "features"), "--config", str(ROOT / "config" / "features.toml"),
               "--dataset-config", str(ROOT / "config" / "dataset.toml"),
               "--from", "2026-09-23", "--to", "2026-09-24"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2026-09-23: no dataset day, skipped" in out and "built 1 day(s)" in out
    day = tmp_path / "features" / "v1" / "service_day=2026-09-24"
    table = pq.read_table(day / "part-0.parquet")
    assert table.num_rows == 6 and str(table.schema.field("t_cut").type) == "timestamp[ns, tz=UTC]"
    m = json.loads((day / "_meta.json").read_text(encoding="utf-8"))
    assert m["rows"] == 6 and m["features"] == ALL_FEATURES and m["config"]["feature_version"] == 1
    assert m["non_missing_pct"]["db_slack_min"] == 100.0
    assert not list((tmp_path / "features" / "v1").glob(".tmp-*"))
