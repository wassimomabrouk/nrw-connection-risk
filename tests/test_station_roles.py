"""Station roles: feeders are polled less often and never become transfer hubs."""
import pytest

from nrw_connection_risk.collector.config import Station, load_settings
from nrw_connection_risk.collector.main import Collector
from nrw_connection_risk.dataset.config import load_config

BASE = """
[api]
base_url = "https://x"
max_calls_per_minute = 50
timeout_s = 30
max_retries = 3
[schedule]
rchg_interval_s = 60
rchg_interval_feeder_s = 120
fchg_interval_s = 1800
plan_interval_s = 3600
plan_hours_behind = 2
plan_hours_ahead = 3
[storage]
data_dir = "data"
flush_interval_s = 600
[health]
heartbeat_file = "data/hb.json"
ping_interval_s = 300
"""


def stations(n_hubs, n_feeders, role_override=None):
    out = ""
    for i in range(n_hubs + n_feeders):
        role = role_override or ("hub" if i < n_hubs else "feeder")
        out += f'[[stations]]\neva = "{8000000 + i}"\nname = "St {i}"\nrole = "{role}"\n'
    return out


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_CLIENT_ID", "id")
    monkeypatch.setenv("DB_API_KEY", "key")
    (tmp_path / "config").mkdir()
    return tmp_path / "config"


def write(cfg_dir, body):
    p = cfg_dir / "collector.toml"
    p.write_text(BASE + body, encoding="utf-8")
    return p


def test_roles_and_rate_budget(cfg_dir):
    s = load_settings(write(cfg_dir, stations(5, 12)))
    assert [st.role for st in s.stations].count("feeder") == 12
    # hubs 5/min + feeders 6/min + fchg 17/30 + plan 17*6/60
    assert s.calls_per_minute() == pytest.approx(5 + 6 + 17 / 30 + 17 * 6 / 60)
    assert s.calls_per_minute() < 0.8 * s.max_calls_per_minute


def test_role_defaults_to_hub(cfg_dir):
    s = load_settings(write(cfg_dir, '[[stations]]\neva = "8000207"\nname = "Köln Hbf"\n'))
    assert s.stations == (Station("8000207", "Köln Hbf", "hub"),)


def test_unknown_role_is_refused(cfg_dir):
    with pytest.raises(RuntimeError, match="role"):
        load_settings(write(cfg_dir, stations(1, 0, role_override="depot")))


def test_schedule_over_budget_is_refused(cfg_dir):
    with pytest.raises(RuntimeError, match="calls/min"):
        load_settings(write(cfg_dir, stations(40, 0)))


def test_feeders_polled_every_two_minutes(cfg_dir):
    s = load_settings(write(cfg_dir, stations(1, 1)))
    jobs = {j.name: j.interval_s for j in Collector(s, client=object()).scheduler.jobs}
    assert jobs["rchg:8000000"] == 60 and jobs["rchg:8000001"] == 120
    assert jobs["fchg:8000001"] == 1800 and jobs["plan:8000001"] == 3600


def test_dataset_uses_hubs_only(cfg_dir):
    write(cfg_dir, stations(2, 3))
    (cfg_dir / "dataset.toml").write_text(
        "builder_version = 1\nmin_parser_version = 2\n[candidates]\nmin_transfer_min = 4\nmax_slack_min = 30\n"
        "[service_day]\nstart_hour_local = 4\n[cutoffs]\nminutes_before_arrival = [60, 30, 10]\n"
        "[quality]\nmax_collector_gap_min = 45\nlabel_horizon_h = 6\n", encoding="utf-8")
    assert load_config(cfg_dir / "dataset.toml").hubs == {"8000000": "St 0", "8000001": "St 1"}
