import json

import pandas as pd

from cryptopred.serve import heartbeat


def test_never_started_when_no_file(tmp_path):
    s = heartbeat.status(tmp_path / "heartbeat.json")
    assert s["state"] == "never_started"
    assert "not completed a cycle" in s["detail"]


def test_alive_right_after_a_cycle(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat.write(path, "1h", {"predictions": 1})
    s = heartbeat.status(path)
    assert s["state"] == "alive"
    assert s["minutes_ago"] < 1


def test_stale_after_two_missed_cycles(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat.write(path, "1h", {})
    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=heartbeat.STALE_AFTER_MINUTES + 10)
    s = heartbeat.status(path, now=later)
    assert s["state"] == "stale"
    assert "stopped" in s["detail"]


def test_a_recent_gap_is_not_yet_stale(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat.write(path, "1h", {})
    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=70)
    assert heartbeat.status(path, now=later)["state"] == "alive"


def test_a_malformed_file_is_reported_not_crashed(tmp_path):
    path = tmp_path / "heartbeat.json"
    path.write_text("{not json", encoding="utf-8")
    assert heartbeat.status(path)["state"] == "unreadable"


def test_a_file_missing_its_timestamp_is_reported(tmp_path):
    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps({"pid": 1}), encoding="utf-8")
    assert heartbeat.status(path)["state"] == "unreadable"


def test_writing_records_the_cycle_counts(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat.write(path, "1h", {"predictions": 2, "scored": 3})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["counts"]["scored"] == 3
    assert payload["interval"] == "1h"
    assert payload["pid"] > 0


def test_a_failed_write_never_takes_down_the_scheduler(tmp_path):
    """A monitoring failure must not break the thing it monitors."""
    unwritable = tmp_path / "a-file"
    unwritable.write_text("x", encoding="utf-8")
    heartbeat.write(unwritable / "nested" / "heartbeat.json", "1h", {})


def test_status_report_leads_with_liveness(tmp_path):
    from cryptopred.config import Config
    from cryptopred.serve.status import collect, format_status

    cfg = Config()
    cfg.data.root = tmp_path
    text = format_status(collect(cfg))
    assert "Scheduler: NEVER STARTED" in text
    assert "run.bat" in text


def test_status_report_says_running_when_the_beat_is_fresh(tmp_path):
    from cryptopred.config import Config
    from cryptopred.serve.status import collect, format_status

    cfg = Config()
    cfg.data.root = tmp_path
    heartbeat.write(tmp_path / "heartbeat.json", "1h", {})
    text = format_status(collect(cfg))
    assert "Scheduler: RUNNING" in text


def test_status_report_flags_a_dead_scheduler(tmp_path):
    from cryptopred.config import Config
    from cryptopred.serve.status import collect, format_status

    cfg = Config()
    cfg.data.root = tmp_path
    stale = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=20)
    (tmp_path / "heartbeat.json").write_text(
        json.dumps({"last_cycle": stale.isoformat(), "pid": 1}), encoding="utf-8"
    )
    text = format_status(collect(cfg))
    assert "Scheduler: STOPPED" in text
    assert "Nothing new is being recorded" in text
