import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.api import create_app
from cryptopred.serve.store import PredictionStore
from tests.conftest import make_ohlcv


@pytest.fixture
def client(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]

    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", make_ohlcv(n=1200, seed=91))

    store = PredictionStore(tmp_path / "predictions.db")
    store.record_prediction(
        symbol="BTCUSDT",
        interval="1h",
        bar_close_time=pd.Timestamp("2024-02-01", tz="UTC"),
        proba=(0.15, 0.20, 0.65),
        signal=1,
        close_price=42000.0,
        model_version="test-v1",
    )
    return TestClient(create_app(cfg))


def test_health_reports_bar_counts(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["data"][0]["bars"] == 1200


def test_config_endpoint_exposes_the_rule_actually_applied(client):
    """Coverage, not a probability threshold: the threshold was a calibration
    artefact and is no longer consulted anywhere in the trading path."""
    body = client.get("/api/config").json()
    assert body["signal_coverage"] == pytest.approx(0.08)
    assert "signal_threshold" not in body
    assert body["horizon_bars"]["1h"] == 24


def test_predict_serves_the_logged_prediction(client):
    body = client.get("/api/predict?symbol=BTCUSDT&interval=1h").json()
    assert body["source"] == "log"
    assert body["prob_up"] == pytest.approx(0.65)


def test_predict_404s_without_a_model(client):
    res = client.get("/api/predict?symbol=ETHUSDT&interval=1h")
    assert res.status_code == 404
    assert "no model" in res.json()["detail"]


def test_history_returns_rows(client):
    body = client.get("/api/history?symbol=BTCUSDT&interval=1h").json()
    assert len(body["rows"]) == 1


def test_metrics_always_carries_the_caveat(client):
    body = client.get("/api/metrics?symbol=BTCUSDT&interval=1h").json()
    # Pin the shape, not the figure. Re-running the validation legitimately
    # changes the pass count, and a test that breaks on an honest re-run
    # pressures the next person to leave the stale number in place.
    assert re.search(r"\d+/20", body["caveat"])
    assert "findings.md" in body["caveat"]
    assert body["live"]["n_scored"] == 0
    assert body["live"]["accuracy"] is None


def test_candles_endpoint(client):
    body = client.get("/api/candles?symbol=BTCUSDT&interval=1h&limit=10").json()
    assert len(body["rows"]) == 10
    assert set(body["rows"][0]) == {"time", "open", "high", "low", "close"}


def test_candles_404_for_unknown_symbol(client):
    assert client.get("/api/candles?symbol=NOPE&interval=1h").status_code == 404


def test_paper_positions_shape(client):
    body = client.get("/api/paper/positions?symbol=BTCUSDT").json()
    assert body["summary"]["n_trades"] == 0
    assert body["open"] == []


def test_dashboard_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Bảng Tín Hiệu Crypto" in res.text


def test_health_measures_age_from_the_bar_close_not_its_open(client):
    """An hourly bar that has just closed is fresh. Measuring from the open made
    every feed look an hour behind and lit a permanent warning."""
    body = client.get("/api/health").json()
    row = body["data"][0]
    open_time = pd.Timestamp(row["last_bar"])
    close_time = pd.Timestamp(row["last_close"])

    assert close_time > open_time
    expected = (pd.Timestamp.now(tz="UTC") - close_time).total_seconds() / 60
    assert row["minutes_behind"] == pytest.approx(expected, abs=1.0)


def test_config_reports_the_execution_style_in_use(client):
    body = client.get("/api/config").json()
    assert body["execution_style"] in {"taker", "maker"}
    assert body["maker_fee"] < body["taker_fee"]


def test_paper_positions_separates_resting_orders_from_positions(client):
    body = client.get("/api/paper/positions?symbol=BTCUSDT").json()
    assert "pending" in body
    assert "open" in body
    assert body["execution"]["style"] in {"taker", "maker"}


def test_health_says_the_scheduler_never_ran_when_there_is_no_heartbeat(client):
    """Data age cannot stand in for this: the API never syncs bars itself."""
    body = client.get("/api/health").json()
    assert body["scheduler"]["state"] == "never_started"


def test_health_reports_a_live_scheduler_from_its_heartbeat(tmp_path):
    from cryptopred.serve import heartbeat

    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    heartbeat.write(tmp_path / "heartbeat.json", "1h", {"predictions": 1})

    sched = TestClient(create_app(cfg)).get("/api/health").json()["scheduler"]
    assert sched["state"] == "alive"
    assert sched["minutes_ago"] < 5
    assert sched["last_cycle"]


def test_alerts_are_served_verbatim_newest_first(tmp_path, monkeypatch):
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    cfg = Config()
    cfg.data.root = tmp_path
    alerts.notify("BTCUSDT — model bắn LONG\nHồ sơ: đúng 7/15", tmp_path / "signals.log")
    alerts.notify("BTCUSDT — model bắn SHORT\nHồ sơ: đúng 7/16", tmp_path / "signals.log")

    body = TestClient(create_app(cfg)).get("/api/alerts").json()
    assert [a["message"].splitlines()[0] for a in body["alerts"]] == [
        "BTCUSDT — model bắn SHORT",
        "BTCUSDT — model bắn LONG",
    ]
    assert "Hồ sơ: đúng 7/16" in body["alerts"][0]["message"]


def test_alerts_are_empty_before_anything_has_fired(client):
    assert client.get("/api/alerts").json() == {"alerts": []}
