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


def test_config_endpoint_exposes_threshold(client):
    body = client.get("/api/config").json()
    assert body["signal_threshold"] == pytest.approx(0.60)
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
    assert "28" in body["caveat"]
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
