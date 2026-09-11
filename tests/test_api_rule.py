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

    ParquetStore(tmp_path / "raw").write("klines", "BTCUSDT", "1h", make_ohlcv(n=1200, seed=5))
    store = PredictionStore(tmp_path / "predictions.db")
    store.record_prediction(
        symbol="BTCUSDT", interval="1h",
        bar_close_time=pd.Timestamp("2024-02-01", tz="UTC"),
        proba=(0.15, 0.20, 0.65), signal=1, close_price=42000.0, model_version="v1",
    )
    return TestClient(create_app(cfg))


def test_config_exposes_coverage_not_a_probability_threshold(client):
    body = client.get("/api/config").json()
    assert body["signal_coverage"] == pytest.approx(0.08)
    assert "signal_threshold" not in body


def test_predict_reports_the_margin_that_decides_the_trade(client):
    body = client.get("/api/predict?symbol=BTCUSDT&interval=1h").json()
    # |0.65 - 0.15|, not the 0.65 maximum
    assert body["margin"] == pytest.approx(0.50)


def test_predict_reports_the_cutoff_as_none_without_a_model(client):
    """A dashboard must not invent a rule the system is not applying."""
    body = client.get("/api/predict?symbol=BTCUSDT&interval=1h").json()
    assert body["margin_cutoff"] is None


def test_the_caveat_describes_the_corrected_result(client):
    body = client.get("/api/metrics?symbol=BTCUSDT&interval=1h").json()
    caveat = body["caveat"]
    assert "7/20" in caveat
    assert "53.98%" in caveat
    assert "gấp đôi" in caveat          # the factor-of-two warning travels with it


def test_the_caveat_no_longer_repeats_the_withdrawn_claim(client):
    body = client.get("/api/metrics?symbol=BTCUSDT&interval=1h").json()
    assert "28 cấu hình" not in body["caveat"]


def test_api_serves_a_newly_saved_model_without_a_restart(tmp_path):
    """The dashboard must not keep quoting a model the runner has replaced.

    The runner rebuilds its predictor every cycle, so after a retrain the two
    disagree until the API process is restarted — and nothing on the page says
    which model produced the number being read.
    """
    from cryptopred.models.registry import ModelRegistry
    from cryptopred.models.train import TrainConfig, train_fold
    from tests.test_train import _learnable_dataset

    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]
    ParquetStore(tmp_path / "raw").write(
        "klines", "BTCUSDT", "1h", make_ohlcv(n=1200, seed=5)
    )
    PredictionStore(tmp_path / "predictions.db").record_prediction(
        symbol="BTCUSDT", interval="1h",
        bar_close_time=pd.Timestamp("2024-02-01", tz="UTC"),
        proba=(0.15, 0.20, 0.65), signal=1, close_price=42000.0, model_version="v1",
    )

    df = _learnable_dataset(n=2000)
    result = train_fold(df.iloc[:1500], df.iloc[1500:], TrainConfig(num_boost_round=20))
    registry = ModelRegistry(tmp_path / "models")
    registry.save(
        result, symbol="BTCUSDT", interval="1h",
        metrics={}, config=TrainConfig(), margin_cutoff=0.11,
    )

    client = TestClient(create_app(cfg))
    assert client.get("/api/predict").json()["margin_cutoff"] == pytest.approx(0.11)

    registry.save(
        result, symbol="BTCUSDT", interval="1h",
        metrics={}, config=TrainConfig(), margin_cutoff=0.06,
    )
    assert client.get("/api/predict").json()["margin_cutoff"] == pytest.approx(0.06)
