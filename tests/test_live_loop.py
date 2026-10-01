"""The hourly loop, end to end, on the failures a live review found (2026-10-01).

Each test drives the real run_cycle over real code paths; only the network
ingest and the model are replaced.
"""

import sqlite3

import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve import runner
from cryptopred.serve.predictor import Prediction
from cryptopred.serve.store import PredictionStore
from tests.conftest import make_ohlcv


class _Client:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePredictor:
    """Signals the same way on every bar; enough to drive a cycle."""

    def __init__(self, parquet, signal=1, version="v1"):
        self.parquet, self.signal = parquet, signal
        self.bundle = type("B", (), {"metadata": {"version": version, "margin_cutoff": 0.1}})()

    def predict_latest(self, symbol, interval):
        bar = self.parquet.read("klines", symbol, interval).iloc[-1]
        return Prediction(
            symbol, interval, bar["close_time"], (0.1, 0.2, 0.7), self.signal, 0.7,
            float(bar["close"]), self.bundle.metadata["version"],
        )


@pytest.fixture
def loop(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.labels.horizon_bars = {"1h": 4}
    cfg.strategy.funding_rate = 0.0
    full = make_ohlcv(n=80, seed=5)
    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", full.iloc[:60])
    position = {"n": 60}

    def advance():
        """One more bar closes."""
        parquet.append("klines", "BTCUSDT", "1h", full.iloc[position["n"] : position["n"] + 1])
        position["n"] += 1

    monkeypatch.setattr(runner, "BinanceClient", _Client)
    monkeypatch.setattr(runner, "run_klines_ingest", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "run_funding_ingest", lambda *a: 0)
    monkeypatch.setattr(runner, "fill_gaps", lambda *a, **k: {"filled": 0})
    monkeypatch.setattr(runner, "_maybe_alert", lambda **k: 0)
    return cfg, full, parquet, advance


def _use(monkeypatch, factory):
    monkeypatch.setattr(
        runner.Predictor, "from_registry", classmethod(lambda cls, c, s, i: factory())
    )


def test_a_market_order_is_held_for_the_full_horizon(loop, monkeypatch):
    """The signal bar's open time was passed as the entry, so the trade exited
    at open[t+H] = close[t+H-1]: one bar short of the backtest and the labels."""
    cfg, full, parquet, advance = loop
    cfg.strategy.execution.style = "taker"
    _use(monkeypatch, lambda: _FakePredictor(parquet))
    runner.run_cycle(cfg, "1h")  # signal on bar 59
    _use(monkeypatch, lambda: _FakePredictor(parquet, signal=0))
    for _ in range(8):
        advance()
        runner.run_cycle(cfg, "1h")

    trade = PredictionStore(cfg.data.root / "predictions.db").closed_trades("BTCUSDT").iloc[0]
    assert pd.Timestamp(trade["signal_time"]) == full.index[59]
    # open[t+1+H] is close[t+H]: H full bars after the signal bar's close.
    assert pd.Timestamp(trade["exit_time"]) == full.index[59] + pd.Timedelta(hours=1 + 4)


def test_a_symbol_without_a_loadable_model_still_scores_and_closes(loop, monkeypatch):
    """`except FileNotFoundError: continue` skipped scoring and closing too, so
    a missing model froze the symbol's ledger while the heartbeat said alive."""
    cfg, full, parquet, advance = loop
    cfg.strategy.execution.style = "taker"
    _use(monkeypatch, lambda: _FakePredictor(parquet))
    runner.run_cycle(cfg, "1h")

    def gone():
        raise FileNotFoundError("metadata.json")

    _use(monkeypatch, gone)
    for _ in range(8):
        advance()
        runner.run_cycle(cfg, "1h")
    store = PredictionStore(cfg.data.root / "predictions.db")
    assert store.unscored("BTCUSDT", "1h").empty
    assert store.open_trades("BTCUSDT").empty


def test_a_cycle_that_dies_after_logging_still_places_the_order_on_retry(loop, monkeypatch):
    """The order depended on `recorded`, which a retry of the same bar never
    sees again: the signal was logged and the ledger never traded it."""
    cfg, full, parquet, advance = loop
    cfg.strategy.execution.style = "maker"
    _use(monkeypatch, lambda: _FakePredictor(parquet))
    real = runner.PaperTrader.post_limit
    calls = []

    def locked_once(self, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(self, **kwargs)

    monkeypatch.setattr(runner.PaperTrader, "post_limit", locked_once)
    with pytest.raises(sqlite3.OperationalError):
        runner.run_cycle(cfg, "1h")
    runner.run_cycle(cfg, "1h")  # the retry, same bar
    runner.run_cycle(cfg, "1h")  # and again: still one order
    assert len(PredictionStore(cfg.data.root / "predictions.db").pending_orders("BTCUSDT")) == 1


def test_a_retrain_within_the_hour_does_not_log_the_bar_twice(loop, monkeypatch):
    """The unique key included the model version: a retrain and restart in the
    same hour logged the bar again, alerted again, and counted it twice."""
    cfg, full, parquet, advance = loop
    cfg.strategy.execution.style = "maker"
    _use(monkeypatch, lambda: _FakePredictor(parquet, signal=1, version="A"))
    runner.run_cycle(cfg, "1h")
    _use(monkeypatch, lambda: _FakePredictor(parquet, signal=-1, version="B"))
    counts = runner.run_cycle(cfg, "1h")

    store = PredictionStore(cfg.data.root / "predictions.db")
    assert counts["predictions"] == 0
    history = store.history("BTCUSDT", "1h")
    assert len(history) == 1 and history.iloc[0]["model_version"] == "A"
    # The order follows the row of record, not the second model's opinion.
    orders = store.pending_orders("BTCUSDT")
    assert len(orders) == 1 and int(orders.iloc[0]["direction"]) == 1


def test_rows_already_duplicated_count_once(tmp_path):
    """A log that already holds two rows for one bar is not rewritten; every
    reader sees only the first."""
    store = PredictionStore(tmp_path / "p.db")
    bar = pd.Timestamp("2026-08-10 10:59:59.999", tz="UTC")
    store.record_prediction("BTCUSDT", "1h", bar, (0.6, 0.2, 0.2), -1, 1.0, "A")
    with sqlite3.connect(tmp_path / "p.db") as conn:  # as the old key allowed
        conn.execute(
            "INSERT INTO predictions (symbol, interval, bar_close_time, prob_down, "
            "prob_flat, prob_up, signal, confidence, close_price, model_version, "
            "created_at) VALUES ('BTCUSDT', '1h', ?, 0.2, 0.2, 0.6, 1, 0.6, 1.0, 'B', ?)",
            (bar.isoformat(), pd.Timestamp.now(tz="UTC").isoformat()),
        )
    for row in store.unscored("BTCUSDT", "1h").itertuples():
        store.score_prediction(row.id, -0.01, 0, row.signal == -1)

    assert len(store.history("BTCUSDT", "1h")) == 1
    assert store.latest_prediction("BTCUSDT", "1h")["model_version"] == "A"
    assert store.accuracy_summary("BTCUSDT", "1h")["n_scored"] == 1
