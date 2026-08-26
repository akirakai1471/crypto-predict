import pandas as pd
import pytest

from cryptopred.serve.store import PredictionStore


@pytest.fixture
def store(tmp_path) -> PredictionStore:
    return PredictionStore(tmp_path / "predictions.db")


def _record(store: PredictionStore, hour: int, signal: int = 1) -> bool:
    return store.record_prediction(
        symbol="BTCUSDT",
        interval="1h",
        bar_close_time=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=hour),
        proba=(0.2, 0.2, 0.6),
        signal=signal,
        close_price=42000.0 + hour,
        model_version="v1",
    )


def test_record_and_read_back(store):
    assert _record(store, 0) is True
    latest = store.latest_prediction("BTCUSDT", "1h")
    assert latest["prob_up"] == pytest.approx(0.6)
    assert latest["model_version"] == "v1"


def test_duplicate_bar_is_ignored(store):
    assert _record(store, 0) is True
    assert _record(store, 0) is False
    assert len(store.history("BTCUSDT", "1h")) == 1


def test_latest_returns_the_newest_bar(store):
    _record(store, 0)
    _record(store, 5)
    _record(store, 3)
    latest = store.latest_prediction("BTCUSDT", "1h")
    assert latest["bar_close_time"].startswith("2024-01-01T05")


def test_unscored_excludes_scored_rows(store):
    _record(store, 0)
    _record(store, 1)
    pending = store.unscored("BTCUSDT", "1h")
    assert len(pending) == 2

    store.score_prediction(int(pending.iloc[0]["id"]), 0.01, 1, True)
    assert len(store.unscored("BTCUSDT", "1h")) == 1


def test_accuracy_summary_counts_only_scored(store):
    _record(store, 0, signal=1)
    _record(store, 1, signal=1)
    _record(store, 2, signal=0)

    pending = store.unscored("BTCUSDT", "1h")
    store.score_prediction(int(pending.iloc[0]["id"]), 0.02, 1, True)
    store.score_prediction(int(pending.iloc[1]["id"]), -0.02, -1, False)

    summary = store.accuracy_summary("BTCUSDT", "1h")
    assert summary["n_scored"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["n_signals"] == 2
    assert summary["signal_accuracy"] == pytest.approx(0.5)


def test_accuracy_summary_is_none_when_nothing_scored(store):
    _record(store, 0)
    summary = store.accuracy_summary("BTCUSDT", "1h")
    assert summary["n_scored"] == 0
    assert summary["accuracy"] is None


def test_paper_trade_lifecycle(store):
    entry = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert store.open_trade("BTCUSDT", "1h", 1, entry, 42000.0, 1000.0, "v1") is True
    assert store.open_trade("BTCUSDT", "1h", 1, entry, 42000.0, 1000.0, "v1") is False

    open_trades = store.open_trades("BTCUSDT")
    assert len(open_trades) == 1

    store.close_trade(
        int(open_trades.iloc[0]["id"]),
        exit_time=entry + pd.Timedelta(hours=24),
        exit_price=43000.0,
        gross_return=0.0238,
        cost=0.0014,
        net_return=0.0224,
        pnl_usd=22.4,
    )
    assert store.open_trades("BTCUSDT").empty
    closed = store.closed_trades("BTCUSDT")
    assert len(closed) == 1
    assert closed.iloc[0]["pnl_usd"] == pytest.approx(22.4)


def test_history_respects_limit(store):
    for hour in range(10):
        _record(store, hour)
    assert len(store.history("BTCUSDT", "1h", limit=3)) == 3


def test_scored_only_filter(store):
    _record(store, 0)
    _record(store, 1)
    pending = store.unscored("BTCUSDT", "1h")
    store.score_prediction(int(pending.iloc[0]["id"]), 0.01, 1, True)
    assert len(store.history("BTCUSDT", "1h", scored_only=True)) == 1


def test_store_survives_reopen(tmp_path):
    path = tmp_path / "predictions.db"
    first = PredictionStore(path)
    _record(first, 0)
    second = PredictionStore(path)
    assert second.latest_prediction("BTCUSDT", "1h") is not None
