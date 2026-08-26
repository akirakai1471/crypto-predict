import numpy as np
import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve.store import PredictionStore


@pytest.fixture
def setup(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.strategy.starting_capital = 10_000.0
    cfg.strategy.taker_fee = 0.0005
    cfg.strategy.slippage = 0.0002
    cfg.strategy.funding_rate = 0.0

    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(100.0 * 1.01 ** np.arange(60), index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "quote_volume": 1.0,
            "trades": 1,
            "taker_buy_base": 0.5,
            "taker_buy_quote": 0.5,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )
    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", bars)

    store = PredictionStore(tmp_path / "predictions.db")
    return PaperTrader(cfg=cfg, store=store, parquet=parquet), bars


def test_position_size_divides_capital_by_horizon(setup):
    trader, _ = setup
    assert trader.position_size("BTCUSDT", "1h", horizon=24) == pytest.approx(10_000 / 24)


def test_no_signal_opens_no_trade(setup):
    trader, bars = setup
    opened = trader.open_from_signal(
        "BTCUSDT", "1h", 0, bars.index[0], 100.0, "v1", horizon=24
    )
    assert opened is False
    assert trader.store.open_trades("BTCUSDT").empty


def test_signal_opens_a_trade(setup):
    trader, bars = setup
    assert trader.open_from_signal(
        "BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=24
    )
    assert len(trader.store.open_trades("BTCUSDT")) == 1


def test_trade_closes_at_horizon_with_costs(setup):
    trader, bars = setup
    entry_time = bars.index[0]
    entry_price = float(bars["open"].iloc[0])
    trader.open_from_signal("BTCUSDT", "1h", 1, entry_time, entry_price, "v1", horizon=10)

    assert trader.close_due_trades("BTCUSDT", "1h", horizon=10) == 1
    closed = trader.store.closed_trades("BTCUSDT")
    assert len(closed) == 1

    exit_price = float(bars["open"].iloc[10])
    expected_gross = exit_price / entry_price - 1
    assert closed.iloc[0]["gross_return"] == pytest.approx(expected_gross, abs=1e-9)
    assert closed.iloc[0]["net_return"] < closed.iloc[0]["gross_return"]


def test_trade_stays_open_before_its_horizon(setup):
    trader, bars = setup
    trader.open_from_signal("BTCUSDT", "1h", 1, bars.index[55], 100.0, "v1", horizon=24)
    assert trader.close_due_trades("BTCUSDT", "1h", horizon=24) == 0
    assert len(trader.store.open_trades("BTCUSDT")) == 1


def test_summary_ignores_open_trades(setup):
    trader, bars = setup
    trader.open_from_signal("BTCUSDT", "1h", 1, bars.index[55], 100.0, "v1", horizon=24)
    summary = trader.summary("BTCUSDT")
    assert summary["n_trades"] == 0
    assert summary["equity"] == pytest.approx(10_000.0)


def test_summary_reports_realised_pnl(setup):
    trader, bars = setup
    trader.open_from_signal(
        "BTCUSDT", "1h", 1, bars.index[0], float(bars["open"].iloc[0]), "v1", horizon=10
    )
    trader.close_due_trades("BTCUSDT", "1h", horizon=10)

    summary = trader.summary("BTCUSDT")
    assert summary["n_trades"] == 1
    assert summary["equity"] != pytest.approx(10_000.0)
    assert summary["win_rate"] == pytest.approx(1.0)


def test_short_trade_loses_in_a_rising_market(setup):
    trader, bars = setup
    trader.open_from_signal(
        "BTCUSDT", "1h", -1, bars.index[0], float(bars["open"].iloc[0]), "v1", horizon=10
    )
    trader.close_due_trades("BTCUSDT", "1h", horizon=10)
    assert trader.store.closed_trades("BTCUSDT").iloc[0]["net_return"] < 0


def test_duplicate_entry_time_is_rejected(setup):
    trader, bars = setup
    assert trader.open_from_signal("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", 24)
    assert not trader.open_from_signal("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", 24)
