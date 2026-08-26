import pandas as pd
import pytest

from cryptopred.backtest.execution import ExecutionModel
from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve.store import PredictionStore

MAKER = ExecutionModel(
    style="maker", maker_fee=0.0002, taker_fee=0.0005, slippage=0.0002,
    limit_offset=0.002, unfilled="chase", fill_buffer=0.0005,
)
SKIPPER = ExecutionModel(
    style="maker", maker_fee=0.0002, taker_fee=0.0005, slippage=0.0002,
    limit_offset=0.002, unfilled="skip", fill_buffer=0.0005,
)


def _make(tmp_path, rows, execution=MAKER):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.strategy.starting_capital = 10_000.0
    cfg.strategy.funding_rate = 0.0

    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    bars["volume"] = 1.0
    bars["quote_volume"] = 1.0
    bars["trades"] = 1
    bars["taker_buy_base"] = 0.5
    bars["taker_buy_quote"] = 0.5
    bars["close_time"] = idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)

    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", bars)
    store = PredictionStore(tmp_path / "paper.db")
    return PaperTrader(cfg=cfg, store=store, parquet=parquet, execution=execution), bars


def test_a_signal_creates_a_resting_order_not_a_position(tmp_path):
    trader, bars = _make(tmp_path, [(100, 101, 99, 100)] * 4)
    assert trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=2)

    assert len(trader.store.pending_orders("BTCUSDT")) == 1
    assert trader.store.open_trades("BTCUSDT").empty      # not a position yet


def test_limit_is_quoted_from_the_last_close(tmp_path):
    trader, bars = _make(tmp_path, [(100, 101, 99, 100)] * 4)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=2)
    order = trader.store.pending_orders("BTCUSDT").iloc[0]
    assert order["limit_price"] == pytest.approx(100.0 * (1 - 0.002))


def test_short_posts_above_the_market(tmp_path):
    trader, bars = _make(tmp_path, [(100, 101, 99, 100)] * 4)
    trader.post_limit("BTCUSDT", "1h", -1, bars.index[0], 100.0, "v1", horizon=2)
    order = trader.store.pending_orders("BTCUSDT").iloc[0]
    assert order["limit_price"] == pytest.approx(100.0 * (1 + 0.002))


def test_order_fills_when_the_next_bar_trades_through_the_limit(tmp_path):
    # limit sits at 99.8; the next bar dips to 99.0
    rows = [(100, 101, 99.5, 100), (100, 101, 99.0, 100.5), (100, 101, 99, 100)]
    trader, bars = _make(tmp_path, rows)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)

    counts = trader.resolve_pending("BTCUSDT", "1h")
    assert counts["filled"] == 1
    opened = trader.store.open_trades("BTCUSDT")
    assert len(opened) == 1
    assert opened.iloc[0]["entry_price"] == pytest.approx(99.8)
    assert opened.iloc[0]["entry_was_maker"] == 1


def test_unfilled_order_is_chased_at_the_bar_close(tmp_path):
    # limit at 99.8; the bar never trades below 100 — the runaway case, which is
    # exactly the move a long signal wanted
    rows = [(100, 101, 100, 100), (100, 102, 100, 101.5), (101, 103, 100, 102)]
    trader, bars = _make(tmp_path, rows)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)

    counts = trader.resolve_pending("BTCUSDT", "1h")
    assert counts["chased"] == 1
    opened = trader.store.open_trades("BTCUSDT")
    assert opened.iloc[0]["entry_was_maker"] == 0
    assert opened.iloc[0]["entry_price"] == pytest.approx(101.5 * 1.0002)


def test_skip_mode_cancels_instead_of_chasing(tmp_path):
    rows = [(100, 101, 100, 100), (100, 102, 100, 101.5), (101, 103, 100, 102)]
    trader, bars = _make(tmp_path, rows, execution=SKIPPER)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)

    counts = trader.resolve_pending("BTCUSDT", "1h")
    assert counts["cancelled"] == 1
    assert trader.store.open_trades("BTCUSDT").empty
    assert trader.store.pending_orders("BTCUSDT").empty


def test_an_order_whose_bar_has_not_closed_is_left_alone(tmp_path):
    trader, bars = _make(tmp_path, [(100, 101, 99, 100)] * 3)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[-1], 100.0, "v1", horizon=1)

    counts = trader.resolve_pending("BTCUSDT", "1h")
    assert counts["waiting"] == 1
    assert len(trader.store.pending_orders("BTCUSDT")) == 1


def test_a_rested_entry_costs_less_than_two_taker_legs(tmp_path):
    rows = [(100, 101, 99.5, 100), (100, 101, 99.0, 100), (100, 101, 99, 100),
            (100, 101, 99, 100)]
    trader, bars = _make(tmp_path, rows)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)
    trader.resolve_pending("BTCUSDT", "1h")
    trader.close_due_trades("BTCUSDT", "1h", horizon=1)

    closed = trader.store.closed_trades("BTCUSDT").iloc[0]
    assert closed["cost"] < 2 * (0.0005 + 0.0002)


def test_exit_limit_fills_when_the_bar_reaches_it(tmp_path):
    rows = [(100, 101, 99.5, 100), (100, 101, 99.0, 100), (100, 105, 99, 104),
            (104, 105, 103, 104)]
    trader, bars = _make(tmp_path, rows)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)
    trader.resolve_pending("BTCUSDT", "1h")
    trader.close_due_trades("BTCUSDT", "1h", horizon=1)

    closed = trader.store.closed_trades("BTCUSDT").iloc[0]
    assert closed["exit_price"] == pytest.approx(100.0 * (1 + 0.002))
    assert closed["net_return"] > 0


def test_exit_falls_back_to_market_when_the_limit_is_missed(tmp_path):
    rows = [(100, 101, 99.5, 100), (100, 101, 99.0, 100), (100, 100.1, 95, 96),
            (96, 97, 95, 96)]
    trader, bars = _make(tmp_path, rows)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)
    trader.resolve_pending("BTCUSDT", "1h")
    trader.close_due_trades("BTCUSDT", "1h", horizon=1)

    closed = trader.store.closed_trades("BTCUSDT").iloc[0]
    assert closed["exit_price"] < 100.0
    assert closed["net_return"] < 0


def test_market_trader_is_unaffected_by_the_maker_code(tmp_path):
    """The default path must behave exactly as it did before maker support."""
    rows = [(100, 101, 99, 100), (100, 111, 99, 110), (110, 122, 109, 121)]
    trader, bars = _make(tmp_path, rows, execution=None)
    trader.open_from_signal("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)
    trader.close_due_trades("BTCUSDT", "1h", horizon=1)

    closed = trader.store.closed_trades("BTCUSDT").iloc[0]
    assert closed["cost"] == pytest.approx(2 * (0.0005 + 0.0002))
    assert trader.store.pending_orders("BTCUSDT").empty


def test_resolve_pending_is_a_no_op_without_an_execution_model(tmp_path):
    trader, _ = _make(tmp_path, [(100, 101, 99, 100)] * 3, execution=None)
    assert trader.resolve_pending("BTCUSDT", "1h") == {
        "filled": 0, "chased": 0, "cancelled": 0, "waiting": 0
    }


def test_duplicate_limit_for_the_same_signal_is_rejected(tmp_path):
    trader, bars = _make(tmp_path, [(100, 101, 99, 100)] * 4)
    assert trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=2)
    assert not trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=2)


def test_cancelled_orders_never_reach_the_summary(tmp_path):
    rows = [(100, 101, 100, 100), (100, 102, 100, 101.5), (101, 103, 100, 102)]
    trader, bars = _make(tmp_path, rows, execution=SKIPPER)
    trader.post_limit("BTCUSDT", "1h", 1, bars.index[0], 100.0, "v1", horizon=1)
    trader.resolve_pending("BTCUSDT", "1h")
    assert trader.summary("BTCUSDT")["n_trades"] == 0


def test_orders_from_consecutive_bars_can_fill_without_colliding(tmp_path):
    """Regression for a real crash: the table was keyed on entry_time, so a
    limit that filled moved onto the bar the next pending order occupied."""
    rows = [(100, 101, 99.0, 100)] * 8
    trader, bars = _make(tmp_path, rows)

    for i in range(4):
        assert trader.post_limit(
            "BTCUSDT", "1h", 1, bars.index[i], 100.0, "v1", horizon=1
        )

    counts = trader.resolve_pending("BTCUSDT", "1h")
    assert counts["filled"] == 4
    assert len(trader.store.open_trades("BTCUSDT")) == 4
