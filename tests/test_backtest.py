import numpy as np
import pandas as pd
import pytest

from cryptopred.backtest.engine import CostModel, backtest, equity_metrics


def _bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {
            # entry happens at the next bar's open; make open == previous close
            # so hand-computed expectations are exact
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=idx,
    )


def _signals(directions: list[int], index: pd.DatetimeIndex) -> pd.DataFrame:
    """directions: +1 long, -1 short, 0 stand aside."""
    return pd.DataFrame({"signal": directions}, index=index)


def test_no_signals_means_flat_equity():
    bars = _bars([100, 110, 120, 130, 140])
    signals = _signals([0, 0, 0, 0, 0], bars.index)
    result = backtest(bars, signals, horizon=1, costs=CostModel())
    assert result.trades.empty
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_a_winning_long_trade_nets_move_minus_costs():
    # entry at open of bar 1 (=100), exit at open of bar 2 (=110): +10%
    bars = _bars([100, 110, 121])
    signals = _signals([1, 0, 0], bars.index)
    costs = CostModel(taker_fee=0.0005, slippage=0.0002, funding_rate=0.0)

    result = backtest(bars, signals, horizon=1, costs=costs)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]

    gross = 110 / 100 - 1
    expected = gross - 2 * (0.0005 + 0.0002)  # entry and exit both pay fee + slippage
    assert trade["net_return"] == pytest.approx(expected, abs=1e-9)


def test_a_short_trade_profits_when_price_falls():
    bars = _bars([100, 90, 81])
    signals = _signals([-1, 0, 0], bars.index)
    costs = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)

    result = backtest(bars, signals, horizon=1, costs=costs)
    assert result.trades.iloc[0]["net_return"] == pytest.approx(0.10, abs=1e-9)


def test_costs_can_turn_a_winner_into_a_loser():
    """A 0.05% edge cannot survive a 0.14% round trip. This is the whole reason
    the backtest exists."""
    bars = _bars([100.0, 100.05, 100.10])
    signals = _signals([1, 0, 0], bars.index)

    free = backtest(bars, signals, horizon=1, costs=CostModel(taker_fee=0.0, slippage=0.0))
    real = backtest(
        bars, signals, horizon=1, costs=CostModel(taker_fee=0.0005, slippage=0.0002)
    )

    assert free.trades.iloc[0]["net_return"] > 0
    assert real.trades.iloc[0]["net_return"] < 0


def test_entry_uses_next_bar_open_not_signal_bar_close():
    """Trading at the close of the bar that produced the signal is impossible in
    reality and inflates every backtest that allows it."""
    bars = _bars([100, 200, 200])
    bars.loc[bars.index[1], "open"] = 150.0  # entry price differs from close[0]
    signals = _signals([1, 0, 0], bars.index)

    result = backtest(bars, signals, horizon=1, costs=CostModel(taker_fee=0.0, slippage=0.0))
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(150.0)


def test_last_signals_without_room_to_exit_are_dropped():
    bars = _bars([100, 110, 120])
    signals = _signals([0, 0, 1], bars.index)  # no bar after the horizon
    result = backtest(bars, signals, horizon=2, costs=CostModel())
    assert result.trades.empty


def test_funding_is_charged_to_longs_and_paid_to_shorts():
    bars = _bars([100, 100, 100])
    costs = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.001)

    long_result = backtest(bars, _signals([1, 0, 0], bars.index), horizon=1, costs=costs)
    short_result = backtest(bars, _signals([-1, 0, 0], bars.index), horizon=1, costs=costs)

    assert long_result.trades.iloc[0]["net_return"] < 0
    assert short_result.trades.iloc[0]["net_return"] > 0


def test_equity_compounds_across_trades():
    bars = _bars([100, 110, 121, 133.1, 146.41])
    signals = _signals([1, 1, 1, 0, 0], bars.index)
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)
    result = backtest(bars, signals, horizon=1, costs=free)

    assert len(result.trades) == 3
    assert result.equity.iloc[-1] == pytest.approx(1.1**3, rel=1e-6)


def test_equity_metrics_report_drawdown_and_sharpe():
    equity = pd.Series(
        [1.0, 1.1, 1.05, 1.2, 0.9, 1.3],
        index=pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
    )
    metrics = equity_metrics(equity, bars_per_year=8760)

    assert metrics["total_return"] == pytest.approx(0.3)
    assert metrics["max_drawdown"] == pytest.approx(-0.25, abs=1e-9)
    assert np.isfinite(metrics["sharpe"])


def test_equity_metrics_on_flat_equity_do_not_explode():
    equity = pd.Series(
        [1.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    )
    metrics = equity_metrics(equity, bars_per_year=8760)
    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["sharpe"] == 0.0


def test_win_rate_and_profit_factor():
    # 5 bars so all three signals have a bar to exit on
    bars = _bars([100, 110, 99, 108.9, 108.9])
    signals = _signals([1, 1, 1, 0, 0], bars.index)
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)
    result = backtest(bars, signals, horizon=1, costs=free)

    assert result.summary["n_trades"] == 3
    assert 0.0 <= result.summary["win_rate"] <= 1.0
    assert result.summary["profit_factor"] > 0
