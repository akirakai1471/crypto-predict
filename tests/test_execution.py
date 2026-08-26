import numpy as np
import pandas as pd
import pytest

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.backtest.execution import (
    ExecutionModel,
    entry_fill,
    exit_fill,
    fill_statistics,
    maker_entry_fill,
    taker_fill,
)

MAKER = ExecutionModel(style="maker", maker_fee=0.0002, limit_offset=0.001, unfilled="skip")


def test_taker_always_fills_and_pays_slippage():
    fill = taker_fill(100.0, direction=1, model=ExecutionModel(slippage=0.001))
    assert fill.filled
    assert fill.price == pytest.approx(100.1)      # buying costs more
    assert not fill.was_maker


def test_taker_short_gets_a_worse_price_too():
    fill = taker_fill(100.0, direction=-1, model=ExecutionModel(slippage=0.001))
    assert fill.price == pytest.approx(99.9)       # selling receives less


def test_maker_long_fills_only_when_price_dips_to_the_limit():
    # limit sits at 99.9; the bar's low reaches 99.5, so it fills
    hit = maker_entry_fill(100.0, high=101.0, low=99.5, close=100.5, direction=1, model=MAKER)
    assert hit.filled
    assert hit.price == pytest.approx(99.9)
    assert hit.was_maker

    # the bar never trades below 99.95: no fill
    miss = maker_entry_fill(100.0, high=101.0, low=99.95, close=100.8, direction=1, model=MAKER)
    assert not miss.filled


def test_maker_short_fills_only_when_price_rises_to_the_limit():
    hit = maker_entry_fill(100.0, high=100.5, low=99.0, close=99.5, direction=-1, model=MAKER)
    assert hit.filled
    assert hit.price == pytest.approx(100.1)

    miss = maker_entry_fill(100.0, high=100.05, low=98.0, close=98.5, direction=-1, model=MAKER)
    assert not miss.filled


def test_chase_crosses_the_spread_when_the_limit_misses():
    chaser = ExecutionModel(style="maker", limit_offset=0.001, unfilled="chase")
    fill = maker_entry_fill(100.0, high=101.0, low=99.95, close=100.8, direction=1, model=chaser)
    assert fill.filled
    assert not fill.was_maker             # it became a market order
    assert fill.cost == chaser.taker_fee


def test_an_exit_can_never_be_skipped():
    """A position must be closed; an unfilled exit limit becomes a market order."""
    fill = exit_fill(100.0, high=100.05, low=99.0, close=99.2, direction=1, model=MAKER)
    assert fill.filled
    assert not fill.was_maker


def test_exit_limit_fills_at_the_better_price_when_reached():
    fill = exit_fill(100.0, high=101.0, low=99.0, close=100.5, direction=1, model=MAKER)
    assert fill.was_maker
    assert fill.price == pytest.approx(100.1)     # sold above the open


def test_maker_fee_is_lower_than_taker_cost():
    assert MAKER.maker_cost() < MAKER.taker_cost()


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="nonsense"):
        entry_fill(100.0, 101.0, 99.0, 100.0, 1, ExecutionModel(style="nonsense"))


def test_fill_statistics_summarise_what_happened():
    stats = fill_statistics(
        entry_filled=np.array([True, True, False, True]),
        entry_was_maker=np.array([True, False, False, True]),
        exit_was_maker=np.array([True, True, False, False]),
    )
    assert stats["n_signals"] == 4
    assert stats["n_filled"] == 3
    assert stats["entry_fill_rate"] == pytest.approx(0.75)


# --- engine integration -------------------------------------------------

def _bars(rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_taker_path_is_unchanged_when_no_execution_model_is_given():
    bars = _bars([(100, 101, 99, 100), (100, 111, 99, 110), (110, 122, 109, 121)])
    signals = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)
    costs = CostModel(taker_fee=0.0005, slippage=0.0002, funding_rate=0.0)

    result = backtest(bars, signals, horizon=1, costs=costs)
    expected = (110 / 100 - 1) - 2 * (0.0005 + 0.0002)
    assert result.trades.iloc[0]["net_return"] == pytest.approx(expected, abs=1e-9)


def test_maker_skips_the_trade_when_the_entry_limit_never_fills():
    """The bar gaps up and never trades down to the limit — exactly the move a
    long signal wanted, and exactly the one a maker misses."""
    bars = _bars([(100, 101, 99, 100), (100, 111, 100, 110), (110, 122, 109, 121)])
    signals = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)

    result = backtest(bars, signals, horizon=1, execution=MAKER)
    assert result.trades.empty
    assert result.summary["n_unfilled"] == 1
    assert result.summary["fill_rate"] == pytest.approx(0.0)


def test_maker_fills_and_beats_taker_when_price_dips_first():
    bars = _bars([(100, 101, 99, 100), (100, 111, 99, 110), (110, 122, 109, 121)])
    signals = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)
    costs = CostModel(taker_fee=0.0005, slippage=0.0002, funding_rate=0.0)

    taker = backtest(bars, signals, horizon=1, costs=costs)
    maker = backtest(bars, signals, horizon=1, costs=costs, execution=MAKER)

    assert not maker.trades.empty
    assert maker.trades.iloc[0]["net_return"] > taker.trades.iloc[0]["net_return"]


def test_fill_rate_is_reported_for_partial_fills():
    rows = []
    for i in range(6):
        # alternate: bars that dip (fillable) and bars that gap up (not)
        rows.append((100, 101, 99 if i % 2 == 0 else 100, 100))
    bars = _bars(rows)
    signals = pd.DataFrame({"signal": [1] * 6}, index=bars.index)

    result = backtest(bars, signals, horizon=1, execution=MAKER)
    assert 0.0 < result.summary["fill_rate"] < 1.0


def test_equity_reflects_the_maker_price_improvement():
    """Entry below the open is a gain, and the equity curve must show it —
    otherwise maker execution looks worse than the trade ledger says."""
    bars = _bars([(100, 101, 99, 100), (100, 111, 99, 110), (110, 122, 109, 121)])
    signals = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)
    free_maker = ExecutionModel(style="maker", maker_fee=0.0, limit_offset=0.001)

    result = backtest(bars, signals, horizon=1, costs=free, execution=free_maker)
    ledger = result.trades.iloc[0]["net_return"]
    equity = result.equity.iloc[-1] - 1
    assert equity == pytest.approx(ledger, rel=1e-3)


def test_maker_leg_share_is_reported():
    bars = _bars([(100, 101, 99, 100), (100, 111, 99, 110), (110, 122, 109, 121)])
    signals = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)
    result = backtest(bars, signals, horizon=1, execution=MAKER)
    assert 0.0 <= result.summary["maker_leg_share"] <= 1.0
