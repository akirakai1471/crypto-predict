import numpy as np
import pandas as pd
import pytest

from cryptopred.backtest.breakeven import (
    analyse,
    breakeven_accuracy,
    format_table,
    round_trip_cost,
    summarise_moves,
)


def _bars(moves_pct: float, n: int = 500) -> pd.DataFrame:
    """Price that alternates up and down by a fixed percentage, so the median
    absolute forward move over one bar is exactly `moves_pct`."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    steps = np.where(np.arange(n) % 2 == 0, moves_pct, -moves_pct / (1 + moves_pct))
    close = 100 * np.cumprod(1 + steps)
    return pd.DataFrame({"close": close}, index=idx)


def test_round_trip_cost_charges_both_legs():
    assert round_trip_cost(0.0005, 0.0002) == pytest.approx(0.0014)


def test_breakeven_is_fifty_percent_when_trading_is_free():
    assert breakeven_accuracy(move=0.01, cost=0.0) == pytest.approx(0.5)


def test_breakeven_rises_as_cost_grows():
    cheap = breakeven_accuracy(move=0.01, cost=0.001)
    dear = breakeven_accuracy(move=0.01, cost=0.005)
    assert dear > cheap > 0.5


def test_breakeven_falls_as_the_move_grows():
    small = breakeven_accuracy(move=0.001, cost=0.0014)
    large = breakeven_accuracy(move=0.05, cost=0.0014)
    assert small > large


def test_cost_double_the_move_needs_impossible_accuracy():
    """The 1m scalping case: a 0.07% move against a 0.14% toll."""
    acc = breakeven_accuracy(move=0.0007, cost=0.0014)
    assert acc > 1.0
    results = analyse(_bars(0.0007), horizons=(1,), cost=0.0014)
    assert results[0].is_possible is False


def test_zero_move_is_never_tradeable():
    assert breakeven_accuracy(move=0.0, cost=0.0014) == float("inf")


def test_analyse_measures_the_real_move_size():
    bars = _bars(0.01)
    result = analyse(bars, horizons=(1,), cost=0.0014)[0]
    assert result.median_move == pytest.approx(0.01, rel=0.02)
    assert result.breakeven_accuracy == pytest.approx((0.0014 / 0.01 + 1) / 2, rel=0.02)


def test_longer_horizons_report_larger_moves():
    rng = np.random.default_rng(0)
    n = 4000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    bars = pd.DataFrame({"close": close}, index=idx)

    results = analyse(bars, horizons=(1, 4, 24, 72))
    moves = [r.median_move for r in results]
    assert moves == sorted(moves)
    accs = [r.breakeven_accuracy for r in results]
    assert accs == sorted(accs, reverse=True)


def test_margin_is_positive_when_accuracy_clears():
    result = analyse(_bars(0.02), horizons=(1,), cost=0.0014)[0]
    assert result.margin(0.60) > 0
    assert result.margin(0.51) < 0


def test_format_table_flags_impossible_horizons():
    results = analyse(_bars(0.0005), horizons=(1,), cost=0.0014)
    text = format_table(results, interval="1m", symbol="BTCUSDT")
    assert "impossible" in text
    assert "oracle loses money" in text


def test_format_table_without_measured_accuracy_omits_margin():
    results = analyse(_bars(0.02), horizons=(1,), cost=0.0014)
    text = format_table(results, interval="1h", symbol="BTCUSDT")
    assert "margin" not in text


def test_summarise_moves_reports_a_distribution():
    stats = summarise_moves(_bars(0.01), horizon=1)
    assert stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"]
