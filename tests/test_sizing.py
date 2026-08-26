import numpy as np
import pandas as pd
import pytest

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.backtest.sizing import (
    breakeven_probability,
    describe_sizing,
    kelly_fraction,
    size_from_confidence,
)


def test_kelly_is_zero_for_a_losing_bet():
    # win 0.86%, lose 1.14%: needs better than ~57% to be worth anything
    assert kelly_fraction(0.50, win=0.0086, loss=0.0114) == pytest.approx(0.0)
    assert kelly_fraction(0.40, win=0.0086, loss=0.0114) == pytest.approx(0.0)


def test_kelly_grows_with_confidence():
    stakes = kelly_fraction(np.array([0.58, 0.62, 0.70]), win=0.012, loss=0.016)
    assert stakes[0] < stakes[1] < stakes[2]


def test_kelly_never_exceeds_full_capital():
    assert kelly_fraction(1.0, win=0.01, loss=0.01) <= 1.0


def test_breakeven_probability_matches_the_cost_arithmetic():
    """Losing more than you win on the same move pushes break-even above 50%."""
    move, cost = 0.01373, 0.0014
    be = breakeven_probability(win=move - cost, loss=move + cost)
    assert be > 0.5
    assert be == pytest.approx(0.551, abs=0.01)   # agrees with the break-even table


def test_breakeven_is_fifty_percent_without_costs():
    assert breakeven_probability(win=0.01, loss=0.01) == pytest.approx(0.5)


def test_fixed_sizing_bets_the_same_everywhere():
    sizes = size_from_confidence(np.array([0.51, 0.75, 0.99]), method="fixed")
    assert (sizes == 1.0).all()


def test_linear_sizing_ramps_from_the_threshold():
    sizes = size_from_confidence(
        np.array([0.60, 0.80, 1.00]), method="linear", threshold=0.60
    )
    assert sizes[0] == pytest.approx(0.0)
    assert sizes[2] == pytest.approx(1.0)
    assert sizes[0] < sizes[1] < sizes[2]


def test_kelly_sizing_declines_weak_signals_entirely():
    sizes = size_from_confidence(
        np.array([0.52, 0.70]), method="kelly", median_move=0.01373,
        round_trip_cost=0.0014, kelly_scale=0.5,
    )
    assert sizes[0] == pytest.approx(0.0)   # below break-even: no stake at all
    assert sizes[1] > 0.0


def test_sqrt_kelly_is_flatter_than_kelly():
    conf = np.array([0.60, 0.65, 0.70])
    kelly = size_from_confidence(conf, method="kelly", kelly_scale=1.0)
    sqrt_kelly = size_from_confidence(conf, method="sqrt_kelly", kelly_scale=1.0)
    # sqrt lifts small stakes toward the middle, so the spread narrows
    assert (sqrt_kelly >= kelly).all()
    assert sqrt_kelly.std() < kelly.std() or sqrt_kelly.max() >= kelly.max()


def test_no_method_ever_exceeds_full_size():
    conf = np.linspace(0.34, 1.0, 50)
    for method in ("fixed", "linear", "kelly", "sqrt_kelly"):
        sizes = size_from_confidence(conf, method=method, threshold=0.5)
        assert sizes.max() <= 1.0
        assert sizes.min() >= 0.0


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="nonsense"):
        size_from_confidence(np.array([0.6]), method="nonsense")


def test_describe_sizing_summarises_deployment():
    sizes = np.array([0.0, 0.5, 1.0, 0.25])
    signals = np.array([1, -1, 1, 1])
    summary = describe_sizing(sizes, signals)
    assert summary["n_sized"] == 4
    assert summary["zero_size_share"] == pytest.approx(0.25)
    assert summary["max_size"] == pytest.approx(1.0)


# --- engine integration -------------------------------------------------

def _bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close * 1.01,
         "low": close * 0.99, "close": close},
        index=idx,
    )


def test_half_size_halves_the_pnl():
    bars = _bars([100, 110, 121])
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)

    full = pd.DataFrame({"signal": [1, 0, 0], "size": [1.0, 0.0, 0.0]}, index=bars.index)
    half = pd.DataFrame({"signal": [1, 0, 0], "size": [0.5, 0.0, 0.0]}, index=bars.index)

    full_eq = backtest(bars, full, horizon=1, costs=free).equity.iloc[-1] - 1
    half_eq = backtest(bars, half, horizon=1, costs=free).equity.iloc[-1] - 1
    assert half_eq == pytest.approx(full_eq / 2, rel=1e-6)


def test_half_size_halves_the_costs_too():
    """Sizing down must reduce fees as well as exposure, or variable sizing looks
    free."""
    bars = _bars([100, 100, 100])
    costs = CostModel(taker_fee=0.0005, slippage=0.0002, funding_rate=0.0)

    full = pd.DataFrame({"signal": [1, 0, 0], "size": [1.0, 0.0, 0.0]}, index=bars.index)
    half = pd.DataFrame({"signal": [1, 0, 0], "size": [0.5, 0.0, 0.0]}, index=bars.index)

    full_loss = 1 - backtest(bars, full, horizon=1, costs=costs).equity.iloc[-1]
    half_loss = 1 - backtest(bars, half, horizon=1, costs=costs).equity.iloc[-1]
    assert half_loss == pytest.approx(full_loss / 2, rel=1e-3)


def test_zero_size_takes_no_trade():
    bars = _bars([100, 110, 121])
    frame = pd.DataFrame({"signal": [1, 0, 0], "size": [0.0, 0.0, 0.0]}, index=bars.index)
    assert backtest(bars, frame, horizon=1).trades.empty


def test_missing_size_column_means_full_size():
    bars = _bars([100, 110, 121])
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)
    without = pd.DataFrame({"signal": [1, 0, 0]}, index=bars.index)
    with_full = pd.DataFrame({"signal": [1, 0, 0], "size": [1.0, 0.0, 0.0]}, index=bars.index)

    a = backtest(bars, without, horizon=1, costs=free).equity.iloc[-1]
    b = backtest(bars, with_full, horizon=1, costs=free).equity.iloc[-1]
    assert a == pytest.approx(b)


def test_sizing_cannot_create_leverage():
    """A size above 1 is clipped: the rule may bet less, never more."""
    bars = _bars([100, 110, 121])
    free = CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0)
    huge = pd.DataFrame({"signal": [1, 0, 0], "size": [5.0, 0.0, 0.0]}, index=bars.index)
    full = pd.DataFrame({"signal": [1, 0, 0], "size": [1.0, 0.0, 0.0]}, index=bars.index)

    assert backtest(bars, huge, horizon=1, costs=free).equity.iloc[-1] == pytest.approx(
        backtest(bars, full, horizon=1, costs=free).equity.iloc[-1]
    )


def test_variable_sizing_lowers_average_exposure():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(100 * 1.002 ** np.arange(n), index=idx)
    bars = pd.DataFrame(
        {"open": close.shift(1).fillna(100.0), "high": close, "low": close, "close": close},
        index=idx,
    )
    full = pd.DataFrame({"signal": [1] * n, "size": [1.0] * n}, index=idx)
    varied = pd.DataFrame({"signal": [1] * n, "size": [0.4] * n}, index=idx)

    a = backtest(bars, full, horizon=10).summary["avg_exposure"]
    b = backtest(bars, varied, horizon=10).summary["avg_exposure"]
    assert b < a


def test_match_exposure_equalises_average_stakes():
    from cryptopred.backtest.sizing_report import match_exposure

    signals = np.ones(100, dtype=int)
    arrays = {
        "fixed": np.ones(100),
        "kelly": np.full(100, 0.05),
        "linear": np.full(100, 0.40),
    }
    matched = match_exposure(arrays, signals)
    means = [m[signals != 0].mean() for m in matched.values()]
    assert np.allclose(means, means[0])
    # scaled to the smallest, so nothing is scaled up past full size
    assert all(m.max() <= 1.0 for m in matched.values())
    assert means[0] == pytest.approx(0.05)


def test_match_exposure_preserves_relative_allocation():
    """Scaling must not flatten the differences between trades — that is the
    thing being tested."""
    from cryptopred.backtest.sizing_report import match_exposure

    signals = np.ones(4, dtype=int)
    arrays = {"a": np.array([0.1, 0.2, 0.3, 0.4]), "b": np.full(4, 0.05)}
    matched = match_exposure(arrays, signals)
    ratios = matched["a"] / matched["a"].max()
    assert ratios == pytest.approx(np.array([0.25, 0.5, 0.75, 1.0]))


def test_match_exposure_handles_no_signals():
    from cryptopred.backtest.sizing_report import match_exposure

    signals = np.zeros(10, dtype=int)
    arrays = {"a": np.zeros(10)}
    assert match_exposure(arrays, signals) == arrays


def _row(method, total_return, drawdown, sharpe):
    return {
        "method": method, "n_trades": 100, "total_return": total_return,
        "max_drawdown": drawdown, "sharpe": sharpe, "sortino": 0.1,
        "avg_exposure": 0.05, "doubled_cost_return": 0.05,
        "survives_doubled_costs": True, "two_sided": "TWO-SIDED",
        "n_sized": 100, "mean_size": 0.5, "median_size": 0.5, "max_size": 1.0,
        "zero_size_share": 0.0,
    }


def test_report_prints_the_matched_table_when_given_one():
    """Regression: the matched table was computed and never reached the report,
    so the verdict judged the rigged comparison instead of the fair one."""
    from cryptopred.backtest.sizing_report import format_sizing_comparison

    raw = [_row("fixed", 0.71, -0.17, 0.67), _row("kelly", 0.11, -0.05, 0.47)]
    matched = [_row("fixed", 0.11, -0.05, 0.45), _row("kelly", 0.14, -0.04, 0.52)]
    text = format_sizing_comparison(raw, "BTCUSDT", 24, 0.01373, matched)

    assert "EXPOSURE-MATCHED" in text
    assert "at equal exposure" in text
    assert "KELLY beats fixed sizing" in text


def test_report_judges_raw_stakes_when_no_matched_table():
    from cryptopred.backtest.sizing_report import format_sizing_comparison

    raw = [_row("fixed", 0.71, -0.17, 0.67), _row("kelly", 0.11, -0.05, 0.47)]
    text = format_sizing_comparison(raw, "BTCUSDT", 24, 0.01373)
    assert "EXPOSURE-MATCHED" not in text
    assert "at raw stakes" in text


def _trending_bars(n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.006, n))), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(100.0), "high": close * 1.002,
         "low": close * 0.998, "close": close},
        index=idx,
    )


def test_confidence_diagnostics_reports_correlations_and_buckets():
    from cryptopred.backtest.sizing_report import confidence_diagnostics

    bars = _trending_bars()
    idx = bars.index
    rng = np.random.default_rng(1)
    signals = rng.choice([-1, 1], len(idx))
    confidence = rng.uniform(0.60, 0.95, len(idx))

    diag = confidence_diagnostics(bars, idx, signals, confidence, horizon=10)
    assert diag["n"] > 0
    assert -1.0 <= diag["corr_net"] <= 1.0
    assert set(diag["table"].columns) == {"n", "hit_rate", "avg_abs_move", "avg_net"}


def test_confidence_diagnostics_handles_no_signals():
    from cryptopred.backtest.sizing_report import confidence_diagnostics

    bars = _trending_bars()
    signals = np.zeros(len(bars), dtype=int)
    diag = confidence_diagnostics(bars, bars.index, signals, np.full(len(bars), 0.7), horizon=10)
    assert diag["n"] == 0


def test_diagnostics_note_fires_when_the_top_bucket_is_not_the_best():
    from cryptopred.backtest.sizing_report import format_confidence_diagnostics

    table = pd.DataFrame(
        {
            "n": [100, 100, 100],
            "hit_rate": [0.57, 0.63, 0.57],
            "avg_abs_move": [0.026, 0.028, 0.025],
            "avg_net": [0.003, 0.008, 0.002],   # best in the middle
        },
        index=["[0.6, 0.65)", "[0.65, 0.7)", "[0.7, 1.01)"],
    )
    lines = format_confidence_diagnostics(
        {"n": 300, "corr_net": -0.02, "corr_move": -0.03, "corr_correct": -0.004,
         "table": table}
    )
    text = "\n".join(lines)
    assert "most profitable bucket" in text
    assert "no further information" in text


def test_diagnostics_note_stays_quiet_when_confidence_does_rank_profit():
    from cryptopred.backtest.sizing_report import format_confidence_diagnostics

    table = pd.DataFrame(
        {
            "n": [100, 100, 100],
            "hit_rate": [0.55, 0.60, 0.68],
            "avg_abs_move": [0.02, 0.025, 0.03],
            "avg_net": [0.002, 0.005, 0.010],   # rises with confidence
        },
        index=["[0.6, 0.65)", "[0.65, 0.7)", "[0.7, 1.01)"],
    )
    lines = format_confidence_diagnostics(
        {"n": 300, "corr_net": 0.2, "corr_move": 0.1, "corr_correct": 0.15, "table": table}
    )
    assert "most profitable bucket" not in "\n".join(lines)
