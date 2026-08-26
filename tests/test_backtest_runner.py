import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel
from cryptopred.backtest.runner import (
    format_backtest,
    probabilities_to_signals,
    run_strategy_backtest,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")


def test_probabilities_to_signals_applies_the_dead_zone():
    proba = np.array(
        [
            [0.1, 0.2, 0.7],   # confident UP
            [0.7, 0.2, 0.1],   # confident DOWN
            [0.4, 0.2, 0.4],   # not confident enough
            [0.2, 0.6, 0.2],   # FLAT prediction
        ]
    )
    signals = probabilities_to_signals(proba, _index(4), threshold=0.5)
    assert list(signals["signal"]) == [1, -1, 0, 0]


def test_threshold_controls_how_many_signals_fire():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), 500)
    loose = probabilities_to_signals(proba, _index(500), threshold=0.35)
    strict = probabilities_to_signals(proba, _index(500), threshold=0.8)
    assert (loose["signal"] != 0).sum() >= (strict["signal"] != 0).sum()


def test_run_strategy_backtest_reports_all_three_cost_regimes():
    n = 300
    idx = _index(n)
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=idx)
    bars = pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close, "low": close,
         "close": close},
        index=idx,
    )
    proba = rng.dirichlet(np.ones(3), n)
    evaluation = {"proba": proba}

    result = run_strategy_backtest(bars, evaluation, idx, horizon=4, threshold=0.4)

    assert set(result) >= {"base", "doubled_costs", "frictionless", "survives_doubled_costs"}
    assert result["cost_drag"] >= 0  # costs can only reduce return


def test_doubled_costs_never_beat_base_costs():
    n = 400
    idx = _index(n)
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n))), index=idx)
    bars = pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close, "low": close,
         "close": close},
        index=idx,
    )
    # always predict UP with high confidence so there are plenty of trades
    proba = np.tile([0.05, 0.05, 0.9], (n, 1))

    result = run_strategy_backtest(bars, {"proba": proba}, idx, horizon=4, threshold=0.5)
    assert (
        result["doubled_costs"].summary["total_return"]
        <= result["base"].summary["total_return"]
    )


def test_format_backtest_mentions_robustness():
    n = 200
    idx = _index(n)
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    bars = pd.DataFrame(
        {"open": close.shift(1).fillna(100.0), "high": close, "low": close, "close": close},
        index=idx,
    )
    proba = np.tile([0.05, 0.05, 0.9], (n, 1))
    result = run_strategy_backtest(
        bars, {"proba": proba}, idx, horizon=4, threshold=0.5, costs=CostModel()
    )
    text = format_backtest(result, "BTCUSDT", "1h")
    assert "ROBUSTNESS" in text
    assert "survives doubled costs" in text
