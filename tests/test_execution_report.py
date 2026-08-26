import pandas as pd
import pytest

from cryptopred.backtest.engine import CostModel
from cryptopred.backtest.execution import ExecutionModel, maker_entry_fill
from cryptopred.backtest.execution_report import execution_verdict, score_execution


def _result(label, total_return, survives=True, decision="TWO-SIDED", fill_rate=0.9):
    return {
        "label": label, "total_return": total_return, "survives_doubled_costs": survives,
        "two_sided": decision, "fill_rate": fill_rate, "n_signals": 1000,
        "n_trades": 900, "win_rate": 0.56, "avg_return": 0.004,
        "max_drawdown": -0.16, "sharpe": 0.7, "doubled_cost_return": 0.3,
        "maker_leg_share": 0.9,
    }


def test_strict_fill_requires_price_to_trade_through():
    """A touch is not a fill when a queue sits ahead of you."""
    loose = ExecutionModel(style="maker", limit_offset=0.001, fill_buffer=0.0)
    strict = ExecutionModel(style="maker", limit_offset=0.001, fill_buffer=0.0005)

    # low lands exactly on the limit: loose fills, strict does not
    assert maker_entry_fill(100.0, 101.0, 99.9, 100.5, 1, loose).filled
    assert not maker_entry_fill(100.0, 101.0, 99.9, 100.5, 1, strict).filled

    # trade well through and both fill
    assert maker_entry_fill(100.0, 101.0, 99.0, 100.5, 1, strict).filled


def test_strict_fill_applies_to_shorts_too():
    strict = ExecutionModel(style="maker", limit_offset=0.001, fill_buffer=0.0005)
    assert not maker_entry_fill(100.0, 100.1, 99.0, 99.5, -1, strict).filled
    assert maker_entry_fill(100.0, 101.0, 99.0, 99.5, -1, strict).filled


def test_verdict_flags_an_advantage_that_only_exists_under_loose_fills():
    results = [
        _result("taker", 0.70),
        _result("maker 0.10% chase", 1.07),
        _result("maker 0.10% chase strict", 0.55),
    ]
    v = execution_verdict(results)
    assert "FILL-ASSUMPTION ARTEFACT" in v


def test_verdict_accepts_an_advantage_that_survives_strict_fills():
    results = [
        _result("taker", 0.70),
        _result("maker 0.10% chase", 1.07),
        _result("maker 0.10% chase strict", 0.95),
    ]
    v = execution_verdict(results)
    assert "beats taker execution" in v
    assert "holds under the strict fill rule" in v


def test_verdict_reports_no_benefit_when_maker_loses():
    results = [
        _result("taker", 0.70),
        _result("maker 0.10% skip", 0.40),
        _result("maker 0.10% skip strict", 0.20),
    ]
    assert "NO BENEFIT" in execution_verdict(results)


def test_verdict_ignores_maker_results_that_go_one_sided():
    results = [
        _result("taker", 0.70),
        _result("maker 0.10% skip strict", 1.20, decision="ONE-SIDED"),
    ]
    assert "NO BENEFIT" in execution_verdict(results)


def test_stress_test_keeps_the_fill_rule():
    """Regression: doubling fees must not quietly relax the fill requirement."""
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {"open": [100.0] * 6, "high": [100.5] * 6, "low": [99.95] * 6, "close": [100.2] * 6},
        index=idx,
    )
    signals = pd.Series([1, 0, 0, 0, 0, 0], index=idx).to_numpy()
    strict = ExecutionModel(
        style="maker", limit_offset=0.001, fill_buffer=0.01, unfilled="skip"
    )
    scored = score_execution(
        bars, idx, signals, horizon=1, label="strict", costs=CostModel(), execution=strict
    )
    # the limit is far out of reach under the strict rule, so nothing trades
    assert scored["n_trades"] == 0
    assert scored["doubled_cost_return"] == pytest.approx(0.0)
