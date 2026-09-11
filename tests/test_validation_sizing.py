"""The sizing arm must be able to return either answer.

A comparison that structurally favours one rule is not a comparison. These
tests pin the decision logic against hand-built results rather than against a
model run, so they fail if the criteria drift from the pre-registration.
"""

import pandas as pd

from cryptopred.models.validation import (
    FrozenConfig,
    SymbolResult,
    evaluate_symbol_sizing,
    format_sizing_validation,
    sizing_verdict,
)


def _result(symbol, fixed_ret, linear_ret, fixed_dd, linear_dd):
    dd_ok = abs(linear_dd) <= abs(fixed_dd) * 1.2
    return SymbolResult(
        symbol,
        status="tested",
        detail={
            "n_signals": 100,
            "fixed": {"total_return": fixed_ret, "max_drawdown": fixed_dd,
                      "sharpe": 0.5, "exposure": 0.01, "n_trades": 100},
            "linear": {"total_return": linear_ret, "max_drawdown": linear_dd,
                       "sharpe": 0.6, "exposure": 0.01, "n_trades": 100},
            "return_gap": linear_ret - fixed_ret,
            "drawdown_ok": dd_ok,
            "favours_linear": linear_ret > fixed_ret and dd_ok,
        },
    )


def test_a_better_return_bought_with_more_risk_does_not_count():
    """The whole point of the drawdown condition."""
    r = _result("X", fixed_ret=0.10, linear_ret=0.20, fixed_dd=-0.10, linear_dd=-0.30)
    assert not r.detail["favours_linear"]


def test_a_better_return_at_similar_risk_counts():
    r = _result("X", fixed_ret=0.10, linear_ret=0.20, fixed_dd=-0.10, linear_dd=-0.11)
    assert r.detail["favours_linear"]


def test_verdict_adopts_only_above_the_registered_rate():
    assert sizing_verdict(13, 20).startswith("ADOPT LINEAR")
    assert sizing_verdict(12, 20).startswith("KEEP FIXED, INCONCLUSIVE")


def test_verdict_rejects_at_or_below_the_registered_rate():
    assert sizing_verdict(7, 20).startswith("KEEP FIXED")
    assert "noise" in sizing_verdict(7, 20)


def test_ties_go_to_the_incumbent():
    """Exactly half favouring linear must not switch the live configuration."""
    assert sizing_verdict(10, 20).startswith("KEEP FIXED")


def test_skipped_symbols_stay_in_the_denominator():
    results = [_result(f"S{i}", 0.1, 0.2, -0.1, -0.1) for i in range(10)]
    results += [SymbolResult(f"T{i}", status="skipped: no bars stored") for i in range(10)]
    report = format_sizing_validation(results, FrozenConfig())
    assert "favour linear 10/20" in report
    assert "KEEP FIXED" in report


def test_symbol_with_too_few_bars_is_skipped_not_tested():
    bars = pd.DataFrame({"close": [1.0] * 10})
    r = evaluate_symbol_sizing("X", pd.DataFrame({"a": [1]}), bars, FrozenConfig())
    assert r.status.startswith("skipped")
