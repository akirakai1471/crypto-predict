import pytest

from cryptopred.models.validation import (
    MIN_BARS,
    FrozenConfig,
    SymbolResult,
    evaluate_symbol,
    format_validation,
    summarise,
    validation_verdict,
)


def _tested(symbol, total_return=0.5, survives=True, two_sided="TWO-SIDED"):
    return SymbolResult(
        symbol,
        status="tested",
        detail={
            "n_bars": 40_000, "n_signals": 3000, "n_trades": 3000,
            "sign_accuracy": 0.57, "total_return": total_return,
            "max_drawdown": -0.2, "sharpe": 0.5,
            "doubled_cost_return": 0.2 if survives else -0.2,
            "survives_doubled_costs": survives,
            "short_trades": 800, "short_win_rate": 0.55, "short_pnl": 300.0,
            "two_sided": two_sided,
        },
    )


def test_passing_requires_all_three_gates():
    assert _tested("A").passed
    assert not _tested("B", total_return=-0.1).passed
    assert not _tested("C", survives=False).passed
    assert not _tested("D", two_sided="ONE-SIDED").passed


def test_a_skipped_symbol_never_passes():
    assert not SymbolResult("E", status="skipped: only 900 bars").passed


def test_pass_rate_denominator_includes_skipped_symbols():
    """Dropping the awkward ones is how a pass rate gets inflated."""
    results = [_tested("A"), _tested("B"), SymbolResult("C", status="skipped: short")]
    stats = summarise(results)
    assert stats["attempted"] == 3
    assert stats["tested"] == 2
    assert stats["passed"] == 2
    assert stats["pass_rate_of_attempted"] == pytest.approx(2 / 3)
    assert stats["pass_rate_of_tested"] == pytest.approx(1.0)


def test_verdict_uses_the_forty_percent_threshold_written_in_advance():
    stats = {"attempted": 10, "pass_rate_of_attempted": 0.4}
    assert "SUPPORTS A REAL EFFECT" in validation_verdict(stats)


def test_verdict_uses_the_ten_percent_threshold_written_in_advance():
    stats = {"attempted": 20, "pass_rate_of_attempted": 0.10}
    assert "CONSISTENT WITH LUCK" in validation_verdict(stats)


def test_verdict_reports_the_middle_as_inconclusive():
    stats = {"attempted": 20, "pass_rate_of_attempted": 0.25}
    v = validation_verdict(stats)
    assert "INCONCLUSIVE" in v
    assert "rather than argued into either camp" in v


def test_verdict_handles_an_empty_run():
    assert "nothing was attempted" in validation_verdict(
        {"attempted": 0, "pass_rate_of_attempted": 0.0}
    )


def test_a_supporting_result_still_states_the_correlation_caveat():
    stats = {"attempted": 10, "pass_rate_of_attempted": 0.8}
    v = validation_verdict(stats)
    assert "not independent tests" in v


def test_short_history_is_skipped_rather_than_tested(tmp_path):
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    result = evaluate_symbol("TINYUSDT", pd.DataFrame(), bars, FrozenConfig())

    assert result.status.startswith("skipped")
    assert str(MIN_BARS) in result.status.replace(",", "")
    assert not result.passed


def test_report_lists_every_symbol_including_skipped():
    results = [_tested("AAAUSDT"), SymbolResult("BBBUSDT", status="skipped: only 900 bars")]
    text = format_validation(results, FrozenConfig())
    assert "AAAUSDT" in text
    assert "BBBUSDT" in text
    assert "skipped" in text


def test_report_marks_passing_symbols():
    text = format_validation([_tested("AAAUSDT")], FrozenConfig())
    assert "PASS" in text


def test_report_names_the_preregistration():
    text = format_validation([_tested("AAAUSDT")], FrozenConfig())
    assert "preregistration" in text


def test_frozen_config_execution_is_maker_and_chases():
    cfg = FrozenConfig()
    ex = cfg.execution()
    assert ex.style == "maker"
    assert ex.unfilled == "chase"        # skip loses to taker, per findings.md
    assert ex.fill_buffer > 0            # a touch is not a fill
    assert ex.limit_reference == "signal_close"
