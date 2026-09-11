from cryptopred.models.meta_report import _verdict
from cryptopred.paper.replay import SideStats


def _scored(total_return, survives=True, decision="TWO-SIDED", drawdown=-0.2):
    return {
        "total_return": total_return,
        "max_drawdown": drawdown,
        "survives_doubled_costs": survives,
        "two_sided": {"decision": decision, "reason": "because"},
        "n_signals": 1000,
        "coverage": 0.08,
        "sign_accuracy": 0.57,
        "win_rate": 0.53,
        "sharpe": 0.5,
        "n_trades": 1000,
        "doubled_cost_return": 0.1,
        "long": SideStats(1000, 0.55, 100.0, 0.001),
        "short": SideStats(300, 0.55, 30.0, 0.001),
    }


def test_no_go_when_doubled_costs_kill_it():
    v = _verdict(_scored(0.1), _scored(0.5, survives=False))
    assert v.startswith("NO-GO")


def test_no_go_when_one_sided():
    v = _verdict(_scored(0.1), _scored(0.5, decision="ONE-SIDED"))
    assert v.startswith("NO-GO") and "one-sided" in v


def test_no_benefit_when_it_loses_to_its_own_primary():
    v = _verdict(_scored(0.6), _scored(0.4))
    assert "does not beat the primary" in v


def test_no_benefit_when_it_loses_to_the_simple_model():
    """Beating a deliberately loose primary is not evidence: the stack must also
    beat one model at the production threshold."""
    v = _verdict(_scored(0.30), _scored(0.48), benchmark=_scored(0.62))
    assert "NO BENEFIT OVER THE SIMPLE MODEL" in v


def test_improvement_requires_clearing_both_comparisons():
    v = _verdict(_scored(0.30), _scored(0.70), benchmark=_scored(0.62))
    assert v.startswith("IMPROVEMENT")


def test_benchmark_is_optional():
    v = _verdict(_scored(0.30), _scored(0.48))
    assert v.startswith("IMPROVEMENT")


def test_report_prints_the_benchmark_line_when_given_one():
    """Regression: the benchmark was computed but never reached the report, so a
    gate that looked active was doing nothing."""
    from cryptopred.models.meta import MetaConfig
    from cryptopred.models.meta_report import format_meta_report

    text = format_meta_report(
        _scored(0.30),
        _scored(0.48),
        symbol="BTCUSDT",
        horizon=24,
        config=MetaConfig(),
        folds=[],
        benchmark=_scored(0.62),
        benchmark_threshold=0.60,
    )
    assert "BENCHMARK" in text
    assert "NO BENEFIT OVER THE SIMPLE MODEL" in text


def test_report_omits_the_benchmark_line_when_absent():
    from cryptopred.models.meta import MetaConfig
    from cryptopred.models.meta_report import format_meta_report

    text = format_meta_report(
        _scored(0.30), _scored(0.48), symbol="BTCUSDT", horizon=24,
        config=MetaConfig(), folds=[],
    )
    assert "BENCHMARK" not in text


def test_report_renders_with_a_real_config():
    """A 20-minute nested-CV run once died at the print because the report read
    a config field that had been renamed. The report must be exercised against
    the actual MetaConfig, not a stub that happens to carry both names."""
    from cryptopred.models.meta import MetaConfig
    from cryptopred.models.meta_report import format_meta_report

    scored = _scored(0.3)
    text = format_meta_report(
        scored, scored, symbol="BTCUSDT", horizon=24,
        config=MetaConfig(), folds=[{"fold": 0, "meta_trained": False, "n_primary": 10}],
        benchmark=scored, benchmark_threshold=0.08,
    )
    assert "Primary coverage" in text
    assert "top 8% by rank" in text


def test_an_unprovable_short_side_does_not_pass():
    """A run reached IMPROVEMENT with 6,732 long trades against 42 short ones.
    Being unable to judge the short side is not evidence for the strategy."""
    meta = _scored(1.20, decision="UNPROVEN")
    v = _verdict(_scored(0.94), meta, _scored(0.80))
    assert v.startswith("NO-GO")
    assert "both sides" in v


def test_a_stack_that_trades_more_is_not_compared_on_return_alone():
    """More exposure earns more in a rising sample without predicting anything."""
    meta = _scored(1.20)
    meta["coverage"] = 0.135
    benchmark = _scored(0.80)
    benchmark["coverage"] = 0.08
    v = _verdict(_scored(0.94), meta, benchmark)
    assert v.startswith("NOT COMPARABLE")


def test_a_matched_stack_that_genuinely_wins_still_passes():
    """The gates must not be unfailable in the other direction."""
    meta = _scored(1.20, drawdown=-0.15)
    meta["coverage"] = 0.085
    benchmark = _scored(0.80)
    benchmark["coverage"] = 0.08
    v = _verdict(_scored(0.94, drawdown=-0.20), meta, benchmark)
    assert v.startswith("IMPROVEMENT")
