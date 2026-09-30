"""docs/preregistration-cost-aware-selection.md: the rule, and the verdict on it."""

import numpy as np
import pandas as pd
import pytest

from cryptopred.models import experiment
from cryptopred.models.selection import (
    realised_volatility,
    signals_by_quantile,
    signals_by_score,
)
from cryptopred.models.validation import SymbolResult
from tests.conftest import make_ohlcv


def _proba(margins, flat=0.1):
    """Rows with the given UP-minus-DOWN margins, all leaning UP."""
    margins = np.asarray(margins, dtype=float)
    up = (1 - flat + margins) / 2
    down = 1 - flat - up
    return np.column_stack([down, np.full(len(margins), flat), up])


# -- the rule -----------------------------------------------------------------


def test_without_a_weight_the_rule_is_exactly_the_old_rank_rule():
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1, 1, 1], size=500)
    np.testing.assert_array_equal(
        signals_by_quantile(raw, 0.08), signals_by_score(raw, 0.08, weight=None)
    )


def test_a_quieter_bar_with_a_wider_margin_can_lose_its_place_to_a_louder_one():
    """The point of the rule: a 0.30 edge on a 1% move beats a 0.40 edge on a
    0.2% move, because only one of them is likely to clear the fees."""
    proba = _proba([0.40, 0.30, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    vol = np.array([0.002, 0.010, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
    assert signals_by_score(proba, 0.1)[0] == 1  # by margin alone, bar 0
    by_move = signals_by_score(proba, 0.1, weight=vol)
    assert by_move[1] == 1 and by_move[0] == 0


def test_a_bar_without_volatility_history_is_never_chosen():
    proba = _proba([0.9, 0.1, 0.1, 0.1])
    vol = np.array([np.nan, 0.01, 0.01, 0.01])
    assert signals_by_score(proba, 0.25, weight=vol)[0] == 0


def test_flat_bars_stay_untradeable_however_volatile():
    proba = np.array([[0.1, 0.8, 0.1], [0.3, 0.2, 0.5]])
    vol = np.array([1.0, 0.001])
    assert list(signals_by_score(proba, 0.5, weight=vol)) == [0, 1]


def test_volatility_uses_only_returns_up_to_the_bar():
    """Poison every close after bar 300: vol at bar 300 and before must not move."""
    bars = make_ohlcv(n=500, seed=4)
    poisoned = bars["close"].copy()
    poisoned.iloc[301:] = 1e9
    clean = realised_volatility(bars["close"], 72).iloc[:301]
    dirty = realised_volatility(poisoned, 72).iloc[:301]
    pd.testing.assert_series_equal(clean, dirty)
    assert clean.iloc[:72].isna().all() and clean.iloc[72:].notna().all()


# -- the verdict --------------------------------------------------------------


def _result(symbol, doubled, sharpe=0.5, two_sided="TWO-SIDED", passed=False):
    detail = {
        "doubled_cost_return": doubled,
        "sharpe": sharpe,
        "two_sided": two_sided,
        "total_return": 0.1 if passed else -0.1,
        "survives_doubled_costs": passed,
    }
    return SymbolResult(symbol, status="tested", detail=detail)


def _pairs(base_doubled, arm_doubled, **arm_kwargs):
    symbols = [f"S{i}USDT" for i in range(len(base_doubled))]
    base = [_result(s, d) for s, d in zip(symbols, base_doubled, strict=True)]
    arm = [_result(s, d, **arm_kwargs) for s, d in zip(symbols, arm_doubled, strict=True)]
    return base, arm


def test_fifteen_of_twenty_better_at_doubled_costs_is_adopted():
    base, arm = _pairs([-0.10] * 20, [0.05] * 15 + [-0.20] * 5)
    assert experiment.judge_selection(base, arm)["decision"] == "ADOPTED"


def test_fourteen_of_twenty_is_rejected():
    base, arm = _pairs([-0.10] * 20, [0.50] * 14 + [-0.20] * 6)
    v = experiment.judge_selection(base, arm)
    assert not v["criteria"]["costs"] and v["decision"] == "REJECTED"


def test_more_return_bought_with_more_risk_is_rejected():
    """Trading bigger moves with a fixed stake is a bigger bet. If Sharpe falls,
    the extra return is the extra risk, not a better rule."""
    base, arm = _pairs([-0.10] * 20, [0.05] * 20, sharpe=0.4)
    v = experiment.judge_selection(base, arm)
    assert v["criteria"]["costs"] and not v["criteria"]["sharpe"]
    assert v["decision"] == "REJECTED"


def test_drifting_into_one_side_is_rejected():
    base, arm = _pairs([-0.10] * 20, [0.05] * 20, two_sided="ONE-SIDED")
    v = experiment.judge_selection(base, arm)
    assert not v["criteria"]["two_sided"] and v["decision"] == "REJECTED"


def test_the_report_names_the_verdict_and_every_criterion():
    base, arm = _pairs([-0.10] * 3, [0.05] * 3)
    text = experiment.format_selection_experiment({"margin": base, "margin_x_vol": arm})
    assert "margin_x_vol: REJECTED" in text  # 3 of 3 cannot reach p <= 0.05
    for line in ("survives costs", "bigger bets", "gate passes", "long-bias"):
        assert line in text


def test_one_symbol_runs_both_rules_on_one_model(tmp_path, monkeypatch):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore
    from cryptopred.models import validation
    from cryptopred.models.validation import FrozenConfig

    monkeypatch.setattr(validation, "MIN_BARS", 1000)
    cfg = Config()
    cfg.data.root = tmp_path
    ParquetStore(tmp_path / "raw").write("klines", "SOLUSDT", "1h", make_ohlcv(n=3000, seed=41))

    out = experiment.run_selection_symbol(
        "SOLUSDT", cfg, "1h", FrozenConfig(rounds=10, calibration_splits=2)
    )
    margin, by_move = out["margin"], out["margin_x_vol"]
    assert margin.status == by_move.status == "tested"
    # One model: identical probabilities, so identical log loss and trade count.
    assert margin.detail["log_loss"] == by_move.detail["log_loss"]
    assert margin.detail["n_signals"] == pytest.approx(by_move.detail["n_signals"], abs=5)
    # Nothing is asserted about which rule picks bigger moves: the synthetic walk
    # has constant volatility, so past volatility predicts nothing here. Whether
    # it does in real markets is what the experiment measures.
