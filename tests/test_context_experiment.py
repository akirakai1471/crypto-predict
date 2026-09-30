"""The two changes pre-registered in docs/preregistration-improvements.md."""

import numpy as np
import pandas as pd

from cryptopred.dataset.builder import build_dataset, feature_columns
from cryptopred.features.context import CONTEXT_COLUMNS, context_features, context_symbol
from cryptopred.models import experiment
from cryptopred.models.train import TrainConfig, recency_weights, walk_forward_evaluate
from cryptopred.models.validation import SymbolResult
from tests.conftest import make_ohlcv
from tests.test_train import _learnable_dataset

# -- recency ------------------------------------------------------------------


def test_the_newest_row_weighs_one_and_a_half_life_back_weighs_half():
    w = recency_weights(10_000, half_life=8760)
    assert w[-1] == 1.0
    assert np.isclose(w[-1 - 8760], 0.5)
    assert np.all(np.diff(w) > 0)  # strictly heavier toward the present


def test_no_half_life_means_no_weights():
    assert recency_weights(100, None) is None


def test_recency_weighting_changes_the_model_and_nothing_else():
    """Same folds, same rows, same labels; only the fitted booster may differ."""
    df = _learnable_dataset(n=3000)
    plain = walk_forward_evaluate(
        df, n_splits=2, horizon=4, config=TrainConfig(num_boost_round=20, calibration_splits=2)
    )
    weighted = walk_forward_evaluate(
        df,
        n_splits=2,
        horizon=4,
        config=TrainConfig(num_boost_round=20, calibration_splits=2, recency_half_life=200),
    )
    np.testing.assert_array_equal(plain["y_true"], weighted["y_true"])
    assert not np.array_equal(plain["proba"], weighted["proba"])


# -- market context -----------------------------------------------------------


def test_context_features_are_the_five_pre_registered_ones():
    bars, context = make_ohlcv(n=400, seed=1), make_ohlcv(n=400, seed=2)
    assert tuple(context_features(bars, context).columns) == CONTEXT_COLUMNS


def test_btc_is_the_market_for_everyone_but_itself():
    assert context_symbol("SOLUSDT") == "BTCUSDT"
    assert context_symbol("ETHUSDT") == "BTCUSDT"
    assert context_symbol("BTCUSDT") == "ETHUSDT"


def test_context_adds_columns_but_never_removes_rows():
    """If the context symbol has a shorter history, its features are NaN there,
    not grounds to drop the row. Otherwise the two arms would be tested on
    different bars and the comparison would measure the sample, not the model."""
    bars = make_ohlcv(n=2000, seed=5)
    context = make_ohlcv(n=2000, seed=6).iloc[1000:]  # listed 1,000 bars later

    plain = build_dataset(bars, interval="1h", horizon=24)
    with_ctx = build_dataset(bars, interval="1h", horizon=24, context_bars=context)

    pd.testing.assert_index_equal(plain.index, with_ctx.index)
    assert set(CONTEXT_COLUMNS) <= set(feature_columns(with_ctx))
    assert with_ctx["ctx_ret_24"].isna().any()
    assert with_ctx["ctx_ret_24"].notna().any()


# -- the decision rule --------------------------------------------------------


def _result(symbol, sign, log_loss=1.0, passed=False):
    detail = {"sign_accuracy": sign, "log_loss": log_loss}
    if passed:
        detail.update(total_return=0.1, survives_doubled_costs=True, two_sided="TWO-SIDED")
    return SymbolResult(symbol, status="tested", detail=detail)


def _runs(base_signs, arm_signs, **arm_kwargs):
    symbols = [f"S{i}USDT" for i in range(len(base_signs))]
    base = [_result(s, b) for s, b in zip(symbols, base_signs, strict=True)]
    arm = [_result(s, a, **arm_kwargs) for s, a in zip(symbols, arm_signs, strict=True)]
    return base, arm


def test_fifteen_of_twenty_by_a_point_is_adopted():
    base, arm = _runs([0.53] * 20, [0.54] * 15 + [0.525] * 5)
    v = experiment.judge(base, arm)
    assert v["wins"] == 15 and v["needed"] == 15
    assert v["decision"] == "ADOPTED"


def test_fourteen_of_twenty_is_rejected_however_large_the_gain():
    """The rule was written before running. A big average carried by a few
    symbols is exactly what it exists to refuse."""
    base, arm = _runs([0.53] * 20, [0.60] * 14 + [0.52] * 6)
    v = experiment.judge(base, arm)
    assert v["mean_gain"] > 0.005
    assert v["decision"] == "REJECTED"
    assert not v["criteria"]["wins"]


def test_winning_everywhere_by_too_little_is_rejected():
    base, arm = _runs([0.53] * 20, [0.532] * 20)
    v = experiment.judge(base, arm)
    assert v["criteria"]["wins"] and not v["criteria"]["margin"]
    assert v["decision"] == "REJECTED"


def test_better_traded_bars_bought_with_worse_probabilities_is_rejected():
    base, arm = _runs([0.53] * 20, [0.55] * 20, log_loss=1.01)
    v = experiment.judge(base, arm)
    assert not v["criteria"]["log_loss"]
    assert v["decision"] == "REJECTED"


def test_losing_gate_passes_is_rejected():
    symbols = [f"S{i}USDT" for i in range(20)]
    base = [_result(s, 0.53, passed=(i < 5)) for i, s in enumerate(symbols)]
    arm = [_result(s, 0.55, passed=(i < 4)) for i, s in enumerate(symbols)]
    v = experiment.judge(base, arm)
    assert (v["gates_base"], v["gates_arm"]) == (5, 4)
    assert v["decision"] == "REJECTED"


def test_a_symbol_skipped_on_either_side_counts_for_neither():
    base, arm = _runs([0.53] * 20, [0.54] * 20)
    arm[0] = SymbolResult("S0USDT", status="skipped: no BTCUSDT bars for context")
    v = experiment.judge(base, arm)
    assert v["n"] == 19 and v["needed"] == 15  # ceil(0.75 * 19)


def test_the_sign_test_matches_the_number_written_in_the_preregistration():
    assert round(experiment.sign_test_p(15, 20), 3) == 0.021


def test_an_arm_runs_end_to_end_on_stored_bars(tmp_path, monkeypatch):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore
    from cryptopred.models import validation
    from cryptopred.models.validation import FrozenConfig

    monkeypatch.setattr(validation, "MIN_BARS", 1000)
    cfg = Config()
    cfg.data.root = tmp_path
    store = ParquetStore(tmp_path / "raw")
    store.write("klines", "SOLUSDT", "1h", make_ohlcv(n=3000, seed=31))
    store.write("klines", "BTCUSDT", "1h", make_ohlcv(n=3000, seed=32))
    frozen = FrozenConfig(rounds=10, calibration_splits=2)

    for arm in ("baseline", "recency", "market_context"):
        result = experiment.run_arm_symbol("SOLUSDT", cfg, "1h", frozen, arm)
        assert result.status == "tested", (arm, result.status)
        assert result.detail["log_loss"] > 0


def test_market_context_without_the_context_symbol_is_a_skip_not_a_crash(tmp_path):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore
    from cryptopred.models.validation import FrozenConfig

    cfg = Config()
    cfg.data.root = tmp_path
    ParquetStore(tmp_path / "raw").write("klines", "SOLUSDT", "1h", make_ohlcv(n=500, seed=33))
    result = experiment.run_arm_symbol("SOLUSDT", cfg, "1h", FrozenConfig(), "market_context")
    assert result.status.startswith("skipped") and "BTCUSDT" in result.status
