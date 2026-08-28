import numpy as np
import pandas as pd
import pytest

from cryptopred.models.train import TrainConfig, train_fold, walk_forward_evaluate


def _learnable_dataset(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A dataset with a real, learnable signal.

    `signal` genuinely predicts the class, `noise_*` do not. A working trainer
    must beat the baselines here; if it cannot learn this, the bug is in the
    trainer, not the market.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    signal = rng.normal(0, 1, n)
    # class follows the signal with 20% label noise
    base = np.digitize(signal, [-0.5, 0.5])
    flip = rng.random(n) < 0.2
    label_class = np.where(flip, rng.integers(0, 3, n), base)

    return pd.DataFrame(
        {
            "signal": signal,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "roc_1": rng.normal(0, 0.01, n),
            "label_class": label_class.astype("int8"),
            "label": label_class - 1.0,
            "forward_return": rng.normal(0, 0.01, n),
            "band": 0.001,
        },
        index=idx,
    )


def test_train_fold_returns_calibrated_probabilities():
    df = _learnable_dataset()
    cfg = TrainConfig(num_boost_round=40, calibrate=True)
    result = train_fold(df.iloc[:3000], df.iloc[3000:], cfg)

    assert result.proba.shape == (1000, 3)
    assert np.allclose(result.proba.sum(axis=1), 1.0)
    assert (result.proba >= 0).all()


def test_trained_model_beats_coin_flip_on_learnable_data():
    df = _learnable_dataset()
    report = walk_forward_evaluate(df, n_splits=3, horizon=4, config=TrainConfig(
        num_boost_round=60, calibrate=True
    ))

    model_acc = report["model"]["directional_accuracy"]
    coin_acc = report["baselines"]["coin_flip"]["directional_accuracy"]
    assert model_acc > coin_acc


def test_walk_forward_reports_every_baseline():
    df = _learnable_dataset(n=3000)
    report = walk_forward_evaluate(df, n_splits=3, horizon=4, config=TrainConfig(
        num_boost_round=20
    ))
    for name in ["coin_flip", "always_up", "prior", "momentum"]:
        assert name in report["baselines"]


def test_walk_forward_records_per_fold_detail():
    df = _learnable_dataset(n=3000)
    report = walk_forward_evaluate(df, n_splits=3, horizon=4, config=TrainConfig(
        num_boost_round=20
    ))
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["test_start"] < fold["test_end"]
        assert fold["n_test"] > 0


def test_feature_importance_ranks_the_real_signal_first():
    df = _learnable_dataset()
    cfg = TrainConfig(num_boost_round=60, calibrate=False)
    result = train_fold(df.iloc[:3000], df.iloc[3000:], cfg)
    top = max(result.importance, key=result.importance.get)
    assert top == "signal"


def test_pure_noise_gives_no_edge_over_prior():
    """The honest negative case: with no signal, the model must not appear to
    win. This is the test that catches a leaking or self-fooling pipeline."""
    rng = np.random.default_rng(7)
    n = 4000
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    df = pd.DataFrame(
        {
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "roc_1": rng.normal(0, 0.01, n),
            "label_class": rng.integers(0, 3, n).astype("int8"),
            "label": 0.0,
            "forward_return": rng.normal(0, 0.01, n),
            "band": 0.001,
        },
        index=idx,
    )
    report = walk_forward_evaluate(df, n_splits=3, horizon=4, config=TrainConfig(
        num_boost_round=40, calibrate=True
    ))
    model_acc = report["model"]["accuracy"]
    # random labels over 3 classes: anything much above chance means leakage
    assert model_acc < 0.45


def test_rejects_dataset_without_label_class():
    df = _learnable_dataset(n=500).drop(columns=["label_class"])
    with pytest.raises(KeyError, match="label_class"):
        walk_forward_evaluate(df, n_splits=2, horizon=4, config=TrainConfig())


def test_oof_calibration_spreads_across_the_window_not_one_tail_block():
    """Regression for a real failure: a final model calibrated on one recent
    block produced a near-constant isotonic map, capping its UP probability at
    0.45 and firing on 0.3% of bars where the folds fired on 8.4%."""
    from cryptopred.dataset.builder import feature_columns
    from cryptopred.models.train import fit_oof_calibrators

    df = _learnable_dataset(n=6000)
    cfg = TrainConfig(num_boost_round=30, calibration_splits=3)
    features = feature_columns(df)

    calibrators = fit_oof_calibrators(df, features, horizon=4, config=cfg)
    assert calibrators is not None and len(calibrators) == 3

    # A usable calibrator must still separate low from high inputs. The broken
    # one returned the same value across almost the whole input range.
    grid = np.linspace(0.05, 0.95, 19)
    for cls, iso in enumerate(calibrators):
        out = iso.predict(grid)
        assert out.max() - out.min() > 0.05, f"class {cls} calibrator is nearly flat"


def test_oof_calibrated_model_keeps_a_usable_signal_rate():
    df = _learnable_dataset(n=6000)
    cfg = TrainConfig(num_boost_round=40, calibration_method="oof", calibration_splits=3)
    result = train_fold(df.iloc[:4500], df.iloc[4500:], cfg, horizon=4)

    confidence = result.proba.max(axis=1)
    assert confidence.max() > 0.6      # the map is not squashed flat


def test_coverage_check_passes_when_the_model_matches_the_folds():
    from cryptopred.models.train import coverage_check

    df = _learnable_dataset(n=4000)
    cfg = TrainConfig(num_boost_round=40, calibration_method="oof", calibration_splits=3)
    result = train_fold(df.iloc[:3000], df.iloc[3000:], cfg, horizon=4)

    from cryptopred.models.selection import margin_cutoff, signals_from_margin

    cutoff = margin_cutoff(result.proba, coverage=0.10)
    observed = float((signals_from_margin(result.proba, cutoff) != 0).mean())

    check = coverage_check(result, df.iloc[3000:], observed, cutoff=cutoff)
    assert check["ok"]
    assert check["ratio"] < 3.0


def test_coverage_check_catches_a_model_that_stopped_firing():
    """The exact failure that shipped: metrics from folds, a saved model that
    almost never signals."""
    from cryptopred.models.train import coverage_check

    df = _learnable_dataset(n=4000)
    cfg = TrainConfig(num_boost_round=40, calibration_method="oof", calibration_splits=3)
    result = train_fold(df.iloc[:3000], df.iloc[3000:], cfg, horizon=4)

    # Folds claimed 8.4% coverage; a cutoff nothing can reach means the saved
    # model fires on nothing, which is the failure that shipped.
    check = coverage_check(result, df.iloc[3000:], expected_coverage=0.084, cutoff=0.99)
    assert not check["ok"]


def test_coverage_check_reports_both_rates_for_the_reader():
    from cryptopred.models.train import coverage_check

    df = _learnable_dataset(n=3000)
    cfg = TrainConfig(num_boost_round=20, calibration_method="oof", calibration_splits=3)
    result = train_fold(df.iloc[:2200], df.iloc[2200:], cfg, horizon=4)

    from cryptopred.models.selection import margin_cutoff

    cutoff = margin_cutoff(result.proba, coverage=0.08)
    check = coverage_check(result, df, expected_coverage=0.08, cutoff=cutoff)
    assert "expected_coverage" in check and "actual_coverage" in check
    assert check["n_sample"] > 0
