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
