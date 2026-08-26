"""Leakage detection.

The point-in-time rule: a feature at bar t may only use data that closed at or
before t. These tests are the enforcement mechanism. If any of them fails,
STOP — do not train a model, do not report a metric. Find the leak first.
"""

import numpy as np
import pandas as pd
import pytest

from cryptopred.features.pipeline import build_features
from cryptopred.labels.barrier import make_labels
from tests.conftest import make_ohlcv


def test_features_are_prefix_invariant():
    """Computing features on a truncated history must give identical values for
    the rows both runs share. If truncation changes a value, that value depended
    on data from the future."""
    bars = make_ohlcv(n=1200, seed=1)
    cut = 800

    full = build_features(bars, interval="1h").iloc[:cut]
    partial = build_features(bars.iloc[:cut], interval="1h")

    # Compare the last 100 shared rows — enough to catch window bugs, cheap to run.
    pd.testing.assert_frame_equal(
        full.tail(100), partial.tail(100), check_freq=False, rtol=1e-9, atol=1e-12
    )


def test_future_bars_cannot_change_past_features():
    """Replace every bar after the cut with an extreme outlier. Features at and
    before the cut must be bit-identical."""
    bars = make_ohlcv(n=1000, seed=2)
    cut = 700

    poisoned = bars.copy()
    for col in ["open", "high", "low", "close"]:
        poisoned.iloc[cut:, poisoned.columns.get_loc(col)] = 1e9
    poisoned.iloc[cut:, poisoned.columns.get_loc("volume")] = 1e12

    clean_feats = build_features(bars, interval="1h").iloc[:cut]
    poisoned_feats = build_features(poisoned, interval="1h").iloc[:cut]

    pd.testing.assert_frame_equal(
        clean_feats.tail(100),
        poisoned_feats.tail(100),
        check_freq=False,
        rtol=1e-9,
        atol=1e-12,
    )


def test_no_feature_is_suspiciously_correlated_with_the_label():
    """A feature correlating above 0.30 with the forward return is a red flag.
    Real technical indicators land far below that on hourly crypto data."""
    bars = make_ohlcv(n=3000, seed=3)
    feats = build_features(bars, interval="1h")
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)

    joined = feats.join(labels[["forward_return"]]).dropna()
    correlations = joined.corr()["forward_return"].drop("forward_return").abs()

    offenders = correlations[correlations > 0.30].sort_values(ascending=False)
    assert offenders.empty, f"Suspicious correlation with the future:\n{offenders}"


def test_label_uses_only_future_and_feature_only_past():
    """Explicit check of the contract: forward_return at t is derived from
    close[t+H], and is NaN for the final H bars where the future is unknown."""
    bars = make_ohlcv(n=100, seed=4)
    horizon = 4
    labels = make_labels(bars, horizon=horizon, atr_period=14, band_k=0.5)

    expected = bars["close"].iloc[10 + horizon] / bars["close"].iloc[10] - 1
    assert labels["forward_return"].iloc[10] == pytest.approx(expected)
    assert labels["forward_return"].iloc[-horizon:].isna().all()
    assert labels["label"].iloc[-horizon:].isna().all()


def test_shuffled_labels_destroy_all_signal():
    """Sanity check on the joining logic: if labels are shuffled, correlation
    with every feature must collapse. If it does not, features and labels are
    being aligned by position somewhere instead of by index."""
    bars = make_ohlcv(n=3000, seed=5)
    feats = build_features(bars, interval="1h")
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)

    joined = feats.join(labels[["forward_return"]]).dropna()
    rng = np.random.default_rng(0)
    joined["forward_return"] = rng.permutation(joined["forward_return"].to_numpy())

    correlations = joined.corr()["forward_return"].drop("forward_return").abs()
    assert correlations.max() < 0.15


def test_features_never_reference_the_next_bar():
    """Regression guard: append one extra bar and confirm the previously-final
    row's features do not change."""
    bars = make_ohlcv(n=600, seed=6)
    shorter = bars.iloc[:-1]

    long_feats = build_features(bars, interval="1h").iloc[:-1]
    short_feats = build_features(shorter, interval="1h")

    pd.testing.assert_frame_equal(
        long_feats.tail(5), short_feats.tail(5), check_freq=False, rtol=1e-9, atol=1e-12
    )
