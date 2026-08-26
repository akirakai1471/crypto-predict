import pandas as pd
import pytest

from cryptopred.features.structure import consecutive_streak, structure_features


def test_consecutive_streak_counts_direction():
    s = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
    streak = consecutive_streak(s)
    assert streak.iloc[2] == 2      # two consecutive rises
    assert streak.iloc[4] == -2     # two consecutive falls
    assert streak.iloc[5] == 1


def test_close_position_in_range_is_bounded(ohlcv):
    feats = structure_features(ohlcv)
    pos = feats["close_pos_in_bar"].dropna()
    assert pos.min() >= 0.0
    assert pos.max() <= 1.0


def test_distance_to_high_is_non_positive(ohlcv):
    """Close can never exceed the rolling max that includes the current bar."""
    feats = structure_features(ohlcv)
    assert feats["dist_to_high_24"].dropna().max() <= 1e-9


def test_structure_features_names(ohlcv):
    feats = structure_features(ohlcv)
    for name in ["close_pos_in_bar", "dist_to_high_24", "dist_to_low_24", "streak"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_doji_bar_does_not_produce_nan():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    feats = structure_features(df)
    assert feats["close_pos_in_bar"].iloc[0] == pytest.approx(0.5)
