import numpy as np
import pandas as pd
import pytest

from cryptopred.features.volatility import atr, true_range, volatility_features


def test_true_range_uses_previous_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"high": [10.0, 12.0], "low": [8.0, 11.0], "close": [9.0, 11.5]}, index=idx
    )
    tr = true_range(df)
    # bar 1: max(12-11, |12-9|, |11-9|) = 3
    assert tr.iloc[1] == pytest.approx(3.0)


def test_atr_is_positive(ohlcv):
    a = atr(ohlcv, period=14).dropna()
    assert (a > 0).all()


def test_atr_rises_with_volatility():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    spread = pd.Series([1.0] * 100 + [10.0] * 100, index=idx)
    df = pd.DataFrame({"high": close + spread, "low": close - spread, "close": close})
    a = atr(df, period=14)
    assert a.iloc[-1] > a.iloc[99] * 3


def test_volatility_features_names(ohlcv):
    feats = volatility_features(ohlcv)
    for name in ["atr_14_norm", "bb_width_20", "realized_vol_24", "vol_ratio_24_72"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_volatility_features_finite(ohlcv):
    feats = volatility_features(ohlcv)
    assert not np.isinf(feats.to_numpy(dtype="float64", na_value=0.0)).any()
