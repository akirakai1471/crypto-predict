import numpy as np

from cryptopred.features.regime import adx, regime_features
from cryptopred.features.timefeat import time_features


def test_adx_is_bounded(ohlcv):
    a = adx(ohlcv, period=14).dropna()
    assert a.min() >= 0.0
    assert a.max() <= 100.0


def test_adx_high_in_strong_trend(rising):
    a = adx(rising, period=14).dropna()
    assert a.iloc[-1] > 40.0


def test_regime_features_names(ohlcv):
    feats = regime_features(ohlcv)
    for name in ["adx_14", "trend_strength", "vol_regime"]:
        assert name in feats.columns


def test_time_features_are_cyclical(ohlcv):
    feats = time_features(ohlcv)
    assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos"} <= set(feats.columns)
    # sin^2 + cos^2 == 1 for a proper cyclical encoding
    total = feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2
    assert np.allclose(total, 1.0)


def test_session_flags_are_mutually_exhaustive(ohlcv):
    feats = time_features(ohlcv)
    total = feats["session_asia"] + feats["session_europe"] + feats["session_us"]
    assert total.min() >= 1
    assert feats["session_asia"].isin([0, 1]).all()
