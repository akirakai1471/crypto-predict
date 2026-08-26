import pandas as pd

from cryptopred.features.momentum import momentum_features, rsi


def test_rsi_is_100_on_monotonic_rise(rising):
    r = rsi(rising["close"], period=14).dropna()
    assert (r > 99.9).all()


def test_rsi_is_bounded(ohlcv):
    r = rsi(ohlcv["close"], period=14).dropna()
    assert r.min() >= 0.0
    assert r.max() <= 100.0


def test_momentum_features_shape_and_names(ohlcv):
    feats = momentum_features(ohlcv)
    assert len(feats) == len(ohlcv)
    assert feats.index.equals(ohlcv.index)
    for name in ["rsi_14", "macd_hist", "roc_12", "ema_dist_50"]:
        assert name in feats.columns
    assert all(c.startswith(("rsi", "macd", "roc", "ema")) for c in feats.columns)


def test_momentum_features_have_no_infinities(ohlcv):
    feats = momentum_features(ohlcv)
    assert feats.replace([float("inf"), float("-inf")], pd.NA).notna().sum().sum() > 0
    assert not feats.isin([float("inf"), float("-inf")]).any().any()


def test_ema_distance_positive_in_uptrend(rising):
    feats = momentum_features(rising)
    assert feats["ema_dist_50"].dropna().iloc[-1] > 0
