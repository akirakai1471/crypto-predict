import pandas as pd
import pytest

from cryptopred.features.volume import obv, volume_features


def test_obv_accumulates_on_up_bars():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 100.0, 102.0], "volume": [10.0, 20.0, 30.0, 40.0]}, index=idx
    )
    o = obv(df)
    assert o.iloc[0] == 0.0
    assert o.iloc[1] == pytest.approx(20.0)      # up bar: +20
    assert o.iloc[2] == pytest.approx(-10.0)     # down bar: -30
    assert o.iloc[3] == pytest.approx(30.0)      # up bar: +40


def test_taker_imbalance_is_bounded(ohlcv):
    feats = volume_features(ohlcv)
    imb = feats["taker_imbalance"].dropna()
    assert imb.min() >= -1.0
    assert imb.max() <= 1.0


def test_volume_features_names(ohlcv):
    feats = volume_features(ohlcv)
    for name in ["volume_z_200", "obv_slope_24", "taker_imbalance", "trades_z_200"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_volume_features_survive_zero_volume_bar(ohlcv):
    df = ohlcv.copy()
    df.iloc[10, df.columns.get_loc("volume")] = 0.0
    df.iloc[10, df.columns.get_loc("taker_buy_base")] = 0.0
    feats = volume_features(df)
    assert feats.loc[df.index[10], "taker_imbalance"] == 0.0
