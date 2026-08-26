import pandas as pd
import pytest

from cryptopred.features.mtf import mtf_features, resample_ohlcv


def test_resample_indexes_by_close_time(ohlcv):
    htf = resample_ohlcv(ohlcv, "4h")
    # right-labelled: the bar covering 00:00-04:00 is stamped 04:00
    assert htf.index[0] == pd.Timestamp("2024-01-01 04:00", tz="UTC")
    assert htf["high"].iloc[0] == pytest.approx(ohlcv["high"].iloc[0:4].max())
    assert htf["low"].iloc[0] == pytest.approx(ohlcv["low"].iloc[0:4].min())
    assert htf["close"].iloc[0] == pytest.approx(ohlcv["close"].iloc[3])


def test_mtf_features_do_not_use_unclosed_higher_bars(ohlcv):
    feats = mtf_features(ohlcv, rule="4h", prefix="h4")
    # bars before the first 4h close have nothing to inherit
    first_valid = feats["h4_rsi_14"].first_valid_index()
    assert first_valid is not None
    assert first_valid >= ohlcv.index[3]


def test_mtf_values_are_constant_within_a_higher_bar(ohlcv):
    feats = mtf_features(ohlcv, rule="4h", prefix="h4")
    window = feats["h4_close_ret_1"].iloc[100:104]
    assert window.nunique(dropna=True) <= 1


def test_mtf_prefixes_all_columns(ohlcv):
    feats = mtf_features(ohlcv, rule="1D", prefix="d1")
    assert all(c.startswith("d1_") for c in feats.columns)
    assert feats.index.equals(ohlcv.index)


def test_mtf_is_prefix_invariant(ohlcv):
    """Truncating the input must not change already-computed values."""
    cut = 500
    full = mtf_features(ohlcv, rule="4h", prefix="h4").iloc[:cut]
    partial = mtf_features(ohlcv.iloc[:cut], rule="4h", prefix="h4")
    pd.testing.assert_frame_equal(full.tail(50), partial.tail(50), check_freq=False)
