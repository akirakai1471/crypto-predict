import pandas as pd

from cryptopred.features.pipeline import build_features


def test_build_features_returns_wide_frame(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert feats.index.equals(ohlcv.index)
    assert len(feats.columns) >= 60


def test_build_features_column_names_are_unique(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert len(feats.columns) == len(set(feats.columns))


def test_build_features_all_float64(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    non_float = [c for c in feats.columns if feats[c].dtype != "float64"]
    assert non_float == []


def test_build_features_contains_no_raw_price_columns(ohlcv):
    """Raw price would let a model memorise date ranges instead of learning."""
    feats = build_features(ohlcv, interval="1h")
    for banned in ["open", "high", "low", "close", "volume", "close_time"]:
        assert banned not in feats.columns


def test_build_features_with_funding(ohlcv):
    idx = pd.date_range("2024-01-01", periods=200, freq="8h", tz="UTC", name="funding_time")
    funding = pd.DataFrame({"funding_rate": [0.0001] * 200}, index=idx)
    feats = build_features(ohlcv, interval="1h", funding=funding)
    assert "funding_rate" in feats.columns


def test_build_features_1m_uses_1m_mtf_rules(ohlcv_1m):
    feats = build_features(ohlcv_1m, interval="1m")
    assert any(c.startswith("m15_") for c in feats.columns)
    assert any(c.startswith("h1_") for c in feats.columns)


def test_no_infinite_values(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert not feats.isin([float("inf"), float("-inf")]).any().any()
