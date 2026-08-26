import pandas as pd
import pytest

from cryptopred.features.derivatives import funding_features


@pytest.fixture
def funding() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="8h", tz="UTC", name="funding_time")
    return pd.DataFrame({"funding_rate": [0.0001] * 50 + [0.0005] * 50}, index=idx)


def test_funding_features_align_to_bar_index(ohlcv, funding):
    feats = funding_features(ohlcv, funding)
    assert feats.index.equals(ohlcv.index)
    assert "funding_rate" in feats.columns


def test_funding_uses_only_already_published_values(ohlcv, funding):
    """A funding value stamped at 08:00 must not appear on bars before 08:00."""
    feats = funding_features(ohlcv, funding)
    first_bar = ohlcv.index[0]
    first_funding = funding.index[0]
    if first_bar < first_funding:
        assert pd.isna(feats.loc[first_bar, "funding_rate"])


def test_funding_features_when_empty(ohlcv):
    empty = pd.DataFrame(columns=["funding_rate"])
    empty.index = pd.DatetimeIndex([], tz="UTC", name="funding_time")
    feats = funding_features(ohlcv, empty)
    assert feats.index.equals(ohlcv.index)
    assert feats["funding_rate"].isna().all()


def test_funding_ma_reflects_regime_change(ohlcv, funding):
    feats = funding_features(ohlcv, funding)
    tail = feats["funding_rate"].dropna()
    assert tail.iloc[-1] == pytest.approx(0.0005)
