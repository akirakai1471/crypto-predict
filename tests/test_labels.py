import numpy as np
import pandas as pd
import pytest

from cryptopred.labels.barrier import LABEL_DOWN, LABEL_FLAT, LABEL_UP, make_labels
from tests.conftest import make_ohlcv


def _flat_market(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": 1.0,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


def test_forward_return_is_computed_from_horizon_bars_ahead():
    bars = make_ohlcv(n=50, seed=11)
    labels = make_labels(bars, horizon=3, atr_period=14, band_k=0.5)
    expected = bars["close"].iloc[20 + 3] / bars["close"].iloc[20] - 1
    assert labels["forward_return"].iloc[20] == pytest.approx(expected)


def test_last_horizon_rows_are_nan():
    bars = make_ohlcv(n=50, seed=12)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert labels["forward_return"].iloc[-4:].isna().all()
    assert labels["label"].iloc[-4:].isna().all()


def test_flat_market_is_all_flat():
    bars = _flat_market()
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    valid = labels["label"].dropna()
    assert (valid == LABEL_FLAT).all()


def test_strong_rise_is_labelled_up():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
            "close_time": idx + pd.Timedelta(hours=1),
        },
        index=idx,
    )
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert (labels["label"].dropna() == LABEL_UP).all()


def test_band_scales_with_k():
    bars = make_ohlcv(n=500, seed=13)
    narrow = make_labels(bars, horizon=4, atr_period=14, band_k=0.1)
    wide = make_labels(bars, horizon=4, atr_period=14, band_k=2.0)
    assert (wide["label"] == LABEL_FLAT).sum() > (narrow["label"] == LABEL_FLAT).sum()


def test_label_values_are_only_minus_one_zero_one():
    bars = make_ohlcv(n=500, seed=14)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert set(labels["label"].dropna().unique()) <= {LABEL_DOWN, LABEL_FLAT, LABEL_UP}


def test_columns_present():
    bars = make_ohlcv(n=100, seed=15)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert list(labels.columns) == ["forward_return", "band", "label"]
    assert labels.index.equals(bars.index)


def test_a_label_does_not_reach_across_a_hole_in_the_data():
    """Six symbols' archives miss days in 2022. Labels looked ahead by rows, so
    beside a hole a "24-bar" label measured a move of up to 96 hours."""
    from cryptopred.labels.barrier import make_labels, make_triple_barrier_labels

    bars = make_ohlcv(n=300, seed=3)
    holed = bars.drop(bars.index[150:200])  # fifty hours missing
    labels = make_labels(holed, horizon=24)
    before = holed.index[holed.index < bars.index[150]]
    across = before[-24:]   # their 24th row ahead lies past the hole
    clear = before[-60:-24]
    assert labels.loc[across, "label"].isna().all()
    assert labels.loc[clear, "label"].notna().all()
    assert make_triple_barrier_labels(holed, horizon=24).loc[across, "label"].isna().all()
    # Unbroken data labels exactly as before.
    assert make_labels(bars, horizon=24)["label"].iloc[:-24].notna().sum() >= 250
