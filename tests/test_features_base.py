import numpy as np
import pandas as pd
import pytest

from cryptopred.features.base import pct_rank, rolling_zscore, safe_divide


def test_rolling_zscore_uses_only_past():
    s = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
    z = rolling_zscore(s, window=3, min_periods=3)
    # index 2 sees [1,2,3] only; the 100 spike at index 3 must not affect it
    expected = (3.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1)
    assert z.iloc[2] == pytest.approx(expected)
    assert pd.isna(z.iloc[0])
    assert pd.isna(z.iloc[1])


def test_rolling_zscore_constant_series_is_zero_not_inf():
    s = pd.Series([5.0] * 10)
    z = rolling_zscore(s, window=5, min_periods=5)
    assert (z.dropna() == 0).all()
    assert np.isfinite(z.dropna()).all()


def test_safe_divide_handles_zero_denominator():
    num = pd.Series([1.0, 2.0, 3.0])
    den = pd.Series([1.0, 0.0, 3.0])
    out = safe_divide(num, den)
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert np.isfinite(out).all()


def test_pct_rank_is_backward_looking():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pct_rank(s, window=3)
    # at index 4 the window is [3,4,5]; 5 is the max -> rank 1.0
    assert r.iloc[4] == pytest.approx(1.0)
    assert pd.isna(r.iloc[0])
