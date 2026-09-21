"""Levels name the prices a user will ask about.

Their value is not that price respects them - untested - but that they turn
"where might it go" into a specific number that touch_probability can measure.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.levels import daily_pivots, swing_levels
from tests.conftest import make_ohlcv


def test_pivots_follow_the_classic_formula():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": np.concatenate([np.full(24, 110.0), np.full(24, 105.0)]),
            "low": np.concatenate([np.full(24, 90.0), np.full(24, 95.0)]),
            "close": np.concatenate([np.full(24, 100.0), np.full(24, 102.0)]),
        },
        index=idx,
    )
    piv = daily_pivots(bars)
    # previous day: H=110, L=90, C=100 -> P = 100
    assert piv["P"]["value"] == 100.0
    assert piv["R1"]["value"] == 110.0  # 2P - L
    assert piv["S1"]["value"] == 90.0  # 2P - H
    assert piv["R2"]["value"] == 120.0  # P + (H - L)
    assert piv["S2"]["value"] == 80.0  # P - (H - L)


def test_pivots_are_conventions():
    bars = make_ohlcv(n=200, seed=41)
    piv = daily_pivots(bars)
    assert all(v["source"] == "convention" for v in piv.values())


def test_a_single_day_of_data_gives_no_pivots():
    bars = make_ohlcv(n=10, seed=42)
    assert daily_pivots(bars) == {}


def test_swing_high_is_the_max_of_its_window():
    close = np.full(41, 100.0)
    high = np.full(41, 100.0)
    high[20] = 130.0
    idx = pd.date_range("2024-01-01", periods=41, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": close, "close": close}, index=idx
    )
    out = swing_levels(bars, k=5, lookback=41)
    assert 130.0 in [lvl["value"] for lvl in out["resistance"]]


def test_recent_unconfirmed_bars_produce_no_swing():
    """A swing needs k bars after it to be confirmed. The last k bars cannot
    have one yet, and inventing one would be a claim about bars that have not
    happened."""
    bars = make_ohlcv(n=200, seed=43)
    out = swing_levels(bars, k=5, lookback=200)
    last_five = bars.index[-5:]
    assert all(pd.Timestamp(lvl["at"]) not in last_five for lvl in out["resistance"])


def test_swing_levels_carry_the_same_warning_as_every_other_convention():
    """A consumer iterating over mixed payloads must not find some conventional
    values warned and others not."""
    bars = make_ohlcv(n=200, seed=44)
    out = swing_levels(bars, k=5, lookback=200)
    for side in ("resistance", "support"):
        for level in out[side]:
            assert level["source"] == "convention"
            assert level["validated"] is False
            assert "warning" in level


def test_the_forward_window_covers_the_next_k_bars_at_every_index():
    """pandas rolling counts rows, not available data.

    `shift(-k).rolling(k)` returns NaN for the first k-1 bars even though their
    forward windows are fully in-bounds. Today the left-hand condition already
    excludes those bars, so the defect changes no output — but the expression
    is wrong, and a future change to the left condition would expose it. This
    pins the helper directly rather than through behaviour it cannot yet affect.
    """
    from cryptopred.briefing.levels import _forward

    series = pd.Series(np.arange(20.0))
    k = 5
    expected = [
        series.iloc[i + 1 : i + 1 + k].max() if i + k < len(series) else np.nan
        for i in range(len(series))
    ]
    got = _forward(series, k, "max")
    for i, want in enumerate(expected):
        if np.isnan(want):
            assert pd.isna(got.iloc[i]), i
        else:
            assert got.iloc[i] == want, i


def test_the_final_k_bars_have_no_forward_window():
    """The reversal must not accidentally confirm bars that have no future."""
    n = 40
    high = np.full(n, 100.0)
    high[n - 2] = 130.0  # inside the unconfirmable tail
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = np.full(n, 100.0)
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": close, "close": close}, index=idx
    )
    out = swing_levels(bars, k=5, lookback=40)
    assert 130.0 not in [lvl["value"] for lvl in out["resistance"]]


def test_a_swing_needs_k_bars_on_both_sides():
    """A spike in the first k bars is not a k-swing: it has no left
    neighbourhood. Excluding it is correct, not a boundary bug."""
    n = 40
    high = np.full(n, 100.0)
    high[2] = 130.0
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = np.full(n, 100.0)
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": close, "close": close}, index=idx
    )
    out = swing_levels(bars, k=5, lookback=40)
    assert 130.0 not in [lvl["value"] for lvl in out["resistance"]]
