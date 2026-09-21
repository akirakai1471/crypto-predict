"""A level that was traded was touched.

Using closes instead of intrabar extremes is the easiest detail to get wrong
here, and it biases every probability downward.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.touch import touch_outcomes


def _bars(close, low=None, high=None):
    n = len(close)
    close = np.asarray(close, dtype=float)
    low = np.asarray(low, dtype=float) if low is not None else close
    high = np.asarray(high, dtype=float) if high is not None else close
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_a_series_that_always_falls_enough_returns_one():
    close = np.full(50, 100.0)
    low = np.full(50, 90.0)  # every bar dips 10%
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=5)
    assert touched.mean() == 1.0


def test_a_series_that_never_falls_enough_returns_zero():
    close = np.full(50, 100.0)
    low = np.full(50, 99.0)  # never more than 1% down
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=5)
    assert touched.mean() == 0.0


def test_an_intrabar_wick_counts_even_when_the_close_does_not():
    """The bar that pierces the level and recovers still traded there."""
    close = np.full(20, 100.0)
    low = np.full(20, 100.0)
    low[5] = 96.0  # a wick, close stays at 100
    bars = _bars(close, low=low)
    touched, _ = touch_outcomes(bars, target_pct=-0.03, horizon=5)
    # bars 0..4 look forward far enough to see bar 5
    assert touched[0] and touched[4]
    assert not touched[5]


def test_time_to_touch_is_the_first_bar_that_reached_it():
    close = np.full(20, 100.0)
    low = np.full(20, 100.0)
    low[3] = 90.0
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=10)
    assert touched[0]
    assert bars_to[0] == 3  # three bars ahead
    assert bars_to[2] == 1


def test_an_upward_target_uses_highs():
    close = np.full(20, 100.0)
    high = np.full(20, 100.0)
    high[4] = 110.0
    touched, _ = touch_outcomes(_bars(close, high=high), target_pct=0.05, horizon=6)
    assert touched[0]


def test_bars_without_a_full_forward_window_are_dropped():
    """A bar whose horizon runs past the end of the data has no outcome, and
    counting it as 'not touched' would bias every probability downward."""
    close = np.full(20, 100.0)
    touched, _ = touch_outcomes(_bars(close), target_pct=-0.03, horizon=5)
    assert len(touched) == 15
