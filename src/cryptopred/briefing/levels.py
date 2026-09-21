"""Prices worth asking about.

Pivots and swing points are conventional constructions, and this project has not
measured whether price respects them. Their job here is narrower and defensible:
they turn "where might it go" into specific numbers that `touch_probability` can
then measure honestly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Convention

PIVOT_READING = "pivot cổ điển tính từ H/L/C của ngày hôm trước"


def daily_pivots(bars: pd.DataFrame) -> dict[str, Any]:
    """Classic floor-trader pivots from the previous completed day."""
    if bars.empty:
        return {}
    daily = bars.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()
    if len(daily) < 2:
        return {}

    prev = daily.iloc[-2]
    high, low, close = float(prev["high"]), float(prev["low"]), float(prev["close"])
    p = (high + low + close) / 3.0
    span = high - low

    values = {
        "P": p,
        "R1": 2 * p - low,
        "S1": 2 * p - high,
        "R2": p + span,
        "S2": p - span,
        "R3": high + 2 * (p - low),
        "S3": low - 2 * (high - p),
    }
    return {
        name: Convention(value=float(value), reading=PIVOT_READING).to_dict()
        for name, value in values.items()
    }


def _forward(series: pd.Series, k: int, how: str) -> pd.Series:
    """`max`/`min` of the next k bars, defined at every bar that has k ahead.

    Reversed so pandas counts its rows from the end of the series, where they
    exist. Still NaN for the final k bars, which is the point: those have no
    forward window and so cannot hold a confirmed swing.
    """
    reversed_ = series.iloc[::-1]
    rolled = getattr(reversed_.rolling(k, min_periods=k), how)()
    return rolled.iloc[::-1].shift(-1)


def swing_levels(
    bars: pd.DataFrame, k: int = 5, lookback: int = 720
) -> dict[str, list[dict[str, Any]]]:
    """Fractal swing highs and lows within the lookback window.

    A swing at bar i must be **strictly** above both of its neighbourhoods:
    higher than the previous k bars and higher than the next k. Requiring only
    that it equal the window maximum marks every bar of a flat stretch as a
    swing, because they all tie — which on quiet data buries the one real peak
    under dozens of fake ones.

    The strictness also confirms the level: `right_max` is undefined for the
    last k bars, so they hold no swing. That is correct rather than incidental —
    a swing that needs future bars to confirm is not a level yet.
    """
    if bars.empty:
        return {"resistance": [], "support": []}

    window = bars.tail(lookback)
    high, low = window["high"], window["low"]
    reading = f"đỉnh/đáy xoay, xác nhận bằng {k} nến hai bên"

    # Bar i itself is in neither neighbourhood.
    #
    # The left side is a plain backward window: shift(1).rolling(k) at i covers
    # i-k .. i-1, and is correctly NaN for the first k bars.
    #
    # The right side has to be built by reversing. The obvious
    # `shift(-k).rolling(k)` is wrong at the start of the series: pandas counts
    # ROWS, not available data, so at i < k-1 it returns NaN even though bars
    # i+1 .. i+k all exist. That silently hides any swing in the oldest k-1 bars
    # of the window — verified: a spike at position 2 with k=5 went undetected
    # while the same spike at position 20 was found. Reversing counts rows from
    # the other end, where they are there.
    left_high = high.shift(1).rolling(k, min_periods=k).max()
    left_low = low.shift(1).rolling(k, min_periods=k).min()
    right_high = _forward(high, k, "max")
    right_low = _forward(low, k, "min")

    is_high = (high > left_high) & (high > right_high)
    is_low = (low < left_low) & (low < right_low)

    def pack(mask: pd.Series, series: pd.Series) -> list[dict[str, Any]]:
        picked = series[mask.fillna(False)]
        # Built through Convention.to_dict() rather than by hand, so these carry
        # the same warning as every other conventional value in this package. A
        # consumer iterating over mixed payloads must not find some warned and
        # some not; `at` is added on top rather than replacing the shape.
        return [
            {**Convention(value=float(value), reading=reading).to_dict(), "at": str(at)}
            for at, value in picked.items()
        ]

    return {
        "resistance": pack(is_high, high)[-8:],
        "support": pack(is_low, low)[-8:],
    }
