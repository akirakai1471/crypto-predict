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
