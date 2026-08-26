"""Where price sits relative to its own recent range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import safe_divide


def consecutive_streak(close: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) or down (-) closes ending at each bar."""
    direction = np.sign(close.diff()).fillna(0.0).to_numpy()
    streak = np.zeros(len(direction), dtype="float64")
    for i in range(1, len(direction)):
        d = direction[i]
        if d == 0:
            streak[i] = 0.0
        elif np.sign(streak[i - 1]) == d:
            streak[i] = streak[i - 1] + d
        else:
            streak[i] = d
    return pd.Series(streak, index=close.index)


def structure_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    bar_range = df["high"] - df["low"]
    # A doji (high == low) has no meaningful position; 0.5 is the neutral answer.
    out["close_pos_in_bar"] = safe_divide(close - df["low"], bar_range, fill=0.5)
    out["bar_range_norm"] = safe_divide(bar_range, close)
    if "open" in df.columns:
        out["body_ratio"] = safe_divide((close - df["open"]).abs(), bar_range, fill=0.0)
        out["upper_wick"] = safe_divide(
            df["high"] - np.maximum(close, df["open"]), bar_range
        )
        out["lower_wick"] = safe_divide(
            np.minimum(close, df["open"]) - df["low"], bar_range
        )

    for window in (24, 72, 168):
        roll_high = df["high"].rolling(window, min_periods=window).max()
        roll_low = df["low"].rolling(window, min_periods=window).min()
        out[f"dist_to_high_{window}"] = safe_divide(close - roll_high, roll_high)
        out[f"dist_to_low_{window}"] = safe_divide(close - roll_low, roll_low)
        out[f"range_pos_{window}"] = safe_divide(
            close - roll_low, roll_high - roll_low, fill=0.5
        )

    out["streak"] = consecutive_streak(close)

    return out
