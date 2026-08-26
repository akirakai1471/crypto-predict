"""Volatility measures. ATR is also used by the label engine, so it lives here
and is imported rather than duplicated."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import pct_rank, safe_divide, wilder_smooth


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    return wilder_smooth(true_range(df), period)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Standard deviation of log returns over a trailing window."""
    log_ret = np.log(close).diff()
    return log_ret.rolling(window, min_periods=window).std(ddof=1)


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    for period in (7, 14, 28):
        out[f"atr_{period}_norm"] = safe_divide(atr(df, period), close)

    for window in (20, 50):
        mid = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=1)
        out[f"bb_width_{window}"] = safe_divide(2 * std, mid)
        out[f"bb_position_{window}"] = safe_divide(close - mid, 2 * std)

    for window in (12, 24, 72, 168):
        out[f"realized_vol_{window}"] = realized_vol(close, window)

    out["vol_ratio_24_72"] = safe_divide(
        out["realized_vol_24"], out["realized_vol_72"], fill=1.0
    )
    out["vol_pct_rank_720"] = pct_rank(out["realized_vol_24"], window=720)

    return out
