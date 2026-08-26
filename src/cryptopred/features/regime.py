"""Market regime: is price trending or ranging, is volatility high or low."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import pct_rank, safe_divide, wilder_smooth
from cryptopred.features.volatility import atr, realized_vol


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index: trend strength regardless of direction."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    atr_series = atr(df, period)
    plus_di = 100.0 * safe_divide(wilder_smooth(plus_dm, period), atr_series)
    minus_di = 100.0 * safe_divide(wilder_smooth(minus_dm, period), atr_series)

    dx = 100.0 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return wilder_smooth(dx, period)


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["adx_14"] = adx(df, 14)
    out["adx_28"] = adx(df, 28)
    out["trend_strength"] = out["adx_14"] / 100.0

    rv = realized_vol(df["close"], 24)
    out["vol_regime"] = pct_rank(rv, window=720)
    out["is_high_vol"] = (out["vol_regime"] > 0.8).astype("float64")
    out["is_low_vol"] = (out["vol_regime"] < 0.2).astype("float64")
    out["is_trending"] = (out["adx_14"] > 25).astype("float64")

    return out
