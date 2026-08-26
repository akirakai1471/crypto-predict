"""Shared numeric helpers.

Every windowed operation here is strictly backward-looking. Never introduce a
function that uses `center=True` or a negative `shift` — that is future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill: float = 0.0
) -> pd.Series:
    """Element-wise division that yields `fill` instead of inf/NaN on zero."""
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def rolling_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Standardise against a trailing window.

    Using a trailing window (rather than whole-series mean/std) is mandatory:
    whole-series statistics encode information from the future into every row.
    """
    min_periods = min_periods or window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=1)
    z = (series - mean) / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    # A perfectly flat window has zero deviation; its correct z-score is 0, not NaN.
    z[(std == 0) & mean.notna()] = 0.0
    return z


def pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile of the current value within its trailing window, in [0, 1].

    Uses pandas' native rolling rank, which ranks the window's final value.
    A `.rolling().apply()` version is ~200x slower and would make the 1m
    dataset build take hours.
    """
    return series.rolling(window, min_periods=window).rank(pct=True)


def ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, used by RSI, ATR and ADX."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
