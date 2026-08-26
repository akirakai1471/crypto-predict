"""Three-class labels with a volatility-scaled dead zone.

Forcing a binary up/down decision on a bar where price barely moved teaches the
model to fit noise. The dead zone is proportional to ATR so it adapts: in a calm
market a 0.2% move is meaningful, in a violent one it is nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.volatility import atr

LABEL_DOWN = -1.0
LABEL_FLAT = 0.0
LABEL_UP = 1.0


def make_labels(
    bars: pd.DataFrame,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 0.5,
) -> pd.DataFrame:
    """Label each bar by where price sits `horizon` bars later.

    Returns a frame indexed like `bars` with:
      - forward_return: (close[t+H] / close[t]) - 1
      - band: the dead-zone half-width at t, as a fraction of price
      - label: -1 down, 0 flat, +1 up; NaN for the last H bars
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = bars["close"]
    forward_return = close.shift(-horizon) / close - 1.0
    band = band_k * atr(bars, atr_period) / close

    label = pd.Series(np.nan, index=bars.index, dtype="float64")
    known = forward_return.notna() & band.notna()
    label[known & (forward_return > band)] = LABEL_UP
    label[known & (forward_return < -band)] = LABEL_DOWN
    label[known & (forward_return.abs() <= band)] = LABEL_FLAT

    return pd.DataFrame({"forward_return": forward_return, "band": band, "label": label})


def make_triple_barrier_labels(
    bars: pd.DataFrame,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 1.0,
) -> pd.DataFrame:
    """Alternative labelling: whichever barrier price touches first wins.

    Used by the backtest in Plan 2 to check that results are not an artefact of
    the fixed-horizon definition. Slower (a Python loop) but only run offline.
    """
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    band = (band_k * atr(bars, atr_period) / bars["close"]).to_numpy()

    n = len(bars)
    label = np.full(n, np.nan)
    for i in range(n - horizon):
        if not np.isfinite(band[i]):
            continue
        upper = close[i] * (1 + band[i])
        lower = close[i] * (1 - band[i])
        outcome = LABEL_FLAT
        for j in range(i + 1, i + horizon + 1):
            if high[j] >= upper:
                outcome = LABEL_UP
                break
            if low[j] <= lower:
                outcome = LABEL_DOWN
                break
        label[i] = outcome

    forward_return = bars["close"].shift(-horizon) / bars["close"] - 1.0
    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "band": pd.Series(band, index=bars.index),
            "label": pd.Series(label, index=bars.index),
        }
    )
