"""Which kind of market hour is this one like?

Touch probabilities conditioned on nothing answer a question nobody asked: the
unconditional chance of a 3% drop mixes calm weeks with crashes. Conditioning on
volatility and trend makes the comparison set resemble now.

Both boundaries come from an expanding window. The tercile edges for bar t use
bars up to t only, so a classification made in 2020 cannot be revised by what
volatility did in 2026.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptopred.features.base import ewma
from cryptopred.features.volatility import atr

# Bars needed before an expanding quantile is stable enough to bucket on. Below
# this the boundaries move with almost every new bar and the label is noise.
MIN_HISTORY_BARS = 2000

VOL_LABELS = {0: "biến động thấp", 1: "biến động vừa", 2: "biến động cao"}
TREND_LABELS = {0: "xu hướng giảm", 1: "đi ngang", 2: "xu hướng tăng"}


@dataclass(frozen=True)
class RegimeCell:
    vol_bucket: int
    trend_bucket: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.vol_bucket, self.trend_bucket)

    @property
    def label(self) -> str:
        return f"{VOL_LABELS[self.vol_bucket]} / {TREND_LABELS[self.trend_bucket]}"


def expanding_tercile(series: pd.Series, min_history: int) -> pd.Series:
    """Which third of its own past does each value sit in?

    `expanding().rank(pct=True)` ranks the window's final value against
    everything before it, which is exactly the point-in-time question. The same
    property is already relied on by `features.base.pct_rank`.
    """
    ranks = series.expanding(min_periods=min_history).rank(pct=True)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[ranks <= 1 / 3] = 0.0
    out[(ranks > 1 / 3) & (ranks <= 2 / 3)] = 1.0
    out[ranks > 2 / 3] = 2.0
    return out


def classify_regimes(
    bars: pd.DataFrame, min_history: int = MIN_HISTORY_BARS
) -> pd.DataFrame:
    """Per-bar volatility and trend buckets. NaN where history is too thin.

    `min_history` counts non-null observations, so each column also waits out its
    own indicator warm-up first: volatility becomes valid around
    `min_history + 14` bars and trend around `min_history + 168`. At the default
    that is roughly a week's difference between the two columns, and callers that
    need both — every caller here does — are gated by the slower one.
    """
    vol = atr(bars, 14) / bars["close"]
    trend = bars["close"] / ewma(bars["close"], 168) - 1.0
    return pd.DataFrame(
        {
            "vol_bucket": expanding_tercile(vol, min_history),
            "trend_bucket": expanding_tercile(trend, min_history),
        },
        index=bars.index,
    )


def current_cell(cells: pd.DataFrame) -> RegimeCell | None:
    """The cell of the most recent fully classified bar, or None."""
    valid = cells.dropna()
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return RegimeCell(int(last["vol_bucket"]), int(last["trend_bucket"]))
