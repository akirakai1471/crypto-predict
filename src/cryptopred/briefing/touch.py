"""Did price reach a level within a horizon, and how long did it take?

This is the measurement that replaces "risk of falling to $2,400 increases".
It is descriptive, not predictive: it reports what price did in the past, and
makes no claim that the future resembles it beyond the regime conditioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def touch_outcomes(
    bars: pd.DataFrame, target_pct: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each bar, did the target get touched in the next `horizon` bars?

    Returns (touched, bars_to_touch), both length `len(bars) - horizon`. Bars
    whose forward window runs past the end of the data are dropped rather than
    counted as misses, which would bias every probability downward.

    `bars_to_touch` is the number of bars ahead of the first touch, and is
    meaningless (0) where `touched` is False.

    A downward target is tested against `low` and an upward one against `high`:
    a level that was traded was touched, whatever the bar closed at. Testing
    closes instead is the easiest error to make here and it biases the answer.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 bar")
    n = len(bars)
    if n <= horizon:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    close = bars["close"].to_numpy(dtype=float)
    extreme = (bars["low"] if target_pct < 0 else bars["high"]).to_numpy(dtype=float)

    # windows[i] == extreme[i : i + horizon]; the window starting at i + 1 is the
    # forward window for bar i, so bar i itself is never its own outcome.
    windows = sliding_window_view(extreme, horizon)[1:]
    targets = close[: len(windows)] * (1.0 + target_pct)

    hit = (
        windows <= targets[:, None] if target_pct < 0 else windows >= targets[:, None]
    )
    touched = hit.any(axis=1)
    bars_to_touch = np.where(touched, hit.argmax(axis=1) + 1, 0)
    return touched, bars_to_touch
