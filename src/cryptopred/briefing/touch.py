"""Did price reach a level within a horizon, and how long did it take?

This is the measurement that replaces "risk of falling to $2,400 increases".
It is descriptive, not predictive: it reports what price did in the past, and
makes no claim that the future resembles it beyond the regime conditioning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cryptopred.briefing.provenance import Measured, Unavailable


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

    A NaN in `low` or `high` never registers a touch, because every comparison
    against NaN is False. That fails open rather than crashing, and the ingest
    pipeline records gaps as missing rows rather than NaN-filled ones, so it
    should not arise — but a silent undercount is the shape of error this
    function is most able to hide, so it is written down.
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
    # argmax on a boolean row returns the FIRST True, which is what makes this
    # the first touch rather than an arbitrary one. +1 because the window starts
    # one bar ahead, so the next bar is 1 rather than 0.
    bars_to_touch = np.where(touched, hit.argmax(axis=1) + 1, 0)
    return touched, bars_to_touch


# Below this many observations a regime cell produces a confident-looking number
# out of noise. The trade is deliberate: conditioning makes the estimate more
# relevant and noisier, and this is where relevance stops being worth the noise.
MIN_CELL_BARS = 500


def block_bootstrap_ci(
    touched: np.ndarray,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval that survives autocorrelation.

    Resamples contiguous blocks rather than individual outcomes, so the
    dependence between neighbouring windows is preserved instead of being
    assumed away. A Wilson interval on the same data reports roughly the width
    it would have if every window were an independent trial, which is wrong by
    a factor of several here.
    """
    n = len(touched)
    if n == 0:
        return (0.0, 1.0)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)

    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n]
        means[b] = touched[idx].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def touch_probability(
    bars: pd.DataFrame,
    target_pct: float,
    horizon: int,
    min_cell_bars: int = MIN_CELL_BARS,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """How often price reached this level, in hours that resembled this one.

    Returns both the conditional and unconditional figures. When they disagree
    sharply that is information; when the cell is too thin the unconditional one
    is still an answer.
    """
    from cryptopred.briefing.regime import classify_regimes, current_cell

    touched, bars_to = touch_outcomes(bars, target_pct, horizon)
    if touched.size == 0:
        reason = f"chỉ có {len(bars):,} nến, cần hơn {horizon:,} nến để nhìn tới đích"
        return {
            "conditional": Unavailable(reason=reason),
            "unconditional": Unavailable(reason=reason),
            "cell_label": "không xác định",
            "wait_hours": {},
        }

    block = max(horizon, 24)
    unconditional = Measured(
        value=float(touched.mean()),
        n=int(touched.size),
        ci95=block_bootstrap_ci(touched, block=block, n_boot=n_boot),
        method=f"mọi nến lịch sử, bootstrap khối {block} nến",
    )

    cells = classify_regimes(bars)
    cell = current_cell(cells)
    if cell is None:
        return {
            "conditional": Unavailable(
                reason="chưa đủ lịch sử để xếp nến hiện tại vào chế độ nào"
            ),
            "unconditional": unconditional,
            "cell_label": "không xác định",
            "wait_hours": _wait_percentiles(touched, bars_to, bars),
        }

    aligned = cells.iloc[: touched.size]
    in_cell = (
        (aligned["vol_bucket"] == cell.vol_bucket)
        & (aligned["trend_bucket"] == cell.trend_bucket)
    ).to_numpy()
    cell_touched = touched[in_cell]
    cell_bars_to = bars_to[in_cell]

    if cell_touched.size < min_cell_bars:
        conditional: Measured | Unavailable = Unavailable(
            reason=(
                f"ô '{cell.label}' chỉ có {cell_touched.size:,} quan sát, "
                f"cần ít nhất {min_cell_bars:,} — số sẽ là nhiễu"
            )
        )
        wait_source = (touched, bars_to)
    else:
        conditional = Measured(
            value=float(cell_touched.mean()),
            n=int(cell_touched.size),
            ci95=block_bootstrap_ci(cell_touched, block=block, n_boot=n_boot),
            method=f"nến cùng chế độ '{cell.label}', bootstrap khối {block} nến",
        )
        wait_source = (cell_touched, cell_bars_to)

    return {
        "conditional": conditional,
        "unconditional": unconditional,
        "cell_label": cell.label,
        "wait_hours": _wait_percentiles(*wait_source, bars),
    }


def _wait_percentiles(
    touched: np.ndarray, bars_to: np.ndarray, bars: pd.DataFrame
) -> dict[str, float]:
    """How long the touches that happened took, in hours.

    Only the windows that touched contribute: the ones that did not have no wait
    time, and filling them with the horizon would invent data.
    """
    hit = bars_to[touched]
    if hit.size == 0:
        return {}
    hours_per_bar = _bar_hours(bars)
    q = np.quantile(hit, [0.25, 0.5, 0.75, 0.90]) * hours_per_bar
    return {
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p90": float(q[3]),
        "n": int(hit.size),
    }


def _bar_hours(bars: pd.DataFrame) -> float:
    if len(bars.index) < 2:
        return 1.0
    delta = bars.index[1] - bars.index[0]
    return float(delta.total_seconds() / 3600.0)
