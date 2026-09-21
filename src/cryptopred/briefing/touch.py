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
from cryptopred.briefing.regime import classify_regimes, current_cell


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

# This interval is NOT calibrated to 95% and must not be presented as one. It is
# labelled "95%" because that is the nominal quantile cut (2.5% / 97.5%) used to
# build it, in the same sense a t-test is still called a "95% CI" even though
# its actual coverage depends on how well its assumptions hold. Here they hold
# poorly: a percentile block-bootstrap on strongly autocorrelated data is known
# to under-cover, and this project measured it rather than assuming it away.
#
# Harness: a two-state Markov chain with a known long-run "touched" rate of 0.4
# and mean run length ~50 bars (state 1 mean run 40, state 0 mean run 60, giving
# stationary P=0.4 and average run length 50). 800 independent realisations per
# (n, block) cell, n_boot=400, checking how often the interval contains the
# true 0.4. Measured on this module's circular block_bootstrap_ci (SE ~1-1.6pp
# at 800 reps):
#
#   n=3000 (large sample):  block=1  23%   block=24  75%   block=48  85%
#                            block=72 89%   block=144 91%   block=200 91%
#   n=500  (MIN_CELL_BARS):  block=24 73%   block=48  80%   block=72  80%
#                            block=144 78%
#
# Switching from a non-circular to a circular bootstrap (positions wrapped
# modulo n so every position is drawn equally often, fixing the ~15x
# under-sampling of the first/last `block` positions that a non-circular
# window has - Künsch 1989) was checked directly against a non-circular clone
# on identical draws: it helps, but only modestly (roughly +1-4 percentage
# points, bigger at small n where the edge zone is a larger share of the
# series) and it does NOT close the gap to 95% at any n or block tested here.
# The dominant remaining shortfall is the known low-order bias of a plain
# percentile block-bootstrap under strong dependence, which this change does
# not attempt to fix.
#
# Conservative headline figure used in the strings this module returns to
# callers: 80%, the worst case actually measured at MIN_CELL_BARS-sized cells
# with the block sizes this module uses by default (block=48 and block=144,
# see the floor formula in touch_probability). Re-run coverage_harness-style
# measurement before changing this constant; do not eyeball it.
MEASURED_COVERAGE = 0.80


def block_bootstrap_ci(
    touched: np.ndarray,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """A block-bootstrap interval that survives autocorrelation better than
    Wilson does, but is not a calibrated 95% interval. See MEASURED_COVERAGE
    above for what it actually delivers and how that was measured.

    Resamples contiguous blocks rather than individual outcomes, so the
    dependence between neighbouring windows is preserved instead of being
    assumed away. A Wilson interval on the same data reports roughly the width
    it would have if every window were an independent trial, which is wrong by
    a factor of several here — but "wider than Wilson" is not the same claim
    as "95% coverage", and this interval only supports the first one.

    Circular: block start positions range over the whole series and wrap with
    `% n`, so every position is equally likely to be drawn, rather than the
    first/last `block` positions being drawn far less often than interior ones
    (see MEASURED_COVERAGE for the measured effect of this).
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
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n] % n
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
    touched, bars_to = touch_outcomes(bars, target_pct, horizon)
    if touched.size == 0:
        reason = f"chỉ có {len(bars):,} nến, cần hơn {horizon:,} nến để nhìn tới đích"
        return {
            "conditional": Unavailable(reason=reason),
            "unconditional": Unavailable(reason=reason),
            "cell_label": "không xác định",
            "wait_hours": {},
            "wait_source": "unconditional",
        }

    # The dependence length between consecutive touch outcomes is about
    # `horizon` bars (they share horizon-1 of horizon forward bars). A block
    # equal to that length is measured to under-cover (see MEASURED_COVERAGE
    # above); doubling it materially improves coverage at horizon=24 — the
    # most common query — at both large-n and MIN_CELL_BARS-sized samples,
    # and is neutral (no measured regression) at horizon=72. The floor of 48
    # keeps short horizons from using a degenerately small block.
    block = max(2 * horizon, 48)
    coverage_note = f"độ phủ đo được ≈{MEASURED_COVERAGE:.0%}, không phải 95%"
    unconditional = Measured(
        value=float(touched.mean()),
        n=int(touched.size),
        interval=block_bootstrap_ci(touched, block=block, n_boot=n_boot),
        method=f"mọi nến lịch sử, bootstrap khối {block} nến ({coverage_note})",
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
            "wait_source": "unconditional",
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
        # The cell itself is too thin to report on, so the wait-time figure
        # below falls back to the unconditional sample — it must NOT be
        # captioned with the cell's label, or a renderer would misattribute
        # unconditional wait times to a named regime.
        wait_inputs = (touched, bars_to)
        wait_source = "unconditional"
    else:
        conditional = Measured(
            value=float(cell_touched.mean()),
            n=int(cell_touched.size),
            interval=block_bootstrap_ci(cell_touched, block=block, n_boot=n_boot),
            method=f"nến cùng chế độ '{cell.label}', bootstrap khối {block} nến ({coverage_note})",
        )
        wait_inputs = (cell_touched, cell_bars_to)
        wait_source = "cell"

    return {
        "conditional": conditional,
        "unconditional": unconditional,
        "cell_label": cell.label,
        "wait_hours": _wait_percentiles(*wait_inputs, bars),
        "wait_source": wait_source,
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
