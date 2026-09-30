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
# built at the nominal 2.5% / 97.5% cut, in the same sense a t-test is still
# called a "95% CI" even though its actual coverage depends on how well its
# assumptions hold. Here they hold imperfectly, and this project measures that
# rather than assuming it away.
#
# Measured with scripts/touch_interval_coverage.py, which is the harness: it
# simulates series with a known true rate, builds this interval on each, and
# counts how often the truth lands inside. Re-run it before changing anything
# below; do not eyeball it. Results are recorded in docs/findings.md under
# "The touch-probability interval".
#
# Headline figure used in the strings this module returns: the worst case
# measured across the harness's two dependence models (a two-state Markov
# chain, and real touch outcomes on a fat-tailed random walk) at MIN_CELL_BARS
# and larger samples, with the block sizes touch_probability uses. That was a
# 72-hour query on 500 bars, at 86.5%, 86.8% and 90.2% on three seeds; the
# label takes the lowest and rounds down for the ~2pp noise.
#
# One situation falls outside that figure and is documented instead of averaged
# in, because every method tried fails it, not just this one: volatility regimes
# that persist for weeks, which make touch outcomes dependent over far longer
# than any block. Coverage there measured 69-85%. The regime-conditional figure
# is less exposed than the unconditional one, since its cell fixes the regime.
MEASURED_COVERAGE = 0.85


def block_length(horizon: int, n: int) -> int:
    """How many consecutive outcomes each bootstrap block keeps together.

    Consecutive touch outcomes share horizon-1 of their horizon forward bars,
    so the dependence runs at least `horizon` bars. A block of exactly that
    under-covers; twice it, floored at 48 so short horizons do not get a
    degenerate block, was measured to do better at every horizon tried.

    It then grows with the sample as n^(1/3), the textbook rate for block
    length. A fixed block is too short for a large sample: its bias, not its
    noise, is what leaves the interval narrow, and only a longer block reduces
    it. Measured at n=3,000 this lifted coverage on every dependence model in
    scripts/touch_interval_coverage.py. Below MIN_CELL_BARS it is left alone.
    """
    base = max(2 * horizon, 48)
    return int(round(base * max(1.0, (n / MIN_CELL_BARS) ** (1 / 3))))


def block_bootstrap_ci(
    touched: np.ndarray,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """A studentized circular block-bootstrap interval for the touch rate.

    Blocks, because consecutive touch outcomes share almost all their forward
    bars: a Wilson interval would report the width of that many independent
    trials, which is wrong by a factor of several. Circular, so every position is
    drawn equally often rather than the first and last `block` positions being
    under-sampled (Künsch 1989).

    Studentized, because the plain percentile version under-covers when there
    are few effective blocks: it treats the spread of ten block means as if it
    were known exactly. Each resample here is scaled by its own standard error,
    and the interval is read off the distribution of those t-ratios, so a small
    number of blocks produces the heavier tails it should. See MEASURED_COVERAGE
    for what this delivers and how it was measured - it is not 95%.
    """
    t = np.asarray(touched, dtype=float)
    n = len(t)
    if n == 0:
        return (0.0, 1.0)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    mean = float(t.mean())
    if n_blocks < 2:
        # One block holds no information about its own spread.
        return (0.0, 1.0)

    # The mean of every circular block, from a single cumulative sum, so a
    # resample is just a choice of block starts - no per-draw copy of the series.
    wrapped = np.concatenate([t, t[: block - 1]])
    csum = np.concatenate([[0.0], np.cumsum(wrapped)])
    block_means = (csum[block:] - csum[:-block]) / block
    se = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
    if se == 0.0:
        return _no_variation_interval(mean, n_blocks)

    rng = np.random.default_rng(seed)
    draws = block_means[rng.integers(0, n, size=(n_boot, n_blocks))]
    mean_star = draws.mean(axis=1)
    se_star = draws.std(axis=1, ddof=1) / np.sqrt(n_blocks)
    # A resample whose blocks all agree has a standard error of zero and an
    # infinite t-ratio. The floor keeps those draws in the distribution - they
    # are real information about how lopsided the sample can look - without
    # letting one of them decide the whole interval.
    t_star = (mean_star - mean) / np.maximum(se_star, se / np.sqrt(n_blocks))
    q_lo, q_hi = np.quantile(t_star, [0.025, 0.975])
    return (max(0.0, mean - q_hi * se), min(1.0, mean - q_lo * se))


def _no_variation_interval(mean: float, n_blocks: int) -> tuple[float, float]:
    """Every block agrees: the level was touched always, or never.

    A zero-width interval would claim certainty from what may be ten effective
    observations. The rule of three - a 95% bound of 3/n for zero events in n
    independent trials - applied to the blocks, which are what is roughly
    independent here, keeps the answer honest about how little was seen.
    """
    reach = min(1.0, 3.0 / n_blocks)
    if mean <= 0.0:
        return (0.0, reach)
    if mean >= 1.0:
        return (1.0 - reach, 1.0)
    return (mean, mean)


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

    coverage_note = f"độ phủ đo được ≈{MEASURED_COVERAGE:.0%}, không phải 95%"
    block = block_length(horizon, touched.size)
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
        cell_block = block_length(horizon, cell_touched.size)
        conditional = Measured(
            value=float(cell_touched.mean()),
            n=int(cell_touched.size),
            interval=block_bootstrap_ci(cell_touched, block=cell_block, n_boot=n_boot),
            method=(
                f"nến cùng chế độ '{cell.label}', bootstrap khối {cell_block} nến "
                f"({coverage_note})"
            ),
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
