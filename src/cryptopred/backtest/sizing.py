"""Position sizing from predicted probability.

A threshold answers one question — act or don't. It cannot express the thing a
probability actually tells you: a 68% call and a 61% call are both "act", and
betting the same amount on each throws away the difference between them.

Every rule here returns a size in [0, 1], read as a fraction of the capital slot
a trade would otherwise take. Sizes never exceed 1, so variable sizing can only
reduce exposure relative to fixed sizing — it cannot quietly introduce leverage.
A rule that sometimes bets more than full size is a different and far more
dangerous thing than a rule that sometimes bets less.
"""

from __future__ import annotations

import numpy as np

SIZING_METHODS = ("fixed", "linear", "kelly", "sqrt_kelly")


def kelly_fraction(p: float | np.ndarray, win: float, loss: float) -> np.ndarray:
    """Kelly stake for a bet that wins `win` and loses `loss`, both as fractions.

    Returns 0 where the bet has negative expectancy — Kelly's answer to a bad bet
    is not a small stake, it is no stake.
    """
    p = np.asarray(p, dtype="float64")
    if win <= 0 or loss <= 0:
        return np.zeros_like(p)
    odds = win / loss
    fraction = p - (1.0 - p) / odds
    return np.clip(fraction, 0.0, 1.0)


def breakeven_probability(win: float, loss: float) -> float:
    """The probability below which Kelly stakes nothing."""
    if win <= 0:
        return 1.0
    odds = win / loss
    return float(1.0 / (1.0 + odds))


def size_from_confidence(
    confidence: np.ndarray,
    method: str = "kelly",
    threshold: float = 0.5,
    median_move: float = 0.01,
    round_trip_cost: float = 0.0014,
    kelly_scale: float = 0.5,
    max_size: float = 1.0,
) -> np.ndarray:
    """Map model confidence to a position size in [0, 1].

    `median_move` and `round_trip_cost` describe the payoff being sized: a trade
    that wins `median_move - cost` and loses `median_move + cost`. Losing more
    than you win on the same move is what the fee does, and it is why the
    break-even probability sits above 50%.

    `kelly_scale` below 1 is deliberate. Full Kelly maximises long-run growth
    only if the probabilities are exactly right; they are estimates here, and
    over-betting an overestimated edge is the fastest way to ruin. Half Kelly
    gives up a quarter of the growth for a large reduction in variance.
    """
    confidence = np.asarray(confidence, dtype="float64")

    if method == "fixed":
        return np.full(confidence.shape, float(max_size))

    if method == "linear":
        # Straight ramp from the threshold to certainty.
        span = max(1.0 - threshold, 1e-9)
        return np.clip((confidence - threshold) / span, 0.0, 1.0) * max_size

    win = median_move - round_trip_cost
    loss = median_move + round_trip_cost
    fraction = kelly_fraction(confidence, win=win, loss=loss)

    if method == "kelly":
        return np.clip(fraction * kelly_scale, 0.0, max_size)

    if method == "sqrt_kelly":
        # Square root flattens the curve: still more on stronger signals, but far
        # less concentrated when a single probability is badly wrong.
        return np.clip(np.sqrt(fraction) * kelly_scale, 0.0, max_size)

    raise ValueError(f"Unknown sizing method: {method!r}. Known: {', '.join(SIZING_METHODS)}")


def describe_sizing(
    sizes: np.ndarray, signals: np.ndarray
) -> dict[str, float]:
    """Summary of how much capital a sizing rule actually deploys."""
    taken = signals != 0
    if not taken.any():
        return {"n_sized": 0, "mean_size": 0.0, "median_size": 0.0, "zero_size_share": 0.0}
    active = sizes[taken]
    return {
        "n_sized": int(taken.sum()),
        "mean_size": float(active.mean()),
        "median_size": float(np.median(active)),
        "max_size": float(active.max()),
        # Signals the rule declined to fund at all: a real decision, not a bug.
        "zero_size_share": float((active <= 1e-9).mean()),
    }
