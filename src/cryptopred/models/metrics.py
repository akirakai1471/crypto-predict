"""Evaluation metrics.

Raw accuracy is the least informative number here and the easiest to fool
yourself with. The metrics that matter are:

- Brier score: are the probabilities themselves any good?
- Directional accuracy on filtered signals: when the model actually commits to
  a direction, how often is it right?
- Accuracy by confidence bucket: does a 70% call really beat a 55% call? If the
  buckets are flat, the probabilities are decoration.

Class order everywhere: 0 = DOWN, 1 = FLAT, 2 = UP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2
N_CLASSES = 3
_EPS = 1e-15


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot truth.

    Range is 0 (perfect) to 2 (confidently wrong on every sample).
    """
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    clipped = np.clip(proba, _EPS, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def directional_accuracy(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy restricted to bars where the model commits to UP or DOWN.

    A FLAT prediction is not a trade, so it is neither right nor wrong here.
    `threshold` is the minimum probability required to count as a commitment;
    it is what implements the spec's dead zone.
    """
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    committed = (predicted != FLAT) & (confidence >= threshold)

    n_signals = int(committed.sum())
    if n_signals == 0:
        return {"accuracy": float("nan"), "n_signals": 0, "coverage": 0.0}

    correct = (predicted[committed] == y_true[committed]).sum()
    return {
        "accuracy": float(correct / n_signals),
        "n_signals": n_signals,
        "coverage": float(n_signals / len(y_true)),
    }


def sign_accuracy(
    forward_return: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy against the raw sign of the move, ignoring the FLAT label.

    This is the number that maps to trading. `directional_accuracy` counts a
    long call as wrong when price drifted inside the dead zone, but such a bar
    is a small scratch, not a loss. Here a call is right when the move went the
    predicted way at all.
    """
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    committed = (predicted != FLAT) & (confidence >= threshold)

    n_signals = int(committed.sum())
    if n_signals == 0:
        return {"accuracy": float("nan"), "n_signals": 0}

    predicted_sign = np.where(predicted[committed] == UP, 1.0, -1.0)
    actual_sign = np.sign(forward_return[committed])
    # A bar that closed exactly flat is neither right nor wrong; count it as half
    # so it cannot inflate the number.
    correct = (predicted_sign == actual_sign).sum() + 0.5 * (actual_sign == 0).sum()
    return {"accuracy": float(correct / n_signals), "n_signals": n_signals}


def accuracy_by_confidence(
    y_true: np.ndarray,
    proba: np.ndarray,
    bins: tuple[float, ...] = (0.0, 0.4, 0.5, 0.6, 0.7, 1.01),
) -> pd.DataFrame:
    """Accuracy within each confidence band. A well-calibrated model shows
    accuracy rising monotonically across the bands."""
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    correct = predicted == y_true

    bucket = pd.cut(confidence, bins=list(bins), right=False)
    frame = pd.DataFrame({"correct": correct, "bucket": bucket})
    grouped = frame.groupby("bucket", observed=False)["correct"]
    return pd.DataFrame({"n": grouped.size(), "accuracy": grouped.mean()})


def evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
    forward_return: np.ndarray | None = None,
) -> dict[str, Any]:
    """Headline metrics for one model on one test window."""
    predicted = proba.argmax(axis=1)
    directional = directional_accuracy(y_true, proba, threshold=threshold)
    report = {
        "accuracy": float((predicted == y_true).mean()),
        "brier": multiclass_brier(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "directional_accuracy": directional["accuracy"],
        "n_signals": directional["n_signals"],
        "coverage": directional["coverage"],
    }
    if forward_return is not None:
        report["sign_accuracy"] = sign_accuracy(forward_return, proba, threshold)["accuracy"]
    return report


def block_bootstrap_mean(
    values: np.ndarray, block: int, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """A studentized circular block-bootstrap 95% interval for a mean.

    Used on paired per-row differences (model right minus baseline right) in
    time order. Rows whose 24-bar labels overlap are not independent, and an
    interval that resamples them one at a time reports the width of that many
    independent trials: on BTC it read [+1.3, +3.1] points where blocks read
    [-0.4, +4.8], and every symbol it had granted a classification GO had a
    block interval that included zero. Same construction as the touch interval
    (briefing/touch.py), without its clipping to [0, 1].
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    block = max(1, min(int(block), max(n, 1)))
    n_blocks = int(np.ceil(n / block)) if n else 0
    result = {
        "mean_difference": mean,
        "ci_lower": float("-inf"),
        "ci_upper": float("inf"),
        "block": block,
        "n_blocks": n_blocks,
    }
    if n_blocks < 2:
        # One block holds no information about its own spread.
        result["excludes_zero"] = False
        return result

    wrapped = np.concatenate([x, x[: block - 1]])
    csum = np.concatenate([[0.0], np.cumsum(wrapped)])
    block_means = (csum[block:] - csum[:-block]) / block
    se = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
    if se == 0.0:
        lower = upper = mean
    else:
        rng = np.random.default_rng(seed)
        draws = block_means[rng.integers(0, n, size=(n_boot, n_blocks))]
        mean_star = draws.mean(axis=1)
        se_star = draws.std(axis=1, ddof=1) / np.sqrt(n_blocks)
        t_star = (mean_star - mean) / np.maximum(se_star, se / np.sqrt(n_blocks))
        q_lo, q_hi = np.quantile(t_star, [0.025, 0.975])
        lower, upper = mean - q_hi * se, mean - q_lo * se
    result.update(
        ci_lower=float(lower),
        ci_upper=float(upper),
        excludes_zero=bool(lower > 0 or upper < 0),
    )
    return result
