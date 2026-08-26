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


def bootstrap_difference(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for (A's directional accuracy - B's).

    The spec's success criterion is that this interval excludes zero. Comparing
    two point estimates is not enough: with a few thousand signals, a 2 point
    accuracy gap is often noise.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        acc_a = directional_accuracy(y_true[sample], proba_a[sample], threshold)["accuracy"]
        acc_b = directional_accuracy(y_true[sample], proba_b[sample], threshold)["accuracy"]
        diffs[i] = (acc_a if np.isfinite(acc_a) else 0.0) - (
            acc_b if np.isfinite(acc_b) else 0.0
        )

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_difference": float(diffs.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "excludes_zero": bool(lower > 0 or upper < 0),
    }
