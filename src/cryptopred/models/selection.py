"""Choosing which bars to trade, without depending on the probability scale.

A fixed probability threshold assumes the probability scale means the same thing
everywhere. It does not. Isotonic calibration fitted on a small window overfits
and emits extreme probabilities; fitted on a large one it converges and rarely
leaves the middle. Measured on BTCUSDT with an identical model and an identical
0.60 threshold, coverage ran 26.6%, 11.1%, 1.0%, 0.03%, 3.3% across five
walk-forward folds — purely because each fold had a different amount of
calibration data. Sixty-three percent of all "signals" came from the first fold,
the one whose calibrator was worst.

That is not a trading rule, it is a calibration artefact wearing one.

Selecting by rank instead removes the dependency entirely: "trade the most
confident 8% of bars" means the same thing whatever the probabilities look like,
and a monotone recalibration cannot change which bars are chosen.
"""

from __future__ import annotations

import numpy as np

DOWN, FLAT, UP = 0, 1, 2


def signals_by_quantile(
    proba: np.ndarray, coverage: float, allow_flat: bool = False
) -> np.ndarray:
    """Trade the most confident `coverage` fraction of bars.

    Confidence is the margin between the strongest directional class and the
    other one, not the raw maximum. A bar where UP and DOWN are 0.40 and 0.39
    is a coin flip however large those numbers look, and using the maximum would
    rank it alongside a genuine 0.40-versus-0.10 call.
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    directional_margin = np.abs(proba[:, UP] - proba[:, DOWN])
    if not allow_flat:
        # A bar whose most likely outcome is FLAT is not a trade at any rank.
        directional_margin = np.where(
            proba.argmax(axis=1) == FLAT, -np.inf, directional_margin
        )

    eligible = np.isfinite(directional_margin)
    n_take = int(round(coverage * len(proba)))
    n_take = min(n_take, int(eligible.sum()))
    if n_take <= 0:
        return np.zeros(len(proba), dtype=int)

    cutoff_idx = np.argsort(directional_margin)[-n_take:]
    signal = np.zeros(len(proba), dtype=int)
    signal[cutoff_idx] = np.where(
        proba[cutoff_idx, UP] > proba[cutoff_idx, DOWN], 1, -1
    )
    return signal


def signals_by_quantile_per_fold(
    proba: np.ndarray, fold_ids: np.ndarray, coverage: float
) -> np.ndarray:
    """Apply the rank rule inside each fold separately.

    Ranking across pooled folds would let a fold with an inflated probability
    scale monopolise the selection — reintroducing exactly the bias this module
    exists to remove. Each fold contributes its own most confident `coverage`.
    """
    signal = np.zeros(len(proba), dtype=int)
    for fold in np.unique(fold_ids):
        mask = fold_ids == fold
        signal[mask] = signals_by_quantile(proba[mask], coverage)
    return signal


def coverage_by_fold(signals: np.ndarray, fold_ids: np.ndarray) -> dict[int, float]:
    """How much of each fold was actually traded — the diagnostic that exposed
    the threshold problem in the first place."""
    return {
        int(fold): float((signals[fold_ids == fold] != 0).mean())
        for fold in np.unique(fold_ids)
    }
