"""Baselines the model must beat before anyone is allowed to deploy it.

A 55% accuracy number means nothing on its own. If 55% of bars are labelled UP,
a model that always says UP scores 55% too. These four baselines make the
comparison explicit and unavoidable.

Class order everywhere: column 0 = DOWN, 1 = FLAT, 2 = UP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = ("coin_flip", "always_up", "prior", "momentum")

N_CLASSES = 3
DOWN, FLAT, UP = 0, 1, 2


def baseline_predictions(
    name: str, test: pd.DataFrame, train: pd.DataFrame
) -> np.ndarray:
    """Probability matrix of shape (len(test), 3) for the named baseline.

    `train` supplies whatever the baseline is allowed to learn — only class
    frequencies today. It must never be the test set in real evaluation.
    """
    n = len(test)

    if name == "coin_flip":
        proba = np.zeros((n, N_CLASSES))
        proba[:, DOWN] = 0.5
        proba[:, UP] = 0.5
        return proba

    if name == "always_up":
        proba = np.zeros((n, N_CLASSES))
        proba[:, UP] = 1.0
        return proba

    if name == "prior":
        counts = train["label_class"].value_counts(normalize=True)
        proba = np.zeros((n, N_CLASSES))
        for cls in range(N_CLASSES):
            proba[:, cls] = counts.get(cls, 0.0)
        total = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, total, out=np.full_like(proba, 1 / N_CLASSES), where=total > 0)

    if name == "momentum":
        # Persistence: assume the next move continues the last one.
        last = test["roc_1"].to_numpy()
        proba = np.full((n, N_CLASSES), 0.2)
        proba[:, FLAT] = 0.2
        proba[last > 0, UP] = 0.6
        proba[last > 0, DOWN] = 0.2
        proba[last < 0, DOWN] = 0.6
        proba[last < 0, UP] = 0.2
        flat_mask = last == 0
        proba[flat_mask] = 1 / N_CLASSES
        return proba / proba.sum(axis=1, keepdims=True)

    raise ValueError(f"Unknown baseline: {name!r}. Known: {', '.join(BASELINES)}")
