"""Purged walk-forward cross-validation.

Ordinary k-fold puts future data in the training set, which on a price series
means the model is trained on the answers. This splitter enforces three rules:

1. Training data always comes strictly before test data (walk-forward).
2. The last `horizon` training bars are dropped, because their labels are
   derived from prices inside the test window (purging).
3. A further slice after the purge is dropped to break the serial correlation
   that survives the gap (embargo).

The result is pessimistic compared to naive validation. That is the point.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForward:
    n_splits: int = 5
    horizon: int = 4
    embargo_frac: float = 0.01

    def split(self, index: pd.Index) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(index)
        # Each fold needs a test block plus at least one test block of training
        # history before it, plus the purge and embargo cut out of the middle.
        test_size = n // (self.n_splits + 1)
        embargo = int(n * self.embargo_frac)
        if test_size <= self.horizon + embargo:
            raise ValueError(
                f"too few samples: {n} rows cannot support {self.n_splits} folds "
                f"with horizon={self.horizon} and embargo_frac={self.embargo_frac}"
            )

        for fold in range(self.n_splits):
            test_start = test_size * (fold + 1)
            test_stop = test_start + test_size if fold < self.n_splits - 1 else n

            train_stop = test_start - self.horizon - embargo
            if train_stop <= 0:
                raise ValueError(
                    f"too few samples: fold {fold} leaves no training data after "
                    f"purging {self.horizon} bars and embargoing {embargo}"
                )

            train_idx = np.arange(0, train_stop)
            test_idx = np.arange(test_start, test_stop)
            yield train_idx, test_idx
