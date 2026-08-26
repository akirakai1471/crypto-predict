"""LightGBM training with purged walk-forward evaluation and calibration.

The trainer never sees a random split. Every reported number comes from data
that lies strictly in the future of the data the model was fitted on, with the
purge and embargo applied. Baselines are evaluated on exactly the same test
windows so the comparison is apples to apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cryptopred.dataset.builder import feature_columns
from cryptopred.models.baselines import BASELINES, baseline_predictions
from cryptopred.models.metrics import evaluate
from cryptopred.models.splits import PurgedWalkForward

N_CLASSES = 3


@dataclass
class TrainConfig:
    num_boost_round: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    calibrate: bool = True
    # Fraction of the training window held back to fit the probability
    # calibrator. It sits at the end of the training window, closest in time to
    # the test window, so calibration reflects the most recent regime.
    calibration_frac: float = 0.15
    signal_threshold: float = 0.5
    seed: int = 42
    extra_params: dict[str, Any] = field(default_factory=dict)

    def lgb_params(self) -> dict[str, Any]:
        params = {
            "objective": "multiclass",
            "num_class": N_CLASSES,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "verbosity": -1,
            "num_threads": 0,
        }
        params.update(self.extra_params)
        return params


@dataclass
class FoldResult:
    proba: np.ndarray
    y_true: np.ndarray
    booster: lgb.Booster
    calibrators: list[IsotonicRegression] | None
    importance: dict[str, float]
    features: list[str]


def _fit_calibrators(
    booster: lgb.Booster, holdout: pd.DataFrame, features: list[str]
) -> list[IsotonicRegression]:
    """One isotonic regressor per class, fitted one-vs-rest.

    LightGBM's softmax output is a ranking score, not a probability. Without
    this step a displayed "65%" has no relationship to how often the call is
    right, which makes the dashboard actively misleading.
    """
    raw = booster.predict(holdout[features], num_iteration=booster.best_iteration)
    raw = np.asarray(raw)
    y = holdout["label_class"].to_numpy()

    calibrators = []
    for cls in range(N_CLASSES):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw[:, cls], (y == cls).astype(float))
        calibrators.append(iso)
    return calibrators


def _apply_calibrators(
    raw: np.ndarray, calibrators: list[IsotonicRegression]
) -> np.ndarray:
    calibrated = np.column_stack(
        [calibrators[cls].predict(raw[:, cls]) for cls in range(N_CLASSES)]
    )
    total = calibrated.sum(axis=1, keepdims=True)
    # A row where every calibrator returned zero carries no information;
    # fall back to a uniform distribution rather than dividing by zero.
    return np.divide(
        calibrated, total, out=np.full_like(calibrated, 1 / N_CLASSES), where=total > 0
    )


def train_fold(
    train: pd.DataFrame, test: pd.DataFrame, config: TrainConfig
) -> FoldResult:
    """Fit one model on `train` and predict `test`."""
    features = feature_columns(train)
    features = [f for f in features if f in test.columns]

    fit_df = train
    calibrators = None
    if config.calibrate and config.calibration_frac > 0:
        split = int(len(train) * (1 - config.calibration_frac))
        fit_df, holdout = train.iloc[:split], train.iloc[split:]
        if holdout.empty or fit_df.empty:
            fit_df, holdout = train, None
    else:
        holdout = None

    dataset = lgb.Dataset(
        fit_df[features], label=fit_df["label_class"], free_raw_data=False
    )
    booster = lgb.train(
        config.lgb_params(), dataset, num_boost_round=config.num_boost_round
    )

    if holdout is not None and config.calibrate:
        calibrators = _fit_calibrators(booster, holdout, features)

    raw = np.asarray(booster.predict(test[features]))
    proba = _apply_calibrators(raw, calibrators) if calibrators else raw

    gains = booster.feature_importance(importance_type="gain")
    importance = dict(zip(features, (float(g) for g in gains), strict=True))

    return FoldResult(
        proba=proba,
        y_true=test["label_class"].to_numpy(),
        booster=booster,
        calibrators=calibrators,
        importance=importance,
        features=features,
    )


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    n_splits: int = 5,
    horizon: int = 4,
    embargo_frac: float = 0.01,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate across every fold, alongside all four baselines.

    Returns pooled out-of-sample metrics for the model and each baseline, plus
    per-fold detail and averaged feature importance.
    """
    if "label_class" not in dataset.columns:
        raise KeyError("dataset is missing 'label_class'; build it with cryptopred-dataset")

    config = config or TrainConfig()
    cv = PurgedWalkForward(n_splits=n_splits, horizon=horizon, embargo_frac=embargo_frac)

    all_proba: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    baseline_proba: dict[str, list[np.ndarray]] = {name: [] for name in BASELINES}
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(dataset.index)):
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]

        result = train_fold(train, test, config)
        all_proba.append(result.proba)
        all_true.append(result.y_true)
        all_returns.append(test["forward_return"].to_numpy())
        importances.append(result.importance)

        for name in BASELINES:
            baseline_proba[name].append(baseline_predictions(name, test, train))

        folds.append(
            {
                "fold": fold_id,
                "n_train": len(train),
                "n_test": len(test),
                "test_start": test.index.min(),
                "test_end": test.index.max(),
                **evaluate(
                    result.y_true,
                    result.proba,
                    threshold=config.signal_threshold,
                    forward_return=test["forward_return"].to_numpy(),
                ),
            }
        )

    y_true = np.concatenate(all_true)
    proba = np.concatenate(all_proba)
    forward_return = np.concatenate(all_returns)

    mean_importance: dict[str, float] = {}
    for imp in importances:
        for name, value in imp.items():
            mean_importance[name] = mean_importance.get(name, 0.0) + value / len(importances)

    return {
        "model": evaluate(
            y_true, proba, threshold=config.signal_threshold, forward_return=forward_return
        ),
        "baselines": {
            name: evaluate(
                y_true,
                np.concatenate(chunks),
                threshold=config.signal_threshold,
                forward_return=forward_return,
            )
            for name, chunks in baseline_proba.items()
        },
        "forward_return": forward_return,
        "folds": folds,
        "feature_importance": dict(
            sorted(mean_importance.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "y_true": y_true,
        "proba": proba,
        "baseline_proba": {
            name: np.concatenate(chunks) for name, chunks in baseline_proba.items()
        },
        "n_test_total": int(len(y_true)),
    }
