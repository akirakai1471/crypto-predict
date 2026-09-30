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
    # "holdout" fits the calibrator on a single contiguous tail block. That
    # produces a different probability scale depending on how much data the block
    # holds, and the scale is what a trading rule is expressed in: coverage ran
    # 26.6% to 0.03% across folds under an identical threshold.
    #
    # "oof" spreads calibration across the whole window via inner out-of-fold
    # predictions. It is the default because the folds and the deployed model
    # must share one scale — otherwise the evaluation measures a model that is
    # not the one being shipped, which is exactly how this went wrong.
    calibration_method: str = "oof"
    calibration_splits: int = 4
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


def fit_oof_calibrators(
    train: pd.DataFrame, features: list[str], horizon: int, config: TrainConfig
) -> list[IsotonicRegression] | None:
    """Fit calibrators on out-of-fold predictions spanning the whole window.

    Each inner fold trains on its own past and predicts its own future, so every
    calibration point is an honest out-of-sample probability, and the points come
    from several different market regimes rather than one recent block.
    """
    cv = PurgedWalkForward(
        n_splits=config.calibration_splits, horizon=horizon, embargo_frac=0.01
    )
    raw_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    inner = TrainConfig(**{**config.__dict__, "calibrate": False})
    for train_idx, test_idx in cv.split(train.index):
        fold_train, fold_test = train.iloc[train_idx], train.iloc[test_idx]
        dataset = lgb.Dataset(
            fold_train[features], label=fold_train["label_class"], free_raw_data=False
        )
        booster = lgb.train(
            inner.lgb_params(), dataset, num_boost_round=inner.num_boost_round
        )
        raw_parts.append(np.asarray(booster.predict(fold_test[features])))
        label_parts.append(fold_test["label_class"].to_numpy())

    if not raw_parts:
        return None

    raw = np.concatenate(raw_parts)
    labels = np.concatenate(label_parts)

    calibrators = []
    for cls in range(N_CLASSES):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw[:, cls], (labels == cls).astype(float))
        calibrators.append(iso)
    return calibrators


def train_fold(
    train: pd.DataFrame, test: pd.DataFrame, config: TrainConfig, *, horizon: int
) -> FoldResult:
    """Fit one model on `train` and predict `test`.

    `horizon` has no default on purpose. It sets the purge inside out-of-fold
    calibration, and a default of 24 was silently applied to every 48- and
    72-bar experiment: inner training labels then overlapped inner test windows,
    and the calibrator learned from probabilities that had seen their answers.
    """
    features = feature_columns(train)
    features = [f for f in features if f in test.columns]

    fit_df = train
    calibrators = None

    if config.calibrate and config.calibration_method == "oof":
        # Train the booster on everything; calibrate from inner out-of-fold runs.
        dataset = lgb.Dataset(
            train[features], label=train["label_class"], free_raw_data=False
        )
        booster = lgb.train(
            config.lgb_params(), dataset, num_boost_round=config.num_boost_round
        )
        calibrators = fit_oof_calibrators(train, features, horizon, config)
        raw = np.asarray(booster.predict(test[features]))
        proba = _apply_calibrators(raw, calibrators) if calibrators else raw
        gains = booster.feature_importance(importance_type="gain")
        return FoldResult(
            proba=proba,
            y_true=test["label_class"].to_numpy(),
            booster=booster,
            calibrators=calibrators,
            importance=dict(zip(features, (float(g) for g in gains), strict=True)),
            features=features,
        )

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

        result = train_fold(train, test, config, horizon=horizon)
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


# A saved model whose signal rate differs from the evaluated one by more than
# this factor is not the model that was evaluated, whatever the metrics said.
# How far the saved model's signal rate may drift from the planned one before it
# is refused. Sampling noise on a 1,000-bar check at 8% coverage is about ±1.3x
# at two sigma, so 2.0 sits clearly outside noise while still catching the real
# failures: a 2.5x shortfall means the strategy trades less than half as often as
# the backtest it was approved on, over different bars.
#
# Was 3.0, which passed a model that fired on 3.2% of unseen bars against an 8%
# target. Loosening this is how a broken rule ships.
COVERAGE_TOLERANCE = 2.0

# Bars withheld from the final fit so the trading rule can be set on data the
# deployed model has never seen. Roughly four months of hourly bars: enough for a
# stable quantile, small enough that the model stays current.
CUTOFF_HOLDOUT_BARS = 3000


def apply_calibrators(raw, calibrators):
    """Public wrapper: callers outside this module need the same mapping the
    trainer applies, or they would compute a cutoff on a different scale."""
    return _apply_calibrators(raw, calibrators)


def coverage_check(
    result: FoldResult,
    dataset: pd.DataFrame,
    expected_coverage: float,
    cutoff: float,
    sample: int = 3000,
) -> dict[str, Any]:
    """Does the final model fire as often as the folds did?

    Metrics come from fold models; the registry stores a differently-fitted final
    model. Nothing checks that the two behave alike, and they can diverge badly:
    a final model calibrated on one recent block once produced signals on 0.3% of
    bars where the folds produced 8.4%, which would have left a live test unable
    to record anything at all while every report still looked healthy.

    `dataset` must be data the model was not fitted on AND the cutoff was not
    derived from. The first version of this check ran on the model's own training
    rows, where margins are wide; it reported 12.87% coverage on a model that then
    fired on 0 of 336 live bars.
    """
    from cryptopred.models.selection import signals_from_margin

    recent = dataset.tail(sample)
    features = [f for f in result.features if f in recent.columns]
    raw = np.asarray(result.booster.predict(recent[features]))
    proba = _apply_calibrators(raw, result.calibrators) if result.calibrators else raw

    actual = float((signals_from_margin(proba, cutoff) != 0).mean())

    if expected_coverage <= 0:
        ratio = float("inf") if actual > 0 else 1.0
    else:
        ratio = actual / expected_coverage if actual > 0 else float("inf")
        ratio = max(ratio, 1 / ratio) if actual > 0 else float("inf")

    return {
        "expected_coverage": expected_coverage,
        "actual_coverage": actual,
        "ratio": ratio,
        "ok": bool(ratio <= COVERAGE_TOLERANCE),
        "n_sample": int(len(recent)),
    }
