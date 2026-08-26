"""Meta-labelling: a second model that learns which signals are worth taking.

The primary model answers "which way?". It is deliberately run at a loose
threshold so it casts a wide net. A secondary model then answers the different
and more useful question: "given this signal, does the trade make money after
costs?" — and it learns that from simulated trade outcomes, not from price
labels.

Splitting the two questions helps because they have different answers. A bar can
be genuinely more likely to rise and still be a bad trade, because the expected
move is smaller than the toll. The primary model has no way to express that; the
secondary one is trained on exactly it.

**The trap this module exists to avoid.** Meta-labels must come from primary
predictions the primary model has never seen the answers to. Training the
secondary on in-sample primary probabilities teaches it to trust a confidence
that will not exist in production: in-sample the primary looks near-perfect, so
the secondary learns "always take the trade", and the whole apparatus becomes an
expensive no-op that only reveals itself in live trading. Every primary
probability fed to the secondary here comes from an inner purged walk-forward
split inside the training window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.dataset.builder import feature_columns
from cryptopred.models.splits import PurgedWalkForward
from cryptopred.models.train import TrainConfig, train_fold

DOWN, FLAT, UP = 0, 1, 2


@dataclass
class MetaConfig:
    """Knobs for the two-model stack."""

    # Loose on purpose: the primary casts a wide net and the secondary filters.
    primary_threshold: float = 0.40
    meta_threshold: float = 0.55
    inner_splits: int = 3
    primary: TrainConfig = field(default_factory=lambda: TrainConfig(num_boost_round=300))
    secondary: TrainConfig = field(
        default_factory=lambda: TrainConfig(num_boost_round=200, min_data_in_leaf=100)
    )
    # Weight each meta-label by how much money the trade made or lost, so the
    # secondary cares about a 3% winner more than a 0.01% scratch.
    weight_by_magnitude: bool = True


def signals_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """+1 / -1 / 0 per row."""
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    signal = np.zeros(len(proba), dtype=int)
    signal[(predicted == UP) & (confidence >= threshold)] = 1
    signal[(predicted == DOWN) & (confidence >= threshold)] = -1
    return signal


def simulate_signal_returns(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    horizon: int,
    costs: CostModel | None = None,
) -> pd.Series:
    """Net return of the trade each signal would have produced.

    Delegates to the backtest engine rather than recomputing entry and exit
    prices, so meta-labels can never drift from what the backtest scores.
    """
    costs = costs or CostModel()
    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals}, index=index)
    result = backtest(window, frame, horizon=horizon, costs=costs)
    if result.trades.empty:
        return pd.Series(dtype="float64")
    return result.trades.set_index("signal_time")["net_return"]


def oof_primary_proba(
    dataset: pd.DataFrame, horizon: int, config: MetaConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold primary probabilities inside a training window.

    Returns (proba, mask) where mask marks the rows that received a genuine
    out-of-fold prediction. Early rows never appear in an inner test block —
    walk-forward has no way to predict them without looking forward — so they
    are excluded rather than filled in.
    """
    n = len(dataset)
    proba = np.full((n, 3), np.nan)
    mask = np.zeros(n, dtype=bool)

    cv = PurgedWalkForward(
        n_splits=config.inner_splits, horizon=horizon, embargo_frac=0.01
    )
    for train_idx, test_idx in cv.split(dataset.index):
        result = train_fold(
            dataset.iloc[train_idx], dataset.iloc[test_idx], config.primary
        )
        proba[test_idx] = result.proba
        mask[test_idx] = True

    return proba, mask


def build_meta_training_set(
    dataset: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
    config: MetaConfig,
    costs: CostModel | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features, labels and weights for the secondary model.

    A row exists for every bar where the primary fired an honest out-of-fold
    signal. The label is whether that trade would have made money after costs.
    """
    proba, mask = oof_primary_proba(dataset, horizon, config)
    signals = np.zeros(len(dataset), dtype=int)
    signals[mask] = signals_from_proba(proba[mask], config.primary_threshold)

    fired = signals != 0
    if not fired.any():
        empty = pd.DataFrame()
        return empty, pd.Series(dtype="int8"), pd.Series(dtype="float64")

    returns = simulate_signal_returns(
        bars, dataset.index, signals, horizon=horizon, costs=costs
    )
    if returns.empty:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype="int8"), pd.Series(dtype="float64")

    rows = dataset.index[fired].intersection(returns.index)
    features = dataset.loc[rows, feature_columns(dataset)].copy()

    positions = dataset.index.get_indexer(rows)
    features["primary_prob_down"] = proba[positions, DOWN]
    features["primary_prob_flat"] = proba[positions, FLAT]
    features["primary_prob_up"] = proba[positions, UP]
    features["primary_confidence"] = proba[positions].max(axis=1)
    features["primary_signal"] = signals[positions]

    net = returns.loc[rows]
    labels = (net > 0).astype("int8")
    weights = net.abs() if config.weight_by_magnitude else pd.Series(1.0, index=rows)
    # A zero-weight row teaches nothing; give every row a floor.
    weights = weights.clip(lower=weights[weights > 0].min() if (weights > 0).any() else 1.0)

    return features, labels, weights


def train_secondary(
    features: pd.DataFrame, labels: pd.Series, weights: pd.Series, config: MetaConfig
) -> lgb.Booster:
    """Binary model: will this signal's trade be profitable after costs?"""
    params = config.secondary.lgb_params()
    params.update({"objective": "binary", "num_class": 1, "metric": "binary_logloss"})
    dataset = lgb.Dataset(features, label=labels, weight=weights, free_raw_data=False)
    return lgb.train(params, dataset, num_boost_round=config.secondary.num_boost_round)


def apply_meta(
    booster: lgb.Booster,
    dataset: pd.DataFrame,
    primary_proba: np.ndarray,
    meta_features: list[str],
    config: MetaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter primary signals through the secondary model.

    Returns (final_signals, meta_probability). Rows the primary skipped get a
    meta probability of NaN — the secondary was never asked about them.
    """
    primary_signals = signals_from_proba(primary_proba, config.primary_threshold)
    meta_prob = np.full(len(dataset), np.nan)
    final = np.zeros(len(dataset), dtype=int)

    fired = primary_signals != 0
    if not fired.any():
        return final, meta_prob

    features = dataset.loc[fired, feature_columns(dataset)].copy()
    features["primary_prob_down"] = primary_proba[fired, DOWN]
    features["primary_prob_flat"] = primary_proba[fired, FLAT]
    features["primary_prob_up"] = primary_proba[fired, UP]
    features["primary_confidence"] = primary_proba[fired].max(axis=1)
    features["primary_signal"] = primary_signals[fired]

    scores = np.asarray(booster.predict(features[meta_features]))
    meta_prob[fired] = scores

    take = fired.copy()
    take[fired] = scores >= config.meta_threshold
    final[take] = primary_signals[take]
    return final, meta_prob


def walk_forward_meta(
    dataset: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
    n_splits: int = 5,
    config: MetaConfig | None = None,
    costs: CostModel | None = None,
) -> dict[str, Any]:
    """Evaluate the two-model stack out-of-sample, fold by fold.

    Within each outer fold the secondary is built entirely from inner-fold
    predictions, so nothing the secondary learns from has leaked across the
    outer boundary.
    """
    config = config or MetaConfig()
    cv = PurgedWalkForward(n_splits=n_splits, horizon=horizon, embargo_frac=0.01)

    final_signals: list[np.ndarray] = []
    primary_signals: list[np.ndarray] = []
    indices: list[pd.Index] = []
    folds: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(dataset.index)):
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]

        meta_x, meta_y, meta_w = build_meta_training_set(
            train, bars, horizon=horizon, config=config, costs=costs
        )
        if meta_x.empty or meta_y.nunique() < 2:
            # Nothing to learn from: fall back to the primary alone rather than
            # inventing a filter.
            result = train_fold(train, test, config.primary)
            primary = signals_from_proba(result.proba, config.primary_threshold)
            final_signals.append(primary)
            primary_signals.append(primary)
            indices.append(test.index)
            folds.append(
                {"fold": fold_id, "meta_trained": False, "n_meta_rows": 0,
                 "n_primary": int((primary != 0).sum()), "n_final": int((primary != 0).sum())}
            )
            continue

        booster = train_secondary(meta_x, meta_y, meta_w, config)
        result = train_fold(train, test, config.primary)
        primary = signals_from_proba(result.proba, config.primary_threshold)
        final, _ = apply_meta(
            booster, test, result.proba, list(meta_x.columns), config
        )

        final_signals.append(final)
        primary_signals.append(primary)
        indices.append(test.index)
        folds.append(
            {
                "fold": fold_id,
                "meta_trained": True,
                "n_meta_rows": int(len(meta_x)),
                "meta_positive_rate": float(meta_y.mean()),
                "n_primary": int((primary != 0).sum()),
                "n_final": int((final != 0).sum()),
            }
        )

    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    return {
        "index": index,
        "final_signals": np.concatenate(final_signals),
        "primary_signals": np.concatenate(primary_signals),
        "folds": folds,
        "config": config,
    }
