import numpy as np
import pytest

from cryptopred.models.selection import (
    coverage_by_fold,
    signals_by_quantile,
    signals_by_quantile_per_fold,
)


def _proba(rows):
    return np.array(rows, dtype="float64")


def test_takes_the_requested_fraction_of_bars():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([2, 1, 2], 1000)
    signals = signals_by_quantile(proba, coverage=0.10)
    assert (signals != 0).sum() == pytest.approx(100, abs=2)


def test_coverage_is_stable_regardless_of_the_probability_scale():
    """The whole point: a monotone recalibration must not change what is traded.

    A fixed threshold fails this — it selected 26.6% of one fold and 0.03% of
    another with the same model, purely from calibration differences.
    """
    rng = np.random.default_rng(1)
    proba = rng.dirichlet([2, 1, 2], 500)

    # Squash everything toward the middle, as a well-fitted calibrator does.
    squashed = 0.5 + (proba - proba.mean(axis=1, keepdims=True)) * 0.15
    squashed = squashed / squashed.sum(axis=1, keepdims=True)

    a = signals_by_quantile(proba, coverage=0.10)
    b = signals_by_quantile(squashed, coverage=0.10)
    assert (a != 0).sum() == (b != 0).sum()


def test_ranks_by_directional_margin_not_raw_maximum():
    """A 0.40 versus 0.39 call is a coin flip however big the numbers look."""
    proba = _proba([
        [0.40, 0.21, 0.39],   # large maximum, tiny margin — a coin flip
        [0.30, 0.35, 0.35],   # smaller maximum, but no directional edge either
        [0.05, 0.15, 0.80],   # genuine conviction
    ])
    signals = signals_by_quantile(proba, coverage=1 / 3)
    assert signals[2] == 1
    assert signals[0] == 0


def test_flat_predictions_are_never_traded():
    proba = _proba([
        [0.10, 0.80, 0.10],
        [0.15, 0.70, 0.15],
        [0.30, 0.60, 0.10],
    ])
    assert (signals_by_quantile(proba, coverage=1.0) == 0).all()


def test_direction_follows_the_stronger_side():
    proba = _proba([
        [0.10, 0.10, 0.80],
        [0.80, 0.10, 0.10],
    ])
    signals = signals_by_quantile(proba, coverage=1.0)
    assert list(signals) == [1, -1]


def test_full_coverage_trades_every_directional_bar():
    proba = _proba([
        [0.10, 0.10, 0.80],
        [0.10, 0.80, 0.10],   # flat: still skipped
        [0.80, 0.10, 0.10],
    ])
    assert (signals_by_quantile(proba, coverage=1.0) != 0).sum() == 2


def test_rejects_a_nonsense_coverage():
    proba = _proba([[0.1, 0.1, 0.8]])
    with pytest.raises(ValueError, match="coverage"):
        signals_by_quantile(proba, coverage=0.0)
    with pytest.raises(ValueError, match="coverage"):
        signals_by_quantile(proba, coverage=1.5)


def test_tiny_coverage_takes_nothing_rather_than_guessing():
    rng = np.random.default_rng(2)
    proba = rng.dirichlet([2, 1, 2], 10)
    assert (signals_by_quantile(proba, coverage=0.001) == 0).all()


def test_per_fold_selection_gives_each_fold_the_same_share():
    rng = np.random.default_rng(3)
    proba = rng.dirichlet([2, 1, 2], 900)
    folds = np.repeat([0, 1, 2], 300)

    # Inflate one fold's probabilities, as an overfitted calibrator would.
    proba[folds == 0] = np.clip(proba[folds == 0] * 3, 0, 1)
    proba[folds == 0] /= proba[folds == 0].sum(axis=1, keepdims=True)

    signals = signals_by_quantile_per_fold(proba, folds, coverage=0.10)
    per_fold = coverage_by_fold(signals, folds)
    assert all(abs(v - 0.10) < 0.02 for v in per_fold.values())


def test_pooled_ranking_would_let_one_fold_dominate():
    """Demonstrates why per-fold selection exists."""
    rng = np.random.default_rng(4)
    proba = rng.dirichlet([2, 1, 2], 900)
    folds = np.repeat([0, 1, 2], 300)
    # Sharpen fold 0 toward the corners, which is what an overfitted isotonic
    # map does: the same underlying opinion expressed far more emphatically.
    sharp = proba[folds == 0] ** 4
    proba[folds == 0] = sharp / sharp.sum(axis=1, keepdims=True)

    pooled = signals_by_quantile(proba, coverage=0.10)
    shares = coverage_by_fold(pooled, folds)
    assert shares[0] > shares[1]           # the inflated fold takes more than its share

    balanced = coverage_by_fold(
        signals_by_quantile_per_fold(proba, folds, coverage=0.10), folds
    )
    assert abs(balanced[0] - balanced[1]) < 0.02


def test_coverage_by_fold_reports_each_fold():
    signals = np.array([1, 0, -1, 0, 0, 1])
    folds = np.array([0, 0, 0, 1, 1, 1])
    assert coverage_by_fold(signals, folds) == {0: pytest.approx(2 / 3), 1: pytest.approx(1 / 3)}
