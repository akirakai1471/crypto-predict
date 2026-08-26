import numpy as np
import pytest

from cryptopred.models.metrics import (
    accuracy_by_confidence,
    directional_accuracy,
    evaluate,
    multiclass_brier,
)


def _perfect(y: np.ndarray) -> np.ndarray:
    proba = np.zeros((len(y), 3))
    proba[np.arange(len(y)), y] = 1.0
    return proba


def test_brier_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 2, 2, 0])
    assert multiclass_brier(y, _perfect(y)) == pytest.approx(0.0)


def test_brier_is_two_for_confidently_wrong_predictions():
    y = np.array([0, 0])
    proba = np.zeros((2, 3))
    proba[:, 2] = 1.0
    # squared error is 1 on the true class and 1 on the predicted class
    assert multiclass_brier(y, proba) == pytest.approx(2.0)


def test_brier_penalises_overconfidence():
    y = np.array([0, 1, 2])
    confident = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    hedged = np.full((3, 3), 1 / 3)
    assert multiclass_brier(y, confident) < multiclass_brier(y, hedged)


def test_directional_accuracy_ignores_flat_predictions():
    y = np.array([0, 1, 2, 2])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],   # predicts DOWN, true DOWN -> correct
            [0.1, 0.8, 0.1],   # predicts FLAT -> not a directional call, skipped
            [0.1, 0.1, 0.8],   # predicts UP, true UP -> correct
            [0.8, 0.1, 0.1],   # predicts DOWN, true UP -> wrong
        ]
    )
    result = directional_accuracy(y, proba, threshold=0.5)
    assert result["n_signals"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_directional_accuracy_respects_threshold():
    y = np.array([2, 2])
    proba = np.array([[0.1, 0.3, 0.6], [0.2, 0.4, 0.4]])
    strict = directional_accuracy(y, proba, threshold=0.55)
    assert strict["n_signals"] == 1
    assert strict["accuracy"] == pytest.approx(1.0)


def test_directional_accuracy_with_no_signals_returns_nan():
    y = np.array([0, 1])
    proba = np.full((2, 3), 1 / 3)
    result = directional_accuracy(y, proba, threshold=0.9)
    assert result["n_signals"] == 0
    assert np.isnan(result["accuracy"])


def test_accuracy_by_confidence_buckets_are_ordered_for_a_good_model():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 3, n)
    proba = np.full((n, 3), 0.2)
    proba[np.arange(n), y] = 0.6
    # add noise so not every high-confidence call is right
    flip = rng.random(n) < 0.2
    y_noisy = np.where(flip, (y + 1) % 3, y)

    table = accuracy_by_confidence(y_noisy, proba, bins=(0.0, 0.4, 0.55, 1.01))
    assert set(table.columns) == {"n", "accuracy"}
    assert table["n"].sum() == n


def test_evaluate_returns_all_headline_metrics():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 3, n)
    proba = rng.dirichlet(np.ones(3), n)
    report = evaluate(y, proba, threshold=0.5)

    for key in ["accuracy", "brier", "directional_accuracy", "n_signals", "log_loss"]:
        assert key in report
