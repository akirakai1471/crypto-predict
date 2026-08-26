import numpy as np

from cryptopred.models.report import format_evaluation, verdict


def _evaluation(model_acc: float, baseline_acc: float, n: int = 4000, seed: int = 0) -> dict:
    """Build a synthetic evaluation where the model and one baseline have known
    directional accuracies, so the verdict logic can be tested directly."""
    rng = np.random.default_rng(seed)
    # Only DOWN and UP, so every row is a directional call and the realised
    # accuracy matches the requested one exactly.
    y_true = rng.choice([0, 2], n)

    def proba_with_accuracy(acc: float) -> np.ndarray:
        proba = np.full((n, 3), 0.1)
        correct = rng.random(n) < acc
        targets = np.where(correct, y_true, 2 - y_true)
        proba[np.arange(n), targets] = 0.8
        return proba / proba.sum(axis=1, keepdims=True)

    model_proba = proba_with_accuracy(model_acc)
    base_proba = proba_with_accuracy(baseline_acc)

    from cryptopred.models.metrics import evaluate

    return {
        "model": evaluate(y_true, model_proba, threshold=0.5),
        "baselines": {"coin_flip": evaluate(y_true, base_proba, threshold=0.5)},
        "baseline_proba": {"coin_flip": base_proba},
        "y_true": y_true,
        "proba": model_proba,
        "folds": [
            {
                "fold": 0,
                "n_test": n,
                "test_start": "2024-01-01",
                "test_end": "2024-06-01",
                "directional_accuracy": model_acc,
            }
        ],
        "feature_importance": {"signal": 100.0, "noise": 1.0},
        "n_test_total": n,
    }


def test_verdict_is_go_when_model_clearly_wins():
    result = verdict(_evaluation(model_acc=0.58, baseline_acc=0.33))
    assert result["decision"] == "GO"


def test_verdict_is_no_go_when_baseline_wins():
    result = verdict(_evaluation(model_acc=0.30, baseline_acc=0.45))
    assert result["decision"] == "NO-GO"
    assert "coin_flip" in result["reason"]


def test_verdict_is_no_go_when_margin_is_noise():
    result = verdict(_evaluation(model_acc=0.335, baseline_acc=0.33, n=300))
    assert result["decision"] == "NO-GO"


def test_implausibly_high_accuracy_triggers_investigate():
    result = verdict(_evaluation(model_acc=0.85, baseline_acc=0.33))
    assert result["decision"] == "INVESTIGATE"
    assert "leakage" in result["reason"]


def test_format_evaluation_mentions_the_verdict():
    text = format_evaluation(_evaluation(0.58, 0.33), symbol="BTCUSDT", interval="1h")
    assert "VERDICT" in text
    assert "BTCUSDT" in text
    assert "BASELINES" in text


def test_vacuous_baseline_is_compared_on_brier_not_accuracy():
    """A baseline that never makes a directional call cannot be beaten on
    directional accuracy; the comparison must fall back to Brier instead of
    silently counting as a failure."""
    evaluation = _evaluation(model_acc=0.58, baseline_acc=0.33)
    n = len(evaluation["y_true"])

    # A baseline that always predicts FLAT: no directional signals at all.
    flat_proba = np.zeros((n, 3))
    flat_proba[:, 1] = 1.0
    from cryptopred.models.metrics import evaluate as eval_metrics

    evaluation["baselines"]["prior"] = eval_metrics(
        evaluation["y_true"], flat_proba, threshold=0.5
    )
    evaluation["baseline_proba"]["prior"] = flat_proba

    result = verdict(evaluation)
    prior = result["comparisons"]["prior"]
    assert prior["basis"] == "brier"
    # the always-FLAT baseline is confidently wrong on every row, so any real
    # model beats it on Brier
    assert prior["beaten"] is True
    assert result["decision"] == "GO"


def test_vacuous_baseline_can_still_fail_the_model():
    """The Brier fallback is a real test, not a rubber stamp: a model with worse
    probabilities than the vacuous baseline must still be rejected."""
    evaluation = _evaluation(model_acc=0.58, baseline_acc=0.33)
    n = len(evaluation["y_true"])

    flat_proba = np.zeros((n, 3))
    flat_proba[:, 1] = 1.0
    from cryptopred.models.metrics import evaluate as eval_metrics

    baseline_metrics = eval_metrics(evaluation["y_true"], flat_proba, threshold=0.5)
    baseline_metrics["brier"] = 0.0  # pretend the baseline is perfectly calibrated
    evaluation["baselines"]["prior"] = baseline_metrics
    evaluation["baseline_proba"]["prior"] = flat_proba

    result = verdict(evaluation)
    assert result["comparisons"]["prior"]["beaten"] is False
    assert result["decision"] == "NO-GO"
