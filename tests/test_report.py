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
    result = verdict(_evaluation(model_acc=0.58, baseline_acc=0.33), coverage=1.0)
    assert result["decision"] == "GO"


def test_verdict_is_no_go_when_baseline_wins():
    result = verdict(_evaluation(model_acc=0.30, baseline_acc=0.45), coverage=1.0)
    assert result["decision"] == "NO-GO"
    assert "coin_flip" in result["reason"]


def test_verdict_is_no_go_when_margin_is_noise():
    result = verdict(_evaluation(model_acc=0.335, baseline_acc=0.33, n=300), coverage=1.0)
    assert result["decision"] == "NO-GO"


def test_implausibly_high_accuracy_triggers_investigate():
    result = verdict(_evaluation(model_acc=0.85, baseline_acc=0.33), coverage=1.0)
    assert result["decision"] == "INVESTIGATE"
    assert "leakage" in result["reason"]


def test_format_evaluation_mentions_the_verdict():
    text = format_evaluation(
        _evaluation(0.58, 0.33), symbol="BTCUSDT", interval="1h", coverage=1.0
    )
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

    result = verdict(evaluation, coverage=1.0)
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

    result = verdict(evaluation, coverage=1.0)
    assert result["comparisons"]["prior"]["beaten"] is False
    assert result["decision"] == "NO-GO"


# -- review findings, 2026-10-01: score the rule that is traded --------------------


def _ranked_evaluation(n=6000, seed=1, uptrend_on_confident=False):
    """Margins vary; the confident 8% of each fold is right 60% of the time and
    the rest is noise. Two folds, so the per-fold ranking matters."""
    rng = np.random.default_rng(seed)
    margin = rng.uniform(0.0, 0.4, n)
    confident = np.zeros(n, dtype=bool)
    for half in (slice(0, n // 2), slice(n // 2, n)):
        idx = np.arange(n)[half]
        top = idx[np.argsort(margin[half])[-int(round(0.08 * len(idx))):]]
        confident[top] = True
    calls_up = rng.random(n) < 0.5
    if uptrend_on_confident:
        calls_up[confident] = True
    y_true = np.where(calls_up, 2, 0)
    wrong = rng.random(n) < np.where(confident, 0.40, 0.50)
    y_true = np.where(wrong, 2 - y_true, y_true)
    if uptrend_on_confident:
        # On the confident bars price simply rose: always_up is right as often.
        y_true[confident] = np.where(rng.random(confident.sum()) < 0.6, 2, 0)
    proba = np.full((n, 3), 0.2)
    proba[:, 2] = np.where(calls_up, 0.4 + margin / 2, 0.4 - margin / 2)
    proba[:, 0] = np.where(calls_up, 0.4 - margin / 2, 0.4 + margin / 2)
    proba = proba / proba.sum(axis=1, keepdims=True)

    import pandas as pd

    from cryptopred.models.baselines import baseline_predictions

    frame = pd.DataFrame({"roc_1": rng.normal(size=n), "label_class": y_true})
    baseline_proba = {
        name: baseline_predictions(name, frame, frame)
        for name in ("coin_flip", "always_up", "always_down", "momentum")
    }
    from cryptopred.models.metrics import evaluate

    return {
        "model": evaluate(y_true, proba),
        "baselines": {k: evaluate(y_true, v) for k, v in baseline_proba.items()},
        "baseline_proba": baseline_proba,
        "y_true": y_true,
        "proba": proba,
        "folds": [
            {"fold": 0, "n_test": n // 2, "test_start": "2024-01-01", "test_end": "2024-06-01"},
            {"fold": 1, "n_test": n - n // 2, "test_start": "2024-06-01", "test_end": "2025-01-01"},
        ],
        "feature_importance": {"signal": 1.0},
        "n_test_total": n,
    }


def test_the_gate_scores_the_top_eight_percent_of_each_fold():
    result = verdict(_ranked_evaluation(), coverage=0.08)
    assert result["n_selected"] == 480
    assert {f["n"] for f in result["per_fold"].values()} == {240}
    assert 0.53 < result["model_accuracy"] < 0.68


def test_a_lead_that_is_only_row_selection_is_not_a_pass():
    """On BTC the model called UP on 99% of the bars it chose, and always_up
    on those same bars scored exactly what it did. The gap the old report
    printed came from scoring the baselines on every bar instead."""
    result = verdict(_ranked_evaluation(uptrend_on_confident=True), coverage=0.08)
    assert result["calls_up"] == 1.0
    assert abs(result["comparisons"]["always_up"]["gap"]) < 1e-12
    assert result["decision"] == "NO-GO"


def test_a_coin_flip_is_scored_as_a_coin_flip_not_as_always_down():
    from cryptopred.models.report import _credit

    proba = np.array([[0.5, 0.0, 0.5]] * 3)
    truth = np.array([0, 2, 1])
    assert list(_credit(proba, truth)) == [0.5, 0.5, 0.0]


def test_overlapping_outcomes_widen_the_interval():
    """Rows that share most of their 24-bar window are not independent; a
    bootstrap that resamples them one by one is several times too narrow."""
    from cryptopred.models.metrics import block_bootstrap_mean

    rng = np.random.default_rng(0)
    runs = np.repeat(rng.choice([-1.0, 1.0], 100), 24)[:2000] + 0.1
    iid = block_bootstrap_mean(runs, block=1)
    blocked = block_bootstrap_mean(runs, block=48)
    assert (blocked["ci_upper"] - blocked["ci_lower"]) > 3 * (iid["ci_upper"] - iid["ci_lower"])
    assert blocked["n_blocks"] == 42
