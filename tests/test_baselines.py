import numpy as np
import pandas as pd

from cryptopred.models.baselines import BASELINES, baseline_predictions


def _frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "roc_1": rng.normal(0, 0.01, n),
            "label_class": rng.integers(0, 3, n),
            "forward_return": rng.normal(0, 0.01, n),
        },
        index=idx,
    )


def test_every_baseline_returns_probabilities_summing_to_one():
    df = _frame()
    for name in BASELINES:
        proba = baseline_predictions(name, df, train=df)
        assert proba.shape == (len(df), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()


def test_always_up_puts_all_mass_on_the_up_class():
    df = _frame()
    proba = baseline_predictions("always_up", df, train=df)
    assert (proba[:, 2] == 1.0).all()


def test_coin_flip_is_uniform_over_up_and_down():
    df = _frame()
    proba = baseline_predictions("coin_flip", df, train=df)
    assert np.allclose(proba[:, 0], 0.5)
    assert np.allclose(proba[:, 2], 0.5)
    assert np.allclose(proba[:, 1], 0.0)


def test_prior_matches_training_class_frequencies():
    df = _frame()
    train = df.copy()
    train["label_class"] = [0] * 100 + [1] * 300 + [2] * 100
    proba = baseline_predictions("prior", df, train=train)
    assert np.isclose(proba[0, 0], 0.2)
    assert np.isclose(proba[0, 1], 0.6)
    assert np.isclose(proba[0, 2], 0.2)


def test_momentum_follows_the_sign_of_the_last_return():
    df = _frame()
    df["roc_1"] = [0.01, -0.01] * (len(df) // 2)
    proba = baseline_predictions("momentum", df, train=df)
    assert proba[0, 2] > proba[0, 0]   # positive roc -> favours up
    assert proba[1, 0] > proba[1, 2]   # negative roc -> favours down


def test_unknown_baseline_name_raises():
    df = _frame()
    try:
        baseline_predictions("nonsense", df, train=df)
    except ValueError as exc:
        assert "nonsense" in str(exc)
    else:
        raise AssertionError("expected ValueError")
