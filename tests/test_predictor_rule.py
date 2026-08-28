import numpy as np
import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.serve.predictor import Predictor


class _Bundle:
    """Minimal stand-in: the rule under test lives in the predictor, not the model."""

    def __init__(self, proba, cutoff):
        self._proba = np.array([proba], dtype="float64")
        self.metadata = {"version": "test", "margin_cutoff": cutoff}
        self.features = []

    def predict(self, frame):
        return self._proba


def _bar():
    return pd.Series(
        {
            "close": 100.0,
            "close_time": pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        }
    )


def _predict(proba, cutoff):
    predictor = Predictor(Config(), _Bundle(proba, cutoff), store=None)
    return predictor._to_prediction("BTCUSDT", "1h", _bar(), np.array(proba))


def test_a_margin_above_the_cutoff_fires():
    p = _predict([0.10, 0.20, 0.70], cutoff=0.30)
    assert p.signal == 1


def test_a_margin_below_the_cutoff_stays_silent():
    p = _predict([0.30, 0.20, 0.50], cutoff=0.30)   # margin 0.20
    assert p.signal == 0


def test_the_short_side_works_the_same_way():
    p = _predict([0.70, 0.20, 0.10], cutoff=0.30)
    assert p.signal == -1


def test_a_flat_prediction_never_fires_however_wide_the_margin():
    p = _predict([0.05, 0.90, 0.05], cutoff=0.0)
    assert p.signal == 0


def test_a_high_maximum_with_a_thin_margin_stays_silent():
    """The failure a probability threshold could not see: 0.45 versus 0.44 looks
    confident and is a coin flip."""
    p = _predict([0.44, 0.11, 0.45], cutoff=0.10)
    assert p.signal == 0


def test_a_model_without_a_cutoff_refuses_to_trade():
    """Older models predate the rule. Falling back to some default threshold
    would trade on a rule nobody chose."""
    predictor = Predictor(Config(), _Bundle([0.05, 0.05, 0.90], cutoff=None), store=None)
    p = predictor._to_prediction("BTCUSDT", "1h", _bar(), np.array([0.05, 0.05, 0.90]))
    assert p.signal == 0


def test_confidence_is_still_reported_even_when_silent():
    p = _predict([0.30, 0.20, 0.50], cutoff=0.30)
    assert p.confidence == pytest.approx(0.50)
    assert p.proba == pytest.approx((0.30, 0.20, 0.50))
