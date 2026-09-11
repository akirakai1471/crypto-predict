"""The drift check must catch the failure it was written for.

A model was saved claiming an 8% signal rate and then fired on 0 of 336 bars for
sixteen days with nothing on screen saying so. These tests pin that case, and the
cases where warning would be wrong.
"""

import numpy as np
import pandas as pd

from cryptopred.serve.drift import DRIFT_TOLERANCE, coverage_drift, format_drift


def _history(margins, version="v1", flat=False):
    """A prediction log with the given directional margins."""
    margins = np.asarray(margins, dtype=float)
    up = 0.4 + margins / 2
    down = 0.4 - margins / 2
    return pd.DataFrame(
        {
            "prob_down": down,
            "prob_flat": np.full(len(margins), 0.9 if flat else 0.2),
            "prob_up": up,
            "model_version": version,
        }
    )


def test_a_model_that_never_fires_is_reported_silent():
    """The exact failure: every margin below the cutoff, nothing on screen."""
    drift = coverage_drift(_history([0.01] * 336), "v1", cutoff=0.0795, target=0.08)
    assert drift["state"] == "silent"
    assert "0 of 336" in drift["detail"]
    assert "SILENT" in format_drift(drift)


def test_a_model_firing_at_its_target_is_ok():
    margins = np.concatenate([np.full(8, 0.20), np.full(92, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "ok"
    assert drift["actual"] == 0.08


def test_firing_far_too_often_also_counts_as_drift():
    """Drift is two-sided. A rule firing five times too often is not the rule
    that was backtested either, and it burns five times the fees."""
    margins = np.concatenate([np.full(40, 0.20), np.full(60, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "drifted"
    assert drift["ratio"] > DRIFT_TOLERANCE


def test_a_flat_argmax_never_counts_as_a_signal():
    """A wide gap between UP and DOWN is not a trade if FLAT is likeliest."""
    drift = coverage_drift(
        _history([0.20] * 100, flat=True), "v1", cutoff=0.10, target=0.08
    )
    assert drift["state"] == "silent"


def test_other_versions_rows_are_not_counted():
    """An older model's probabilities came from a different calibrator. Judging
    today's cutoff against them compares two unrelated scales."""
    old = _history([0.20] * 300, version="v0")
    new = _history([0.01] * 100, version="v1")
    drift = coverage_drift(pd.concat([old, new]), "v1", cutoff=0.10, target=0.08)
    assert drift["n_rows"] == 100
    assert drift["state"] == "silent"


def test_too_few_rows_is_not_an_alarm():
    """Seeing no signals in 20 bars at an 8% rate is unremarkable."""
    drift = coverage_drift(_history([0.01] * 20), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "too_few"


def test_no_saved_rule_is_reported_rather_than_assumed():
    drift = coverage_drift(_history([0.2] * 100), "v1", cutoff=None, target=0.08)
    assert drift["state"] == "no_rule"


def test_only_the_most_recent_window_is_judged():
    """Drift means 'now', so a long-dead stretch must not dilute it."""
    margins = np.concatenate([np.full(1000, 0.20), np.full(500, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08, window=500)
    assert drift["n_rows"] == 500
    assert drift["state"] == "silent"
