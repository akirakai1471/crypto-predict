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
    margins = np.concatenate([np.full(40, 0.20), np.full(460, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "ok"
    assert drift["actual"] == 0.08


def test_firing_far_too_often_also_counts_as_drift():
    """Drift is two-sided. A rule firing nine times too often is not the rule
    that was backtested either, and it burns nine times the fees."""
    margins = np.concatenate([np.full(350, 0.20), np.full(150, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "drifted"
    assert drift["ratio"] > DRIFT_TOLERANCE


def test_a_flat_argmax_never_counts_as_a_signal():
    """A wide gap between UP and DOWN is not a trade if FLAT is likeliest."""
    drift = coverage_drift(
        _history([0.20] * 400, flat=True), "v1", cutoff=0.10, target=0.08
    )
    assert drift["state"] == "silent"


def test_other_versions_rows_are_not_counted():
    """An older model's probabilities came from a different calibrator. Judging
    today's cutoff against them compares two unrelated scales."""
    old = _history([0.20] * 300, version="v0")
    new = _history([0.01] * 400, version="v1")
    drift = coverage_drift(pd.concat([old, new]), "v1", cutoff=0.10, target=0.08)
    assert drift["n_rows"] == 400
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


def test_the_window_is_the_newest_bars_even_when_the_log_arrives_newest_first(tmp_path):
    """PredictionStore.history returns newest first, and every caller passes it
    straight in. A window taken off the end of that frame is the OLDEST bars:
    past `window` rows the check would report a model's first weeks forever and
    never see it go silent.
    """
    from cryptopred.serve.store import PredictionStore

    store = PredictionStore(tmp_path / "predictions.db")
    start = pd.Timestamp("2026-09-01T00:00Z")
    for i in range(900):
        fires = i < 500  # fired for 500 bars, then went quiet
        proba = (0.1, 0.3, 0.6) if fires else (0.33, 0.33, 0.34)
        store.record_prediction(
            symbol="BTCUSDT",
            interval="1h",
            bar_close_time=start + pd.Timedelta(hours=i),
            proba=proba,
            signal=int(fires),
            close_price=1.0,
            model_version="v1",
        )

    history = store.history("BTCUSDT", "1h", limit=100_000)
    drift = coverage_drift(history, "v1", cutoff=0.10, target=0.08, window=400)
    assert drift["state"] == "silent"
    assert "0 of 400" in drift["detail"]



# -- review findings, 2026-10-01: drift measured against a sound model's noise ------


def test_a_rate_a_sound_model_often_shows_is_not_called_drift():
    """3.2% against an 8% target over 500 bars is 2.5x. A model working as
    built strays past 2x in 43% of 500-bar windows; this used to be "drifted"
    and was read as evidence that something broke."""
    margins = np.concatenate([np.full(16, 0.20), np.full(484, 0.01)])
    drift = coverage_drift(_history(margins), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "ok"
    assert "4 windows of 10" in drift["detail"]


def test_silence_over_a_few_days_is_not_yet_an_alarm():
    """A sound model fires on nothing in one 200-bar window in eight."""
    drift = coverage_drift(_history([0.01] * 200), "v1", cutoff=0.10, target=0.08)
    assert drift["state"] == "too_few"
