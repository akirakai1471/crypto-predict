import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.status import (
    MIN_SIGNALS_FOR_A_CLAIM,
    collect,
    format_status,
    wilson_interval,
)
from cryptopred.serve.store import PredictionStore
from tests.conftest import make_ohlcv


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.data.root = tmp_path
    c.data.symbols = ["BTCUSDT"]
    ParquetStore(tmp_path / "raw").write("klines", "BTCUSDT", "1h", make_ohlcv(n=200, seed=3))
    return c


def _log(cfg, n_signals, n_correct, n_flat=0, scored=True):
    store = PredictionStore(cfg.data.root / "predictions.db")
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n_signals + n_flat):
        store.record_prediction(
            symbol="BTCUSDT", interval="1h",
            bar_close_time=base + pd.Timedelta(hours=i),
            proba=(0.2, 0.2, 0.6),
            signal=1 if i < n_signals else 0,
            close_price=100.0, model_version="v1",
        )

    if scored:
        pending = store.unscored("BTCUSDT", "1h")
        for i, (_, r) in enumerate(pending.iterrows()):
            correct = int(r["signal"]) != 0 and i < n_correct
            store.score_prediction(int(r["id"]), 0.01 if correct else -0.01, 1, correct)
    return store


def test_wilson_interval_is_wide_at_tiny_samples():
    low, high = wilson_interval(3, 4)
    assert high - low > 0.4      # four observations say almost nothing


def test_wilson_interval_narrows_as_evidence_accumulates():
    small = wilson_interval(60, 100)
    large = wilson_interval(600, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_handles_zero_samples():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(0, 5)
    assert low >= 0.0 and high <= 1.0


def test_collect_reports_nothing_before_any_prediction(cfg):
    assert collect(cfg)["symbols"] == []


def test_collect_counts_signals_separately_from_bars(cfg):
    _log(cfg, n_signals=5, n_correct=3, n_flat=10)
    s = collect(cfg)["symbols"][0]
    assert s["n_predictions"] == 15
    assert s["n_signals"] == 5
    assert s["n_scored_signals"] == 5
    assert s["signal_accuracy"] == pytest.approx(3 / 5)


def test_unscored_predictions_are_excluded_from_the_rate(cfg):
    _log(cfg, n_signals=4, n_correct=4, scored=False)
    s = collect(cfg)["symbols"][0]
    assert s["n_scored"] == 0
    assert s["signal_accuracy"] is None


def test_small_samples_are_labelled_not_yet_meaningful(cfg):
    _log(cfg, n_signals=6, n_correct=5)
    text = format_status(collect(cfg))
    assert "NOT YET MEANINGFUL" in text
    assert "83.3%" in text          # the number is shown, but framed


def test_a_large_winning_sample_is_allowed_to_claim(cfg):
    n = MIN_SIGNALS_FOR_A_CLAIM + 50
    _log(cfg, n_signals=n, n_correct=int(n * 0.62))
    text = format_status(collect(cfg))
    assert "NOT YET MEANINGFUL" not in text
    assert "clears 50%" in text


def test_a_large_losing_sample_says_the_edge_is_negative(cfg):
    n = MIN_SIGNALS_FOR_A_CLAIM + 50
    _log(cfg, n_signals=n, n_correct=int(n * 0.35))
    text = format_status(collect(cfg))
    assert "BELOW 50%" in text


def test_a_large_ambiguous_sample_says_so(cfg):
    n = MIN_SIGNALS_FOR_A_CLAIM + 50
    _log(cfg, n_signals=n, n_correct=int(n * 0.51))
    text = format_status(collect(cfg))
    assert "straddles 50%" in text


def test_status_reports_when_nothing_is_running(cfg):
    """Liveness is stated outright, not left for the reader to infer from an
    empty table."""
    text = format_status(collect(cfg))
    assert "Scheduler: NEVER STARTED" in text
    assert "Nothing recorded yet" in text


def test_status_always_shows_the_sample_size(cfg):
    _log(cfg, n_signals=8, n_correct=6)
    text = format_status(collect(cfg))
    assert "6/8" in text
    assert "95% CI" in text


def test_backfilled_rows_are_excluded_from_the_headline_rate(cfg):
    """A row written after its outcome existed cannot prove it was not
    influenced by it, so it must not feed the number used as evidence."""
    store = PredictionStore(cfg.data.root / "predictions.db")
    base = pd.Timestamp("2024-01-01", tz="UTC")

    for i in range(10):
        store.record_prediction(
            symbol="BTCUSDT", interval="1h",
            bar_close_time=base + pd.Timedelta(hours=i),
            proba=(0.2, 0.2, 0.6), signal=1, close_price=100.0,
            model_version="v1",
            was_backfilled=i >= 4,        # last six were filled after a shutdown
        )
    for _, r in store.unscored("BTCUSDT", "1h").iterrows():
        store.score_prediction(int(r["id"]), 0.01, 1, True)

    s = collect(cfg)["symbols"][0]
    assert s["n_predictions"] == 10
    assert s["n_live"] == 4
    assert s["n_backfilled"] == 6
    assert s["n_scored_signals"] == 4      # only the live rows count

    text = format_status(collect(cfg))
    assert "backfilled" in text
    assert "cannot prove" in text


# -- funding freshness, and what drift is allowed to recommend ------------------


def _funding_at(cfg, hours_ago):
    t = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(hours=hours_ago)
    frame = pd.DataFrame(
        {"funding_rate": [0.0001]}, index=pd.DatetimeIndex([t], name="funding_time")
    )
    ParquetStore(cfg.data.root / "raw").write("funding", "BTCUSDT", "8h", frame)


def test_frozen_funding_is_named_before_anything_else_is_judged(cfg):
    """176 hours old on the live machine on 2026-09-30, and nothing said so."""
    _log(cfg, n_signals=3, n_correct=2)
    _funding_at(cfg, 176)
    text = format_status(collect(cfg))
    assert "BTCUSDT funding: last rate 176" in text
    assert "not current" in text
    assert "funding:            STALE" in text


def test_current_funding_raises_no_alarm(cfg):
    _log(cfg, n_signals=3, n_correct=2)
    _funding_at(cfg, 2)
    text = format_status(collect(cfg))
    assert "funding:            FRESH" in text
    assert "not current" not in text


def test_no_funding_at_all_is_reported_missing(tmp_path):
    from cryptopred.serve.freshness import funding_freshness

    assert funding_freshness(ParquetStore(tmp_path), "BTCUSDT")["state"] == "missing"


def test_drift_never_hands_over_a_retrain_command(cfg):
    """It used to print `train ... --save`. The first time it fired after a fix,
    its window was bars predicted on frozen funding: following it would have
    fitted a new model to a diagnosis of the bug."""
    _log(cfg, n_signals=3, n_correct=2)
    status = collect(cfg)
    status["symbols"][0]["drift"] = {
        "state": "drifted",
        "detail": "fires on 3.2% of the last 189 bars against a 8% target (2.5x)",
    }
    text = format_status(status)
    assert "3.2%" in text
    assert "--save" not in text
    assert "Do NOT retrain on this line alone" in text
    assert "BOTH verdict lines" in text
