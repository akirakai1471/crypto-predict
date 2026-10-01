import numpy as np
import pandas as pd

from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.scoring import _predicted_label, score_pending
from cryptopred.serve.store import PredictionStore


def test_the_model_prediction_is_the_argmax_not_the_signal():
    """A bar the strategy skipped still carries a prediction. Scoring it as FLAT
    marks the model wrong for being right but cautious."""
    leaning_up = {"prob_down": 0.38, "prob_flat": 0.15, "prob_up": 0.47}
    assert _predicted_label(leaning_up) == 1

    leaning_down = {"prob_down": 0.47, "prob_flat": 0.15, "prob_up": 0.38}
    assert _predicted_label(leaning_down) == -1

    genuinely_flat = {"prob_down": 0.20, "prob_flat": 0.60, "prob_up": 0.20}
    assert _predicted_label(genuinely_flat) == 0


def _env(tmp_path, up_move=True):
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC", name="open_time")
    step = 1.004 if up_move else 0.996
    close = pd.Series(100.0 * step ** np.arange(60), index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )
    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", bars)
    return PredictionStore(tmp_path / "p.db"), parquet, bars


def test_an_untraded_but_correct_lean_is_scored_correct(tmp_path):
    """The exact case the dashboard was getting wrong: no signal, market rose,
    model had leaned up, row read 'wrong'."""
    store, parquet, bars = _env(tmp_path, up_move=True)
    store.record_prediction(
        symbol="BTCUSDT", interval="1h",
        bar_close_time=bars["close_time"].iloc[25],
        proba=(0.38, 0.15, 0.47),      # leans up
        signal=0,                      # but below the cutoff, so not traded
        close_price=100.0, model_version="v1",
    )
    score_pending(store, parquet, "BTCUSDT", "1h", horizon=4, band_k=0.1)

    row = store.history("BTCUSDT", "1h", limit=5).iloc[0]
    assert row["actual_label"] == 1
    assert row["is_correct"] == 1


def test_an_untraded_wrong_lean_is_still_scored_wrong(tmp_path):
    store, parquet, bars = _env(tmp_path, up_move=True)
    store.record_prediction(
        symbol="BTCUSDT", interval="1h",
        bar_close_time=bars["close_time"].iloc[25],
        proba=(0.47, 0.15, 0.38),      # leans down into a rising market
        signal=0, close_price=100.0, model_version="v1",
    )
    score_pending(store, parquet, "BTCUSDT", "1h", horizon=4, band_k=0.1)
    assert store.history("BTCUSDT", "1h", limit=5).iloc[0]["is_correct"] == 0


def test_a_traded_signal_scores_the_same_way_as_before(tmp_path):
    """Bars the strategy did act on are unaffected: the signal follows the
    argmax whenever it fires."""
    store, parquet, bars = _env(tmp_path, up_move=True)
    store.record_prediction(
        symbol="BTCUSDT", interval="1h",
        bar_close_time=bars["close_time"].iloc[25],
        proba=(0.10, 0.15, 0.75), signal=1, close_price=100.0, model_version="v1",
    )
    score_pending(store, parquet, "BTCUSDT", "1h", horizon=4, band_k=0.1)
    assert store.history("BTCUSDT", "1h", limit=5).iloc[0]["is_correct"] == 1


def test_a_genuinely_flat_prediction_is_right_when_nothing_moves(tmp_path):
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC", name="open_time")
    flat = pd.Series(100.0, index=idx)
    bars = pd.DataFrame(
        {"open": flat, "high": flat * 1.0001, "low": flat * 0.9999, "close": flat,
         "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)},
        index=idx,
    )
    parquet = ParquetStore(tmp_path / "raw")
    parquet.write("klines", "BTCUSDT", "1h", bars)
    store = PredictionStore(tmp_path / "p.db")
    store.record_prediction(
        symbol="BTCUSDT", interval="1h", bar_close_time=bars["close_time"].iloc[25],
        proba=(0.20, 0.60, 0.20), signal=0, close_price=100.0, model_version="v1",
    )
    score_pending(store, parquet, "BTCUSDT", "1h", horizon=4, band_k=0.5)
    assert store.history("BTCUSDT", "1h", limit=5).iloc[0]["is_correct"] == 1


def test_an_outcome_across_a_hole_in_the_bars_is_not_scored(tmp_path):
    """Training labels give no label where `horizon` rows ahead is not
    `horizon` bars ahead in time; live scoring follows the same rule."""
    _, _, bars = _env(tmp_path, up_move=True)
    holed = bars.drop(bars.index[20:30])  # ten hours missing
    parquet = ParquetStore(tmp_path / "holed")
    parquet.write("klines", "BTCUSDT", "1h", holed)
    store = PredictionStore(tmp_path / "holed.db")
    # Bar 18: four rows ahead is bar 32, fourteen hours later. Bars 15 and 35
    # have clean four-hour windows.
    for i in (15, 18, 35):
        store.record_prediction(
            symbol="BTCUSDT", interval="1h",
            bar_close_time=bars["close_time"].iloc[i],
            proba=(0.1, 0.2, 0.7), signal=1, close_price=100.0, model_version="v1",
        )
    assert score_pending(store, parquet, "BTCUSDT", "1h", horizon=4) == 2
    left = pd.to_datetime(store.unscored("BTCUSDT", "1h")["bar_close_time"], utc=True)
    assert list(left) == [bars["close_time"].iloc[18]]
