import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.gapfill import MAX_GAP_BARS, fill_gaps, find_gap_bars
from cryptopred.serve.store import PredictionStore
from tests.conftest import make_ohlcv


class FakePredictor:
    """Returns a fixed prediction for whichever bar it is asked about."""

    def __init__(self, bars: pd.DataFrame, signal: int = 1):
        self.bars = bars
        self.signal = signal
        self.asked: list[pd.Timestamp] = []

    def predict_at(self, symbol, interval, upto):
        from cryptopred.serve.predictor import Prediction

        window = self.bars if upto is None else self.bars[self.bars.index <= upto]
        if window.empty:
            return None
        bar = window.iloc[-1]
        self.asked.append(window.index[-1])
        return Prediction(
            symbol=symbol, interval=interval,
            bar_close_time=bar["close_time"],
            proba=(0.2, 0.2, 0.6), signal=self.signal, confidence=0.6,
            close_price=float(bar["close"]), model_version="fake",
        )


@pytest.fixture
def env(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]

    # bars well in the past so every horizon has elapsed
    bars = make_ohlcv(n=60, seed=11)
    bars.index = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC", name="open_time")
    bars["close_time"] = bars.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)

    ParquetStore(tmp_path / "raw").write("klines", "BTCUSDT", "1h", bars)
    store = PredictionStore(tmp_path / "predictions.db")
    return cfg, bars, store, ParquetStore(tmp_path / "raw")


def _record(store, close_time, backfilled=False):
    store.record_prediction(
        symbol="BTCUSDT", interval="1h", bar_close_time=close_time,
        proba=(0.2, 0.2, 0.6), signal=1, close_price=100.0,
        model_version="v1", was_backfilled=backfilled,
    )


def test_no_gaps_when_nothing_was_ever_logged(env):
    cfg, bars, store, parquet = env
    assert find_gap_bars(store, parquet, "BTCUSDT", "1h", horizon=4) == []


def test_gap_detection_ignores_bars_before_the_log_began(env):
    """Bars from before the system existed were not missed."""
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[50])

    gaps = find_gap_bars(store, parquet, "BTCUSDT", "1h", horizon=4)
    assert all(g >= bars["close_time"].iloc[50] for g in gaps)


def test_gap_detection_finds_bars_missed_in_the_middle(env):
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[40])
    _record(store, bars["close_time"].iloc[45])

    gaps = find_gap_bars(store, parquet, "BTCUSDT", "1h", horizon=4)
    assert bars["close_time"].iloc[42] in gaps
    assert bars["close_time"].iloc[40] not in gaps


def test_fill_gaps_flags_every_row_it_writes(env):
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[40])
    _record(store, bars["close_time"].iloc[45])

    result = fill_gaps(
        cfg, store, parquet, "BTCUSDT", "1h", horizon=4,
        predictor=FakePredictor(bars),
    )
    assert result["filled"] > 0

    history = store.history("BTCUSDT", "1h", limit=1000)
    written = history[history["was_backfilled"] == 1]
    assert len(written) == result["filled"]
    # the two originals stay unflagged
    assert (history["was_backfilled"] == 0).sum() == 2


def test_backfilled_rows_do_not_enter_the_live_accuracy(env):
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[40])
    _record(store, bars["close_time"].iloc[45])
    fill_gaps(cfg, store, parquet, "BTCUSDT", "1h", horizon=4,
              predictor=FakePredictor(bars))

    for _, r in store.unscored("BTCUSDT", "1h").iterrows():
        store.score_prediction(int(r["id"]), 0.01, 1, True)

    summary = store.accuracy_summary("BTCUSDT", "1h")
    assert summary["n_scored"] == 2      # only the rows written live


def test_filling_is_idempotent(env):
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[40])
    _record(store, bars["close_time"].iloc[45])

    first = fill_gaps(cfg, store, parquet, "BTCUSDT", "1h", horizon=4,
                      predictor=FakePredictor(bars))
    second = fill_gaps(cfg, store, parquet, "BTCUSDT", "1h", horizon=4,
                       predictor=FakePredictor(bars))
    assert first["filled"] > 0
    assert second["filled"] == 0


def test_no_model_means_nothing_is_invented(env):
    """Without a model in the registry the gap stays a gap."""
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[40])
    _record(store, bars["close_time"].iloc[45])

    result = fill_gaps(cfg, store, parquet, "BTCUSDT", "1h", horizon=4)
    assert result["filled"] == 0
    assert result["skipped"] == result["gap_bars"]


def test_an_enormous_gap_is_capped(env):
    cfg, bars, store, parquet = env
    _record(store, bars["close_time"].iloc[0])

    gaps = find_gap_bars(store, parquet, "BTCUSDT", "1h", horizon=4)
    assert len(gaps) < MAX_GAP_BARS      # this fixture is small; the cap is real


def test_bars_too_recent_to_score_are_left_for_the_normal_cycle(env, tmp_path):
    """A bar whose horizon has not elapsed will be predicted live shortly, and
    flagging it as backfilled would be wrong."""
    cfg = Config()
    cfg.data.root = tmp_path / "fresh"
    cfg.data.symbols = ["BTCUSDT"]

    now = pd.Timestamp.now(tz="UTC").floor("h")
    idx = pd.date_range(end=now, periods=40, freq="1h", tz="UTC", name="open_time")
    bars = make_ohlcv(n=40, seed=5)
    bars.index = idx
    bars["close_time"] = idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)

    parquet = ParquetStore(cfg.data.root / "raw")
    parquet.write("klines", "BTCUSDT", "1h", bars)
    store = PredictionStore(cfg.data.root / "predictions.db")
    _record(store, bars["close_time"].iloc[0])

    gaps = find_gap_bars(store, parquet, "BTCUSDT", "1h", horizon=24)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    assert all(g <= cutoff for g in gaps)
