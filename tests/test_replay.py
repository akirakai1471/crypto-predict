import numpy as np
import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.replay import (
    SideStats,
    format_replay,
    replay_predictions,
    two_sided_verdict,
)
from cryptopred.serve.store import PredictionStore


@pytest.fixture
def env(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.strategy.starting_capital = 10_000.0
    cfg.strategy.taker_fee = 0.0005
    cfg.strategy.slippage = 0.0002
    cfg.strategy.funding_rate = 0.0

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    # steady uptrend: longs should win, shorts should lose
    close = pd.Series(100.0 * 1.004 ** np.arange(n), index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
            "quote_volume": 1.0,
            "trades": 1,
            "taker_buy_base": 0.5,
            "taker_buy_quote": 0.5,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )
    ParquetStore(tmp_path / "raw").write("klines", "BTCUSDT", "1h", bars)
    store = PredictionStore(tmp_path / "replay.db")
    return cfg, bars, idx, store


def _proba(n: int, cls: int, conf: float = 0.9) -> np.ndarray:
    p = np.full((n, 3), (1 - conf) / 2)
    p[:, cls] = conf
    return p


def test_all_long_signals_open_long_trades(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 2), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["long"].n > 0
    assert result["short"].n == 0


def test_shorts_lose_in_an_uptrend(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 0), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["short"].n > 0
    assert result["short"].total_pnl < 0
    assert result["long"].n == 0


def test_longs_win_in_an_uptrend(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 2), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["long"].total_pnl > 0
    assert result["summary"]["equity"] > cfg.strategy.starting_capital


def test_low_confidence_predictions_are_ignored(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 2, conf=0.45), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["opened"] == 0
    assert result["summary"]["n_trades"] == 0


def test_flat_predictions_open_nothing(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 1), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["opened"] == 0


def test_mixed_signals_produce_both_sides(env):
    cfg, bars, idx, store = env
    proba = _proba(len(idx), 2)
    proba[::2] = _proba(len(idx), 0)[::2]   # alternate long and short
    result = replay_predictions(
        cfg, bars, proba, idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    assert result["long"].n > 0
    assert result["short"].n > 0


def test_position_size_is_capital_over_horizon(env):
    cfg, bars, idx, store = env
    replay_predictions(
        cfg, bars, _proba(len(idx), 2), idx, "BTCUSDT", "1h",
        horizon=20, threshold=0.6, store=store,
    )
    trades = store.closed_trades("BTCUSDT")
    assert trades.iloc[0]["size_usd"] == pytest.approx(10_000 / 20)


def test_equity_curve_is_ordered_by_exit_time(env):
    cfg, bars, idx, store = env
    result = replay_predictions(
        cfg, bars, _proba(len(idx), 2), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    equity = result["equity_curve"]
    assert equity.index.is_monotonic_increasing


def test_format_warns_when_only_longs_make_money(env):
    cfg, bars, idx, store = env
    proba = _proba(len(idx), 2)
    proba[::2] = _proba(len(idx), 0)[::2]
    result = replay_predictions(
        cfg, bars, proba, idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    text = format_replay(result, "BTCUSDT", 0.6)
    assert "LONG" in text and "SHORT" in text
    assert "DIRECTIONAL VERDICT" in text


def test_replay_does_not_touch_the_live_database(env, tmp_path):
    cfg, bars, idx, store = env
    replay_predictions(
        cfg, bars, _proba(len(idx), 2), idx, "BTCUSDT", "1h",
        horizon=10, threshold=0.6, store=store,
    )
    live = PredictionStore(tmp_path / "predictions.db")
    assert live.closed_trades("BTCUSDT").empty


def _side(n, win, pnl, avg=0.001):
    return SideStats(n=n, win_rate=win, total_pnl=pnl, avg_return=avg)


def test_two_sided_when_shorts_also_profit():
    v = two_sided_verdict(_side(3000, 0.55, 4700.0), _side(1100, 0.551, 793.0))
    assert v["decision"] == "TWO-SIDED"


def test_one_sided_when_shorts_lose_money():
    v = two_sided_verdict(_side(3000, 0.55, 4544.0), _side(542, 0.509, -357.0))
    assert v["decision"] == "ONE-SIDED"
    assert "long side alone" in v["reason"]


def test_one_sided_when_short_win_rate_is_below_chance():
    v = two_sided_verdict(_side(3400, 0.53, 4029.0), _side(530, 0.436, 58.0))
    assert v["decision"] == "ONE-SIDED"
    assert "coin flip" in v["reason"]


def test_unproven_when_too_few_shorts():
    v = two_sided_verdict(_side(3000, 0.55, 4000.0), _side(12, 0.60, 90.0))
    assert v["decision"] == "UNPROVEN"


def test_no_trades_case():
    v = two_sided_verdict(_side(0, None, 0.0), _side(0, None, 0.0))
    assert v["decision"] == "NO TRADES"


def test_one_sided_when_shorts_win_often_but_earn_nothing():
    """A high win rate with no profit means small wins and large losses — the
    short side is being carried, not contributing."""
    v = two_sided_verdict(_side(3000, 0.55, 5000.0), _side(1500, 0.571, 20.0))
    assert v["decision"] == "ONE-SIDED"
    assert "small wins and large losses" in v["reason"]


def test_two_sided_needs_a_material_short_contribution():
    v = two_sided_verdict(_side(3000, 0.55, 5000.0), _side(1100, 0.551, 800.0))
    assert v["decision"] == "TWO-SIDED"


# -- the same tests for the long side (review, 2026-10-01) ------------------------


def test_one_sided_when_longs_lose_in_a_falling_market():
    """OPUSDT: shorts +5,287, longs -765 at 43.3% over a 96% decline - recorded
    TWO-SIDED because only the shorts were ever tested."""
    v = two_sided_verdict(_side(410, 0.433, -765.0), _side(900, 0.56, 5287.0))
    assert v["decision"] == "ONE-SIDED"
    assert "short side alone" in v["reason"]


def test_one_sided_when_long_win_rate_is_below_chance():
    """INJUSDT's longs made money but won 49.3% of the time."""
    v = two_sided_verdict(_side(700, 0.493, 300.0), _side(800, 0.55, 2000.0))
    assert v["decision"] == "ONE-SIDED"
    assert "longs win only" in v["reason"]


def test_unproven_when_too_few_longs():
    v = two_sided_verdict(_side(40, 0.60, 90.0), _side(3000, 0.55, 4000.0))
    assert v["decision"] == "UNPROVEN"
    assert "long" in v["reason"]


def test_the_verdict_is_symmetric():
    long, short = _side(3000, 0.55, 5000.0), _side(1500, 0.571, 20.0)
    assert two_sided_verdict(long, short)["decision"] == two_sided_verdict(short, long)["decision"]
