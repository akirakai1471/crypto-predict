import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1000, seed: int = 42, freq: str = "1h") -> pd.DataFrame:
    """Synthetic but well-formed OHLCV: geometric random walk with valid highs/lows."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.004, n)
    close = 30_000 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(1000, 300, n))
    taker_buy = volume * rng.uniform(0.3, 0.7, n)

    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="open_time")
    delta = pd.tseries.frequencies.to_offset(freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(100, 5000, n),
            "taker_buy_base": taker_buy,
            "taker_buy_quote": taker_buy * close,
            "close_time": idx + delta - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def ohlcv_1m() -> pd.DataFrame:
    return make_ohlcv(n=2000, seed=7, freq="1min")


@pytest.fixture
def rising() -> pd.DataFrame:
    """Strictly increasing closes — useful for asserting bounded indicators."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(np.linspace(100, 300, n), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
            "quote_volume": 1000.0 * close,
            "trades": 100,
            "taker_buy_base": 500.0,
            "taker_buy_quote": 500.0 * close,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


def record_listening(store, start, end, every: str = "1min", feeds_ok: int = 5) -> None:
    """Log a completed news poll every `every` from start to end, inclusive.

    One executemany instead of thousands of store.record_poll calls, each of
    which opens a connection and commits.
    """
    import sqlite3

    from cryptopred.news.store import utc_stamp

    times = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=every)
    with sqlite3.connect(store.path) as conn:
        conn.executemany(
            "INSERT INTO polls (polled_at, feeds_ok, feeds_failed, inserted) VALUES (?, ?, 0, 0)",
            [(utc_stamp(t), feeds_ok) for t in times],
        )
