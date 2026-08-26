"""Bar-grid arithmetic. Every timestamp in this project is UTC."""

from __future__ import annotations

import pandas as pd

_INTERVALS: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    try:
        return _INTERVALS[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval!r}") from exc


def to_ms(ts: pd.Timestamp) -> int:
    """Binance speaks epoch milliseconds."""
    return int(ts.timestamp() * 1000)


def from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def align_down(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Snap a timestamp back to the open of the bar containing it."""
    delta = interval_to_timedelta(interval)
    return ts.floor(delta)


def expected_open_times(
    start: pd.Timestamp, end: pd.Timestamp, interval: str
) -> pd.DatetimeIndex:
    """Every bar open time in [start, end). Used to detect missing bars."""
    delta = interval_to_timedelta(interval)
    return pd.date_range(
        start=align_down(start, interval),
        end=align_down(end, interval) - delta,
        freq=delta,
        tz="UTC",
    )
