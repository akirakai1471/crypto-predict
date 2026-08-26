import pandas as pd

from cryptopred.ingest.backfill import backfill_klines, drop_unclosed, find_gaps


def _kline_row(open_ms: int, interval_ms: int, close: float) -> list:
    return [
        open_ms, "1", "2", "0.5", str(close), "10",
        open_ms + interval_ms - 1, "100", 5, "5", "50", "0",
    ]


class FakeClient:
    """Serves klines from an in-memory grid, mimicking Binance pagination."""

    def __init__(self, start_ms: int, n_bars: int, interval_ms: int, limit: int = 3):
        self.rows = [
            _kline_row(start_ms + i * interval_ms, interval_ms, 100.0 + i) for i in range(n_bars)
        ]
        self.limit = limit
        self.calls = 0

    def fetch_klines(self, symbol, interval, start_ms, end_ms, limit=1500):
        self.calls += 1
        window = [r for r in self.rows if start_ms <= r[0] <= end_ms]
        return window[: self.limit]


def test_drop_unclosed_removes_forming_bar():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC", name="open_time")
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0], "close_time": idx + pd.Timedelta(hours=1)}, index=idx
    )
    now = pd.Timestamp("2024-01-01 02:30", tz="UTC")
    out = drop_unclosed(df, now=now)
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_find_gaps_returns_missing_bar_ranges():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            # 02:00 and 03:00 missing
            pd.Timestamp("2024-01-01 04:00", tz="UTC"),
        ],
        name="open_time",
    )
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    gaps = find_gaps(df, "1h")
    assert gaps == [
        (pd.Timestamp("2024-01-01 02:00", tz="UTC"), pd.Timestamp("2024-01-01 03:00", tz="UTC"))
    ]


def test_find_gaps_empty_when_contiguous():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC", name="open_time")
    df = pd.DataFrame({"close": range(10)}, index=idx)
    assert find_gaps(df, "1h") == []


def test_backfill_paginates_until_end():
    interval_ms = 3_600_000
    start = pd.Timestamp("2024-01-01", tz="UTC")
    client = FakeClient(int(start.timestamp() * 1000), n_bars=10, interval_ms=interval_ms, limit=3)

    df = backfill_klines(
        client,
        "BTCUSDT",
        "1h",
        start=start,
        end=start + pd.Timedelta(hours=10),
        now=start + pd.Timedelta(hours=10),
    )

    assert len(df) == 10
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert client.calls >= 4  # 10 bars at 3 per page


def test_backfill_stops_when_page_empty():
    interval_ms = 3_600_000
    start = pd.Timestamp("2024-01-01", tz="UTC")
    client = FakeClient(int(start.timestamp() * 1000), n_bars=0, interval_ms=interval_ms)

    df = backfill_klines(
        client, "BTCUSDT", "1h", start=start, end=start + pd.Timedelta(hours=5), now=start
    )
    assert df.empty
