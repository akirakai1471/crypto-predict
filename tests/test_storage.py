import pandas as pd

from cryptopred.ingest.storage import ParquetStore, merge_frames


def _frame(start: str, periods: int, close_start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(periods),
            "low": range(periods),
            "close": [close_start + i for i in range(periods)],
            "volume": [1.0] * periods,
            "quote_volume": [1.0] * periods,
            "trades": [1] * periods,
            "taker_buy_base": [1.0] * periods,
            "taker_buy_quote": [1.0] * periods,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    ).astype({"open": "float64", "high": "float64", "low": "float64"})


def test_merge_frames_dedupes_keeping_newest():
    old = _frame("2024-01-01", 3, close_start=100.0)
    new = _frame("2024-01-01 02:00", 3, close_start=999.0)
    merged = merge_frames(old, new)

    assert len(merged) == 5
    assert merged.index.is_monotonic_increasing
    assert not merged.index.has_duplicates
    # overlapping bar takes the value from the newer frame
    assert merged.loc[pd.Timestamp("2024-01-01 02:00", tz="UTC"), "close"] == 999.0


def test_merge_frames_with_empty_old():
    new = _frame("2024-01-01", 3)
    assert len(merge_frames(pd.DataFrame(), new)) == 3


def test_store_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    df = _frame("2024-01-01", 5)
    store.write("klines", "BTCUSDT", "1h", df)

    loaded = store.read("klines", "BTCUSDT", "1h")
    # parquet does not preserve the DatetimeIndex freq attribute
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_store_append_merges_and_persists(tmp_path):
    store = ParquetStore(tmp_path)
    store.write("klines", "BTCUSDT", "1h", _frame("2024-01-01", 3))
    store.append("klines", "BTCUSDT", "1h", _frame("2024-01-01 02:00", 3))

    loaded = store.read("klines", "BTCUSDT", "1h")
    assert len(loaded) == 5
    assert not loaded.index.has_duplicates


def test_store_splits_by_year(tmp_path):
    store = ParquetStore(tmp_path)
    df = _frame("2023-12-31 22:00", 4)
    store.write("klines", "BTCUSDT", "1h", df)

    assert (tmp_path / "klines" / "BTCUSDT" / "1h" / "2023.parquet").exists()
    assert (tmp_path / "klines" / "BTCUSDT" / "1h" / "2024.parquet").exists()
    assert len(store.read("klines", "BTCUSDT", "1h")) == 4


def test_read_missing_returns_empty(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.read("klines", "NOPE", "1h").empty


def test_last_open_time(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.last_open_time("klines", "BTCUSDT", "1h") is None
    store.write("klines", "BTCUSDT", "1h", _frame("2024-01-01", 3))
    assert store.last_open_time("klines", "BTCUSDT", "1h") == pd.Timestamp(
        "2024-01-01 02:00", tz="UTC"
    )
