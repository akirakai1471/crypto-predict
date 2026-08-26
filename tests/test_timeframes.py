import pandas as pd
import pytest

from cryptopred.timeframes import (
    align_down,
    expected_open_times,
    interval_to_timedelta,
    to_ms,
)


def test_interval_to_timedelta():
    assert interval_to_timedelta("1m") == pd.Timedelta(minutes=1)
    assert interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert interval_to_timedelta("4h") == pd.Timedelta(hours=4)
    assert interval_to_timedelta("1d") == pd.Timedelta(days=1)


def test_unknown_interval_raises():
    with pytest.raises(ValueError, match="Unsupported interval"):
        interval_to_timedelta("7s")


def test_to_ms():
    ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert to_ms(ts) == 1704067200000


def test_align_down_snaps_to_bar_open():
    ts = pd.Timestamp("2024-01-01 03:47:31", tz="UTC")
    assert align_down(ts, "1h") == pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    assert align_down(ts, "4h") == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert align_down(ts, "1m") == pd.Timestamp("2024-01-01 03:47:00", tz="UTC")


def test_align_down_is_idempotent():
    ts = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    assert align_down(align_down(ts, "1h"), "1h") == ts


def test_expected_open_times_is_half_open():
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01 05:00:00", tz="UTC")
    grid = expected_open_times(start, end, "1h")
    assert len(grid) == 5
    assert grid[0] == start
    assert grid[-1] == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
