"""The table must be readable without the LLM layer, and must not quietly
present a convention as a measurement."""

import pandas as pd

from cryptopred.briefing.report import format_brief
from tests.conftest import make_ohlcv


def _brief_input(stale: bool = False):
    bars = make_ohlcv(n=3000, seed=51)
    now = bars["close_time"].max() + (
        pd.Timedelta(days=10) if stale else pd.Timedelta(minutes=10)
    )
    return bars, now


def test_the_table_separates_measured_from_convention():
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "ĐO ĐƯỢC" in text
    assert "QUY ƯỚC" in text


def test_staleness_appears_at_the_top_when_data_is_old():
    bars, now = _brief_input(stale=True)
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    head = text.split("QUY ƯỚC")[0]
    assert "cũ" in head


def test_touch_probabilities_are_shown_with_their_sample_size():
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "n=" in text


def test_wait_times_say_which_sample_they_came_from():
    """Printed under a cell label, unconditional wait times would read as if
    they belonged to that cell."""
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "cùng chế độ" in text or "MỌI chế độ" in text


def test_the_interval_is_never_presented_as_95_percent():
    """The block bootstrap measures ~80% coverage. Printing "95%" next to it
    would be the overclaim findings.md exists to record."""
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "95%" not in text


def test_it_runs_on_an_empty_store_without_raising():
    text = format_brief("BTCUSDT", "1h", pd.DataFrame(), pd.DataFrame())
    assert "Không có dữ liệu" in text
