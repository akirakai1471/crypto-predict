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


def test_the_interval_states_its_measured_coverage_and_disclaims_95():
    """The block bootstrap measures ~80% coverage.

    An earlier version of this test just asserted "95%" was absent, which review
    called out as guarding only against a hardcoded label - it would have passed
    on a table that said nothing at all about coverage, leaving a reader to
    default to the 95% the measurement refuted. It now checks the real property:
    the table says what the interval covers, and says it is not 95%.
    """
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "độ phủ" in text
    assert "80%" in text
    assert "không phải 95%" in text


def test_it_runs_on_an_empty_store_without_raising():
    text = format_brief("BTCUSDT", "1h", pd.DataFrame(), pd.DataFrame())
    assert "Không có dữ liệu" in text
