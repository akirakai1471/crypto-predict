"""Freshness is a top-level field, not a footnote.

An answer computed on ten-day-old bars is wrong in a way the prose will not
reveal. The live experiment lost sixteen days to a model that had silently
stopped firing; stale data is the same class of failure.
"""

import pandas as pd

from cryptopred.briefing.snapshot import STALE_AFTER_HOURS, market_snapshot
from tests.conftest import make_ohlcv


def test_snapshot_reports_price_and_changes():
    bars = make_ohlcv(n=1000, seed=21)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=bars["close_time"].max())
    assert snap["last_close"]["value"] == bars["close"].iloc[-1]
    assert "change_24h" in snap
    assert "change_7d" in snap


def test_freshness_is_top_level_and_flags_stale_data():
    bars = make_ohlcv(n=1000, seed=22)
    late = bars["close_time"].max() + pd.Timedelta(days=10)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=late)
    assert snap["data_age_hours"] > 200
    assert snap["is_stale"] is True
    assert "cũ" in snap["staleness_note"]


def test_fresh_data_says_so_without_a_warning():
    bars = make_ohlcv(n=1000, seed=23)
    snap = market_snapshot(
        bars,
        funding=pd.DataFrame(),
        now=bars["close_time"].max() + pd.Timedelta(minutes=30),
    )
    assert snap["is_stale"] is False
    assert snap["staleness_note"] == ""


def test_missing_funding_is_unavailable_not_zero():
    bars = make_ohlcv(n=1000, seed=24)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=bars["close_time"].max())
    assert snap["funding_now"]["source"] == "unavailable"


def test_funding_present_is_measured_with_its_sample():
    bars = make_ohlcv(n=1000, seed=25)
    idx = pd.date_range(
        "2024-01-01", periods=120, freq="8h", tz="UTC", name="funding_time"
    )
    funding = pd.DataFrame({"funding_rate": [0.0001] * 120}, index=idx)
    snap = market_snapshot(bars, funding=funding, now=bars["close_time"].max())
    assert snap["funding_now"]["source"] == "measured"
    assert snap["funding_mean_30d"]["n"] > 0


def test_stale_threshold_is_three_hours():
    assert STALE_AFTER_HOURS == 3.0
