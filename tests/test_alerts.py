"""An alert that arrives without its track record is an invitation to trust it.

This project's tradeable edge is unproven - 6 of 20 symbols pass its gates, and
the live paper record is currently 0 of 3. A notification that says "LONG signal"
and nothing else hides all of that behind a word that sounds like advice.
"""

import pandas as pd
import pytest

from cryptopred.serve.alerts import (
    MIN_SIGNALS_FOR_A_CLAIM,
    format_alert,
    should_alert,
    track_record_line,
)


def _history(rows):
    """rows: list of (bar_close_time, signal, is_correct or None)"""
    return pd.DataFrame(
        [
            {
                "bar_close_time": t,
                "signal": s,
                "is_correct": c,
                "actual_return": None if c is None else 0.01,
            }
            for t, s, c in rows
        ]
    )


# -- clustering ---------------------------------------------------------------


def test_the_first_signal_of_an_episode_alerts():
    assert should_alert(
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-23T09:00Z"),
        history=_history([]),
        horizon_hours=24,
    )


def test_a_repeat_of_the_same_direction_inside_the_horizon_is_suppressed():
    """Eight signals in eight hours is one bet, not eight.

    A fixed 24-hour horizon means a signal every bar while conditions hold
    opens overlapping positions on the same view. Alerting each one sends eight
    messages for one decision and makes the record look busier than it is.
    """
    history = _history([("2026-09-23T09:00:00+00:00", 1, None)])
    assert not should_alert(
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-23T13:00Z"),
        history=history,
        horizon_hours=24,
    )


def test_the_same_direction_after_the_horizon_alerts_again():
    history = _history([("2026-09-23T09:00:00+00:00", 1, None)])
    assert should_alert(
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-24T10:00Z"),
        history=history,
        horizon_hours=24,
    )


def test_the_opposite_direction_always_alerts():
    """A flip is new information even if it arrives an hour later."""
    history = _history([("2026-09-23T09:00:00+00:00", 1, None)])
    assert should_alert(
        signal=-1,
        bar_close_time=pd.Timestamp("2026-09-23T10:00Z"),
        history=history,
        horizon_hours=24,
    )


def test_no_signal_never_alerts():
    assert not should_alert(
        signal=0,
        bar_close_time=pd.Timestamp("2026-09-23T09:00Z"),
        history=_history([]),
        horizon_hours=24,
    )


def test_bars_without_a_signal_do_not_count_as_the_previous_episode():
    """Only signals define an episode; quiet bars in between are irrelevant."""
    history = _history(
        [
            ("2026-09-23T09:00:00+00:00", 1, None),
            ("2026-09-24T20:00:00+00:00", 0, None),
        ]
    )
    assert should_alert(
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-24T21:00Z"),
        history=history,
        horizon_hours=24,
    )


# -- the record that travels with every alert ---------------------------------


def test_the_record_states_correct_out_of_scored():
    history = _history(
        [
            ("2026-09-20T01:00:00+00:00", 1, 1),
            ("2026-09-20T02:00:00+00:00", 1, 0),
            ("2026-09-20T03:00:00+00:00", 1, 0),
        ]
    )
    line = track_record_line(history, paper_equity=9971.0, starting_capital=10000.0)
    assert "1/3" in line
    assert "33" in line  # 33.3%


def test_the_record_warns_while_the_sample_is_too_small():
    history = _history([("2026-09-20T01:00:00+00:00", 1, 1)])
    line = track_record_line(history, paper_equity=10000.0, starting_capital=10000.0)
    assert str(MIN_SIGNALS_FOR_A_CLAIM) in line
    assert "chưa" in line.lower()


def test_the_record_says_so_when_nothing_has_been_scored():
    line = track_record_line(_history([]), paper_equity=10000.0, starting_capital=10000.0)
    assert "chưa có" in line.lower()


def test_the_record_shows_the_paper_loss_not_just_the_hit_rate():
    history = _history([("2026-09-20T01:00:00+00:00", 1, 1)])
    line = track_record_line(history, paper_equity=9971.0, starting_capital=10000.0)
    assert "-29" in line or "−29" in line


# -- the message itself -------------------------------------------------------


def test_the_alert_never_says_buy_or_sell():
    """It reports what the model did. Telling someone to buy is advice, and this
    project does not give advice - its edge is unproven."""
    history = _history([("2026-09-20T01:00:00+00:00", 1, 1)])
    for signal in (1, -1):
        text = format_alert(
            symbol="BTCUSDT",
            signal=signal,
            bar_close_time=pd.Timestamp("2026-09-26T09:00Z"),
            close_price=84560.6,
            margin=0.0605,
            cutoff=0.06,
            history=history,
            paper_equity=9971.0,
            starting_capital=10000.0,
        )
        lowered = text.lower()
        assert "mua ngay" not in lowered
        assert "bán ngay" not in lowered
        assert "nên mua" not in lowered
        assert "nên bán" not in lowered


def test_the_alert_carries_the_record_and_the_unproven_caveat():
    history = _history([("2026-09-20T01:00:00+00:00", 1, 1)])
    text = format_alert(
        symbol="BTCUSDT",
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-26T09:00Z"),
        close_price=84560.6,
        margin=0.0605,
        cutoff=0.06,
        history=history,
        paper_equity=9971.0,
        starting_capital=10000.0,
    )
    assert "BTCUSDT" in text
    assert "84,560" in text or "84560" in text
    assert "chưa chứng minh" in text.lower()
    assert "1/1" in text or "1 /1" in text


def test_the_alert_names_the_direction_plainly():
    history = _history([])
    long_text = format_alert(
        symbol="BTCUSDT", signal=1, bar_close_time=pd.Timestamp("2026-09-26T09:00Z"),
        close_price=1.0, margin=0.07, cutoff=0.06, history=history,
        paper_equity=10000.0, starting_capital=10000.0,
    )
    short_text = format_alert(
        symbol="BTCUSDT", signal=-1, bar_close_time=pd.Timestamp("2026-09-26T09:00Z"),
        close_price=1.0, margin=0.07, cutoff=0.06, history=history,
        paper_equity=10000.0, starting_capital=10000.0,
    )
    assert "LONG" in long_text
    assert "SHORT" in short_text


def test_a_zero_signal_is_not_an_alert():
    with pytest.raises(ValueError, match="signal"):
        format_alert(
            symbol="BTCUSDT", signal=0, bar_close_time=pd.Timestamp("2026-09-26T09:00Z"),
            close_price=1.0, margin=0.0, cutoff=0.06, history=_history([]),
            paper_equity=10000.0, starting_capital=10000.0,
        )
