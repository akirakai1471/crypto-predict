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


# -- delivery -----------------------------------------------------------------


def test_the_log_is_written_even_when_the_popup_fails(tmp_path, monkeypatch):
    """The log is the durable record; the popup is a courtesy.

    Same principle as report_io.emit: a display that fails must not be able to
    destroy the thing that succeeded.
    """
    from cryptopred.serve import alerts

    def boom(*_a, **_k):
        raise OSError("no window station")

    monkeypatch.setattr(alerts.subprocess, "Popen", boom)
    log = tmp_path / "signals.log"
    alerts.notify("BTCUSDT — model bắn LONG\nHồ sơ: đúng 7/15", log_path=log)
    assert "LONG" in log.read_text(encoding="utf-8")


def test_the_log_appends_rather_than_overwrites(tmp_path, monkeypatch):
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    log = tmp_path / "signals.log"
    alerts.notify("first", log_path=log)
    alerts.notify("second", log_path=log)
    text = log.read_text(encoding="utf-8")
    assert "first" in text and "second" in text


def test_the_log_records_when_each_alert_was_written(tmp_path, monkeypatch):
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    log = tmp_path / "signals.log"
    alerts.notify("x", log_path=log)
    assert "UTC" in log.read_text(encoding="utf-8")


def test_the_popup_summary_fits_a_balloon_tip():
    """Windows truncates balloon text, so the popup gets a summary and the log
    keeps the full message."""
    from cryptopred.serve.alerts import BALLOON_TEXT_LIMIT, balloon_parts

    long_message = "\n".join(
        [
            "BTCUSDT — model bắn LONG",
            "nến đóng 2026-09-26 09:59 UTC, giá 84,560.60",
            "margin 0.0631 so với cutoff 0.0600",
            "",
            "Hồ sơ: đúng 7/15 (47%). Vốn ảo 9,971 (-29). CHƯA ĐỦ ĐỂ KẾT LUẬN — "
            "15 tín hiệu; cần khoảng 100.",
            "",
            "Đây là báo model đã bắn gì, KHÔNG phải khuyến nghị. " * 3,
        ]
    )
    title, text = balloon_parts(long_message)
    assert title == "BTCUSDT — model bắn LONG"
    assert len(text) <= BALLOON_TEXT_LIMIT
    assert "7/15" in text


def test_the_popup_never_swallows_the_unproven_caveat_silently(tmp_path, monkeypatch):
    """Truncating for the balloon must not lose the caveat from the log."""
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    log = tmp_path / "signals.log"
    message = "BTCUSDT — LONG\nHồ sơ: 7/15\n\nCHƯA CHỨNG MINH ĐƯỢC"
    alerts.notify(message, log_path=log)
    assert "CHƯA CHỨNG MINH" in log.read_text(encoding="utf-8")


# -- wired into the cycle -----------------------------------------------------


def test_a_signal_already_in_the_log_does_not_suppress_itself():
    """The scheduler records a prediction, then asks whether to alert on it.

    So the history it passes in holds this very row. Compared against itself, a
    signal is always zero hours into its own horizon - which silenced every
    alert, including the first one ever fired.
    """
    history = _history([("2026-09-26T06:00:00+00:00", 1, None)])
    assert should_alert(
        signal=1,
        bar_close_time=pd.Timestamp("2026-09-26T06:00Z"),
        history=history,
        horizon_hours=24,
    )


class _Bundle:
    metadata = {"margin_cutoff": 0.06}


class _Predictor:
    bundle = _Bundle()


class _Trader:
    def summary(self, symbol):
        return {"equity": 9_971.0}


def _fire(cfg, store, when, signal=1):
    """What run_cycle does for one bar: record first, then consider alerting."""
    from cryptopred.serve.predictor import Prediction
    from cryptopred.serve.runner import _maybe_alert

    proba = (0.2, 0.2, 0.6) if signal > 0 else (0.6, 0.2, 0.2)
    prediction = Prediction(
        symbol="BTCUSDT",
        interval="1h",
        bar_close_time=pd.Timestamp(when),
        proba=proba,
        signal=signal,
        confidence=max(proba),
        close_price=84_560.6,
        model_version="v1",
    )
    store.record_prediction(
        symbol="BTCUSDT",
        interval="1h",
        bar_close_time=prediction.bar_close_time,
        proba=proba,
        signal=signal,
        close_price=prediction.close_price,
        model_version="v1",
    )
    return _maybe_alert(
        cfg=cfg,
        symbol="BTCUSDT",
        interval="1h",
        prediction=prediction,
        close_price=prediction.close_price,
        horizon=24,
        predictions=store,
        trader=_Trader(),
        predictor=_Predictor(),
    )


def test_the_cycle_alerts_on_a_fresh_signal_against_the_real_store(tmp_path, monkeypatch):
    """End to end through SQLite, in the order the scheduler actually runs.

    The unit tests above passed while this path alerted on nothing: they built
    the history by hand, and the real one already contains the row being judged.
    """
    from cryptopred.config import Config
    from cryptopred.serve import alerts
    from cryptopred.serve.store import PredictionStore

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    cfg = Config()
    cfg.data.root = tmp_path
    store = PredictionStore(tmp_path / "predictions.db")

    assert _fire(cfg, store, "2026-09-26T06:00Z") == 1
    assert _fire(cfg, store, "2026-09-26T07:00Z") == 0  # same episode
    assert _fire(cfg, store, "2026-09-26T08:00Z", signal=-1) == 1  # a flip
    assert _fire(cfg, store, "2026-09-27T07:00Z") == 1  # a new episode

    log = (tmp_path / "signals.log").read_text(encoding="utf-8")
    assert log.count("model bắn") == 3


# -- reading the log back -----------------------------------------------------


def test_the_log_reads_back_newest_first_and_verbatim(tmp_path, monkeypatch):
    """The dashboard shows what was sent, caveat included, not a summary of it."""
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    log = tmp_path / "signals.log"
    alerts.notify("BTCUSDT — model bắn LONG\n\nCHƯA CHỨNG MINH ĐƯỢC", log_path=log)
    alerts.notify("BTCUSDT — model bắn SHORT\n\nCHƯA CHỨNG MINH ĐƯỢC", log_path=log)

    entries = alerts.read_log(log)
    assert [e["message"].splitlines()[0] for e in entries] == [
        "BTCUSDT — model bắn SHORT",
        "BTCUSDT — model bắn LONG",
    ]
    assert all("CHƯA CHỨNG MINH ĐƯỢC" in e["message"] for e in entries)
    assert all(e["written_at"].endswith("UTC") for e in entries)


def test_reading_the_log_respects_the_limit(tmp_path, monkeypatch):
    from cryptopred.serve import alerts

    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)
    log = tmp_path / "signals.log"
    for i in range(5):
        alerts.notify(f"alert {i}", log_path=log)
    assert [e["message"] for e in alerts.read_log(log, limit=2)] == ["alert 4", "alert 3"]


def test_a_missing_log_reads_as_no_alerts(tmp_path):
    from cryptopred.serve.alerts import read_log

    assert read_log(tmp_path / "never-written.log") == []


# -- the popup command cannot be rewritten by what it shows ----------------------


@pytest.mark.parametrize("quote", ["'", "‘", "’", "‚", "‛", '"', "`", "$"])
def test_no_character_in_a_headline_reaches_the_powershell_script(quote):
    """PowerShell closes a single-quoted string on ' and also on the curly
    quotes, which about one real headline in five contains. The text now travels
    in the environment, so the script is the same constant whatever it shows."""
    from cryptopred.serve.alerts import _POPUP, popup_command

    message = f"BTCUSDT {quote}; Write-Output X; # headline\nHồ sơ: đúng 1/2"
    argv, env = popup_command(message)
    assert argv[-1] == _POPUP  # one constant script, whatever the headline
    assert "headline" not in argv[-1] and "Write-Output" not in argv[-1]
    assert env["CRYPTOPRED_BALLOON_TITLE"] == f"BTCUSDT {quote}; Write-Output X; # headline"
    assert env["CRYPTOPRED_BALLOON_TEXT"] == "Hồ sơ: đúng 1/2"


def test_the_popup_is_spawned_with_the_text_in_its_environment(tmp_path, monkeypatch):
    from cryptopred.serve import alerts

    calls = []
    monkeypatch.setattr(alerts, "_on_windows", lambda: True)
    monkeypatch.setattr(alerts.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)))
    alerts.notify("ETH’s ETF approved\nHồ sơ: x", log_path=tmp_path / "news.log")
    argv, kwargs = calls[0]
    assert argv[-1] == alerts._POPUP
    assert kwargs["env"]["CRYPTOPRED_BALLOON_TITLE"] == "ETH’s ETF approved"
