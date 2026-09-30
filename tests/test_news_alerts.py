"""News alerts.

Three ways a headline alert goes wrong, each worse than no alert: it reads as
a trading signal, it repeats, or it arrives late enough to be yesterday's news
presented as breaking. The tests below pin the message and the three rules
(never twice, rate limit, nothing old) against a real SQLite store.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptopred.news.alerts import (
    FRESH_FOR,
    MAX_PUBLISHER_LAG,
    alert_fresh_headlines,
    format_news_alert,
)
from cryptopred.news.parse import Entry
from cryptopred.news.store import NewsStore
from cryptopred.serve.alerts import balloon_parts, read_log

NOW = pd.Timestamp("2026-09-30T12:00:00Z")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


class _Outbox:
    """Stands in for serve.alerts.notify, recording instead of popping up."""

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    def __call__(self, message, log_path, popup=True):
        if self.fail:
            raise OSError("log unwritable")
        self.sent.append(message)


@pytest.fixture
def store(tmp_path):
    return NewsStore(tmp_path / "news.db")


def _add(store, uid, title, at=NOW, backlog=False, published=None, link=None):
    written = store.add(
        [Entry(uid=uid, title=title, link=link, published_at=published)],
        source="Example", backlog=backlog, received_at=at,
    )
    assert written


def _run(store, tmp_path, outbox, now=NOW, max_per_hour=3):
    return alert_fresh_headlines(
        store, SYMBOLS, log_path=tmp_path / "news.log", max_per_hour=max_per_hour,
        now=now, popup=False, send=outbox,
    )


# -- what qualifies -----------------------------------------------------------------


def test_a_fresh_high_impact_headline_about_a_traded_coin_alerts(store, tmp_path):
    _add(store, "1", "SEC approves spot bitcoin ETF options", at=NOW - pd.Timedelta(minutes=1))
    outbox = _Outbox()
    assert _run(store, tmp_path, outbox)["sent"] == 1
    assert "SEC approves spot bitcoin ETF options" in outbox.sent[0]


@pytest.mark.parametrize(
    "title",
    [
        "Bitcoin miners expand in Texas",       # tagged, not high impact
        "SEC sues a stock broker",              # high impact, no coin
        "Solana exchange hacked",               # high impact, a coin not traded here
    ],
)
def test_anything_less_does_not(store, tmp_path, title):
    _add(store, "1", title)
    outbox = _Outbox()
    _run(store, tmp_path, outbox)
    assert outbox.sent == []


# -- never twice --------------------------------------------------------------------


def test_the_same_headline_is_never_alerted_twice(store, tmp_path):
    """The job runs every minute and the headline stays fresh for thirty."""
    _add(store, "1", "Bitcoin exchange hacked")
    outbox = _Outbox()
    for minute in range(10):
        _run(store, tmp_path, outbox, now=NOW + pd.Timedelta(minutes=minute))
    assert len(outbox.sent) == 1


def test_never_twice_survives_a_restart(store, tmp_path):
    """The decision lives in news.db, not in memory."""
    _add(store, "1", "Bitcoin exchange hacked")
    _run(store, tmp_path, _Outbox())
    reopened = NewsStore(store.path)
    outbox = _Outbox()
    _run(reopened, tmp_path, outbox, now=NOW + pd.Timedelta(minutes=1))
    assert outbox.sent == []


def test_a_failed_send_is_recorded_and_not_retried_every_minute(store, tmp_path):
    """A broken log path retried each pass would be a failure every minute and,
    once fixed, a burst of stale alerts."""
    _add(store, "1", "Bitcoin exchange hacked")
    counts = _run(store, tmp_path, _Outbox(fail=True))
    assert counts["failed"] == 1
    outbox = _Outbox()
    _run(store, tmp_path, outbox, now=NOW + pd.Timedelta(minutes=1))
    assert outbox.sent == []
    assert store.recent()[0]["alert"] == "failed"


# -- rate limit ---------------------------------------------------------------------


def test_no_more_than_the_hourly_limit_and_the_rest_are_never_sent_late(store, tmp_path):
    for i in range(5):
        _add(store, f"h{i}", f"Bitcoin ETF story number {i}", at=NOW - pd.Timedelta(minutes=5))
    outbox = _Outbox()
    counts = _run(store, tmp_path, outbox, max_per_hour=3)
    assert counts == {"sent": 3, "rate_limited": 2, "stale": 0, "failed": 0}

    # An hour later the budget is back, but the two held back are not news any more.
    later = _Outbox()
    _run(store, tmp_path, later, now=NOW + pd.Timedelta(hours=1, minutes=1))
    assert later.sent == []


def test_the_limit_is_a_trailing_hour_not_a_clock_hour(store, tmp_path):
    outbox = _Outbox()
    for i in range(3):
        _add(store, f"a{i}", f"Bitcoin ETF story {i}", at=NOW + pd.Timedelta(minutes=50 + i))
    _run(store, tmp_path, outbox, now=NOW + pd.Timedelta(minutes=55))
    # 13:05 is a new clock hour but only fifteen minutes later.
    _add(store, "b", "Ether ETF approved", at=NOW + pd.Timedelta(minutes=64))
    counts = _run(store, tmp_path, outbox, now=NOW + pd.Timedelta(minutes=65))
    assert counts["rate_limited"] == 1
    assert len(outbox.sent) == 3


def test_a_limit_of_zero_sends_nothing(store, tmp_path):
    _add(store, "1", "Bitcoin exchange hacked")
    outbox = _Outbox()
    _run(store, tmp_path, outbox, max_per_hour=0)
    assert outbox.sent == []


# -- nothing old ---------------------------------------------------------------------


def test_backlog_is_never_alerted(store, tmp_path):
    """The first fetch after a start delivers days of RSS. Every item is new to
    us and none is new to the world."""
    _add(store, "1", "Bitcoin exchange hacked", backlog=True)
    outbox = _Outbox()
    _run(store, tmp_path, outbox)
    assert outbox.sent == []


def test_a_headline_received_longer_ago_than_the_fresh_window_is_not_alerted(store, tmp_path):
    _add(store, "1", "Bitcoin exchange hacked", at=NOW - FRESH_FOR - pd.Timedelta(seconds=1))
    outbox = _Outbox()
    _run(store, tmp_path, outbox)
    assert outbox.sent == []


def test_a_headline_its_own_publisher_dates_hours_earlier_is_stale(store, tmp_path):
    """Re-surfaced or edited old stories. A publisher date can suppress an alert
    here but never cause one, so a wrong date costs at most a missed message."""
    published = (NOW - MAX_PUBLISHER_LAG - pd.Timedelta(minutes=1)).to_pydatetime()
    _add(store, "1", "Bitcoin exchange hacked", published=published)
    outbox = _Outbox()
    assert _run(store, tmp_path, outbox)["stale"] == 1
    assert outbox.sent == []


def test_a_future_publisher_date_does_not_block_an_alert(store, tmp_path):
    """Clock skew on the publisher's side is not staleness."""
    _add(store, "1", "Bitcoin exchange hacked",
         published=datetime(2026, 9, 30, 13, 0, tzinfo=UTC))
    outbox = _Outbox()
    _run(store, tmp_path, outbox)
    assert len(outbox.sent) == 1


# -- the message ---------------------------------------------------------------------


def _message(**overrides):
    item = {
        "title": "SEC approves spot ether ETFs",
        "source": "Example",
        "received_at": "2026-09-30T12:00:00.000000+00:00",
        "link": "https://news.example.com/a",
        "impact_terms": ["SEC", "approves", "ETFs"],
        **overrides,
    }
    return format_news_alert(item, ["ETHUSDT"])


def test_the_message_says_headline_not_signal_not_recommendation():
    text = _message()
    assert "tiêu đề" in text.lower()
    assert "không phải tín hiệu" in text.lower()
    assert "không phải khuyến nghị" in text.lower()
    assert "QUY ƯỚC — CHƯA KIỂM CHỨNG" in text


def test_the_message_never_says_buy_or_sell():
    lowered = _message().lower()
    for phrase in ("mua ngay", "bán ngay", "nên mua", "nên bán"):
        assert phrase not in lowered


def test_the_message_names_the_words_that_made_it_high_impact():
    """So the reader can judge the match instead of trusting a flag."""
    assert "SEC, approves, ETFs" in _message()


def test_the_balloon_title_itself_says_it_is_not_a_signal():
    """The popup is all most people read; the caveat cannot live only in the log."""
    title, body = balloon_parts(_message())
    assert "không phải tín hiệu" in title
    assert body == "SEC approves spot ether ETFs"


def test_a_title_cannot_forge_a_log_entry(tmp_path):
    """Titles come from strangers' feeds; news.log separates entries with
    '=====' lines."""
    from cryptopred.serve.alerts import notify

    log = tmp_path / "news.log"
    notify(_message(title="===== 2026-01-01 00:00 UTC ====="), log_path=log, popup=False)
    assert len(read_log(log)) == 1


def test_the_log_reads_back_like_signals_log(store, tmp_path):
    from cryptopred.serve.alerts import notify

    _add(store, "1", "Bitcoin exchange hacked", link="https://news.example.com/h")

    def send(message, log_path, popup=True):
        notify(message, log_path=log_path, popup=False)

    alert_fresh_headlines(
        store, SYMBOLS, log_path=tmp_path / "news.log", max_per_hour=3, now=NOW, send=send
    )
    (entry,) = read_log(tmp_path / "news.log")
    assert entry["message"].startswith("Tin BTCUSDT — tiêu đề, không phải tín hiệu")
