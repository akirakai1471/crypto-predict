"""The headline store.

received_at is the column everything point-in-time depends on, so it is ours,
set once, and never moved. Dedupe has to be idempotent because the same feed
is read every minute and the same story is syndicated across outlets.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptopred.news.parse import Entry
from cryptopred.news.store import OLD_ON_ARRIVAL, NewsStore, utc_stamp

T0 = pd.Timestamp("2026-09-29T12:00:00Z")


def _entry(uid: str, title: str, link: str | None = None, published=None) -> Entry:
    return Entry(uid=uid, title=title, link=link, published_at=published)


@pytest.fixture
def store(tmp_path):
    return NewsStore(tmp_path / "news.db")


def test_adding_the_same_entries_twice_writes_them_once(store):
    entries = [_entry("a", "Bitcoin up"), _entry("b", "Ether down")]
    assert len(store.add(entries, source="S", received_at=T0)) == 2
    assert store.add(entries, source="S", received_at=T0 + pd.Timedelta(minutes=1)) == []
    assert store.count() == 2


def test_received_at_is_the_first_sighting_and_never_moves(store):
    """A later poll that sees the same item must not restamp it: that would make
    a headline look newer than it is, and a bar that closed in between would
    lose it."""
    store.add([_entry("a", "Bitcoin up")], source="S", received_at=T0)
    store.add([_entry("a", "Bitcoin up")], source="S", received_at=T0 + pd.Timedelta(hours=1))
    (row,) = store.recent()
    assert pd.Timestamp(row["received_at"]) == T0


def test_the_same_title_from_another_source_within_a_day_is_skipped(store):
    store.add([_entry("x.io/1", "SEC sues exchange")], source="A", received_at=T0)
    written = store.add(
        [_entry("y.io/9", "  sec SUES   exchange ")],
        source="B",
        received_at=T0 + pd.Timedelta(hours=3),
    )
    assert written == []
    assert store.count() == 1


def test_the_same_title_a_day_later_is_a_new_headline(store):
    """Recurring titles ("Bitcoin price today") are not duplicates forever."""
    store.add([_entry("x.io/1", "Bitcoin price today")], source="A", received_at=T0)
    written = store.add(
        [_entry("x.io/2", "Bitcoin price today")],
        source="A",
        received_at=T0 + pd.Timedelta(hours=25),
    )
    assert len(written) == 1


def test_a_duplicate_title_inside_one_batch_is_written_once(store):
    written = store.add(
        [_entry("x.io/1", "Same"), _entry("x.io/2", "Same")], source="A", received_at=T0
    )
    assert len(written) == 1


def test_published_at_is_kept_beside_received_at_not_instead_of_it(store):
    backdated = datetime(2026, 9, 27, 8, 0, tzinfo=UTC)
    store.add([_entry("a", "Old story", published=backdated)], source="S", received_at=T0)
    (row,) = store.recent(include_old=True)
    assert pd.Timestamp(row["published_at"]) == pd.Timestamp(backdated)
    assert pd.Timestamp(row["received_at"]) == T0


def test_a_missing_published_at_is_stored_as_null(store):
    store.add([_entry("a", "No date")], source="S", received_at=T0)
    assert store.recent()[0]["published_at"] is None


def test_stamps_sort_as_text_in_time_order():
    """received_at is compared as text in SQL. Variable-width ISO strings, which
    drop the fraction when it is zero, sort wrongly against ones that keep it."""
    a = utc_stamp(pd.Timestamp("2026-09-29T12:00:00Z"))
    b = utc_stamp(pd.Timestamp("2026-09-29T12:00:00.000001Z"))
    c = utc_stamp(pd.Timestamp("2026-09-29T19:00:00+07:00"))  # same instant as a
    assert a < b
    assert a == c
    assert len(a) == len(b)


def test_stamps_round_up_never_down():
    """An early stamp could admit a headline into a bar that closed before it
    arrived. A late one only delays it by a microsecond."""
    stamp = utc_stamp(pd.Timestamp("2026-09-29T12:00:00.000000001Z"))
    assert stamp == "2026-09-29T12:00:00.000001+00:00"


def test_a_naive_timestamp_is_refused():
    with pytest.raises(ValueError, match="naive"):
        utc_stamp(pd.Timestamp("2026-09-29T12:00:00"))


def test_recent_is_newest_first_and_carries_tags(store):
    store.add([_entry("a", "Bitcoin ETF approved")], source="S", received_at=T0)
    store.add(
        [_entry("b", "Ether upgrade scheduled")],
        source="S",
        received_at=T0 + pd.Timedelta(minutes=5),
    )
    rows = store.recent()
    assert [r["uid"] for r in rows] == ["b", "a"]
    assert rows[1]["symbols"] == ["BTCUSDT"]
    assert rows[1]["high_impact"] is True
    assert rows[1]["impact_terms"] == ["ETF", "approved"]
    assert rows[0]["high_impact"] is False


def test_recent_filters_by_symbol_across_pages(store):
    """Tags are not stored, so the symbol filter reads page after page. A filter
    that only looked at the newest page would report 'no BTC news' whenever
    the newest page happened to be about something else."""
    store.add([_entry("btc", "Bitcoin miners sell")], source="S", received_at=T0)
    store.add(
        [_entry(f"n{i}", f"Unrelated story {i}") for i in range(30)],
        source="S",
        received_at=T0 + pd.Timedelta(minutes=1),
    )
    rows = store.recent(limit=5, symbol="BTCUSDT", page=7)
    assert [r["uid"] for r in rows] == ["btc"]


def test_the_backlog_flag_is_stored(store):
    store.add([_entry("a", "Old")], source="S", backlog=True, received_at=T0)
    assert store.recent()[0]["is_backlog"] == 1


def test_polls_record_when_anything_was_listening(store):
    store.record_poll(feeds_ok=5, feeds_failed=0, inserted=2, polled_at=T0)
    store.record_poll(
        feeds_ok=0, feeds_failed=5, inserted=0, polled_at=T0 + pd.Timedelta(minutes=1)
    )
    times = store.listening_times()
    assert list(times) == [T0]
    last = store.last_poll(now=T0 + pd.Timedelta(minutes=3))
    assert last["feeds_failed"] == 5
    assert last["minutes_ago"] == pytest.approx(2.0)


def test_last_poll_is_none_before_anything_ran(store):
    assert store.last_poll() is None


def test_an_alert_decision_can_be_claimed_only_once(store):
    """The primary key is the never-twice guarantee, across restarts and across
    two processes polling the same database."""
    assert store.claim_alert("uid-1", "sent", at=T0)
    assert not store.claim_alert("uid-1", "sent", at=T0 + pd.Timedelta(minutes=1))
    assert store.alerts_sent_since(T0 - pd.Timedelta(minutes=1)) == 1


def test_a_headline_old_on_arrival_is_flagged_and_left_out_of_latest(store):
    """Measured on a real feed: a poll fifteen seconds after the first brought
    twenty items dated nine months back. They are not the latest news."""
    store.add(
        [_entry("old", "Video from last winter",
                published=(T0 - OLD_ON_ARRIVAL - pd.Timedelta(minutes=1)).to_pydatetime())],
        source="S", received_at=T0,
    )
    store.add(
        [_entry("new", "Just out", published=(T0 - pd.Timedelta(minutes=4)).to_pydatetime())],
        source="S", received_at=T0,
    )
    assert [r["uid"] for r in store.recent()] == ["new"]
    flags = {r["uid"]: r["old_on_arrival"] for r in store.recent(include_old=True)}
    assert flags == {"old": True, "new": False}


def test_a_publisher_date_in_the_future_or_missing_is_not_old(store):
    """Clock skew is not age, and no date is no evidence of age."""
    store.add([_entry("skew", "From the future",
                      published=(T0 + pd.Timedelta(hours=3)).to_pydatetime())],
              source="S", received_at=T0)
    store.add([_entry("none", "No date")], source="S", received_at=T0)
    assert {r["uid"] for r in store.recent()} == {"skew", "none"}


def test_the_features_frame_carries_the_old_flag_but_not_the_date(store):
    """The only use of a publisher's date downstream is one that can drop a row."""
    store.add([_entry("old", "Old",
                      published=(T0 - pd.Timedelta(days=3)).to_pydatetime())],
              source="S", received_at=T0)
    frame = store.headlines_frame()
    assert "published_at" not in frame.columns
    assert frame["old_on_arrival"].tolist() == [True]


def test_rows_from_one_pass_are_listed_newest_published_first(store):
    """A pass stamps its rows alike, so received_at cannot order them."""
    store.add(
        [
            _entry("older", "Older", published=(T0 - pd.Timedelta(minutes=50)).to_pydatetime()),
            _entry("newer", "Newer", published=(T0 - pd.Timedelta(minutes=5)).to_pydatetime()),
        ],
        source="S", received_at=T0,
    )
    assert [r["uid"] for r in store.recent()] == ["newer", "older"]
