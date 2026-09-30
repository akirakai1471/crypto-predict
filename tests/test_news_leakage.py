"""Leakage detection for news features.

The point-in-time rule for news: a bar may count a headline only if our own
clock had stored it by the bar's close. Not the publisher's date - publishers
backdate - and not "the same day". These tests are the enforcement. News
features are not in the model yet; when someone proposes adding them, these
are what make the measurement worth reading. If one fails, the counts carry
the future and any result built on them is fiction.
"""

import pandas as pd

from cryptopred.news.features import news_features
from cryptopred.news.parse import Entry
from cryptopred.news.store import NewsStore
from tests.conftest import make_ohlcv, record_listening

ONE_MS = pd.Timedelta(milliseconds=1)


def _store(tmp_path, bars) -> NewsStore:
    """A store that was listening from a day before the first bar to the last close."""
    store = NewsStore(tmp_path / "news.db")
    first_open = bars.index[0]
    record_listening(store, first_open - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    return store


def _headline(store, uid, title, received_at, published_at=None) -> None:
    written = store.add(
        [Entry(uid=uid, title=title, link=None, published_at=published_at)],
        source="Test",
        received_at=received_at,
    )
    assert written, "fixture headline was deduplicated away"


def test_a_headline_received_one_millisecond_after_the_close_is_not_in_that_bar(tmp_path):
    """The whole rule in one test. Bar k closes at close_time[k]; a headline we
    stored 1 ms later was not knowable when bar k's features were computed."""
    bars = make_ohlcv(n=48, seed=11)
    store = _store(tmp_path, bars)
    k = 30
    close_k = bars["close_time"].iloc[k]
    _headline(store, "late", "Bitcoin ETF approved", close_k + ONE_MS)

    feats = news_features(bars, store, "BTCUSDT")

    assert feats["news_tagged_1h"].iloc[k] == 0
    assert feats["news_high_impact_1h"].iloc[k] == 0
    assert feats["news_tagged_24h"].iloc[k] == 0
    # It belongs to the next bar, whose close is after it.
    assert feats["news_tagged_1h"].iloc[k + 1] == 1
    assert feats["news_high_impact_1h"].iloc[k + 1] == 1


def test_a_headline_received_exactly_at_the_close_is_in_that_bar(tmp_path):
    """At or before the close, inclusive - the same boundary price features use."""
    bars = make_ohlcv(n=48, seed=12)
    store = _store(tmp_path, bars)
    k = 30
    _headline(store, "on-time", "Bitcoin ETF approved", bars["close_time"].iloc[k])

    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].iloc[k] == 1
    assert feats["news_tagged_1h"].iloc[k + 1] == 0
    assert feats["news_tagged_4h"].iloc[k + 3] == 1


def test_the_publishers_date_is_never_what_admits_a_headline(tmp_path):
    """A story the publisher dates an hour before the close, but that we only
    stored after it, stays out. Backdating must not be able to leak it in."""
    bars = make_ohlcv(n=48, seed=13)
    store = _store(tmp_path, bars)
    k = 30
    close_k = bars["close_time"].iloc[k]
    _headline(
        store, "backdated", "Bitcoin exchange hacked",
        received_at=close_k + pd.Timedelta(seconds=30),
        published_at=(close_k - pd.Timedelta(hours=1)).to_pydatetime(),
    )
    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].iloc[k] == 0
    assert feats["news_tagged_1h"].iloc[k + 1] == 1


def test_future_headlines_and_outages_cannot_change_past_features(tmp_path):
    """Poison everything after the cut: a flood of headlines, a backlog burst,
    and a poller that stops. Features at and before the cut must be identical."""
    bars = make_ohlcv(n=72, seed=14)
    cut = 50
    cut_close = bars["close_time"].iloc[cut - 1]

    clean = NewsStore(tmp_path / "clean.db")
    poisoned = NewsStore(tmp_path / "poisoned.db")
    start = bars.index[0] - pd.Timedelta(days=1)
    last_close = bars["close_time"].iloc[-1]
    record_listening(clean, start, last_close)
    # The poisoned poller stops at the cut and comes back five hours later.
    record_listening(poisoned, start, cut_close)
    record_listening(poisoned, cut_close + pd.Timedelta(hours=5), last_close)
    for store in (clean, poisoned):
        for i in range(0, cut - 1, 5):
            _headline(store, f"past-{i}", f"Bitcoin story {i}", bars["close_time"].iloc[i])

    # Headlines and a backlog burst, beginning 1 ms after the last shared close.
    for i in range(20):
        _headline(
            poisoned, f"future-{i}", f"Bitcoin SEC ETF hack {i}",
            cut_close + ONE_MS + pd.Timedelta(minutes=i),
        )
    poisoned.add(
        [Entry(uid="burst", title="Old bitcoin backlog", link=None, published_at=None)],
        source="Test", backlog=True, received_at=cut_close + ONE_MS,
    )

    clean_feats = news_features(bars, clean, "BTCUSDT").iloc[:cut]
    poisoned_feats = news_features(bars, poisoned, "BTCUSDT").iloc[:cut]
    pd.testing.assert_frame_equal(clean_feats, poisoned_feats)
    # And the poison did land after the cut, so the test is not vacuous.
    assert not news_features(bars, poisoned, "BTCUSDT").iloc[cut:].equals(
        news_features(bars, clean, "BTCUSDT").iloc[cut:]
    )


def test_news_features_are_prefix_invariant(tmp_path):
    """Computing on a truncated bar history must give the same values for the
    bars both runs share. The store holds headlines and polls after the cut;
    a value that changes with truncation depended on them."""
    bars = make_ohlcv(n=72, seed=15)
    store = _store(tmp_path, bars)
    for i in range(0, 72, 3):
        _headline(store, f"h{i}", f"Ether ETF story {i}", bars["close_time"].iloc[i] - ONE_MS)

    cut = 40
    full = news_features(bars, store, "ETHUSDT").iloc[:cut]
    partial = news_features(bars.iloc[:cut], store, "ETHUSDT")
    pd.testing.assert_frame_equal(full, partial)
    assert full["news_tagged_24h"].iloc[-1] > 0
