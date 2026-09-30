"""What the news counts mean, apart from leakage (tests/test_news_leakage.py).

The failure these guard against is quieter than a leak: a count of zero that
really means "nothing was listening". An outage then reads as a calm market
and the restart after it as a news storm, and a later measurement of whether
headlines predict anything would be measuring the machine's uptime.
"""

import pandas as pd
import pytest

from cryptopred.features.pipeline import build_features
from cryptopred.news.features import MAX_POLL_GAP, news_features
from cryptopred.news.parse import Entry
from cryptopred.news.store import NewsStore
from tests.conftest import make_ohlcv, record_listening


def _add(store, uid, title, at, backlog=False):
    store.add(
        [Entry(uid=uid, title=title, link=None, published_at=None)],
        source="Test", backlog=backlog, received_at=at,
    )


@pytest.fixture
def bars():
    return make_ohlcv(n=72, seed=21)


@pytest.fixture
def store(tmp_path):
    return NewsStore(tmp_path / "news.db")


def test_columns_cover_each_window_for_tagged_and_high_impact(bars, store):
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    feats = news_features(bars, store, "BTCUSDT")
    assert list(feats.columns) == [
        "news_tagged_1h", "news_high_impact_1h",
        "news_tagged_4h", "news_high_impact_4h",
        "news_tagged_24h", "news_high_impact_24h",
    ]
    assert feats.index.equals(bars.index)


def test_only_headlines_tagged_with_the_symbol_count(bars, store):
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    k = 40
    at = bars["close_time"].iloc[k] - pd.Timedelta(minutes=10)
    _add(store, "1", "Bitcoin ETF inflows", at)                  # BTC, high impact
    _add(store, "2", "Bitcoin miners expand", at)                # BTC
    _add(store, "3", "Ether staking climbs", at)                 # ETH only
    _add(store, "4", "Fed cuts rates", at)                       # no coin

    btc = news_features(bars, store, "BTCUSDT")
    eth = news_features(bars, store, "ETHUSDT")
    assert btc["news_tagged_1h"].iloc[k] == 2
    assert btc["news_high_impact_1h"].iloc[k] == 1
    assert eth["news_tagged_1h"].iloc[k] == 1
    assert eth["news_high_impact_1h"].iloc[k] == 0


def test_windows_trail_by_their_width(bars, store):
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    k = 30
    _add(store, "1", "Bitcoin story", bars["close_time"].iloc[k] - pd.Timedelta(minutes=1))
    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].iloc[k] == 1
    assert feats["news_tagged_1h"].iloc[k + 1] == 0      # left the 1h window
    assert feats["news_tagged_4h"].iloc[k + 3] == 1
    assert feats["news_tagged_4h"].iloc[k + 4] == 0
    assert feats["news_tagged_24h"].iloc[k + 23] == 1
    assert feats["news_tagged_24h"].iloc[k + 24] == 0


def test_windows_before_collection_began_are_unknown_not_zero(bars, store):
    """The first poll is half an hour before bar 30 opens. Any window reaching
    back past it - bar 29's 1h window, every 24h window up to bar 52 - also
    holds the first fetch's backlog, and cannot be counted."""
    start = bars.index[30] - pd.Timedelta(minutes=30)
    record_listening(store, start, bars["close_time"].iloc[-1])
    feats = news_features(bars, store, "BTCUSDT")

    assert feats["news_tagged_1h"].iloc[:30].isna().all()
    assert feats["news_tagged_1h"].iloc[30:].notna().all()
    assert feats["news_tagged_24h"].iloc[:53].isna().all()
    assert feats["news_tagged_24h"].iloc[53:].notna().all()


def test_a_poller_that_stopped_leaves_unknowns_not_zeros(bars, store):
    """The scheduler died at bar 40. Everything after is not a quiet market."""
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars.index[40])
    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].iloc[:40].notna().all()
    assert feats["news_tagged_1h"].iloc[41:].isna().all()


def test_an_outage_blanks_every_window_that_overlaps_it(bars, store):
    """Off from bar 20's open to bar 25's open. A 1h window touching that span
    either undercounts it or holds the catch-up burst from the restart."""
    before = bars.index[0] - pd.Timedelta(days=1)
    record_listening(store, before, bars.index[20])
    record_listening(store, bars.index[25], bars["close_time"].iloc[-1])
    feats = news_features(bars, store, "BTCUSDT")

    col = feats["news_tagged_1h"]
    assert col.iloc[:20].notna().all()
    assert col.iloc[20:25].isna().all()     # not listening at these closes
    assert pd.isna(col.iloc[25])            # holds the restart, and its catch-up burst
    assert col.iloc[26:].notna().all()
    # The 4h window keeps the restart in view for three bars longer.
    assert feats["news_tagged_4h"].iloc[25:29].isna().all()
    assert feats["news_tagged_4h"].iloc[29:].notna().all()


def test_a_short_gap_is_not_an_outage(bars, store):
    """A missed pass or two is normal - RSS itself lags by minutes."""
    before = bars.index[0] - pd.Timedelta(days=1)
    gap_start = bars.index[20]
    record_listening(store, before, gap_start)
    record_listening(store, gap_start + MAX_POLL_GAP - pd.Timedelta(minutes=1),
                     bars["close_time"].iloc[-1])
    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].notna().all()


def test_a_backlog_burst_blanks_the_windows_that_contain_it(bars, store):
    """The first fetch after a restart delivers days of old items at one
    instant. Counted, they would look like a news storm."""
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    k = 40
    burst_at = bars["close_time"].iloc[k] - pd.Timedelta(minutes=5)
    for i in range(20):
        _add(store, f"old-{i}", f"Bitcoin old story {i}", burst_at, backlog=True)
    feats = news_features(bars, store, "BTCUSDT")

    assert pd.isna(feats["news_tagged_1h"].iloc[k])
    assert feats["news_tagged_1h"].iloc[k + 1] == 0
    assert feats["news_tagged_4h"].iloc[k:k + 4].isna().all()
    assert feats["news_tagged_4h"].iloc[k + 4] == 0


def test_a_store_that_never_polled_gives_all_unknowns(bars, store):
    assert news_features(bars, store, "BTCUSDT").isna().all().all()


def test_bars_without_a_close_time_are_refused(bars, store):
    """Guessing the close from the open is how an off-by-one-bar leak starts."""
    with pytest.raises(ValueError, match="close_time"):
        news_features(bars.drop(columns=["close_time"]), store, "BTCUSDT")


def test_news_features_are_not_in_the_training_pipeline():
    """Deliberate. There is no history to put them through the walk-forward
    gates yet; a feature that has not passed them does not reach the model.
    Remove this test only together with a findings.md entry that measured them.
    """
    feats = build_features(make_ohlcv(n=600, seed=22), interval="1h")
    assert not [c for c in feats.columns if "news" in c.lower()]


def test_a_headline_old_on_arrival_is_not_counted_and_does_not_blank_the_window(bars, store):
    """A nine-month-old post rotated back into a feed is not news flow, but
    nothing was missed either: drop it, keep the window."""
    record_listening(store, bars.index[0] - pd.Timedelta(days=1), bars["close_time"].iloc[-1])
    k = 40
    at = bars["close_time"].iloc[k] - pd.Timedelta(minutes=10)
    store.add(
        [Entry(uid="old", title="Bitcoin ETF video from last winter", link=None,
               published_at=(at - pd.Timedelta(days=270)).to_pydatetime())],
        source="Decrypt", received_at=at,
    )
    _add(store, "new", "Bitcoin ETF inflows", at)
    feats = news_features(bars, store, "BTCUSDT")
    assert feats["news_tagged_1h"].iloc[k] == 1
    assert feats["news_high_impact_1h"].iloc[k] == 1
