"""The poll pass and the scheduler job that repeats it.

The hourly prediction cycle is the experiment; news is a display feature
beside it. Nothing the news job does - a dead feed, a locked database, a
broken alert - may reach the prediction job, and these tests hold that line.
"""

from pathlib import Path

import httpx
import pandas as pd
import pytest

from cryptopred.config import Config, FeedConfig
from cryptopred.news.fetch import FeedFetcher
from cryptopred.news.poller import NewsJob, poll_once
from cryptopred.news.store import NewsStore

FIXTURES = Path(__file__).parent / "fixtures" / "news"
RSS = (FIXTURES / "rss2.xml").read_bytes()
ATOM = (FIXTURES / "atom.xml").read_bytes()

FEEDS = [
    FeedConfig(name="Rss", url="https://rss.example.com/feed"),
    FeedConfig(name="Atom", url="https://atom.example.com/feed.atom"),
    FeedConfig(name="Dead", url="https://dead.example.com/rss"),
]


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "dead.example.com":
        raise httpx.ConnectTimeout("no route", request=request)
    return httpx.Response(200, content=RSS if request.url.host == "rss.example.com" else ATOM)


def _fetcher(handler=_handler) -> FeedFetcher:
    return FeedFetcher(FEEDS, transport=httpx.MockTransport(handler))


@pytest.fixture
def cfg(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.news.feeds = FEEDS
    return cfg


def test_a_pass_stores_new_headlines_and_logs_itself(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    summary = poll_once(_fetcher(), store)
    assert len(summary.inserted) == 5
    assert (summary.feeds_ok, summary.feeds_failed) == (2, 1)
    assert store.count() == 5
    assert store.last_poll()["feeds_failed"] == 1


def test_the_first_pass_is_backlog_and_the_next_is_not(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    fresh = b"""<rss version="2.0"><channel><title>R</title><item>
        <title>Ether ETF approved</title><link>https://rss.example.com/new</link>
    </item></channel></rss>"""
    state = {"published": False}

    def handler(request):
        if request.url.host == "rss.example.com" and state["published"]:
            return httpx.Response(200, content=fresh)
        return _handler(request)

    fetcher = _fetcher(handler)
    first = poll_once(fetcher, store)
    assert first.backlog == 5

    state["published"] = True
    second = poll_once(fetcher, store)
    assert [r["title"] for r in second.inserted] == ["Ether ETF approved"]
    assert second.backlog == 0


def test_the_pass_is_stamped_no_earlier_than_anything_it_wrote(tmp_path):
    """news/features.py treats the pass's stamp as the moment the record is
    complete up to; a stamp before its own rows would leave them uncovered."""
    store = NewsStore(tmp_path / "news.db")
    ticks = iter(pd.date_range("2026-09-30T12:00Z", periods=10, freq="s"))
    poll_once(_fetcher(), store, clock=lambda: next(ticks))
    newest = max(
        pd.Timestamp(r["received_at"]) for r in store.recent(limit=50, include_old=True)
    )
    assert pd.Timestamp(store.last_poll()["polled_at"]) >= newest


def test_a_second_pass_over_unchanged_feeds_writes_nothing(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    fetcher = _fetcher()
    poll_once(fetcher, store)
    assert poll_once(fetcher, store).inserted == []
    assert store.count() == 5


# -- the scheduler job -------------------------------------------------------------------


def test_the_job_never_raises_when_the_database_fails(cfg, monkeypatch):
    job = NewsJob(cfg, fetcher=_fetcher(), popup=False)

    def locked(*_a, **_k):
        raise OSError("database is locked")

    monkeypatch.setattr(job.store, "add", locked)
    assert job()["inserted"] == 0


def test_the_job_never_raises_when_every_feed_fails(cfg):
    def dead(request):
        raise httpx.ConnectError("offline", request=request)

    counts = NewsJob(cfg, fetcher=_fetcher(dead), popup=False)()
    assert counts["feeds_failed"] == 3


def test_the_job_never_raises_when_alerting_fails(cfg, monkeypatch):
    from cryptopred.news import poller

    def broken(*_a, **_k):
        raise RuntimeError("alert code bug")

    monkeypatch.setattr(poller, "alert_fresh_headlines", broken)
    counts = NewsJob(cfg, fetcher=_fetcher(), popup=False)()
    assert counts["inserted"] == 5
    assert counts["alerts"] == 0


def test_alerts_can_be_switched_off(cfg, monkeypatch):
    from cryptopred.news import poller

    called = []
    monkeypatch.setattr(poller, "alert_fresh_headlines", lambda *a, **k: called.append(1))
    cfg.news.alert_high_impact = False
    NewsJob(cfg, fetcher=_fetcher(), popup=False)()
    assert called == []


def test_the_scheduler_polls_news_on_its_own_interval_beside_the_cycle(cfg):
    from cryptopred.serve.cli import build_scheduler

    cfg.news.poll_seconds = 90
    scheduler = build_scheduler(cfg, "1h", 2)
    cycle, news = scheduler.get_job("cycle"), scheduler.get_job("news")

    assert cycle is not None and news is not None
    assert news.trigger.interval.total_seconds() == 90
    # One at a time, and missed runs collapse into one - a hung feed cannot
    # pile up threads behind itself.
    assert news.max_instances == 1
    assert news.coalesce is True
    assert isinstance(news.func, NewsJob)


def test_a_news_job_that_cannot_start_leaves_the_prediction_job_scheduled(cfg, monkeypatch):
    from cryptopred.news import poller
    from cryptopred.serve.cli import build_scheduler

    class Broken:
        def __init__(self, *_a, **_k):
            raise OSError("disk full")

    monkeypatch.setattr(poller, "NewsJob", Broken)
    scheduler = build_scheduler(cfg, "1h", 2)
    assert scheduler.get_job("cycle") is not None
    assert scheduler.get_job("news") is None


def test_no_feeds_means_no_news_job(cfg):
    from cryptopred.serve.cli import build_scheduler

    cfg.news.feeds = []
    assert build_scheduler(cfg, "1h", 2).get_job("news") is None


def test_the_news_jobs_routine_log_lines_are_dropped_and_the_cycles_are_not(cfg):
    """2,880 "Running job" lines a day would bury the hourly cycle's own log."""
    import logging

    from cryptopred.serve.cli import QuietNewsRuns, build_scheduler

    scheduler = build_scheduler(cfg, "1h", 2)
    news, cycle = scheduler.get_job("news"), scheduler.get_job("cycle")

    def record(level, msg, job):
        return logging.LogRecord(
            "apscheduler.executors.default", level, __file__, 1, msg, (job,), None
        )

    quiet = QuietNewsRuns()
    assert not quiet.filter(record(logging.INFO, 'Running job "%s"', news))
    assert not quiet.filter(record(logging.INFO, 'Job "%s" executed successfully', news))
    assert quiet.filter(record(logging.INFO, 'Running job "%s"', cycle))
    assert quiet.filter(record(logging.WARNING, 'Run time of job "%s" was missed', news))
    assert any(
        isinstance(f, QuietNewsRuns)
        for f in logging.getLogger("apscheduler.executors.default").filters
    )
