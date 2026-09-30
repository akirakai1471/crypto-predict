"""Feed fetching, offline, through httpx.MockTransport.

Two properties matter more than any single response. Polling every minute has
to be cheap for the publisher, which only conditional GET makes true. And one
broken feed must cost its own headlines and nothing else - an exception out of
a single fetch would take down the pass, and inside the scheduler, the job.
"""

from pathlib import Path

import httpx
import pytest

from cryptopred.config import FeedConfig
from cryptopred.news.fetch import BACKLOG_AFTER_SECONDS, FeedFetcher

FIXTURES = Path(__file__).parent / "fixtures" / "news"
RSS = (FIXTURES / "rss2.xml").read_bytes()
ATOM = (FIXTURES / "atom.xml").read_bytes()

GOOD = FeedConfig(name="Good", url="https://good.example.com/rss")
ALSO_GOOD = FeedConfig(name="AlsoGood", url="https://atom.example.com/feed.atom")
BAD = FeedConfig(name="Bad", url="https://bad.example.com/rss")


def _fetcher(handler, feeds, **kwargs) -> FeedFetcher:
    return FeedFetcher(feeds, transport=httpx.MockTransport(handler), **kwargs)


class _Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_a_200_is_parsed_into_entries():
    fetcher = _fetcher(lambda req: httpx.Response(200, content=RSS), [GOOD])
    (result,) = fetcher.fetch_all()
    assert result.status == "ok"
    assert result.http_status == 200
    assert len(result.entries) == 3
    assert fetcher.failures["Good"] == 0


def test_the_second_poll_is_conditional_and_a_304_costs_nothing():
    """Without If-None-Match / If-Modified-Since, a one-minute poll downloads
    every feed in full 1,440 times a day."""
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        if request.headers.get("If-None-Match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=RSS,
            headers={"ETag": '"v1"', "Last-Modified": "Tue, 29 Sep 2026 14:05:00 GMT"},
        )

    fetcher = _fetcher(handler, [GOOD])
    first = fetcher.fetch(GOOD)
    second = fetcher.fetch(GOOD)

    assert "If-None-Match" not in seen[0]
    assert seen[1]["If-None-Match"] == '"v1"'
    assert seen[1]["If-Modified-Since"] == "Tue, 29 Sep 2026 14:05:00 GMT"
    assert first.status == "ok"
    assert second.status == "not_modified"
    assert second.entries == []
    assert second.ok


def test_validators_are_kept_per_feed():
    """One feed's ETag sent to another publisher would get a 200 at best and a
    wrong 304 at worst."""
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = request.headers.get("If-None-Match")
        body = RSS if request.url.host == "good.example.com" else ATOM
        return httpx.Response(200, content=body, headers={"ETag": f'"{request.url.host}"'})

    fetcher = _fetcher(handler, [GOOD, ALSO_GOOD])
    fetcher.fetch_all()
    fetcher.fetch_all()
    assert seen == {
        "good.example.com": '"good.example.com"',
        "atom.example.com": '"atom.example.com"',
    }


def test_a_timeout_is_a_counted_failure_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    fetcher = _fetcher(handler, [GOOD])
    (result,) = fetcher.fetch_all()
    assert result.status == "error"
    assert "ReadTimeout" in result.error
    assert fetcher.failures["Good"] == 1


def test_one_failing_feed_never_stops_the_others():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad.example.com":
            raise httpx.ConnectError("connection refused", request=request)
        body = RSS if request.url.host == "good.example.com" else ATOM
        return httpx.Response(200, content=body)

    fetcher = _fetcher(handler, [GOOD, BAD, ALSO_GOOD])
    results = fetcher.fetch_all()

    assert [r.status for r in results] == ["ok", "error", "ok"]
    assert len(results[0].entries) == 3
    assert len(results[2].entries) == 2
    assert fetcher.failures == {"Bad": 1}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="maintenance"),
        httpx.Response(404, text="moved"),
        httpx.Response(200, text="<html><body>Just a moment...</body></html>"),
        httpx.Response(200, text="<rss><channel><item><title>cut off"),
    ],
    ids=["503", "404", "html-with-200", "truncated-xml"],
)
def test_every_kind_of_bad_answer_is_an_error_result(response):
    fetcher = _fetcher(lambda req: response, [GOOD])
    result = fetcher.fetch(GOOD)
    assert result.status == "error"
    assert result.entries == []
    assert fetcher.failures["Good"] == 1


def test_an_unexpected_exception_is_still_contained():
    """Not only httpx errors: anything a feed can provoke stays inside fetch."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("something nobody anticipated")

    result = _fetcher(handler, [GOOD]).fetch(GOOD)
    assert result.status == "error"
    assert "RuntimeError" in result.error


def test_an_oversized_body_is_refused():
    big = b"<rss><channel>" + b" " * 2_000 + b"</channel></rss>"
    fetcher = _fetcher(lambda req: httpx.Response(200, content=big), [GOOD], max_bytes=1_000)
    result = fetcher.fetch(GOOD)
    assert result.status == "error"
    assert "larger than" in result.error


def test_a_body_that_failed_to_parse_does_not_leave_its_etag_behind():
    """Otherwise every later poll is a 304 for content never read, until the
    publisher happens to change it."""
    calls = {"n": 0}
    sent: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        sent.append(request.headers.get("If-None-Match"))
        if calls["n"] == 1:
            return httpx.Response(200, text="<html/>", headers={"ETag": '"broken"'})
        return httpx.Response(200, content=RSS, headers={"ETag": '"fixed"'})

    fetcher = _fetcher(handler, [GOOD])
    assert fetcher.fetch(GOOD).status == "error"
    assert fetcher.fetch(GOOD).status == "ok"
    assert sent == [None, None]


def test_redirects_are_followed_and_relative_links_resolve_against_the_final_url():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rss":
            return httpx.Response(301, headers={"Location": "https://atom.example.com/feed.atom"})
        return httpx.Response(200, content=ATOM)

    result = _fetcher(handler, [GOOD]).fetch(GOOD)
    assert result.status == "ok"
    assert result.entries[0].link == "https://atom.example.com/2026/09/29/exchange-exploit"


def test_a_user_agent_is_sent():
    """Some publishers' CDNs reject the default python-httpx agent outright."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, content=RSS)

    _fetcher(handler, [GOOD]).fetch(GOOD)
    assert seen["ua"].startswith("cryptopred-news/")


# -- backlog ----------------------------------------------------------------------------


def test_the_first_success_after_start_is_backlog_and_later_ones_are_not():
    """RSS keeps days of items; the first fetch delivers all of them at once.
    Every one is new to us and none is new to the world."""
    clock = _Clock()
    fetcher = _fetcher(lambda req: httpx.Response(200, content=RSS), [GOOD], wall_clock=clock)
    assert fetcher.fetch(GOOD).backlog is True
    clock.t += 60
    assert fetcher.fetch(GOOD).backlog is False


def test_the_first_success_after_an_outage_is_backlog_again():
    clock = _Clock()
    state = {"down": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["down"]:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, content=RSS)

    fetcher = _fetcher(handler, [GOOD], wall_clock=clock)
    fetcher.fetch(GOOD)
    state["down"] = True
    for _ in range(30):
        clock.t += 60
        assert fetcher.fetch(GOOD).status == "error"
    state["down"] = False
    clock.t += 60
    assert clock.t - 1_000_000.0 > BACKLOG_AFTER_SECONDS
    assert fetcher.fetch(GOOD).backlog is True


# -- redirects and logging --------------------------------------------------------------


def test_a_redirect_from_https_to_plain_http_is_refused():
    """Bitcoin Magazine's old feed URL does exactly this (measured 2026-09-30).
    Content that crossed the network in the clear could have been rewritten on
    the way; it must not be stored as though it had not."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "https":
            return httpx.Response(301, headers={"Location": "http://good.example.com/feed"})
        return httpx.Response(200, content=RSS)

    fetcher = _fetcher(handler, [GOOD])
    result = fetcher.fetch(GOOD)
    assert result.status == "error"
    assert "plain http" in result.error
    assert result.entries == []
    assert fetcher.failures["Good"] == 1


def test_a_dead_feed_warns_once_then_hourly_and_says_when_it_recovers(caplog):
    """A feed down for a day must not print 1,440 warnings into the log someone
    reads to see whether the hourly predictions ran."""
    import logging

    from cryptopred.news.fetch import REPEAT_WARNING_EVERY

    state = {"down": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["down"]:
            return httpx.Response(503)
        return httpx.Response(200, content=RSS)

    fetcher = _fetcher(handler, [GOOD])
    with caplog.at_level(logging.INFO, logger="cryptopred.news.fetch"):
        for _ in range(REPEAT_WARNING_EVERY * 2):
            fetcher.fetch(GOOD)
        state["down"] = False
        fetcher.fetch(GOOD)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3  # the first failure, then the 60th and the 120th
    assert "recovered after 120 failures" in caplog.records[-1].getMessage()
    assert fetcher.failures["Good"] == 120
    assert fetcher.consecutive_failures["Good"] == 0


def test_httpx_request_lines_are_dropped_only_inside_a_news_fetch(caplog):
    """httpx logs every request at INFO: five or more lines a minute from news
    alone. The prediction cycle's own requests must keep logging."""
    import logging

    fetcher = _fetcher(lambda req: httpx.Response(200, content=RSS), [GOOD])
    other = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    with caplog.at_level(logging.INFO, logger="httpx"):
        fetcher.fetch(GOOD)
        other.get("https://fapi.binance.example/klines")

    lines = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert len(lines) == 1
    assert "fapi.binance.example" in lines[0]
