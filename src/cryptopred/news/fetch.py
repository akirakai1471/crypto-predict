"""Poll feeds cheaply, and never let one publisher stop the others.

Every minute is affordable only because an unchanged feed costs one
conditional request answered by 304 and a few hundred bytes: the ETag and
Last-Modified from the last good response are sent back as If-None-Match and
If-Modified-Since. Validators are kept in memory per feed, so the first poll
after a restart downloads each feed once in full.

Failure is the normal case for a set of third-party feeds - a timeout, a 503,
a Cloudflare page served with status 200, a feed that moved. `fetch` turns
every one of those into a FeedResult with status "error", logs it and counts
it. It never raises, so a broken feed costs its own headlines and nothing else.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from cryptopred.config import FeedConfig
from cryptopred.news.parse import Entry, parse_feed

logger = logging.getLogger(__name__)

USER_AGENT = "cryptopred-news/0.1 (personal research; RSS reader; polls about once a minute)"
ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, "
    "text/xml;q=0.9, */*;q=0.1"
)
# Short, because a poll every minute is its own retry. Worst case with every
# host black-holing packets: five feeds x (connect 5s + deadline) - long, but
# the scheduler job runs one instance at a time and coalesces the rest.
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
# Wall-clock cap on one download, which a per-read timeout does not give: a
# server dripping a byte every nine seconds never trips READ_TIMEOUT.
FETCH_DEADLINE = 15.0
# Real feeds are well under 1 MB. The cap applies after decompression, so a
# gzip bomb costs a failed fetch rather than the process's memory.
MAX_FEED_BYTES = 5_000_000
# A feed silent for longer than this has left a gap: the next successful fetch
# may carry items published while nobody was listening.
BACKLOG_AFTER_SECONDS = 15 * 60


@dataclass
class FeedResult:
    name: str
    url: str
    status: str  # "ok", "not_modified" or "error"
    entries: list[Entry] = field(default_factory=list)
    error: str | None = None
    http_status: int | None = None
    # True when this is the feed's first success since the process started or
    # since an outage: its items may be days old. See news/store.py.
    backlog: bool = False

    @property
    def ok(self) -> bool:
        return self.status != "error"


class FeedFetcher:
    """One httpx client for every feed, with conditional-GET state per feed."""

    def __init__(
        self,
        feeds: Sequence[FeedConfig],
        transport: httpx.BaseTransport | None = None,
        max_bytes: int = MAX_FEED_BYTES,
        deadline: float = FETCH_DEADLINE,
        clock=time.monotonic,
        wall_clock=time.time,
    ) -> None:
        self.feeds = list(feeds)
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            # http->https and trailing-slash redirects are routine for feeds.
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": ACCEPT},
        )
        self._max_bytes = max_bytes
        self._deadline = deadline
        self._clock = clock
        # Wall time for gaps, not the monotonic clock: on some platforms the
        # monotonic clock stops while a laptop sleeps, and eight hours asleep
        # would then look like one missed minute.
        self._wall_clock = wall_clock
        self._validators: dict[str, dict[str, str]] = {}
        self._last_success: dict[str, float] = {}
        # Failures since this fetcher was created, per feed name.
        self.failures: Counter[str] = Counter()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FeedFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_all(self) -> list[FeedResult]:
        return [self.fetch(feed) for feed in self.feeds]

    def fetch(self, feed: FeedConfig) -> FeedResult:
        """Fetch and parse one feed. Never raises."""
        try:
            result = self._fetch(feed)
        except Exception as exc:  # noqa: BLE001 - one publisher must never stop the others
            result = FeedResult(
                feed.name, feed.url, "error", error=f"{type(exc).__name__}: {exc}"
            )

        if not result.ok:
            self.failures[feed.name] += 1
            logger.warning("news feed %s failed: %s", feed.name, result.error)
            return result

        now = self._wall_clock()
        last = self._last_success.get(feed.url)
        result.backlog = last is None or (now - last) > BACKLOG_AFTER_SECONDS
        self._last_success[feed.url] = now
        return result

    def _fetch(self, feed: FeedConfig) -> FeedResult:
        headers = self._validators.get(feed.url, {})
        started = self._clock()
        with self._client.stream("GET", feed.url, headers=headers) as response:
            if response.status_code == 304:
                return FeedResult(feed.name, feed.url, "not_modified", http_status=304)
            if response.status_code != 200:
                return FeedResult(
                    feed.name, feed.url, "error",
                    error=f"HTTP {response.status_code}",
                    http_status=response.status_code,
                )
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > self._max_bytes:
                    return FeedResult(
                        feed.name, feed.url, "error",
                        error=f"larger than {self._max_bytes:,} bytes", http_status=200,
                    )
                if self._clock() - started > self._deadline:
                    return FeedResult(
                        feed.name, feed.url, "error",
                        error=f"slower than {self._deadline:.0f}s", http_status=200,
                    )
            validators = {}
            if etag := response.headers.get("ETag"):
                validators["If-None-Match"] = etag
            if modified := response.headers.get("Last-Modified"):
                validators["If-Modified-Since"] = modified
            base_url = str(response.url)

        entries = parse_feed(bytes(body), base_url=base_url, source=feed.name)
        # Remembered only once the body parsed. Keeping the validators of a
        # body that failed would turn every later poll into a 304 for content
        # never read, until the publisher happened to change it.
        self._validators[feed.url] = validators
        return FeedResult(feed.name, feed.url, "ok", entries=entries, http_status=200)
