"""One pass over every feed, and the scheduler job that repeats it.

A pass fetches every feed, stores what is new, and then - last - logs itself in
the polls table. Last, so the pass's stamp is never earlier than any headline
it wrote: news/features.py reads that table to tell "no headlines" from
"nobody listening", and it needs the pass to end after its rows arrive.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cryptopred.config import Config
from cryptopred.news.alerts import alert_fresh_headlines
from cryptopred.news.fetch import FeedFetcher, FeedResult
from cryptopred.news.store import NewsStore

logger = logging.getLogger(__name__)


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass
class PollSummary:
    results: list[FeedResult]
    inserted: list[dict[str, Any]] = field(default_factory=list)

    @property
    def feeds_ok(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def feeds_failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def new_by_source(self) -> Counter[str]:
        return Counter(row["source"] for row in self.inserted)

    @property
    def backlog(self) -> int:
        return sum(1 for row in self.inserted if row["is_backlog"])


def poll_once(
    fetcher: FeedFetcher,
    store: NewsStore,
    clock: Callable[[], pd.Timestamp] = _utcnow,
) -> PollSummary:
    """Fetch every feed, store the new headlines, log the pass.

    Feed failures are already contained by the fetcher. A store failure is not
    caught here: a database that cannot be written means the pass did not
    happen, and it must not be logged as though it did.
    """
    results = fetcher.fetch_all()
    inserted: list[dict[str, Any]] = []
    for result in results:
        if result.status == "ok" and result.entries:
            inserted += store.add(
                result.entries, source=result.name, backlog=result.backlog,
                received_at=clock(),
            )
    summary = PollSummary(results=results, inserted=inserted)
    store.record_poll(
        feeds_ok=summary.feeds_ok,
        feeds_failed=summary.feeds_failed,
        inserted=len(inserted),
        polled_at=clock(),
    )
    return summary


class NewsJob:
    """The scheduler's news pass: poll, store, maybe alert. Never raises.

    APScheduler already runs each job in its own pool thread, so an exception
    here would not reach the hourly prediction job - but "a library default
    happens to isolate it" is not a guarantee worth relying on, and a traceback
    every minute would bury the cycle's own log. Nothing is shared with the
    prediction cycle: its own database file, its own HTTP client, its own
    thread.
    """

    def __init__(
        self,
        cfg: Config,
        fetcher: FeedFetcher | None = None,
        store: NewsStore | None = None,
        popup: bool = True,
    ) -> None:
        self.cfg = cfg
        self.store = store or NewsStore(cfg.data.root / "news.db")
        self.fetcher = fetcher or FeedFetcher(cfg.news.feeds)
        self.popup = popup

    def __call__(self) -> dict[str, int]:
        counts = {"feeds_ok": 0, "feeds_failed": 0, "inserted": 0, "alerts": 0}
        try:
            summary = poll_once(self.fetcher, self.store)
        except Exception:  # noqa: BLE001 - news must never cost the scheduler anything
            logger.exception("news pass failed")
            return counts

        counts.update(
            feeds_ok=summary.feeds_ok,
            feeds_failed=summary.feeds_failed,
            inserted=len(summary.inserted),
        )
        if self.cfg.news.alert_high_impact:
            try:
                decided = alert_fresh_headlines(
                    self.store,
                    symbols=self.cfg.data.symbols,
                    log_path=self.cfg.data.root / "news.log",
                    max_per_hour=self.cfg.news.max_alerts_per_hour,
                    popup=self.popup,
                )
                counts["alerts"] = decided["sent"]
            except Exception:  # noqa: BLE001 - a lost alert must not lose the pass
                logger.exception("news alerts failed")

        # Quiet unless something was stored: a line every minute would drown
        # the prediction cycle's hourly log. Feed failures log themselves, at
        # a rate that does not repeat every minute (news/fetch.py).
        if summary.inserted:
            logger.info("news pass: %s (%d backlog)", counts, summary.backlog)
        return counts
