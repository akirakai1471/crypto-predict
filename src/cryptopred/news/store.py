"""SQLite store for headlines, stamped with the moment we first had them.

`received_at` is the timestamp that matters, and it is ours: the UTC clock of
this machine when the row is first written. The feed's own date is stored
beside it as `published_at` and never used where point-in-time matters,
because publishers backdate, re-date on every edit, leave out the zone, or
report when the draft was opened. Only our own clock can prove a headline was
known before a bar closed - the same property that makes the prediction log
worth more than a backtest (serve/store.py). Rows are never updated, so the
stamp cannot drift away from the moment it records.

`is_backlog` marks rows that arrived on a feed's first successful fetch after
the poller started or after an outage. RSS keeps days of items, so such a row
may be old news: its received_at is correct as "when we could first have
known", but it is not "when the story broke", and nothing that alerts or
measures news flow may treat it as fresh.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cryptopred.news.parse import Entry, title_key
from cryptopred.news.tags import DEFAULT_TAGGER, Tagger

# The same wire story syndicated by two outlets is one headline, not two. A
# day is long enough to catch syndication and short enough that a recurring
# title ("Bitcoin price today") is not suppressed forever.
DUPLICATE_TITLE_WINDOW = pd.Timedelta(hours=24)

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    uid          TEXT    NOT NULL UNIQUE,
    source       TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    title_key    TEXT    NOT NULL,
    link         TEXT,
    -- The publisher's claim. Nullable, and may be wrong.
    published_at TEXT,
    -- Our UTC clock when the row was first written. Fixed-width ISO with
    -- microseconds, so text order is time order.
    received_at  TEXT    NOT NULL,
    is_backlog   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_headlines_received ON headlines (received_at);
CREATE INDEX IF NOT EXISTS idx_headlines_title ON headlines (title_key, received_at);

-- One row per completed pass. Zero headlines while nothing was listening is not
-- zero news, and this table is the only way to tell the two apart later.
CREATE TABLE IF NOT EXISTS polls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    polled_at    TEXT    NOT NULL,
    feeds_ok     INTEGER NOT NULL,
    feeds_failed INTEGER NOT NULL,
    inserted     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_polls_at ON polls (polled_at);

-- Every alert decision, including the ones that did not send. The primary key
-- is what makes "never the same headline twice" hold across restarts and
-- across two processes polling at once.
CREATE TABLE IF NOT EXISTS alerts (
    uid        TEXT PRIMARY KEY,
    decided_at TEXT NOT NULL,
    outcome    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_decided ON alerts (decided_at);
"""


def utc_stamp(ts: datetime | pd.Timestamp) -> str:
    """Fixed-width UTC ISO text, rounded UP to the microsecond.

    Up, never down: a stamp earlier than the truth is the one error this column
    must not make, because it would let a headline into a bar that closed
    before we had it.
    """
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        raise ValueError("naive timestamp: a time with no zone cannot be placed")
    stamp = stamp.tz_convert("UTC").ceil("us")
    return stamp.to_pydatetime().isoformat(timespec="microseconds")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601")


class NewsStore:
    def __init__(self, path: Path, tagger: Tagger = DEFAULT_TAGGER) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tagger = tagger
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        # A fresh connection per operation, as in serve/store.py: the scheduler
        # thread, the API and a `cryptopred-news watch` window can all be open
        # on this file at once.
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writing ---------------------------------------------------------------

    def add(
        self,
        entries: Iterable[Entry],
        source: str,
        backlog: bool = False,
        received_at: pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Store the entries that are new. Returns the rows actually written.

        Idempotent: an entry whose uid is already stored, or whose title was
        stored in the last 24 hours under any source, is skipped. The check and
        the insert share one IMMEDIATE transaction, so two pollers racing on the
        same headline cannot both decide it is new.
        """
        now = received_at if received_at is not None else _now()
        stamp = utc_stamp(now)
        window_start = utc_stamp(pd.Timestamp(now) - DUPLICATE_TITLE_WINDOW)
        written = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for entry in entries:
                key = title_key(entry.title)
                if not key:
                    continue
                duplicate = conn.execute(
                    "SELECT 1 FROM headlines WHERE title_key = ? AND received_at >= ? LIMIT 1",
                    (key, window_start),
                ).fetchone()
                if duplicate:
                    continue
                published = (
                    utc_stamp(entry.published_at) if entry.published_at is not None else None
                )
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO headlines
                        (uid, source, title, title_key, link, published_at,
                         received_at, is_backlog)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.uid, source, entry.title, key, entry.link,
                        published, stamp, int(backlog),
                    ),
                )
                if cursor.rowcount > 0:
                    written.append(
                        {
                            "id": cursor.lastrowid,
                            "uid": entry.uid,
                            "source": source,
                            "title": entry.title,
                            "link": entry.link,
                            "published_at": published,
                            "received_at": stamp,
                            "is_backlog": int(backlog),
                        }
                    )
        return written

    def record_poll(
        self,
        feeds_ok: int,
        feeds_failed: int,
        inserted: int,
        polled_at: pd.Timestamp | None = None,
    ) -> None:
        """Record a completed pass. Called at the END of the pass, so the stamp
        is never earlier than any headline the pass wrote."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO polls (polled_at, feeds_ok, feeds_failed, inserted) "
                "VALUES (?, ?, ?, ?)",
                (utc_stamp(polled_at or _now()), feeds_ok, feeds_failed, inserted),
            )

    # -- reading ---------------------------------------------------------------

    def annotate(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add tags computed now, from the title, under today's keyword lists."""
        tags = self.tagger.tag(row["title"])
        return {
            **row,
            "symbols": list(tags.symbols),
            "high_impact": tags.high_impact,
            "impact_terms": list(tags.impact_terms),
        }

    def recent(
        self, limit: int = 20, symbol: str | None = None, page: int = 500
    ) -> list[dict[str, Any]]:
        """Newest headlines first, tagged, with any alert decision attached.

        Filtering by symbol happens after tagging, since tags are not stored, so
        this reads pages of rows until it has `limit` matches or runs out.
        """
        out: list[dict[str, Any]] = []
        offset = 0
        with self._connect() as conn:
            while len(out) < limit:
                rows = conn.execute(
                    """
                    SELECT h.id, h.uid, h.source, h.title, h.link, h.published_at,
                           h.received_at, h.is_backlog, a.outcome AS alert
                    FROM headlines h LEFT JOIN alerts a ON a.uid = h.uid
                    ORDER BY h.received_at DESC, h.id DESC LIMIT ? OFFSET ?
                    """,
                    (page, offset),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    item = self.annotate(dict(row))
                    if symbol is None or symbol in item["symbols"]:
                        out.append(item)
                offset += page
        return out[:limit]

    def headlines_frame(self, until: pd.Timestamp | None = None) -> pd.DataFrame:
        """Every headline received at or before `until`, oldest first.

        received_at is parsed to UTC timestamps; published_at is left out on
        purpose - nothing that reads this frame should be able to reach it.
        """
        query = "SELECT received_at, source, title, is_backlog FROM headlines"
        params: tuple = ()
        if until is not None:
            query += " WHERE received_at <= ?"
            params = (utc_stamp(until),)
        with self._connect() as conn:
            frame = pd.read_sql_query(query + " ORDER BY received_at, id", conn, params=params)
        frame["received_at"] = _to_utc(frame["received_at"])
        return frame

    def listening_times(self, until: pd.Timestamp | None = None) -> pd.Series:
        """When a pass completed with at least one feed answering, oldest first."""
        query = "SELECT polled_at FROM polls WHERE feeds_ok > 0"
        params: tuple = ()
        if until is not None:
            query += " AND polled_at <= ?"
            params = (utc_stamp(until),)
        with self._connect() as conn:
            frame = pd.read_sql_query(query + " ORDER BY polled_at", conn, params=params)
        return _to_utc(frame["polled_at"])

    def last_poll(self, now: pd.Timestamp | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT polled_at, feeds_ok, feeds_failed, inserted FROM polls "
                "ORDER BY polled_at DESC, id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        at = pd.Timestamp(out["polled_at"])
        out["minutes_ago"] = round(((now or _now()) - at).total_seconds() / 60, 1)
        return out

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0])

    # -- alerts ----------------------------------------------------------------

    def alert_candidates(self, since: pd.Timestamp) -> list[dict[str, Any]]:
        """Fresh rows with no alert decision yet, oldest first. Backlog is never
        a candidate: every row in it is new to us and none is new to the world."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT h.id, h.uid, h.source, h.title, h.link, h.published_at,
                       h.received_at, h.is_backlog
                FROM headlines h LEFT JOIN alerts a ON a.uid = h.uid
                WHERE a.uid IS NULL AND h.is_backlog = 0 AND h.received_at >= ?
                ORDER BY h.received_at, h.id
                """,
                (utc_stamp(since),),
            ).fetchall()
        return [self.annotate(dict(row)) for row in rows]

    def claim_alert(self, uid: str, outcome: str, at: pd.Timestamp | None = None) -> bool:
        """Record the decision for a headline. False if one was already recorded,
        by this process or any other - which is the never-twice guarantee."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO alerts (uid, decided_at, outcome) VALUES (?, ?, ?)",
                (uid, utc_stamp(at or _now()), outcome),
            )
            return cursor.rowcount > 0

    def set_alert_outcome(self, uid: str, outcome: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE alerts SET outcome = ? WHERE uid = ?", (outcome, uid))

    def alerts_sent_since(self, since: pd.Timestamp) -> int:
        with self._connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE outcome = 'sent' AND decided_at >= ?",
                    (utc_stamp(since),),
                ).fetchone()[0]
            )
