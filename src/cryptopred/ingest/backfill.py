"""Pagination, unclosed-bar filtering, and gap detection for kline downloads."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from cryptopred.ingest.binance import KLINE_LIMIT, parse_klines
from cryptopred.timeframes import expected_open_times, interval_to_timedelta, to_ms

logger = logging.getLogger(__name__)


def drop_unclosed(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove bars that have not finished forming.

    A bar is usable only once its close_time has passed. Keeping a forming bar
    would leak partial future information into features.
    """
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] <= now]


def find_gaps(df: pd.DataFrame, interval: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of missing bar open times, as (first_missing, last_missing)."""
    if df.empty or len(df) < 2:
        return []
    delta = interval_to_timedelta(interval)
    expected = expected_open_times(df.index.min(), df.index.max() + delta, interval)
    missing = expected.difference(df.index)
    if missing.empty:
        return []

    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev != delta:
            gaps.append((run_start, prev))
            run_start = ts
        prev = ts
    gaps.append((run_start, prev))
    return gaps


def _finalise(
    frames: list[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    now: pd.Timestamp | None,
) -> pd.DataFrame:
    """Deduplicate, clip to the requested window, and drop unclosed bars."""
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= start) & (df.index < end)]
    return drop_unclosed(df, now=now)


def backfill_klines(
    client,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    now: pd.Timestamp | None = None,
    limit: int = KLINE_LIMIT,
    sink: Callable[[pd.DataFrame], None] | None = None,
    flush_every: int = 40,
) -> pd.DataFrame | int:
    """Download every closed bar with open_time in [start, end).

    Pagination advances from the last returned bar. If a page comes back empty
    the download stops — Binance has no data before a contract's listing date.

    With no `sink`, every bar is accumulated and returned as one frame. That is
    fine for hourly data but not for 1m history, which runs to millions of bars:
    pass a `sink` to have batches handed over every `flush_every` pages, so
    memory stays flat and an interrupted run leaves usable partial data behind.
    Returns the frame when no sink is given, otherwise the bar count written.
    """
    delta = interval_to_timedelta(interval)
    frames: list[pd.DataFrame] = []
    cursor = start
    written = 0
    pages = 0

    while cursor < end:
        rows = client.fetch_klines(
            symbol, interval, start_ms=to_ms(cursor), end_ms=to_ms(end), limit=limit
        )
        if not rows:
            break
        page = parse_klines(rows)
        frames.append(page)
        pages += 1

        next_cursor = page.index.max() + delta
        if next_cursor <= cursor:  # defensive: never loop forever
            break
        cursor = next_cursor

        if sink is not None and pages % flush_every == 0:
            batch = _finalise(frames, start, end, now)
            frames = []
            if not batch.empty:
                sink(batch)
                written += len(batch)
                logger.info("%s %s: flushed %d bars up to %s",
                            symbol, interval, len(batch), batch.index.max())

    if frames:
        batch = _finalise(frames, start, end, now)
        if sink is None:
            return batch
        if not batch.empty:
            sink(batch)
            written += len(batch)
    elif sink is None:
        return parse_klines([])

    return written
