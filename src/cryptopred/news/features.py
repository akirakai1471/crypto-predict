"""Per-bar headline counts: built to be measured later, not used now.

Not in features/pipeline.py, and a test keeps it that way. Free RSS keeps a
few days of items, so today there is no history to put these columns through
the walk-forward gates every price feature had to pass. A feature that has not
passed them has no business in the model; wiring it in anyway would be the
"try it and see" that rule 5 of the README calls a bug until proven otherwise.
Collect three months or more, then measure.

Point-in-time: a bar's value counts only headlines whose `received_at` is at or
before that bar's `close_time` - the same rule tests/test_leakage.py enforces
for price features. A publisher's date never places a headline: it can be
earlier than the moment anyone could have seen it. It is used only to drop
rows that were old on arrival (news/store.py) - a direction in which a wrong
date can remove a headline but never admit one early.

Zero headlines while nothing was listening is not zero news. A window is NaN,
not 0, when it reaches back before collection began, when the poller was not
running at the bar's close, when it overlaps a stretch with no successful poll,
or when it holds backlog rows (the burst of old items delivered by the first
fetch after a start or an outage). Otherwise every outage would read as a
quiet market and every restart as a news storm. The listening test itself uses
only polls completed by the bar's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.news.store import NewsStore
from cryptopred.news.tags import DEFAULT_TAGGER, Tagger

WINDOWS: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "24h": pd.Timedelta(hours=24),
}
# Longer than this without a successful pass is a hole in the record. At a
# 60-second poll that is fourteen missed passes; RSS itself lags by minutes.
MAX_POLL_GAP = pd.Timedelta(minutes=15)


def _ns(values) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds, for searchsorted."""
    return pd.DatetimeIndex(values).tz_convert("UTC").as_unit("ns").asi8


def _in_window(sorted_times: np.ndarray, close: np.ndarray, width: int) -> np.ndarray:
    """How many of sorted_times fall in (close - width, close], per bar."""
    upper = np.searchsorted(sorted_times, close, side="right")
    lower = np.searchsorted(sorted_times, close - width, side="right")
    return upper - lower


def news_features(
    bars: pd.DataFrame,
    store: NewsStore,
    symbol: str,
    windows: dict[str, pd.Timedelta] = WINDOWS,
    max_poll_gap: pd.Timedelta = MAX_POLL_GAP,
    tagger: Tagger = DEFAULT_TAGGER,
) -> pd.DataFrame:
    """Trailing counts of headlines tagged `symbol`, and of those also flagged
    high impact, for every bar.

    Columns: news_tagged_<w> and news_high_impact_<w> for each window. Tags and
    the high-impact flag are CONVENTION, UNVALIDATED (news/tags.py).
    """
    if "close_time" not in bars.columns:
        raise ValueError("bars need a close_time column: the cutoff is the close, not the open")

    close_times = pd.DatetimeIndex(bars["close_time"])
    close = _ns(close_times)
    out = pd.DataFrame(index=bars.index)
    if len(bars) == 0:
        for name in windows:
            out[f"news_tagged_{name}"] = pd.Series(dtype="float64")
            out[f"news_high_impact_{name}"] = pd.Series(dtype="float64")
        return out

    until = close_times.max()
    heads = store.headlines_frame(until=until)
    polls = np.sort(_ns(store.listening_times(until=until)))

    tags = [tagger.tag(title) for title in heads["title"]]
    tagged = np.array([symbol in t.symbols for t in tags], dtype=bool)
    impact = np.array([t.high_impact for t in tags], dtype=bool)
    backlog = heads["is_backlog"].to_numpy(dtype=bool) if len(heads) else np.zeros(0, bool)
    stale = heads["old_on_arrival"].to_numpy(dtype=bool) if len(heads) else np.zeros(0, bool)
    received = _ns(heads["received_at"]) if len(heads) else np.zeros(0, np.int64)

    # Old-on-arrival rows are not news flow, but nothing was missed either, so
    # they are dropped from the counts rather than blanking the window.
    counted = tagged & ~backlog & ~stale
    tagged_times = np.sort(received[counted])
    impact_times = np.sort(received[counted & impact])
    backlog_times = np.sort(received[backlog])

    # The end of every hole in the record, with collection start as the first.
    # A window containing one of these either reaches back before we listened
    # or includes the catch-up burst that follows an outage.
    gap = max_poll_gap.value
    if len(polls):
        hole_ends = np.concatenate([polls[:1], polls[1:][np.diff(polls) > gap]])
        previous = np.searchsorted(polls, close, side="right") - 1
        listening_at_close = (previous >= 0) & (
            close - polls[np.clip(previous, 0, None)] <= gap
        )
    else:
        hole_ends = np.zeros(0, np.int64)
        listening_at_close = np.zeros(len(close), dtype=bool)

    for name, width in windows.items():
        w = width.value
        unknown = (
            ~listening_at_close
            | (_in_window(hole_ends, close, w) > 0)
            | (_in_window(backlog_times, close, w) > 0)
        )
        n_tagged = _in_window(tagged_times, close, w).astype("float64")
        n_impact = _in_window(impact_times, close, w).astype("float64")
        n_tagged[unknown] = np.nan
        n_impact[unknown] = np.nan
        out[f"news_tagged_{name}"] = n_tagged
        out[f"news_high_impact_{name}"] = n_impact
    return out
