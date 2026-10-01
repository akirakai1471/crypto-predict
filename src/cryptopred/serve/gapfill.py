"""Fill prediction gaps left by downtime — visibly, never silently.

If the machine is off for a day, the bars that closed meanwhile get no
prediction row and the paper strategy simply did not trade them. Leaving the
hole is defensible but produces an equity curve full of unexplained pauses.

Filling it is also defensible, because features at a bar use only data that
closed at or before that bar: the model produces exactly the probability it
would have produced live, with no future information. What is *not* defensible
is filling it quietly. A row written after its outcome is known cannot prove it
was not influenced by that outcome, and the entire reason this log outranks a
backtest is that its rows can prove it.

So gaps are filled and every filled row is flagged. `accuracy_summary` ignores
flagged rows, and the status report shows them on a separate line.
"""

from __future__ import annotations

import logging

import pandas as pd

from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.predictor import Predictor
from cryptopred.serve.store import PredictionStore
from cryptopred.timeframes import interval_to_timedelta

logger = logging.getLogger(__name__)

# Filling more than this many bars at once means the machine was off for weeks.
# At that point the honest move is to notice, not to quietly manufacture a month
# of history.
MAX_GAP_BARS = 24 * 14


def find_gap_bars(
    store: PredictionStore,
    parquet: ParquetStore,
    symbol: str,
    interval: str,
    horizon: int | None = None,
) -> list[pd.Timestamp]:
    """Closed bars with no prediction recorded, oldest first.

    Every closed bar except the newest, which the cycle predicts live straight
    after this. Bars whose horizon has not elapsed used to be left out, on the
    grounds that "the next normal cycle will predict it" - but the normal cycle
    only ever predicts the newest bar, so a six-hour outage left five holes for
    a day, and the first alert after the restart could not see the signals
    those hours would have shown.

    Only the last MAX_GAP_BARS bars are considered, every cycle. The cap used to
    trim the list instead, and the next cycle filled the older remainder anyway.

    `horizon` is accepted for compatibility and no longer used.
    """
    bars = parquet.read("klines", symbol, interval)
    if bars.empty:
        return []

    history = store.history(symbol, interval, limit=1_000_000)
    if history.empty:
        return []

    recorded = set(pd.to_datetime(history["bar_close_time"], utc=True))
    first_recorded = min(recorded)
    delta = interval_to_timedelta(interval)

    now = pd.Timestamp.now(tz="UTC")
    closed = bars[bars["close_time"] <= now]
    if len(closed) < 2:
        return []
    newest = closed["close_time"].iloc[-1]
    # Never reach back before the log started: those bars were not missed, they
    # were before the system existed. Nor past the cap.
    window_start = max(first_recorded, newest - MAX_GAP_BARS * delta)
    candidates = closed["close_time"].iloc[:-1]
    candidates = candidates[candidates >= window_start]
    return [ts for ts in candidates if ts not in recorded]


def _missed_before_window(
    store: PredictionStore, parquet: ParquetStore, symbol: str, interval: str
) -> int:
    """How many missed bars lie beyond the cap - reported, never filled."""
    bars = parquet.read("klines", symbol, interval)
    history = store.history(symbol, interval, limit=1_000_000)
    if bars.empty or history.empty:
        return 0
    recorded = set(pd.to_datetime(history["bar_close_time"], utc=True))
    delta = interval_to_timedelta(interval)
    window_start = bars["close_time"].iloc[-1] - MAX_GAP_BARS * delta
    old = bars["close_time"][
        (bars["close_time"] >= min(recorded)) & (bars["close_time"] < window_start)
    ]
    return int(sum(ts not in recorded for ts in old))


def fill_gaps(
    cfg,
    store: PredictionStore,
    parquet: ParquetStore,
    symbol: str,
    interval: str,
    horizon: int,
    predictor: Predictor | None = None,
) -> dict[str, int]:
    """Record a flagged prediction for every bar missed while offline."""
    missing = find_gap_bars(store, parquet, symbol, interval, horizon)
    if not missing:
        return {"gap_bars": 0, "filled": 0, "skipped": 0}

    if len(missing) >= MAX_GAP_BARS and (
        older := _missed_before_window(store, parquet, symbol, interval)
    ):
        # Said once: after this fill the window is complete and nothing more
        # is attempted, so the next cycle has nothing to repeat.
        logger.warning(
            "%s %s: %d more missing bars lie beyond the %d-bar limit and are left "
            "empty. A gap this large should be looked at, not papered over.",
            symbol, interval, older, MAX_GAP_BARS,
        )

    if predictor is None:
        try:
            predictor = Predictor.from_registry(cfg, symbol, interval)
        except FileNotFoundError:
            return {"gap_bars": len(missing), "filled": 0, "skipped": len(missing)}

    bars = parquet.read("klines", symbol, interval)
    close_to_open = dict(zip(bars["close_time"], bars.index, strict=True))

    filled = skipped = 0
    for close_time in missing:
        open_time = close_to_open.get(close_time)
        if open_time is None:
            skipped += 1
            continue

        prediction = predictor.predict_at(symbol, interval, upto=open_time)
        if prediction is None or prediction.bar_close_time != close_time:
            skipped += 1
            continue

        if store.record_prediction(
            symbol=symbol,
            interval=interval,
            bar_close_time=prediction.bar_close_time,
            proba=prediction.proba,
            signal=prediction.signal,
            close_price=prediction.close_price,
            model_version=prediction.model_version,
            was_backfilled=True,
        ):
            filled += 1
        else:
            skipped += 1

    if filled:
        logger.info(
            "%s %s: filled %d prediction(s) missed while offline, all flagged",
            symbol, interval, filled,
        )
    return {"gap_bars": len(missing), "filled": filled, "skipped": skipped}
