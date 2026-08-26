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
    horizon: int,
) -> list[pd.Timestamp]:
    """Bars that closed with no prediction recorded, newest-first order preserved.

    Only bars old enough to be scoreable are returned. A bar whose horizon has
    not elapsed will be predicted by the next normal cycle anyway, and doing it
    here would flag a row as backfilled for no reason.
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

    # Never reach back before the log started: those bars were not missed, they
    # were before the system existed.
    candidates = bars[bars["close_time"] >= first_recorded]
    # A bar is only worth filling once its outcome could be scored.
    now = pd.Timestamp.now(tz="UTC")
    scoreable = candidates[candidates["close_time"] + horizon * delta <= now]

    missing = [
        ts for ts in scoreable["close_time"] if ts not in recorded
    ]
    return missing


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

    if len(missing) > MAX_GAP_BARS:
        logger.warning(
            "%s %s: %d missing bars exceeds the %d-bar limit; filling only the most "
            "recent. A gap this large should be looked at, not papered over.",
            symbol, interval, len(missing), MAX_GAP_BARS,
        )
        missing = missing[-MAX_GAP_BARS:]

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
