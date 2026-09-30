"""Is the funding the model is reading still current?

The model trains on five funding features and the live predictor reads them
from the local store. For as long as this project had a live scheduler, the
cycle refreshed bars and never funding: on 2026-09-30 the store on the machine
running it was 176 hours old, against a training maximum of 8 for
`hours_since_funding`. Nothing said so. A frozen input looks exactly like a
quiet one.

Binance settles funding every 8 hours (4 on some symbols), and the cycle now
fetches it every hour, so the newest rate should never be more than about 8
hours old. Past 9, something has stopped refreshing it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.ingest.storage import ParquetStore

FUNDING_STALE_AFTER_HOURS = 9.0


def funding_freshness(
    parquet: ParquetStore, symbol: str, now: pd.Timestamp | None = None
) -> dict[str, Any]:
    """{"state": "fresh" | "stale" | "missing", "hours": float | None, "detail": str}"""
    last = parquet.last_open_time("funding", symbol, "8h")
    if last is None:
        return {
            "state": "missing",
            "hours": None,
            "detail": "no funding stored - every funding feature is empty",
        }
    now = now or pd.Timestamp.now(tz="UTC")
    hours = (now - pd.Timestamp(last)).total_seconds() / 3600
    state = "stale" if hours > FUNDING_STALE_AFTER_HOURS else "fresh"
    return {
        "state": state,
        "hours": hours,
        "detail": f"last rate {hours:.1f} hours old (limit {FUNDING_STALE_AFTER_HOURS:.0f})",
    }
