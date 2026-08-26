"""Score past predictions once their outcome is known.

This is the only honest measure of the system: predictions written before the
outcome existed, graded afterwards against the same label definition used in
training. Backtests can be re-run until they look good. This log cannot.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.features.volatility import atr
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.store import PredictionStore

SIGNAL_TO_LABEL = {1: 1, -1: -1, 0: 0}


def score_pending(
    prediction_store: PredictionStore,
    parquet: ParquetStore,
    symbol: str,
    interval: str,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 0.5,
    now: pd.Timestamp | None = None,
) -> int:
    """Fill in outcomes for every prediction old enough to have one.

    Returns the number of predictions scored.
    """
    pending = prediction_store.unscored(symbol, interval)
    if pending.empty:
        return 0

    bars = parquet.read("klines", symbol, interval)
    if bars.empty:
        return 0

    now = now or pd.Timestamp.now(tz="UTC")
    band = (band_k * atr(bars, atr_period) / bars["close"]).rename("band")
    closes = bars["close"]

    scored = 0
    for _, row in pending.iterrows():
        bar_close = pd.Timestamp(row["bar_close_time"])
        # The prediction was made at this bar's close; find that bar by matching
        # close_time rather than assuming positions line up.
        matches = bars.index[bars["close_time"] == bar_close]
        if len(matches) == 0:
            continue

        position = bars.index.get_loc(matches[0])
        target = position + horizon
        if target >= len(bars):
            continue  # the future this prediction referred to has not happened yet

        entry_close = float(closes.iloc[position])
        exit_close = float(closes.iloc[target])
        actual_return = exit_close / entry_close - 1.0

        band_value = float(band.iloc[position])
        if not pd.notna(band_value):
            continue

        if actual_return > band_value:
            actual_label = 1
        elif actual_return < -band_value:
            actual_label = -1
        else:
            actual_label = 0

        predicted_label = SIGNAL_TO_LABEL[int(row["signal"])]
        # A no-signal bar is not a directional call; it counts as correct only
        # when the market really did go nowhere.
        is_correct = predicted_label == actual_label

        prediction_store.score_prediction(
            int(row["id"]), actual_return, actual_label, is_correct
        )
        scored += 1

    return scored
