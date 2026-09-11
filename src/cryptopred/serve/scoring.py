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


def _predicted_label(row) -> int:
    """What the model said, which is not the same as what the strategy did.

    `signal` is 0 both when the model predicts FLAT and when it leans a direction
    too weakly for the strategy to act. Scoring the second case as a FLAT
    prediction marks the model wrong for being right but cautious — it predicted
    UP, the market went up, and the row reads "sai" because no trade was placed.

    The model's prediction is the argmax of the probabilities it recorded.
    Whether that prediction was traded is a separate question, answered by
    `signal` and reported separately.
    """
    probs = (float(row["prob_down"]), float(row["prob_flat"]), float(row["prob_up"]))
    return int(max(range(3), key=lambda i: probs[i])) - 1


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

        predicted_label = _predicted_label(row)
        is_correct = predicted_label == actual_label

        prediction_store.score_prediction(
            int(row["id"]), actual_return, actual_label, is_correct
        )
        scored += 1

    return scored
