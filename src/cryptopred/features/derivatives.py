"""Perpetual-futures funding features.

Funding is published every 8 hours. A value stamped at time T only becomes known
at T, so it is joined onto bars with `merge_asof(direction="backward")` against
each bar's close_time — never forward-filled from the future.
"""

from __future__ import annotations

import pandas as pd

FUNDING_COLUMNS = [
    "funding_rate",
    "funding_ma_3",
    "funding_ma_21",
    "funding_deviation",
    "hours_since_funding",
]


def funding_features(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Attach funding features to a kline frame, indexed like `bars`."""
    out = pd.DataFrame(index=bars.index, columns=FUNDING_COLUMNS, dtype="float64")
    if funding.empty:
        return out

    enriched = funding.copy()
    enriched["funding_ma_3"] = enriched["funding_rate"].rolling(3, min_periods=3).mean()
    enriched["funding_ma_21"] = enriched["funding_rate"].rolling(21, min_periods=21).mean()
    enriched["funding_deviation"] = enriched["funding_rate"] - enriched["funding_ma_21"]
    enriched = enriched.reset_index()
    enriched = enriched.rename(columns={enriched.columns[0]: "event_time"})

    decision_time = bars["close_time"] if "close_time" in bars.columns else bars.index
    left = pd.DataFrame({"decision_time": pd.Series(decision_time).to_numpy()}, index=bars.index)
    left.index.name = "open_time"
    left = left.reset_index().sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        enriched.sort_values("event_time"),
        left_on="decision_time",
        right_on="event_time",
        direction="backward",
    ).set_index("open_time")

    out["funding_rate"] = merged["funding_rate"]
    out["funding_ma_3"] = merged["funding_ma_3"]
    out["funding_ma_21"] = merged["funding_ma_21"]
    out["funding_deviation"] = merged["funding_deviation"]
    out["hours_since_funding"] = (
        merged["decision_time"] - merged["event_time"]
    ).dt.total_seconds() / 3600.0

    return out.reindex(bars.index)
