"""Higher-timeframe context, joined point-in-time.

The resampled frame is indexed by the higher-timeframe bar's CLOSE time, and the
join is `direction="backward"` against each base bar's close time. That means a
base bar can only see higher-timeframe bars that had already closed when the
prediction would have been made.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.features.base import ewma, safe_divide
from cryptopred.features.momentum import rsi
from cryptopred.features.volatility import atr

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a higher timeframe, indexed by each higher bar's close time."""
    cols = {k: v for k, v in _AGG.items() if k in df.columns}
    out = df.resample(rule, label="right", closed="left").agg(cols)
    return out.dropna(subset=["close"])


def _core_features(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=htf.index)
    close = htf["close"]

    out[f"{prefix}_rsi_14"] = rsi(close, 14)
    out[f"{prefix}_close_ret_1"] = close.pct_change(1)
    out[f"{prefix}_close_ret_3"] = close.pct_change(3)
    out[f"{prefix}_atr_norm"] = safe_divide(atr(htf, 14), close)
    ema20 = ewma(close, 20)
    out[f"{prefix}_ema_dist_20"] = safe_divide(close - ema20, ema20)
    roll_high = htf["high"].rolling(20, min_periods=20).max()
    roll_low = htf["low"].rolling(20, min_periods=20).min()
    out[f"{prefix}_range_pos_20"] = safe_divide(
        close - roll_low, roll_high - roll_low, fill=0.5
    )
    return out


def mtf_features(bars: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    """Higher-timeframe features aligned onto the base bar index."""
    htf = resample_ohlcv(bars, rule)
    if htf.empty:
        return pd.DataFrame(index=bars.index)

    feats = _core_features(htf, prefix).reset_index()
    feats = feats.rename(columns={feats.columns[0]: "htf_close_time"})

    decision_time = bars["close_time"] if "close_time" in bars.columns else bars.index
    left = pd.DataFrame(
        {"decision_time": pd.Series(decision_time).to_numpy()}, index=bars.index
    )
    left.index.name = "open_time"
    left = left.reset_index().sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        feats.sort_values("htf_close_time"),
        left_on="decision_time",
        right_on="htf_close_time",
        direction="backward",
    ).set_index("open_time")

    keep = [c for c in merged.columns if c.startswith(f"{prefix}_")]
    return merged[keep].reindex(bars.index)
