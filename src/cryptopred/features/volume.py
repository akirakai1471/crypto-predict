"""Volume and order-flow proxies derived from kline aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import rolling_zscore, safe_divide


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative signed volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def volume_features(df: pd.DataFrame, zscore_window: int = 200) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out[f"volume_z_{zscore_window}"] = rolling_zscore(df["volume"], zscore_window)
    out[f"trades_z_{zscore_window}"] = rolling_zscore(
        df["trades"].astype("float64"), zscore_window
    )
    out["quote_volume_z_200"] = rolling_zscore(df["quote_volume"], 200)

    obv_series = obv(df)
    for window in (12, 24, 72):
        out[f"obv_slope_{window}"] = safe_divide(
            obv_series.diff(window), df["volume"].rolling(window, min_periods=window).sum()
        )

    # taker_buy_base is the portion of volume that hit the ask (aggressive buying).
    # Imbalance in [-1, 1]: +1 = all aggressive buying, -1 = all aggressive selling.
    taker_sell = df["volume"] - df["taker_buy_base"]
    out["taker_imbalance"] = safe_divide(df["taker_buy_base"] - taker_sell, df["volume"])
    out["taker_imbalance_ma_24"] = out["taker_imbalance"].rolling(24, min_periods=24).mean()

    out["avg_trade_size_z_200"] = rolling_zscore(
        safe_divide(df["volume"], df["trades"].astype("float64")), 200
    )

    return out
