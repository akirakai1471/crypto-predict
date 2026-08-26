"""Calendar features. Crypto trades 24/7 but liquidity is not uniform: Asian,
European and US hours have distinct volatility signatures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)

    hour = idx.hour.to_numpy(dtype="float64")
    dow = idx.dayofweek.to_numpy(dtype="float64")

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # UTC session windows, deliberately overlapping at the handovers
    out["session_asia"] = ((hour >= 0) & (hour < 9)).astype("float64")
    out["session_europe"] = ((hour >= 7) & (hour < 16)).astype("float64")
    out["session_us"] = ((hour >= 13) & (hour < 22)).astype("float64")
    # 22:00-00:00 UTC belongs to no major session; assign it to Asia's run-up
    out.loc[(hour >= 22), "session_asia"] = 1.0

    out["is_weekend"] = (dow >= 5).astype("float64")

    return out
