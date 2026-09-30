"""What the rest of the market did, as of the same bar close.

A single-symbol model sees only its own candles. Most coins move with BTC, and
a BTC move over the last few hours is known at the moment a bar closes, so it
is information the model is missing rather than information from the future.

Alignment is by bar open time with no filling. The context bar that opens at t
closes at the same instant as this symbol's bar at t, so using it is exactly as
point-in-time as using this symbol's own close. A context bar that is missing
stays missing (NaN) rather than being carried forward: carrying it would state a
price at t that nobody observed at t.

These features exist for the experiment pre-registered in
docs/preregistration-improvements.md. They are not in the default pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = ("ctx_ret_1", "ctx_ret_4", "ctx_ret_24", "ctx_vol_24", "rel_ret_24")


def context_features(bars: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Five features from `context` (another symbol's klines), indexed like `bars`.

    - ctx_ret_{1,4,24}: the context symbol's log return over that many bars
    - ctx_vol_24: its realised volatility, the std of 24 one-bar log returns
    - rel_ret_24: this symbol's 24-bar log return minus the context's, i.e.
      whether this coin has been leading or lagging the market
    """
    own = np.log(bars["close"].astype(float))
    ctx = np.log(context["close"].astype(float).reindex(bars.index))

    out = pd.DataFrame(index=bars.index)
    for k in (1, 4, 24):
        out[f"ctx_ret_{k}"] = ctx - ctx.shift(k)
    out["ctx_vol_24"] = (ctx - ctx.shift(1)).rolling(24, min_periods=24).std()
    out["rel_ret_24"] = (own - own.shift(24)) - out["ctx_ret_24"]
    return out.replace([np.inf, -np.inf], np.nan)


def context_symbol(symbol: str) -> str:
    """BTC is the market for every coin but itself; for BTC, ETH is."""
    return "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
