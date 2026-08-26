"""Momentum indicators: how fast and in which direction price has been moving."""

from __future__ import annotations

import pandas as pd

from cryptopred.features.base import ewma, safe_divide, wilder_smooth


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 0 = only losses in the window, 100 = only gains."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = safe_divide(avg_gain, avg_loss, fill=0.0)
    out = 100.0 - 100.0 / (1.0 + rs)
    # when there are no losses at all, rs is 0 by safe_divide's fill; fix to 100
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram), normalised by price so the
    values are comparable across symbols with very different price levels."""
    line = (ewma(close, fast) - ewma(close, slow)) / close
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    for period in (7, 14, 28):
        out[f"rsi_{period}"] = rsi(close, period)

    macd_line, macd_signal, macd_hist = macd(close)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    for period in (1, 3, 6, 12, 24, 72):
        out[f"roc_{period}"] = close.pct_change(period)

    for span in (20, 50, 200):
        ema = ewma(close, span)
        out[f"ema_dist_{span}"] = safe_divide(close - ema, ema)

    out["ema_20_50_spread"] = safe_divide(ewma(close, 20) - ewma(close, 50), close)

    return out
