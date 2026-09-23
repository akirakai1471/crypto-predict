"""Conventional indicators, reported as conventions.

Every value here is computed by a formula the industry agrees on and whose
predictive value this project has not measured on this data. That is not a
reason to hide them — they are what a user will ask about — but it is a reason
that none of them can leave as a `Measured`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Convention
from cryptopred.features.base import pct_rank
from cryptopred.features.momentum import macd, rsi
from cryptopred.features.regime import adx
from cryptopred.features.volatility import atr, realized_vol

READINGS = {
    "rsi_7": "quy ước: >70 quá mua, <30 quá bán",
    "rsi_14": "quy ước: >70 quá mua, <30 quá bán",
    "macd_line": "quy ước: dương là đà tăng",
    "macd_signal": "quy ước: đường tín hiệu của MACD",
    "macd_histogram": "quy ước: histogram dương và mở rộng là đà tăng mạnh lên",
    "atr_pct": "biên độ thật trung bình, tính theo % giá",
    "atr_percentile": "ATR hiện tại đứng ở đâu so với 720 nến gần nhất, 0–1",
    "adx_14": "quy ước: >25 là có xu hướng rõ",
    "realized_vol_168": "độ lệch chuẩn lợi suất 168 nến, chưa quy năm",
}


def _now(series: pd.Series) -> float | None:
    """The value at the LAST bar, or None if the indicator has not warmed up.

    Not the last non-NaN value. A user asking what RSI is now is asking about
    this bar; answering with whatever value happened to exist most recently
    would answer a question about a different bar and never say so.
    """
    if series.empty:
        return None
    value = series.iloc[-1]
    return float(value) if pd.notna(value) else None


def current_indicators(bars: pd.DataFrame) -> dict[str, Any]:
    """Every indicator that has enough history, each as a Convention."""
    if bars.empty:
        return {}

    close = bars["close"]
    line, signal, hist = macd(close)
    atr_pct = atr(bars, 14) / close
    values = {
        "rsi_7": _now(rsi(close, 7)),
        "rsi_14": _now(rsi(close, 14)),
        "macd_line": _now(line),
        "macd_signal": _now(signal),
        "macd_histogram": _now(hist),
        "atr_pct": _now(atr_pct),
        "atr_percentile": _now(pct_rank(atr_pct, 720)),
        "adx_14": _now(adx(bars, 14)),
        "realized_vol_168": _now(realized_vol(close, 168)),
    }
    return {
        name: Convention(value=value, reading=READINGS[name]).to_dict()
        for name, value in values.items()
        if value is not None
    }
