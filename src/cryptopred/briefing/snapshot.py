"""Where the market is, and how old that statement is.

Freshness sits at the top of the payload rather than in a footnote. An answer
computed on ten-day-old bars reads exactly like one computed on current bars,
and this project has already lost sixteen days to a failure whose only symptom
was silence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Measured, Unavailable

# An hourly feed more than three hours behind has missed at least two bars,
# which is past the point where a single slow fetch explains it.
STALE_AFTER_HOURS = 3.0


def _change(close: pd.Series, bars_back: int) -> Measured | Unavailable:
    if len(close) <= bars_back:
        return Unavailable(reason=f"chưa có đủ {bars_back:,} nến lịch sử")
    now, then = float(close.iloc[-1]), float(close.iloc[-1 - bars_back])
    return Measured(
        value=now / then - 1.0,
        n=bars_back,
        method=f"so với {bars_back:,} nến trước",
    )


def market_snapshot(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Price, recent changes, funding, and how stale the data is."""
    if bars.empty:
        return {
            "last_close": Unavailable(reason="chưa có nến nào trong kho").to_dict(),
            "data_age_hours": None,
            "is_stale": True,
            "staleness_note": "Không có dữ liệu.",
        }

    now = now or pd.Timestamp.now(tz="UTC")
    close = bars["close"]
    last_close_time = (
        bars["close_time"].max() if "close_time" in bars.columns else bars.index.max()
    )
    age_hours = float((now - last_close_time).total_seconds() / 3600.0)
    is_stale = age_hours > STALE_AFTER_HOURS

    if funding.empty:
        funding_now: Measured | Unavailable = Unavailable(
            reason="chưa có lịch sử funding trong kho"
        )
        funding_mean: Measured | Unavailable = Unavailable(
            reason="chưa có lịch sử funding trong kho"
        )
    else:
        recent = funding["funding_rate"].tail(90)  # 30 days at 8h intervals
        funding_now = Measured(
            value=float(funding["funding_rate"].iloc[-1]),
            n=1,
            method="lần funding gần nhất",
        )
        funding_mean = Measured(
            value=float(recent.mean()),
            n=int(recent.size),
            method="trung bình 30 ngày",
        )

    volume = bars["quote_volume"] if "quote_volume" in bars.columns else None
    volume_24h: Measured | Unavailable = (
        Measured(
            value=float(volume.tail(24).sum()), n=24, method="tổng 24 nến gần nhất"
        )
        if volume is not None
        else Unavailable(reason="kho không có cột quote_volume")
    )

    return {
        "last_close": Measured(
            value=float(close.iloc[-1]), n=1, method="nến đóng gần nhất"
        ).to_dict(),
        "change_24h": _change(close, 24).to_dict(),
        "change_7d": _change(close, 168).to_dict(),
        "change_30d": _change(close, 720).to_dict(),
        "quote_volume_24h": volume_24h.to_dict(),
        "funding_now": funding_now.to_dict(),
        "funding_mean_30d": funding_mean.to_dict(),
        "last_bar_close_time": str(last_close_time),
        "data_age_hours": round(age_hours, 1),
        "is_stale": is_stale,
        "staleness_note": (
            f"Dữ liệu cũ {age_hours:.0f} giờ. Mọi con số dưới đây mô tả thời điểm đó, "
            "không phải bây giờ."
            if is_stale
            else ""
        ),
    }
