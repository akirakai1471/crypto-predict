"""The briefing as text, readable with no API key and no model.

Measured figures and conventional ones are in separate sections with a heading
between them. Interleaving would let a reader's eye carry the authority of the
first into the second, which is the whole failure being designed against.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.indicators import current_indicators
from cryptopred.briefing.levels import daily_pivots
from cryptopred.briefing.snapshot import market_snapshot
from cryptopred.briefing.touch import touch_probability

# The drops a buyer actually asks about, and the horizons they wait.
DEFAULT_TARGETS = (-0.03, -0.05, -0.10)
DEFAULT_HORIZONS = (24, 72, 168)


def format_brief(
    symbol: str,
    interval: str,
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> str:
    if bars.empty:
        return f"{symbol} {interval}: Không có dữ liệu trong kho."

    snap = market_snapshot(bars, funding, now=now)
    lines = [
        "=" * 72,
        f"{symbol} {interval} — bảng số liệu",
        "=" * 72,
    ]
    if snap["staleness_note"]:
        lines += ["", f"!! {snap['staleness_note']}"]

    lines += [
        "",
        "ĐO ĐƯỢC",
        f"  giá đóng gần nhất: {snap['last_close']['value']:,.2f}",
        f"  đổi 24h: {_pct(snap['change_24h'])}   "
        f"7d: {_pct(snap['change_7d'])}   30d: {_pct(snap['change_30d'])}",
        f"  dữ liệu cũ: {snap['data_age_hours']} giờ",
        "",
        "  xác suất chạm mức (đo từ lịch sử, không phải dự báo):",
    ]

    for target in DEFAULT_TARGETS:
        for horizon in DEFAULT_HORIZONS:
            result = touch_probability(bars, target_pct=target, horizon=horizon)
            lines.append(
                f"    {target:+.0%} trong {horizon:>3}h: "
                f"{_prob(result['conditional'])}  |  "
                f"mọi chế độ: {_prob(result['unconditional'])}"
            )

    sample = touch_probability(bars, target_pct=DEFAULT_TARGETS[0], horizon=72)
    lines.append(f"    chế độ hiện tại: {sample['cell_label']}")
    wait = sample["wait_hours"]
    if wait:
        # Name the sample the wait times came from. When the regime cell is too
        # thin they come from every bar instead, and printing them directly
        # under the cell label would attribute them to a regime they were not
        # measured in — the misreading provenance.py exists to prevent.
        scope = (
            "cùng chế độ"
            if sample["wait_source"] == "cell"
            else "MỌI chế độ, không riêng chế độ trên"
        )
        lines.append(
            f"    thời gian chờ khi có chạm ({scope}): trung vị {wait['median']:.0f}h, "
            f"p90 {wait['p90']:.0f}h (n={wait['n']:,})"
        )

    lines += ["", "QUY ƯỚC — CHƯA KIỂM CHỨNG", ""]
    for name, payload in current_indicators(bars).items():
        lines.append(f"  {name:<18} {payload['value']:>10.4f}   {payload['reading']}")

    pivots = daily_pivots(bars)
    if pivots:
        lines.append("")
        lines.append(
            "  pivot ngày: "
            + "  ".join(f"{k}={v['value']:,.0f}" for k, v in pivots.items())
        )

    lines += [
        "",
        "  Không có dòng nào trong mục này được đo là có giá trị dự báo trên dữ liệu",
        "  này. Chúng có mặt vì bạn sẽ hỏi tới, không phải vì chúng đã được chứng minh.",
        "=" * 72,
    ]
    return "\n".join(lines)


def _pct(payload: dict[str, Any]) -> str:
    if payload.get("source") != "measured":
        return "n/a"
    return f"{payload['value']:+.2%}"


def _prob(payload: Any) -> str:
    d = payload.to_dict() if hasattr(payload, "to_dict") else payload
    if d.get("source") != "measured":
        return "không đủ mẫu"
    lo, hi = d["interval"]
    return f"{d['value']:.1%} n={d['n']:,} khoảng[{lo:.1%},{hi:.1%}]"
