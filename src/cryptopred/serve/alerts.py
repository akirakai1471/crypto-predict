"""Tell the user the model fired, and tell them what its record is.

An alert that says "LONG signal" and nothing else reads as advice. This one
carries the hit rate, the sample size, the paper equity and the fact that the
edge is unproven, every single time — because the moment those are somewhere
else, nobody reads them.

Two things it deliberately does not do. It does not say buy or sell: the project
reports what the model did and the person decides, and its tradeable edge is
unproven at 6 of 20 symbols. And it does not alert on every signal — a fixed
24-hour horizon means a signal every bar while conditions hold, so eight
messages in eight hours would describe one bet as though it were eight.
"""

from __future__ import annotations

import pandas as pd

# Below this many scored signals a hit rate is noise. Matches serve/status.py.
MIN_SIGNALS_FOR_A_CLAIM = 100


def _signals(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "signal" not in history.columns:
        return history.iloc[0:0]
    out = history[history["signal"] != 0].copy()
    out["t"] = pd.to_datetime(out["bar_close_time"], utc=True)
    return out.sort_values("t")


def should_alert(
    signal: int,
    bar_close_time: pd.Timestamp,
    history: pd.DataFrame,
    horizon_hours: int,
) -> bool:
    """Is this signal the start of a new episode rather than a repeat?

    A same-direction signal inside the horizon is the same view held one bar
    longer: the position it opens overlaps the one already open. A flip is new
    information whenever it arrives.
    """
    if signal == 0:
        return False

    prior = _signals(history)
    if prior.empty:
        # An empty frame may carry no columns at all, so this guard has to come
        # before any column is indexed.
        return True

    same = prior[prior["signal"] == signal]
    if same.empty:
        return True

    last = same["t"].iloc[-1]
    now = pd.Timestamp(bar_close_time).tz_convert("UTC")
    return (now - last) >= pd.Timedelta(hours=horizon_hours)


def track_record_line(
    history: pd.DataFrame, paper_equity: float, starting_capital: float
) -> str:
    """What this model has actually done, in one line."""
    scored = _signals(history)
    scored = scored[scored["is_correct"].notna()] if not scored.empty else scored
    pnl = paper_equity - starting_capital

    if scored.empty:
        return (
            f"Hồ sơ: chưa có tín hiệu nào được chấm điểm. "
            f"Vốn ảo {paper_equity:,.0f} ({pnl:+,.0f})."
        )

    n = len(scored)
    correct = int(scored["is_correct"].sum())
    line = (
        f"Hồ sơ: đúng {correct}/{n} ({correct / n:.0%}). "
        f"Vốn ảo {paper_equity:,.0f} ({pnl:+,.0f})."
    )
    if n < MIN_SIGNALS_FOR_A_CLAIM:
        line += (
            f" CHƯA ĐỦ ĐỂ KẾT LUẬN — {n} tín hiệu; cần khoảng "
            f"{MIN_SIGNALS_FOR_A_CLAIM}."
        )
    return line


def format_alert(
    symbol: str,
    signal: int,
    bar_close_time: pd.Timestamp,
    close_price: float,
    margin: float,
    cutoff: float,
    history: pd.DataFrame,
    paper_equity: float,
    starting_capital: float,
) -> str:
    """The message. States what happened, then what it is worth."""
    if signal == 0:
        raise ValueError("signal 0 is not an alert; nothing fired")

    side = "LONG" if signal > 0 else "SHORT"
    when = pd.Timestamp(bar_close_time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            f"{symbol} — model bắn {side}",
            f"nến đóng {when}, giá {close_price:,.2f}",
            f"margin {margin:.4f} so với cutoff {cutoff:.4f}",
            "",
            track_record_line(history, paper_equity, starting_capital),
            "",
            "Đây là báo model đã bắn gì, KHÔNG phải khuyến nghị. Khả năng sinh lời "
            "của hệ này CHƯA CHỨNG MINH ĐƯỢC: 6/20 coin qua cổng kiểm, mức đã ghi "
            "trước là không kết luận được. Xem docs/findings.md.",
        ]
    )
