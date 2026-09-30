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

import contextlib
import re
import subprocess
from pathlib import Path

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
    horizon_hours: float,
) -> bool:
    """Is this signal the start of a new episode rather than a repeat?

    A same-direction signal inside the horizon is the same view held one bar
    longer: the position it opens overlaps the one already open. A flip is new
    information whenever it arrives.

    Only signals strictly before this bar count as history. The scheduler
    records a prediction before it considers alerting on it, so the log it
    passes in already holds this very row - and a signal compared against
    itself is always "inside the horizon", which silenced every alert.
    """
    if signal == 0:
        return False

    now = pd.Timestamp(bar_close_time).tz_convert("UTC")
    prior = _signals(history)
    if not prior.empty:
        prior = prior[prior["t"] < now]
    if prior.empty:
        # An empty frame may carry no columns at all, so this guard has to come
        # before any column is indexed.
        return True

    same = prior[prior["signal"] == signal]
    if same.empty:
        return True

    last = same["t"].iloc[-1]
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


# Windows truncates balloon text well before this, but the cap keeps the popup
# readable rather than a wall. The log always holds the whole message.
BALLOON_TEXT_LIMIT = 200

# PowerShell + NotifyIcon rather than a WinRT toast: the WinRT type accelerator
# fails to load in a plain `powershell -NoProfile` session on this machine, and
# NotifyIcon is .NET Framework, present on every Windows install, and needs no
# registered AppID.
_POPUP = """
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$i = New-Object System.Windows.Forms.NotifyIcon
$i.Icon = [System.Drawing.SystemIcons]::Information
$i.BalloonTipTitle = {title}
$i.BalloonTipText = {text}
$i.Visible = $true
$i.ShowBalloonTip(10000)
Start-Sleep -Seconds 11
$i.Dispose()
"""


def balloon_parts(message: str) -> tuple[str, str]:
    """Split a full alert into a balloon title and a short body.

    The first line is the headline; the record line is what a glance needs. The
    rest - the caveat, the margin - stays in the log, which is why the log is
    written first and unconditionally.
    """
    lines = [ln for ln in message.splitlines() if ln.strip()]
    title = lines[0] if lines else "cryptopred"
    record = next((ln for ln in lines[1:] if ln.startswith("Hồ sơ")), "")
    body = record or (lines[1] if len(lines) > 1 else "")
    if len(body) > BALLOON_TEXT_LIMIT:
        body = body[: BALLOON_TEXT_LIMIT - 1].rstrip() + "…"
    return title, body


# Each entry in signals.log opens with "===== <stamp> =====" on a line of its own.
_ENTRY_MARK = "====="
_ENTRY_HEADER = re.compile(rf"^{_ENTRY_MARK} (.+?) {_ENTRY_MARK}$", re.MULTILINE)


def read_log(log_path: Path, limit: int = 20) -> list[dict[str, str]]:
    """The most recent alerts in signals.log, newest first.

    The balloon is gone after eleven seconds, so anything that wants to show
    what the model fired reads it back from here - the same text, caveat and
    all, rather than a summary that could drift away from what was sent.
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return []

    headers = list(_ENTRY_HEADER.finditer(text))
    entries = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        entries.append(
            {"written_at": header.group(1), "message": text[header.end() : end].strip()}
        )
    return entries[::-1][: max(limit, 0)]


def _ps_quote(value: str) -> str:
    """A PowerShell single-quoted string. Doubling the quote is the escape."""
    return "'" + value.replace("'", "''") + "'"


def notify(message: str, log_path: Path, popup: bool = True) -> None:
    """Append the alert to the log, then try to show it.

    The log comes first and without a try/except, because it is the durable
    record. The popup is a courtesy and any failure of it is swallowed: a
    machine with no window station, a locked session, a missing PowerShell -
    none of those are reasons to lose the alert or to break the scheduler cycle
    that produced it.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_ENTRY_MARK} {stamp} {_ENTRY_MARK}\n{message}\n")

    if not popup:
        return

    title, body = balloon_parts(message)
    script = _POPUP.format(title=_ps_quote(title), text=_ps_quote(body))
    # A machine with no window station, a locked session, a missing PowerShell -
    # none of those are reasons to lose the alert or break the cycle that made
    # it. The log above already holds the message.
    with contextlib.suppress(Exception):
        # Detached and not waited on: the balloon lives for eleven seconds and
        # the scheduler cycle has no reason to sit through it.
        subprocess.Popen(  # noqa: S603
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
