"""`cryptopred-news` - collect headlines and read them back. The model never sees them."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from cryptopred.config import load_config
from cryptopred.news.fetch import FeedFetcher
from cryptopred.news.poller import PollSummary, poll_once
from cryptopred.news.store import NewsStore
from cryptopred.news.tags import TAG_CAVEAT
from cryptopred.report_io import safe_echo

app = typer.Typer(
    help="Thu thập tiêu đề tin crypto từ RSS và đọc lại. Chỉ để xem — model không dùng tin."
)
logger = logging.getLogger(__name__)

HEADER = "Giờ hiển thị là lúc hệ thống NHẬN tin (UTC), không phải giờ toà soạn ghi."


def _store(config: Path | None):
    cfg = load_config(config)
    return cfg, NewsStore(cfg.data.root / "news.db")


def _quiet_logs() -> None:
    # Failures are already in the printed summary; the logger's English
    # warnings would say the same thing twice.
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")


def format_summary(summary: PollSummary) -> str:
    """One line per feed, then the totals."""
    new = summary.new_by_source
    lines = []
    for r in summary.results:
        if r.status == "ok":
            state = f"OK · {len(r.entries)} mục · {new.get(r.name, 0)} mới"
        elif r.status == "not_modified":
            state = "không đổi (304)"
        else:
            state = f"LỖI · {r.error}"
        lines.append(f"  {r.name:<18} {state}")
    lines.append(
        f"Tổng: {len(summary.inserted)} tin mới · {summary.feeds_ok}/{len(summary.results)} "
        f"nguồn trả lời · {summary.feeds_failed} lỗi"
    )
    if summary.backlog:
        lines.append(
            f"  ({summary.backlog} tin là tồn đọng của RSS từ lượt đầu — có thể đã cũ vài "
            "ngày; không báo động, không tính là tin mới.)"
        )
    return "\n".join(lines)


def format_headline(row: dict[str, Any]) -> str:
    when = pd.Timestamp(row["received_at"]).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    head = [when, row["source"]]
    if row.get("symbols"):
        head.append(", ".join(row["symbols"]))
    if row.get("high_impact"):
        head.append(f"TÁC ĐỘNG CAO? ({', '.join(row['impact_terms'])})")
    if row.get("is_backlog"):
        head.append("tồn đọng")
    lines = [" · ".join(head), f"  {row['title']}"]
    if row.get("link"):
        lines.append(f"  {row['link']}")
    return "\n".join(lines)


def run_watch(
    fetcher: FeedFetcher,
    store: NewsStore,
    every: float,
    echo: Callable[[str], None] = safe_echo,
    sleep: Callable[[float], None] = time.sleep,
    max_passes: int | None = None,
) -> int:
    """Poll until interrupted. Each pass stands alone: one that raises is
    reported and the loop goes on, because a watch that dies on the first
    network hiccup is a watch nobody can leave running. Returns passes run."""
    passes = 0
    while max_passes is None or passes < max_passes:
        try:
            summary = poll_once(fetcher, store)
            fresh = [row for row in summary.inserted if not row["is_backlog"]]
            for row in fresh:
                echo(format_headline(store.annotate(row)))
            if summary.backlog:
                echo(
                    f"(nạp {summary.backlog} tin tồn đọng — xem bằng "
                    "`cryptopred-news recent`)"
                )
            if summary.feeds_failed:
                failed = ", ".join(r.name for r in summary.results if not r.ok)
                echo(f"(lượt này {summary.feeds_failed} nguồn lỗi: {failed})")
        except Exception as exc:  # noqa: BLE001 - one bad pass must not end the watch
            logger.exception("news watch pass failed")
            echo(f"Lượt quét lỗi: {type(exc).__name__}: {exc} — thử lại sau {every:g} giây.")
        passes += 1
        if max_passes is not None and passes >= max_passes:
            break
        sleep(every)
    return passes


@app.command()
def poll(
    config: Path = typer.Option(None, help="Đường dẫn file YAML cấu hình."),
) -> None:
    """Quét mọi nguồn một lượt và in số tin mới."""
    _quiet_logs()
    cfg, store = _store(config)
    with FeedFetcher(cfg.news.feeds) as fetcher:
        summary = poll_once(fetcher, store)
    safe_echo(format_summary(summary))


@app.command()
def watch(
    every: int = typer.Option(60, min=10, help="Số giây giữa hai lượt quét."),
    config: Path = typer.Option(None, help="Đường dẫn file YAML cấu hình."),
) -> None:
    """Quét liên tục tới khi bấm Ctrl-C; in tin mới ngay khi nhận.

    Không gửi thông báo - việc đó là của scheduler, để một tin không bị báo
    hai lần từ hai cửa sổ.
    """
    _quiet_logs()
    cfg, store = _store(config)
    safe_echo(
        f"Quét {len(cfg.news.feeds)} nguồn mỗi {every} giây. Ctrl-C để dừng.\n"
        f"{HEADER}\n{TAG_CAVEAT}\n"
    )
    with FeedFetcher(cfg.news.feeds) as fetcher:
        try:
            run_watch(fetcher, store, every=every)
        except KeyboardInterrupt:
            safe_echo("\nĐã dừng.")


@app.command()
def recent(
    limit: int = typer.Option(20, min=1, max=500, help="Số tin tối đa."),
    symbol: str = typer.Option(None, help="Chỉ tin gắn nhãn coin này, ví dụ BTCUSDT."),
    config: Path = typer.Option(None, help="Đường dẫn file YAML cấu hình."),
) -> None:
    """In các tiêu đề nhận gần nhất, mới nhất trước."""
    _, store = _store(config)
    rows = store.recent(limit=limit, symbol=symbol.upper() if symbol else None)
    if not rows:
        where = f" gắn nhãn {symbol.upper()}" if symbol else ""
        safe_echo(
            f"Chưa có tin nào{where}. Chạy `cryptopred-news poll`, hoặc để scheduler "
            "chạy (nó quét tin mỗi phút)."
        )
        return
    safe_echo(f"{HEADER}\n{TAG_CAVEAT}\n")
    safe_echo("\n\n".join(format_headline(row) for row in rows))


if __name__ == "__main__":
    app()
