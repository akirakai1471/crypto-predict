"""Check a machine before trusting it to run unattended.

A server that cannot reach Binance, has no model, or cannot write its data
directory does not fail loudly. It starts, logs one warning an hour, and
records nothing - which is the failure the heartbeat exists for, discovered a
day late. These checks find it in the first minute.

The one that matters most on a new VPS: Binance answers HTTP 451 to regions it
does not serve, the United States among them. That depends on where the
machine is, not on anything in this project, and the only fix is a machine
somewhere else.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx

from cryptopred.config import Config
from cryptopred.ingest.binance import FUTURES_BASE
from cryptopred.models.registry import ModelRegistry
from cryptopred.news.fetch import USER_AGENT
from cryptopred.serve import telegram


@dataclass
class Check:
    name: str
    state: str  # "ok", "warn" or "fail"
    detail: str
    # A failed critical check means the scheduler would record nothing.
    critical: bool = False


def _data_dir(cfg: Config) -> Check:
    root = cfg.data.root
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".doctor-{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check("thư mục dữ liệu", "ok", f"ghi được vào {root}", critical=True)
    except OSError as exc:
        return Check(
            "thư mục dữ liệu", "fail", f"không ghi được vào {root}: {exc}", critical=True
        )


def _binance(cfg: Config, transport: httpx.BaseTransport | None) -> Check:
    name = "Binance futures API"
    symbol = cfg.data.symbols[0] if cfg.data.symbols else "BTCUSDT"
    try:
        with httpx.Client(base_url=FUTURES_BASE, transport=transport, timeout=10) as client:
            ping = client.get("/fapi/v1/ping")
            if ping.status_code == 451:
                return Check(
                    name, "fail",
                    "HTTP 451: Binance từ chối vùng/quốc gia của máy chủ này. Không có "
                    "cách sửa trong code - thuê máy chủ ở vùng khác.",
                    critical=True,
                )
            ping.raise_for_status()
            bars = client.get(
                "/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 1}
            )
            bars.raise_for_status()
            return Check(name, "ok", f"kết nối được, {symbol} trả về nến mới nhất", critical=True)
    except Exception as exc:  # noqa: BLE001 - every failure is a finding here
        return Check(name, "fail", f"không kết nối được: {type(exc).__name__}", critical=True)


def _models(cfg: Config) -> list[Check]:
    registry = ModelRegistry(cfg.data.root / "models")
    out = []
    for symbol in cfg.data.symbols:
        version = registry.latest(symbol, "1h")
        if version:
            out.append(Check(f"model {symbol}", "ok", version))
        else:
            out.append(
                Check(
                    f"model {symbol}", "warn",
                    "chưa có model: scheduler sẽ bỏ qua coin này. Chép data/models từ "
                    "máy cũ, hoặc train --save sau khi đọc cả hai VERDICT.",
                )
            )
    return out


def _feeds(cfg: Config, transport: httpx.BaseTransport | None) -> Check:
    if not cfg.news.feeds:
        return Check("nguồn tin", "warn", "không cấu hình nguồn nào")
    reachable = 0
    # The poller's own User-Agent: Bitcoin Magazine answers 403 to httpx's
    # default one, so checking with anything else reports a feed as dead that
    # the scheduler reads fine.
    with httpx.Client(
        transport=transport,
        timeout=8,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for feed in cfg.news.feeds:
            try:
                if client.get(feed.url).status_code < 400:
                    reachable += 1
            except Exception:  # noqa: BLE001, S112 - one dead feed is a count, not a crash
                continue
    total = len(cfg.news.feeds)
    state = "ok" if reachable == total else ("warn" if reachable else "fail")
    return Check("nguồn tin", state, f"{reachable}/{total} nguồn RSS phản hồi")


def _telegram(send_test: bool, transport: httpx.BaseTransport | None) -> Check:
    if telegram.configured() is None:
        return Check(
            "Telegram", "warn",
            "chưa cấu hình (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID): trên máy chủ sẽ "
            "không ai được báo khi model bắn tín hiệu hay khi scheduler chết",
        )
    if not send_test:
        return Check("Telegram", "ok", "đã cấu hình (thêm --send-test để gửi tin thử)")
    sent = telegram.send(
        "cryptopred: tin nhắn thử từ `doctor`. Nhận được tin này là cấu hình đúng.",
        transport=transport,
    )
    if sent:
        return Check("Telegram", "ok", "đã gửi tin thử")
    return Check("Telegram", "fail", "gửi tin thử thất bại: kiểm tra token và chat id")


def _external_ping() -> Check:
    if os.environ.get("HEALTHCHECK_PING_URL", "").strip():
        return Check("giám sát ngoài", "ok", "HEALTHCHECK_PING_URL đã cấu hình")
    return Check(
        "giám sát ngoài", "warn",
        "chưa cấu hình HEALTHCHECK_PING_URL: nếu cả máy chủ sập sẽ không có cảnh báo nào",
    )


def run_checks(
    cfg: Config,
    send_test: bool = False,
    binance_transport: httpx.BaseTransport | None = None,
    web_transport: httpx.BaseTransport | None = None,
) -> list[Check]:
    return [
        _data_dir(cfg),
        _binance(cfg, binance_transport),
        *_models(cfg),
        _feeds(cfg, web_transport),
        _telegram(send_test, web_transport),
        _external_ping(),
    ]


def format_checks(checks: list[Check]) -> str:
    mark = {"ok": "OK  ", "warn": "CHÚ Ý", "fail": "LỖI "}
    lines = [f"  [{mark[c.state]}] {c.name}: {c.detail}" for c in checks]
    failed = [c for c in checks if c.state == "fail" and c.critical]
    lines.append("")
    lines.append(
        "KHÔNG SẴN SÀNG - sửa các dòng LỖI ở trên trước." if failed
        else "Sẵn sàng chạy. Xem lại các dòng CHÚ Ý."
    )
    return "\n".join(lines)


def healthy(checks: list[Check]) -> bool:
    return not any(c.state == "fail" and c.critical for c in checks)
