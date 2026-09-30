"""Telegram: the alert channel for a machine nobody is sitting in front of.

On a server there is no desktop for a Windows balloon to appear on, and a log
file on a machine in a data centre is read by nobody. A Telegram message
reaches the phone.

Configured only through the environment - TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID - never through config/default.yaml, because that file is in
git and a bot token is a credential. Unset means off, silently.

Two rules this module keeps:

- It never raises. A failed message must cost the user a message, not the
  prediction cycle that produced it.
- The token never reaches a log. It is part of the request URL, and httpx logs
  every request URL at INFO - so a filter rewrites it before any handler sees
  it, and failures are logged by exception type only, since httpx puts the URL
  in its error messages too.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
# Telegram rejects longer messages outright; cutting is better than losing it.
MAX_TEXT = 4096

_TOKEN_IN_URL = re.compile(r"/bot[^/\s]+/")


class RedactBotToken(logging.Filter):
    """Rewrite /bot<token>/ to /bot<redacted>/ in any record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "/bot" in message:
            record.msg = _TOKEN_IN_URL.sub("/bot<redacted>/", message)
            record.args = ()
        return True


def _install_redaction() -> None:
    httpx_log = logging.getLogger("httpx")
    if not any(isinstance(f, RedactBotToken) for f in httpx_log.filters):
        httpx_log.addFilter(RedactBotToken())


def configured() -> tuple[str, str] | None:
    """(token, chat_id) from the environment, or None if either is missing."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def send(
    text: str,
    token: str | None = None,
    chat_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 10.0,
) -> bool:
    """Send one message. True if Telegram accepted it; never raises."""
    if token is None or chat_id is None:
        creds = configured()
        if creds is None:
            return False
        token, chat_id = creds

    _install_redaction()
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 1] + "…"

    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.post(
                f"{API_BASE}/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            )
        if response.status_code == 200:
            return True
        # The body names the problem ("chat not found", "bot was blocked")
        # and does not contain the token.
        logger.warning(
            "telegram rejected the message: HTTP %s %s",
            response.status_code,
            response.text[:200],
        )
        return False
    except Exception as exc:  # noqa: BLE001 - a lost message must not cost the caller
        logger.warning("telegram unreachable: %s", type(exc).__name__)
        return False
