"""Tell the user a headline that sounds important has arrived, and what it is not.

It is a headline. It is not a signal - the model does not read news - and it is
not a recommendation. "High impact" is a keyword match nobody has validated
(news/tags.py). Every alert says all three, in the message itself, because
the moment the caveat lives somewhere else nobody reads it. Delivery reuses
serve/alerts.notify, into data/news.log, apart from the model's signals.log.

Four rules keep it from turning into noise or a stale echo:

- Only a high-impact headline tagged with a symbol this system trades.
- Never the same headline twice. The decision is written to news.db before the
  notification goes out, so a crash in between costs one alert and can never
  repeat one - across restarts, and across two processes on the same file.
- At most `max_alerts_per_hour` in any trailing hour. Headlines over the limit
  are recorded as rate_limited and never sent later: an alert about something
  from an hour ago is not news, and the dashboard shows it anyway.
- Nothing old. Backlog rows (the first fetch after a start or an outage) are
  never candidates, a headline is only considered for FRESH_FOR after we
  received it, and one the publisher itself dates more than MAX_PUBLISHER_LAG
  earlier is recorded as stale. A publisher date can only suppress an alert
  here, never cause one, so a wrong date costs at most a missed message.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from cryptopred.news.store import NewsStore
from cryptopred.news.tags import TAG_CAVEAT
from cryptopred.serve.alerts import notify

logger = logging.getLogger(__name__)

FRESH_FOR = pd.Timedelta(minutes=30)
MAX_PUBLISHER_LAG = pd.Timedelta(hours=2)
RATE_WINDOW = pd.Timedelta(hours=1)

NOT_A_SIGNAL = (
    "Đây là một TIÊU ĐỀ TIN — KHÔNG phải tín hiệu của model, KHÔNG phải khuyến nghị."
)


def format_news_alert(item: dict[str, Any], symbols: Iterable[str]) -> str:
    """The message: what arrived, then what it is not."""
    when = pd.Timestamp(item["received_at"]).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    # A title comes from a stranger's feed. The log marks entries with lines of
    # "=====", so a title must not be able to forge one.
    title = re.sub(r"={3,}", "==", item["title"])
    lines = [
        f"Tin {', '.join(symbols)} — tiêu đề, không phải tín hiệu",
        title,
        f"{item['source']} · nhận lúc {when}",
    ]
    if item.get("link"):
        lines.append(item["link"])
    lines += [
        "",
        f"Gắn “tác động cao” vì khớp từ khoá: {', '.join(item['impact_terms'])}.",
        TAG_CAVEAT,
        NOT_A_SIGNAL,
    ]
    return "\n".join(lines)


def alert_fresh_headlines(
    store: NewsStore,
    symbols: Iterable[str],
    log_path: Path,
    max_per_hour: int,
    now: pd.Timestamp | None = None,
    popup: bool = True,
    send: Callable[..., None] = notify,
) -> dict[str, int]:
    """Decide on every fresh, undecided headline. Returns counts by outcome."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    wanted = set(symbols)
    counts = {"sent": 0, "rate_limited": 0, "stale": 0, "failed": 0}

    for item in store.alert_candidates(since=now - FRESH_FOR):
        hit = [s for s in item["symbols"] if s in wanted]
        if not item["high_impact"] or not hit:
            continue
        uid = item["uid"]

        published = item.get("published_at")
        if published and pd.Timestamp(item["received_at"]) - pd.Timestamp(
            published
        ) > MAX_PUBLISHER_LAG:
            if store.claim_alert(uid, "stale", at=now):
                counts["stale"] += 1
            continue

        if store.alerts_sent_since(now - RATE_WINDOW) >= max_per_hour:
            if store.claim_alert(uid, "rate_limited", at=now):
                counts["rate_limited"] += 1
            continue

        # Claimed before sending: if another process already decided on this
        # headline, the claim fails and nothing is sent twice.
        if not store.claim_alert(uid, "sent", at=now):
            continue
        try:
            send(format_news_alert(item, hit), log_path=log_path, popup=popup)
            counts["sent"] += 1
        except Exception:  # noqa: BLE001 - a lost alert must not lose the pass
            logger.exception("could not send the news alert for %s", uid)
            store.set_alert_outcome(uid, "failed")
            counts["failed"] += 1
    return counts
