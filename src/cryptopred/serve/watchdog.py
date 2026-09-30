"""Say so when the scheduler stops, to someone who is not looking at it.

The heartbeat makes a dead scheduler visible - on the dashboard and in
status.bat - but only to someone who looks. On a server nobody looks, and the
failure this project fears most is a scheduler that stopped days ago while
everything else kept working.

So the dashboard process, which is separate from the scheduler and outlives
it, checks the heartbeat every few minutes and sends one Telegram message when
it goes stale, and one more when it comes back. One per incident: a message
every ten minutes for a day is a message nobody reads by the afternoon.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from cryptopred.serve import heartbeat, telegram

logger = logging.getLogger(__name__)

CHECK_EVERY_SECONDS = 600


class SchedulerWatchdog:
    def __init__(
        self, heartbeat_path: Path, send: Callable[[str], bool] = telegram.send
    ) -> None:
        self.heartbeat_path = Path(heartbeat_path)
        self.send = send
        self.down = False

    def check(self, now=None) -> str:
        """One look. Returns "down", "recovered" or "ok" for what it did."""
        beat = heartbeat.status(self.heartbeat_path, now=now)
        alive = beat["state"] == "alive"

        if not alive and not self.down:
            self.down = True
            self.send(
                "cryptopred — SCHEDULER ĐÃ DỪNG\n"
                f"{beat['detail']}\n\n"
                "Không có dự đoán nào đang được ghi. Trên máy chủ: "
                "`docker compose ps` và `docker compose logs --tail 50 scheduler`."
            )
            logger.warning("scheduler down: %s", beat["detail"])
            return "down"

        if alive and self.down:
            self.down = False
            self.send(f"cryptopred — scheduler chạy lại. {beat['detail']}.")
            logger.info("scheduler recovered: %s", beat["detail"])
            return "recovered"

        return "ok"


def start(heartbeat_path: Path, every: float = CHECK_EVERY_SECONDS) -> threading.Thread | None:
    """Run the watchdog in a daemon thread, if Telegram is configured to hear it."""
    if telegram.configured() is None:
        return None
    dog = SchedulerWatchdog(heartbeat_path)
    stop = threading.Event()

    def loop() -> None:
        # The first look waits a full interval: a freshly started scheduler
        # has not finished its first cycle yet, and that is not an outage.
        while not stop.wait(every):
            try:
                dog.check()
            except Exception:  # noqa: BLE001 - the watchdog must outlive its own bugs
                logger.exception("watchdog check failed")

    thread = threading.Thread(target=loop, name="scheduler-watchdog", daemon=True)
    thread.start()
    return thread
