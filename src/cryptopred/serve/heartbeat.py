"""A record of when the scheduler last did anything.

The largest risk to a multi-day experiment is not a wrong number, it is a
scheduler that stopped an hour after it started and left no trace. Everything
downstream keeps working — the dashboard renders, the status report prints, the
database opens — and the only symptom is a prediction log that quietly stops
growing, which nobody notices until they come back expecting a week of data.

So each cycle writes a timestamp, and the status report treats a stale one as a
failure rather than an absence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from cryptopred.atomic import write_text_atomically

# A 1h scheduler runs hourly. Two missed cycles is a problem worth naming.
STALE_AFTER_MINUTES = 150


def write(path: Path, interval: str, counts: dict[str, int]) -> None:
    """Record the end of a cycle. Never raises: a failed heartbeat must not take
    down the thing it is monitoring."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_cycle": pd.Timestamp.now(tz="UTC").isoformat(),
            "interval": interval,
            "pid": os.getpid(),
            "counts": counts,
        }
        # A torn file reads as "corrupt", which status reports as a fault.
        write_text_atomically(path, json.dumps(payload, indent=2))
    except OSError:
        pass
    ping_external()


def ping_external(timeout: float = 5.0, transport=None) -> bool:
    """Tell an outside dead-man's switch that a cycle finished.

    Everything else here watches the scheduler from the same machine, so it
    goes quiet exactly when the machine does - a VPS that is down sends no
    alert about being down. A service such as healthchecks.io, given
    HEALTHCHECK_PING_URL, expects a ping every hour and alerts when one does not
    arrive. Unset means off. Never raises.
    """
    url = os.environ.get("HEALTHCHECK_PING_URL", "").strip()
    if not url:
        return False
    try:
        import httpx

        with httpx.Client(timeout=timeout, transport=transport) as client:
            return client.get(url).status_code < 400
    except Exception:  # noqa: BLE001 - the ping must never cost the cycle
        return False


def read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(path: Path, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Is the scheduler alive, and when did it last run?"""
    # A corrupt file and a missing one mean different things: one says the
    # scheduler never ran, the other says something wrote nonsense. Reporting
    # the first for the second would send the reader to the wrong problem.
    if not path.exists():
        return {
            "state": "never_started",
            "detail": "no heartbeat file — the scheduler has not completed a cycle",
        }

    beat = read(path)
    if beat is None:
        return {"state": "unreadable", "detail": "heartbeat file is malformed"}

    now = now or pd.Timestamp.now(tz="UTC")
    try:
        last = pd.Timestamp(beat["last_cycle"])
    except (KeyError, ValueError):
        return {"state": "unreadable", "detail": "heartbeat file is malformed"}

    minutes = (now - last).total_seconds() / 60
    state = "stale" if minutes > STALE_AFTER_MINUTES else "alive"
    return {
        "state": state,
        "last_cycle": last,
        "minutes_ago": minutes,
        "pid": beat.get("pid"),
        "detail": (
            f"last cycle {minutes:.0f} minutes ago"
            if state == "alive"
            else f"last cycle {minutes / 60:.1f} hours ago — the scheduler has stopped"
        ),
    }
