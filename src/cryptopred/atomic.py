"""Write a file so a reader sees the old one or the new one, never half of either.

The current year's kline file is rewritten every hour. Written in place, a kill
at the wrong moment - a task stopped, a `docker stop` that ran out its ten
seconds, the power going - left a truncated parquet that every later read
raised on, and a scheduler that died on every cycle from then on, over a file
that had been fine the hour before.

So the new contents go to a temporary file beside the target, are flushed to
disk, and then replace the target in one rename. A kill before the rename
leaves the old file untouched and a stray `.tmp` that nothing reads.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

# Windows refuses to replace a file another process has open - the dashboard
# reading the same parquet for a moment. That passes; a few short waits ride it
# out rather than failing a cycle over it.
_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SECONDS = 0.2


def replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Call write(tmp) on a temporary path, then move it over `path`.

    The temporary name starts with a dot and ends in .tmp, so a glob for
    "*.parquet" or "*.json" never picks up a half-written file.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        with tmp.open("rb+") as handle:
            os.fsync(handle.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_WAIT_SECONDS)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomically(path: Path, text: str) -> None:
    replace_atomically(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
