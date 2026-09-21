"""Write a report to disk before printing it, and never lose one to a console.

Two long runs died at `typer.echo`. The first read a renamed config field; the
second hit a Windows console whose encoding is cp1252, which cannot represent
the arrow in "1,000 → 400 signals". Both had finished twenty minutes of nested
cross-validation, and both threw the result away because the *display* failed.

The output is the artifact. Printing it is a convenience.
"""

from __future__ import annotations

from pathlib import Path

import typer


def safe_echo(text: str) -> None:
    """Print text a Windows cp1252 console cannot represent.

    The briefing is written entirely in Vietnamese, so this is not a corner
    case for it — redirecting `cryptopred-brief` to a file would die on the
    first accented character. `emit` has a file to fall back on; the brief does
    not, so the fallback lives here where both can reach it.
    """
    try:
        typer.echo(text)
    except UnicodeEncodeError:
        # ASCII rather than the console's declared encoding: the declared one is
        # what just failed, and the point of a fallback is that it cannot fail.
        typer.echo(text.encode("ascii", errors="replace").decode("ascii"))


def emit(report: str, path: Path) -> None:
    """Save `report` to `path`, then print it.

    Saving first is the whole point: a print that fails must not be able to
    destroy work that succeeded. The file is always UTF-8; only the console has
    an encoding that depends on the machine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    safe_echo(report)
    typer.echo(f"\nSaved to {path}")
