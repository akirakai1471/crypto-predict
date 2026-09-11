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


def emit(report: str, path: Path) -> None:
    """Save `report` to `path`, then print it.

    Saving first is the whole point: a print that fails must not be able to
    destroy work that succeeded. The file is always UTF-8; only the console has
    an encoding that depends on the machine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    try:
        typer.echo(report)
    except UnicodeEncodeError:
        # A console that cannot show an em-dash is not a reason to show nothing.
        # ASCII rather than the console's declared encoding: the declared one is
        # what just failed, and the point of a fallback is that it cannot fail.
        typer.echo(report.encode("ascii", errors="replace").decode("ascii"))

    typer.echo(f"\nSaved to {path}")
