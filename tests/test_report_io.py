"""A finished report must survive a console that cannot print it.

Two long runs were lost at the display step. The file is the artifact; echoing
is a convenience, and a convenience must not be able to destroy the artifact.
"""

import contextlib

import typer

from cryptopred.report_io import emit


def test_the_file_is_written_before_anything_is_printed(tmp_path, monkeypatch):
    """A console encoding that cannot represent an arrow killed a 20-minute
    nested-CV run after it had finished."""
    def explode(*_args, **_kwargs):
        raise UnicodeEncodeError("charmap", "→", 0, 1, "undefined")

    monkeypatch.setattr(typer, "echo", explode)
    target = tmp_path / "nested" / "report.txt"
    # The fallback print may fail too; the file must not.
    with contextlib.suppress(UnicodeEncodeError):
        emit("1,000 → 400 signals", target)
    assert target.read_text(encoding="utf-8") == "1,000 → 400 signals"


def test_a_console_that_cannot_encode_still_shows_the_report(tmp_path, monkeypatch):
    printed = []

    def picky(text=""):
        text.encode("cp1252")  # raises on the arrow, like a Windows console
        printed.append(text)

    monkeypatch.setattr(typer, "echo", picky)
    emit("1,000 → 400 signals", tmp_path / "r.txt")
    assert any("400 signals" in p for p in printed)


def test_the_directory_is_created(tmp_path):
    emit("hello", tmp_path / "a" / "b" / "c.txt")
    assert (tmp_path / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "hello"
