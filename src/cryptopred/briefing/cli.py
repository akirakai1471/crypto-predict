"""`cryptopred-brief` — the numbers, with no model and no API key."""

from __future__ import annotations

from pathlib import Path

import typer

from cryptopred.briefing.report import format_brief
from cryptopred.config import load_config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.report_io import safe_echo

app = typer.Typer(help="Print the measured market briefing for a symbol.")


@app.command()
def show(
    symbol: str = typer.Argument("BTCUSDT", help="Symbol to describe."),
    interval: str = typer.Option("1h", help="Bar interval."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    cfg = load_config(config)
    parquet = ParquetStore(cfg.data.root / "raw")
    bars = parquet.read("klines", symbol, interval)
    funding = parquet.read("funding", symbol, "8h")
    # Vietnamese text through a Windows console: see safe_echo.
    safe_echo(format_brief(symbol, interval, bars, funding))


if __name__ == "__main__":
    app()
