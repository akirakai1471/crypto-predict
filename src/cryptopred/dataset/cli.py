"""Command line entry point for dataset construction."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from cryptopred.config import Config, load_config
from cryptopred.dataset.builder import build_dataset
from cryptopred.dataset.quality import format_report, quality_report
from cryptopred.ingest.storage import ParquetStore

app = typer.Typer(help="Build model-ready datasets from stored raw data.")
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """Typer collapses a single-command app into a bare command; this callback
    keeps `cryptopred-dataset build` working as a named subcommand."""


def run_build(cfg: Config, store: ParquetStore, quiet: bool = False) -> list[Path]:
    """Build one dataset per symbol/interval. Returns the written file paths."""
    out_dir = cfg.dataset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for symbol in cfg.data.symbols:
        for interval in cfg.data.intervals:
            bars = store.read("klines", symbol, interval)
            if bars.empty:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: no raw data, skipping")
                continue

            funding = store.read("funding", symbol, "8h")
            horizon = cfg.labels.horizon_bars.get(interval)
            if horizon is None:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: no horizon configured, skipping")
                continue

            dataset = build_dataset(
                bars,
                interval=interval,
                horizon=horizon,
                atr_period=cfg.labels.atr_period,
                band_k=cfg.labels.band_k,
                funding=funding if not funding.empty else None,
                symbol=symbol,
                feature_config=cfg.features,
            )
            if dataset.empty:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: dataset empty after cleaning")
                continue

            path = out_dir / f"{symbol}_{interval}.parquet"
            dataset.to_parquet(path, engine="pyarrow", index=True)
            written.append(path)

            if not quiet:
                typer.echo(f"\n{symbol} {interval} -> {path}")
                typer.echo(format_report(quality_report(dataset)))

    return written


@app.command()
def build(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Build datasets for every configured symbol and interval."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    paths = run_build(cfg, store)
    typer.echo(f"\nWrote {len(paths)} dataset file(s).")


if __name__ == "__main__":
    app()
