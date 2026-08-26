"""Command line entry point for the meta-labelled stack."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from cryptopred.config import load_config
from cryptopred.dataset.builder import dataset_path
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.meta import MetaConfig, signals_from_proba, walk_forward_meta
from cryptopred.models.meta_report import format_meta_report, score_signals
from cryptopred.models.train import TrainConfig

app = typer.Typer(help="Train and evaluate the meta-labelled two-model stack.")
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """Typer collapses a single-command app into a bare command; this callback
    keeps `cryptopred-meta run` working as a named subcommand."""


@app.command()
def run(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to evaluate."),
    interval: str = typer.Option("1h", help="Bar interval."),
    horizon: int = typer.Option(None, help="Label horizon in bars, overriding the config."),
    primary_threshold: float = typer.Option(
        0.40, help="Primary signal threshold. Loose values give the secondary more to filter."
    ),
    meta_threshold: float = typer.Option(
        0.55, help="Minimum probability of profit required to take a signal."
    ),
    n_splits: int = typer.Option(5, help="Outer walk-forward folds."),
    inner_splits: int = typer.Option(3, help="Inner folds used to build honest meta-labels."),
    rounds: int = typer.Option(300, help="Primary boosting rounds."),
    meta_rounds: int = typer.Option(200, help="Secondary boosting rounds."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Compare the primary model alone against the primary plus meta filter."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)

    path = dataset_path(cfg, symbol, interval, horizon)
    if not path.exists():
        typer.echo(
            f"No dataset at {path}. Run `cryptopred-dataset build --horizon {horizon}` first."
        )
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)

    meta_config = MetaConfig(
        primary_threshold=primary_threshold,
        meta_threshold=meta_threshold,
        inner_splits=inner_splits,
        primary=TrainConfig(num_boost_round=rounds, signal_threshold=primary_threshold),
        secondary=TrainConfig(num_boost_round=meta_rounds, min_data_in_leaf=100),
    )

    typer.echo(
        f"Training on {len(dataset):,} rows, horizon {horizon} bars. "
        f"Nested CV: {n_splits} outer x {inner_splits} inner folds ..."
    )
    result = walk_forward_meta(
        dataset, bars, horizon=horizon, n_splits=n_splits, config=meta_config
    )

    index = result["index"]
    forward = dataset.loc[index, "forward_return"].to_numpy()

    primary_scored = score_signals(
        bars, index, result["primary_signals"], horizon=horizon, forward_return=forward
    )
    meta_scored = score_signals(
        bars, index, result["final_signals"], horizon=horizon, forward_return=forward
    )

    # The honest benchmark is not the loose primary the stack filters — that is
    # easy to beat — but one model alone at the threshold already in production.
    benchmark_threshold = cfg.strategy.signal_threshold
    benchmark_signals = signals_from_proba(result["primary_proba"], benchmark_threshold)
    benchmark_scored = score_signals(
        bars, index, benchmark_signals, horizon=horizon, forward_return=forward
    )

    report = format_meta_report(
        primary_scored, meta_scored, symbol=symbol, horizon=horizon,
        config=meta_config, folds=result["folds"],
        benchmark=benchmark_scored, benchmark_threshold=benchmark_threshold,
    )
    typer.echo(report)

    reports_dir = cfg.data.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (reports_dir / f"meta_{symbol}_{interval}_h{horizon}_{stamp}.txt").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    app()
