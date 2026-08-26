"""Command line entry point for paper-trading experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from cryptopred.config import load_config
from cryptopred.dataset.builder import dataset_path
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.train import TrainConfig, walk_forward_evaluate
from cryptopred.paper.replay import format_replay, replay_predictions
from cryptopred.serve.store import PredictionStore

app = typer.Typer(help="Paper-trading tools. No exchange account is ever touched.")
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """Typer collapses a single-command app into a bare command; this callback
    keeps `cryptopred-paper replay` working as a named subcommand."""


@app.command()
def replay(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to replay."),
    interval: str = typer.Option("1h", help="Bar interval."),
    horizon: int = typer.Option(None, help="Label horizon in bars, overriding the config."),
    threshold: float = typer.Option(None, help="Signal threshold, overriding the config."),
    n_splits: int = typer.Option(5, help="Number of walk-forward folds."),
    rounds: int = typer.Option(400, help="LightGBM boosting rounds."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Run out-of-sample predictions through the live paper trader.

    Writes to a separate replay database so the forward-running paper log stays
    untouched: mixing simulated history into the live record would destroy the
    one measurement in this project that hindsight cannot edit.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)
    threshold = threshold if threshold is not None else cfg.strategy.signal_threshold

    path = dataset_path(cfg, symbol, interval, horizon)
    if not path.exists():
        typer.echo(
            f"No dataset at {path}. Run `cryptopred-dataset build --horizon {horizon}` first."
        )
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)

    typer.echo(f"Training on {len(dataset):,} rows, horizon {horizon} bars ...")
    evaluation = walk_forward_evaluate(
        dataset,
        n_splits=n_splits,
        horizon=horizon,
        config=TrainConfig(num_boost_round=rounds, signal_threshold=threshold),
    )
    test_index = dataset.index[-evaluation["n_test_total"] :]

    replay_db = cfg.data.root / f"replay_{symbol}_{interval}_h{horizon}_t{threshold}.db"
    replay_db.unlink(missing_ok=True)
    store = PredictionStore(replay_db)

    result = replay_predictions(
        cfg,
        bars,
        evaluation["proba"],
        test_index,
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        threshold=threshold,
        store=store,
    )
    typer.echo(format_replay(result, symbol=symbol, threshold=threshold))
    typer.echo(f"\nTrade ledger: {replay_db}")


if __name__ == "__main__":
    app()
