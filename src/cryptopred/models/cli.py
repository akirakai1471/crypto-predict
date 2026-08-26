"""Command line entry point for training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from cryptopred.config import load_config
from cryptopred.models.registry import ModelRegistry
from cryptopred.models.report import format_evaluation, verdict
from cryptopred.models.train import TrainConfig, train_fold, walk_forward_evaluate

app = typer.Typer(help="Train models and run walk-forward evaluation.")
logger = logging.getLogger(__name__)


@app.command()
def train(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to train on."),
    interval: str = typer.Option("1h", help="Bar interval."),
    n_splits: int = typer.Option(5, help="Number of walk-forward folds."),
    rounds: int = typer.Option(400, help="LightGBM boosting rounds."),
    threshold: float = typer.Option(0.5, help="Minimum probability to count as a signal."),
    save: bool = typer.Option(
        False, help="Save a final model to the registry (only sensible after a GO verdict)."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Evaluate walk-forward, print the report, and optionally save the model."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    path = cfg.dataset_dir() / f"{symbol}_{interval}.parquet"
    if not path.exists():
        typer.echo(f"No dataset at {path}. Run `cryptopred-dataset build` first.")
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
    horizon = cfg.labels.horizon_bars.get(interval, 4)
    train_config = TrainConfig(
        num_boost_round=rounds, signal_threshold=threshold, calibrate=True
    )

    typer.echo(f"Training on {len(dataset):,} rows from {path.name} ...")
    evaluation = walk_forward_evaluate(
        dataset, n_splits=n_splits, horizon=horizon, config=train_config
    )

    report = format_evaluation(evaluation, symbol=symbol, interval=interval, threshold=threshold)
    typer.echo(report)

    reports_dir = cfg.data.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (reports_dir / f"{symbol}_{interval}_{stamp}.txt").write_text(report, encoding="utf-8")

    decision = verdict(evaluation, threshold=threshold)
    summary = {
        "symbol": symbol,
        "interval": interval,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "model": evaluation["model"],
    }
    (reports_dir / f"{symbol}_{interval}_{stamp}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    if not save:
        return

    if decision["decision"] != "GO":
        typer.echo(
            f"\nRefusing to save: verdict is {decision['decision']}. "
            "A model that has not beaten its baselines does not belong in the registry."
        )
        raise typer.Exit(code=1)

    # Final model: fit on everything except the last horizon bars, whose labels
    # depend on prices that do not exist yet.
    final_train = dataset.iloc[: -horizon or None]
    result = train_fold(final_train, final_train.tail(1), train_config)
    registry = ModelRegistry(cfg.data.root / "models")
    version = registry.save(
        result,
        symbol=symbol,
        interval=interval,
        metrics=evaluation["model"],
        config=train_config,
        n_train_rows=len(final_train),
    )
    typer.echo(f"\nSaved model {version}")


@app.command()
def registry(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """List every model in the registry."""
    cfg = load_config(config)
    index = ModelRegistry(cfg.data.root / "models").index()
    if index.empty:
        typer.echo("Registry is empty.")
        return
    typer.echo(index.to_string(index=False))


if __name__ == "__main__":
    app()
