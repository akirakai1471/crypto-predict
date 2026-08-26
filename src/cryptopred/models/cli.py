"""Command line entry point for training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from cryptopred.backtest.breakeven import analyse, format_table, round_trip_cost
from cryptopred.backtest.runner import format_backtest, run_strategy_backtest
from cryptopred.config import load_config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.metrics import evaluate
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
    override: str = typer.Option(
        None,
        help=(
            "Save despite a failing verdict. Requires a written justification, which is "
            "stored in the model metadata and displayed wherever the model is used."
        ),
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

    # The classification report says whether the model knows anything. The
    # backtest says whether that knowledge survives contact with fees.
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)
    if not bars.empty:
        test_index = dataset.index[-evaluation["n_test_total"] :]
        bt = run_strategy_backtest(
            bars, evaluation, test_index, horizon=horizon, threshold=threshold
        )
        report += "\n\n" + format_backtest(bt, symbol=symbol, interval=interval)

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

    if decision["decision"] != "GO" and not override:
        typer.echo(
            f"\nRefusing to save: verdict is {decision['decision']}. "
            "A model that has not beaten its baselines does not belong in the registry.\n"
            "If you have a reason to save it anyway, pass --override with a written "
            "justification; it is recorded in the model's metadata and shown on the "
            "dashboard, permanently."
        )
        raise typer.Exit(code=1)

    # Final model: fit on everything except the last horizon bars, whose labels
    # depend on prices that do not exist yet.
    final_train = dataset.iloc[: -horizon or None]
    result = train_fold(final_train, final_train.tail(1), train_config)
    registry = ModelRegistry(cfg.data.root / "models")
    metrics = dict(evaluation["model"])
    metrics["gate_decision"] = decision["decision"]
    metrics["gate_reason"] = decision["reason"]
    if override:
        metrics["override_reason"] = override

    version = registry.save(
        result,
        symbol=symbol,
        interval=interval,
        metrics=metrics,
        config=train_config,
        n_train_rows=len(final_train),
    )
    typer.echo(f"\nSaved model {version}")
    if override:
        typer.echo(
            f"  SAVED UNDER OVERRIDE — gate said {decision['decision']}: {decision['reason']}"
        )


@app.command()
def breakeven(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to analyse."),
    interval: str = typer.Option("1h", help="Bar interval."),
    accuracy: float = typer.Option(
        None, help="Measured directional accuracy, to show the margin per horizon."
    ),
    maker: bool = typer.Option(
        False, help="Price limit orders (0.02% fee, less slippage) instead of market orders."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """How accurate a model must be for each horizon to pay for itself.

    Run this BEFORE training on a new symbol or horizon. It needs no model and
    answers the question that decides everything: is the move big enough to
    cover the cost of capturing it?
    """
    cfg = load_config(config)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)
    if bars.empty:
        typer.echo(f"No bars for {symbol} {interval}. Run `cryptopred-ingest klines` first.")
        raise typer.Exit(code=1)

    cost = round_trip_cost(0.0002, 0.0001) if maker else round_trip_cost(
        cfg.strategy.taker_fee, cfg.strategy.slippage
    )
    horizons = (1, 2, 4, 8, 12, 24, 48, 72, 168) if interval == "1h" else (1, 3, 5, 15, 30, 60)
    results = analyse(bars, horizons=horizons, cost=cost)
    typer.echo(format_table(results, interval=interval, symbol=symbol, measured_accuracy=accuracy))


@app.command()
def sweep(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to analyse."),
    interval: str = typer.Option("1h", help="Bar interval."),
    n_splits: int = typer.Option(5, help="Number of walk-forward folds."),
    rounds: int = typer.Option(400, help="LightGBM boosting rounds."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Diagnostic: how signal count, accuracy and PnL vary with the threshold.

    READ THIS BEFORE USING THE OUTPUT. Picking the best row of this table and
    trading it is overfitting — the table is computed on the same out-of-sample
    data used to judge the model, so the winning threshold is partly fitted to
    that data's noise. Use it to understand the shape of the trade-off, then
    validate any chosen threshold on data this sweep never touched.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    path = cfg.dataset_dir() / f"{symbol}_{interval}.parquet"
    if not path.exists():
        typer.echo(f"No dataset at {path}. Run `cryptopred-dataset build` first.")
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
    horizon = cfg.labels.horizon_bars.get(interval, 4)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)

    evaluation = walk_forward_evaluate(
        dataset,
        n_splits=n_splits,
        horizon=horizon,
        config=TrainConfig(num_boost_round=rounds, calibrate=True),
    )
    test_index = dataset.index[-evaluation["n_test_total"] :]

    typer.echo("\n*** DIAGNOSTIC ONLY — choosing a threshold from this table overfits it ***\n")
    header = (
        f"{'thresh':>7} {'signals':>9} {'sign_acc':>9} {'net_ret':>10} "
        f"{'maxDD':>9} {'2x_cost':>10} {'robust':>7}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        metrics = evaluate(
            evaluation["y_true"],
            evaluation["proba"],
            threshold=threshold,
            forward_return=evaluation["forward_return"],
        )
        bt = run_strategy_backtest(
            bars, evaluation, test_index, horizon=horizon, threshold=threshold
        )
        base = bt["base"].summary
        doubled = bt["doubled_costs"].summary
        typer.echo(
            f"{threshold:>7.2f} {metrics['n_signals']:>9,} "
            f"{_num(metrics.get('sign_accuracy')):>9} "
            f"{_num(base.get('total_return'), pct=True):>10} "
            f"{_num(base.get('max_drawdown'), pct=True):>9} "
            f"{_num(doubled.get('total_return'), pct=True):>10} "
            f"{str(bt['survives_doubled_costs']):>7}"
        )


def _num(value: float | None, pct: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%" if pct else f"{value:.4f}"


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
