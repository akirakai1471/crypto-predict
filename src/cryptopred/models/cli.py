"""Command line entry point for training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from cryptopred.backtest.breakeven import analyse, format_table, round_trip_cost
from cryptopred.backtest.engine import CostModel
from cryptopred.backtest.execution_report import compare_execution, format_execution_comparison
from cryptopred.backtest.runner import format_backtest, run_strategy_backtest
from cryptopred.backtest.sizing_report import (
    compare_sizing,
    compare_sizing_matched,
    confidence_diagnostics,
    format_sizing_comparison,
)
from cryptopred.config import load_config
from cryptopred.dataset.builder import build_dataset, dataset_path
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.registry import ModelRegistry
from cryptopred.models.report import format_evaluation, verdict
from cryptopred.models.selection import margin_cutoff, signals_by_quantile_per_fold
from cryptopred.models.train import (
    COVERAGE_TOLERANCE,
    CUTOFF_HOLDOUT_BARS,
    TrainConfig,
    apply_calibrators,
    coverage_check,
    train_fold,
    walk_forward_evaluate,
)
from cryptopred.models.validation import (
    FrozenConfig,
    SymbolResult,
    evaluate_symbol,
    format_validation,
)

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
    horizon: int = typer.Option(
        None, help="Label horizon in bars, overriding the config."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Evaluate walk-forward, print the report, and optionally save the model."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)
    path = dataset_path(cfg, symbol, interval, horizon)
    if not path.exists():
        typer.echo(
            f"No dataset at {path}. Run "
            f"`cryptopred-dataset build --horizon {horizon}` first."
        )
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
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
        # Score the rule that gets deployed, not the withdrawn threshold. The
        # two disagree: the threshold run printed a -42.3% drawdown for a
        # strategy this project does not trade.
        bt = run_strategy_backtest(
            bars,
            evaluation,
            test_index,
            horizon=horizon,
            coverage=cfg.strategy.signal_coverage,
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
    # depend on prices that do not exist yet, and a holdout block reserved for
    # setting the trading rule.
    #
    # The holdout is not optional. A cutoff derived from the fold models is too
    # high for the final one: the final model trains on more data, calibrates
    # better, and therefore produces narrower margins. The first attempt at this
    # measured coverage on the final model's own training data, where margins are
    # inflated, reported a healthy 12.87%, and then fired on 0 of 336 live bars.
    # A rule can only be set on data the model has not seen.
    usable = dataset.iloc[: -horizon or None]
    holdout_n = min(CUTOFF_HOLDOUT_BARS, len(usable) // 5)
    final_train = usable.iloc[:-holdout_n]
    holdout = usable.iloc[-holdout_n:]
    # Split again. A cutoff is a quantile of some sample, so measuring coverage on
    # the sample it came from returns the target every time — a check that cannot
    # fail. The rule is set on the older block and verified on the newest one, so
    # a regime where margins have narrowed shows up as a failed check instead of
    # as zero live signals.
    rule_block = holdout.iloc[: int(len(holdout) * 2 / 3)]
    verify_block = holdout.iloc[int(len(holdout) * 2 / 3) :]
    # Calibrate the final model from out-of-fold runs. The tail-block method is
    # fine inside a fold but degenerate here, where the tail is one recent regime.
    final_config = TrainConfig(
        **{**train_config.__dict__, "calibration_method": "oof"}
    )
    result = train_fold(final_train, final_train.tail(1), final_config, horizon=horizon)

    # Both the cutoff and the check it is verified against come from the holdout,
    # which this model never saw. Anything measured on its training data would
    # describe a model that does not exist outside the fit.
    rule_proba = np.asarray(result.booster.predict(rule_block[result.features]))
    if result.calibrators:
        rule_proba = apply_calibrators(rule_proba, result.calibrators)
    cutoff = margin_cutoff(rule_proba, cfg.strategy.signal_coverage)

    check = coverage_check(
        result, verify_block, cfg.strategy.signal_coverage, cutoff
    )
    typer.echo(
        f"\nSignal rule: trade the top {cfg.strategy.signal_coverage:.0%} by directional "
        f"margin, cutoff {cutoff:.4f}"
    )
    typer.echo(
        f"Coverage check: target {check['expected_coverage']:.2%}, "
        f"saved model fires on {check['actual_coverage']:.2%} "
        f"of {check['n_sample']:,} later unseen bars (ratio {check['ratio']:.1f}x)"
    )
    if not check["ok"]:
        typer.echo(
            "\nRefusing to save: the final model does not behave like the models "
            "that were evaluated.\n"
            "Its signal rate differs by more than "
            f"{int(COVERAGE_TOLERANCE)}x, so the metrics above do not describe it. "
            "This is usually a calibration failure — check the isotonic maps before "
            "trusting anything downstream."
        )
        raise typer.Exit(code=1)

    registry = ModelRegistry(cfg.data.root / "models")
    metrics = dict(evaluation["model"])
    metrics["gate_decision"] = decision["decision"]
    metrics["margin_cutoff"] = cutoff
    metrics["signal_coverage"] = cfg.strategy.signal_coverage
    metrics["gate_reason"] = decision["reason"]
    # Only record an override that was actually needed. Marking a model that
    # passed its gate as overridden would put a permanent warning on the
    # dashboard for a model that earned its place.
    override_used = bool(override) and decision["decision"] != "GO"
    if override_used:
        metrics["override_reason"] = override

    version = registry.save(
        result,
        symbol=symbol,
        interval=interval,
        metrics=metrics,
        config=train_config,
        n_train_rows=len(final_train),
        margin_cutoff=cutoff,
        signal_coverage=cfg.strategy.signal_coverage,
    )
    typer.echo(f"\nSaved model {version}")
    if override_used:
        typer.echo(
            f"  SAVED UNDER OVERRIDE — gate said {decision['decision']}: "
            f"{decision['reason']}"
        )
    elif override:
        typer.echo("  (--override was not needed: the gate passed)")


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
    horizon: int = typer.Option(
        None, help="Label horizon in bars, overriding the config."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Diagnostic: how signal count, accuracy and PnL vary with coverage.

    Sweeps the rank rule, which is what gets deployed. It used to sweep the
    probability threshold; that number no longer means anything, because the
    same threshold selected 26.6% of one fold's bars and 0.03% of another's.

    READ THIS BEFORE USING THE OUTPUT. Picking the best row of this table and
    trading it is overfitting — the table is computed on the same out-of-sample
    data used to judge the model, so the winning coverage is partly fitted to
    that data's noise. Use it to understand the shape of the trade-off, then
    validate any chosen coverage on data this sweep never touched.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)
    path = dataset_path(cfg, symbol, interval, horizon)
    if not path.exists():
        typer.echo(
            f"No dataset at {path}. Run "
            f"`cryptopred-dataset build --horizon {horizon}` first."
        )
        raise typer.Exit(code=1)

    dataset = pd.read_parquet(path)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", symbol, interval)

    evaluation = walk_forward_evaluate(
        dataset,
        n_splits=n_splits,
        horizon=horizon,
        config=TrainConfig(num_boost_round=rounds, calibrate=True),
    )
    test_index = dataset.index[-evaluation["n_test_total"] :]

    typer.echo("\n*** DIAGNOSTIC ONLY — choosing a coverage from this table overfits it ***\n")
    header = (
        f"{'cover':>7} {'signals':>9} {'sign_acc':>9} {'net_ret':>10} "
        f"{'maxDD':>9} {'2x_cost':>10} {'robust':>7}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    forward = np.asarray(evaluation["forward_return"])
    for coverage in [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25]:
        bt = run_strategy_backtest(
            bars, evaluation, test_index, horizon=horizon, coverage=coverage
        )
        taken = bt["signals"]["signal"].to_numpy()
        picked = taken != 0
        sign_acc = (
            float((np.sign(forward[picked]) == taken[picked]).mean())
            if picked.any()
            else None
        )
        base = bt["base"].summary
        doubled = bt["doubled_costs"].summary
        typer.echo(
            f"{coverage:>6.0%} {int(picked.sum()):>9,} "
            f"{_num(sign_acc):>9} "
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
def sizing(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to evaluate."),
    interval: str = typer.Option("1h", help="Bar interval."),
    horizon: int = typer.Option(None, help="Label horizon in bars, overriding the config."),
    coverage: float = typer.Option(
        None, help="Fraction of bars to trade, overriding the config."
    ),
    n_splits: int = typer.Option(5, help="Walk-forward folds."),
    rounds: int = typer.Option(400, help="LightGBM boosting rounds."),
    kelly_scale: float = typer.Option(
        0.5, help="Fraction of full Kelly to stake. Below 1 on purpose — see the module docs."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Does betting more on stronger signals beat betting the same every time?

    One model, one set of signals, several staking rules. Any difference in the
    results comes from the stake alone.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)
    coverage = coverage if coverage is not None else cfg.strategy.signal_coverage

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
        config=TrainConfig(num_boost_round=rounds),
    )
    index = dataset.index[-evaluation["n_test_total"] :]
    proba = evaluation["proba"]
    fold_ids = np.concatenate(
        [np.full(f["n_test"], f["fold"]) for f in evaluation["folds"]]
    )
    signals = signals_by_quantile_per_fold(proba, fold_ids, coverage)
    confidence = proba.max(axis=1)

    # The sizing rules scale a stake upward from a floor. That floor used to be
    # the selection threshold; selection is now by rank, so the floor is the
    # weakest confidence that got through. Sizes then run from zero at the
    # marginal signal to full at the strongest, which is what the Kelly formula
    # is being asked to express.
    taken = confidence[signals != 0]
    threshold = float(taken.min()) if taken.size else 0.5

    # The payoff being sized: how far price actually travels over this horizon.
    forward = (bars["close"].shift(-horizon) / bars["close"] - 1).dropna()
    median_move = float(forward.abs().median())

    results = compare_sizing(
        bars, index, signals, confidence, horizon=horizon, threshold=threshold,
        median_move=median_move, kelly_scale=kelly_scale,
    )
    matched = compare_sizing_matched(
        bars, index, signals, confidence, horizon=horizon, threshold=threshold,
        median_move=median_move, kelly_scale=kelly_scale,
    )
    diagnostics = confidence_diagnostics(
        bars, index, signals, confidence, horizon=horizon
    )
    report = format_sizing_comparison(
        results, symbol, horizon, median_move, matched, diagnostics
    )
    typer.echo(report)

    reports_dir = cfg.data.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (reports_dir / f"sizing_{symbol}_{interval}_h{horizon}_{stamp}.txt").write_text(
        report, encoding="utf-8"
    )


@app.command()
def execution(
    symbol: str = typer.Option("BTCUSDT", help="Symbol to evaluate."),
    interval: str = typer.Option("1h", help="Bar interval."),
    horizon: int = typer.Option(None, help="Label horizon in bars, overriding the config."),
    coverage: float = typer.Option(
        None, help="Fraction of bars to trade, overriding the config."
    ),
    maker_fee: float = typer.Option(0.0002, help="Maker fee per side."),
    n_splits: int = typer.Option(5, help="Walk-forward folds."),
    rounds: int = typer.Option(400, help="LightGBM boosting rounds."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Do limit orders beat market orders once missed fills are counted?

    The maker fee is roughly a third of the taker cost, which sharply lowers the
    break-even accuracy. The catch is that a limit order fills only when price
    comes to it, so it skips a biased sample of trades.

    Selects trades by the rank rule. The first run of this comparison used the
    probability threshold, which selected 26.6% of one fold and 0.03% of another;
    the maker/taker gap it reported was measured on that lopsided sample.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    horizon = horizon or cfg.labels.horizon_bars.get(interval, 24)
    coverage = coverage if coverage is not None else cfg.strategy.signal_coverage

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
        config=TrainConfig(num_boost_round=rounds),
    )
    index = dataset.index[-evaluation["n_test_total"] :]
    fold_ids = np.concatenate(
        [np.full(f["n_test"], f["fold"]) for f in evaluation["folds"]]
    )
    signals = signals_by_quantile_per_fold(evaluation["proba"], fold_ids, coverage)

    costs = CostModel(
        taker_fee=cfg.strategy.taker_fee,
        slippage=cfg.strategy.slippage,
        funding_rate=cfg.strategy.funding_rate,
    )
    results = compare_execution(
        bars, index, signals, horizon=horizon, costs=costs, maker_fee=maker_fee
    )
    report = format_execution_comparison(results, symbol, horizon)
    typer.echo(report)

    reports_dir = cfg.data.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (reports_dir / f"execution_{symbol}_{interval}_h{horizon}_{stamp}.txt").write_text(
        report, encoding="utf-8"
    )


@app.command()
def validate(
    symbols: str = typer.Option(
        None, help="Comma-separated symbols. Defaults to the config's symbol list."
    ),
    interval: str = typer.Option("1h", help="Bar interval."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Run the frozen configuration across many symbols and report the spread.

    Nothing is tuned per symbol, so this validates a fixed hypothesis rather
    than searching for a new one. The criteria were committed before the first
    run — see docs/preregistration-multisymbol.md — and every symbol attempted
    is reported, including the ones that fail.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    frozen = FrozenConfig()
    names = [s.strip() for s in symbols.split(",")] if symbols else cfg.data.symbols

    parquet = ParquetStore(cfg.data.root / "raw")
    results = []
    for symbol in names:
        typer.echo(f"  {symbol} ...")
        bars = parquet.read("klines", symbol, interval)
        if bars.empty:
            results.append(SymbolResult(symbol, status="skipped: no bars stored"))
            continue

        funding = parquet.read("funding", symbol, "8h")
        dataset = build_dataset(
            bars,
            interval=interval,
            horizon=frozen.horizon,
            atr_period=cfg.labels.atr_period,
            band_k=cfg.labels.band_k,
            funding=funding if not funding.empty else None,
            symbol=symbol,
            feature_config=cfg.features,
        )
        results.append(evaluate_symbol(symbol, dataset, bars, frozen))

    report = format_validation(results, frozen)
    typer.echo("\n" + report)

    reports_dir = cfg.data.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (reports_dir / f"validation_{interval}_{stamp}.txt").write_text(report, encoding="utf-8")


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
