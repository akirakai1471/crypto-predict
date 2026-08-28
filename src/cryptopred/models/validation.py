"""Run one frozen configuration across many symbols and report the distribution.

This is validation, not search. Every symbol gets identical settings, so there
is no best-of to pick and no parameter that could quietly be fitted to the
results. What comes out is a pass rate, and a pass rate only means something if
the criteria were fixed first — see docs/preregistration-multisymbol.md, which
was committed before this ran.

The pass criterion is deliberately not total return. Over a period when most of
crypto rose several-fold, a long-biased strategy shows a positive return on
almost any symbol while predicting nothing at all, so return alone would
guarantee a flattering answer. A symbol passes only if it also earns on the
short side, which market drift cannot manufacture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.backtest.execution import ExecutionModel
from cryptopred.models.selection import signals_by_quantile_per_fold
from cryptopred.models.train import TrainConfig, walk_forward_evaluate
from cryptopred.paper.replay import SideStats, two_sided_verdict

# Five folds need roughly two years of hourly bars to leave a usable test block.
MIN_BARS = 15_000


@dataclass(frozen=True)
class FrozenConfig:
    """The configuration under test. Nothing here varies by symbol."""

    horizon: int = 24
    # Fraction of bars to trade, applied by rank within each fold. This replaced
    # a fixed 0.60 probability threshold, which selected 26.6% of one fold and
    # 0.03% of another with the same model — see docs/findings.md.
    coverage: float = 0.08
    n_splits: int = 5
    rounds: int = 400
    # Inner folds for out-of-fold calibration. Three rather than four keeps a
    # twenty-symbol run tractable; the scale-invariance of rank selection means
    # the exact number matters far less than it did under a threshold.
    calibration_splits: int = 3
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    slippage: float = 0.0002
    funding_rate: float = 0.0001
    limit_offset: float = 0.002
    fill_buffer: float = 0.0005
    starting_capital: float = 10_000.0

    def costs(self) -> CostModel:
        return CostModel(
            taker_fee=self.taker_fee,
            slippage=self.slippage,
            funding_rate=self.funding_rate,
        )

    def execution(self) -> ExecutionModel:
        return ExecutionModel(
            style="maker",
            taker_fee=self.taker_fee,
            maker_fee=self.maker_fee,
            slippage=self.slippage,
            limit_offset=self.limit_offset,
            unfilled="chase",
            fill_buffer=self.fill_buffer,
            limit_reference="signal_close",
        )


@dataclass
class SymbolResult:
    symbol: str
    status: str                      # "tested" or a reason it could not be
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        d = self.detail
        return bool(
            self.status == "tested"
            and d.get("total_return", 0) > 0
            and d.get("survives_doubled_costs")
            and d.get("two_sided") == "TWO-SIDED"
        )


def _side_stats(trades: pd.DataFrame, direction: int, notional: float) -> SideStats:
    if trades.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    side = trades[trades["direction"] == direction]
    if side.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    return SideStats(
        n=int(len(side)),
        win_rate=float((side["net_return"] > 0).mean()),
        total_pnl=float(side["net_return"].sum() * notional),
        avg_return=float(side["net_return"].mean()),
    )


def evaluate_symbol(
    symbol: str,
    dataset: pd.DataFrame,
    bars: pd.DataFrame,
    config: FrozenConfig,
) -> SymbolResult:
    """Train and score one symbol under the frozen configuration."""
    if len(bars) < MIN_BARS:
        return SymbolResult(
            symbol,
            status=f"skipped: only {len(bars):,} bars, need {MIN_BARS:,}",
        )
    if dataset.empty:
        return SymbolResult(symbol, status="skipped: dataset empty after cleaning")

    evaluation = walk_forward_evaluate(
        dataset,
        n_splits=config.n_splits,
        horizon=config.horizon,
        config=TrainConfig(
            num_boost_round=config.rounds,
            calibration_method="oof",
            calibration_splits=config.calibration_splits,
        ),
    )
    index = dataset.index[-evaluation["n_test_total"] :]
    fold_ids = np.concatenate(
        [np.full(f["n_test"], f["fold"]) for f in evaluation["folds"]]
    )
    signals = signals_by_quantile_per_fold(
        evaluation["proba"], fold_ids, coverage=config.coverage
    )

    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals}, index=index)
    execution = config.execution()

    base = backtest(
        window, frame, horizon=config.horizon, costs=config.costs(), execution=execution
    )
    stressed = backtest(
        window,
        frame,
        horizon=config.horizon,
        costs=config.costs(),
        execution=ExecutionModel(
            style="maker",
            taker_fee=config.taker_fee * 2,
            maker_fee=config.maker_fee * 2,
            slippage=config.slippage * 2,
            limit_offset=config.limit_offset,
            unfilled="chase",
            fill_buffer=config.fill_buffer,
            limit_reference="signal_close",
        ),
    )

    notional = config.starting_capital / config.horizon
    long_s = _side_stats(base.trades, 1, notional)
    short_s = _side_stats(base.trades, -1, notional)
    verdict = two_sided_verdict(long_s, short_s)

    forward = dataset.loc[index, "forward_return"].to_numpy()
    taken = signals != 0
    sign_acc = None
    if taken.any():
        sign_acc = float((np.sign(signals[taken]) == np.sign(forward[taken])).mean())

    return SymbolResult(
        symbol,
        status="tested",
        detail={
            "n_bars": int(len(bars)),
            "n_signals": int(taken.sum()),
            "n_trades": base.summary.get("n_trades", 0),
            "sign_accuracy": sign_acc,
            "total_return": base.summary.get("total_return", 0.0),
            "max_drawdown": base.summary.get("max_drawdown", 0.0),
            "sharpe": base.summary.get("sharpe", 0.0),
            "doubled_cost_return": stressed.summary.get("total_return", 0.0),
            "survives_doubled_costs": bool(
                stressed.summary.get("total_return", 0) > 0
            ),
            "short_trades": short_s.n,
            "short_win_rate": short_s.win_rate,
            "short_pnl": short_s.total_pnl,
            "two_sided": verdict["decision"],
        },
    )


def summarise(results: list[SymbolResult]) -> dict[str, Any]:
    tested = [r for r in results if r.status == "tested"]
    skipped = [r for r in results if r.status != "tested"]
    passed = [r for r in tested if r.passed]

    # The denominator is every symbol attempted, not every symbol that happened
    # to produce a number. Quietly dropping the awkward ones is how a pass rate
    # gets inflated.
    attempted = len(results)
    return {
        "attempted": attempted,
        "tested": len(tested),
        "skipped": len(skipped),
        "passed": len(passed),
        "pass_rate_of_attempted": len(passed) / attempted if attempted else 0.0,
        "pass_rate_of_tested": len(passed) / len(tested) if tested else 0.0,
        "two_sided_count": sum(1 for r in tested if r.detail.get("two_sided") == "TWO-SIDED"),
        "positive_return_count": sum(
            1 for r in tested if r.detail.get("total_return", 0) > 0
        ),
        "survives_costs_count": sum(
            1 for r in tested if r.detail.get("survives_doubled_costs")
        ),
    }


def format_validation(results: list[SymbolResult], config: FrozenConfig) -> str:
    stats = summarise(results)
    lines = [
        "=" * 104,
        "MULTI-SYMBOL VALIDATION — one frozen configuration, no per-symbol tuning",
        "=" * 104,
        f"horizon {config.horizon} bars   top {config.coverage:.0%} by rank   "
        f"maker {config.limit_offset * 100:.2f}% chase, strict fill   "
        f"{config.n_splits} folds   {config.rounds} rounds",
        "Criteria fixed in advance: docs/preregistration-multisymbol.md",
        "",
        f"{'symbol':>10} {'bars':>8} {'signals':>8} {'sign%':>7} {'return':>9} "
        f"{'maxDD':>8} {'2x fee':>9} {'shorts':>7} {'short%':>7} {'verdict':>11} {'':>5}",
        "-" * 104,
    ]

    for r in sorted(results, key=lambda x: (x.status != "tested", x.symbol)):
        if r.status != "tested":
            lines.append(f"{r.symbol:>10}   {r.status}")
            continue
        d = r.detail
        sign = f"{d['sign_accuracy']:.1%}" if d["sign_accuracy"] is not None else "—"
        swin = f"{d['short_win_rate']:.1%}" if d["short_win_rate"] is not None else "—"
        mark = "PASS" if r.passed else ""
        lines.append(
            f"{r.symbol:>10} {d['n_bars']:>8,} {d['n_signals']:>8,} {sign:>7} "
            f"{d['total_return']:>+8.1%} {d['max_drawdown']:>7.1%} "
            f"{d['doubled_cost_return']:>+8.1%} {d['short_trades']:>7,} {swin:>7} "
            f"{d['two_sided']:>11} {mark:>5}"
        )

    lines += [
        "",
        "-" * 104,
        f"attempted {stats['attempted']}   tested {stats['tested']}   "
        f"skipped {stats['skipped']}",
        f"positive return: {stats['positive_return_count']}/{stats['tested']}   "
        f"survives 2x fees: {stats['survives_costs_count']}/{stats['tested']}   "
        f"two-sided: {stats['two_sided_count']}/{stats['tested']}",
        f"PASSED ALL THREE: {stats['passed']}/{stats['attempted']} "
        f"({stats['pass_rate_of_attempted']:.0%} of attempted)",
        "",
        "=" * 104,
        f"VERDICT: {validation_verdict(stats)}",
        "=" * 104,
    ]
    return "\n".join(lines)


def validation_verdict(stats: dict[str, Any]) -> str:
    """Apply the thresholds written down before the experiment."""
    rate = stats["pass_rate_of_attempted"]
    if stats["attempted"] == 0:
        return "nothing was attempted"

    if rate >= 0.40:
        return (
            f"SUPPORTS A REAL EFFECT — {rate:.0%} of symbols pass all three gates, "
            "above the 40% written down in advance. Encouraging, not conclusive: "
            "crypto assets move together, so these are not independent tests"
        )
    if rate <= 0.10:
        return (
            f"CONSISTENT WITH LUCK — {rate:.0%} of symbols pass, at or below the 10% "
            "written down in advance. The BTCUSDT result does not generalise, which "
            "is what a lucky draw out of a 63-configuration search looks like"
        )
    return (
        f"INCONCLUSIVE — {rate:.0%} of symbols pass, between the 10% and 40% written "
        "down in advance. Either a real effect concentrated in a subset, or a weak "
        "one near the noise floor. Reported as inconclusive rather than argued into "
        "either camp"
    )
