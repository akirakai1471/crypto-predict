"""Turn walk-forward model probabilities into trades and score them after costs.

This is where the model stops being a classifier and starts being a strategy.
Everything up to here can look good while still losing money; this module is the
one that decides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.models.selection import signals_by_quantile_per_fold
from cryptopred.paper.replay import SideStats, two_sided_verdict

DOWN, FLAT, UP = 0, 1, 2


def probabilities_to_signals(
    proba: np.ndarray, index: pd.Index, threshold: float = 0.5
) -> pd.DataFrame:
    """+1 / -1 / 0 per bar, applying the confidence dead zone."""
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    signal = np.zeros(len(proba), dtype=int)
    signal[(predicted == UP) & (confidence >= threshold)] = 1
    signal[(predicted == DOWN) & (confidence >= threshold)] = -1
    return pd.DataFrame({"signal": signal, "confidence": confidence}, index=index)


def signals_by_coverage(
    evaluation: dict[str, Any], index: pd.Index, coverage: float
) -> pd.DataFrame:
    """The rank rule: trade each fold's most confident `coverage` of bars.

    Ranking is scale-invariant, so a fold whose calibrated probabilities run
    high cannot monopolise the trades. A fixed threshold is not: under one it
    produced 26.6% of one fold's bars and 0.03% of another's.
    """
    fold_ids = np.concatenate(
        [np.full(f["n_test"], f["fold"]) for f in evaluation["folds"]]
    )
    signal = signals_by_quantile_per_fold(evaluation["proba"], fold_ids, coverage)
    confidence = evaluation["proba"].max(axis=1)
    return pd.DataFrame({"signal": signal, "confidence": confidence}, index=index)


def run_strategy_backtest(
    bars: pd.DataFrame,
    evaluation: dict[str, Any],
    test_index: pd.Index,
    horizon: int,
    threshold: float = 0.5,
    costs: CostModel | None = None,
    coverage: float | None = None,
) -> dict[str, Any]:
    """Backtest the out-of-sample predictions, plus robustness variants.

    The doubled-cost run is not decoration: an edge that only survives at the
    quoted fee is an edge that will not survive a bad fill.

    Pass `coverage` to score the rank rule that actually gets deployed. Without
    it this scores the probability threshold, which the project has withdrawn:
    the report then prints a drawdown belonging to a strategy nobody trades.
    """
    costs = costs or CostModel()
    signals = (
        signals_by_coverage(evaluation, test_index, coverage)
        if coverage is not None
        else probabilities_to_signals(evaluation["proba"], test_index, threshold)
    )
    window = bars.loc[test_index.min() : test_index.max()]

    base = backtest(window, signals, horizon=horizon, costs=costs)
    doubled = backtest(
        window,
        signals,
        horizon=horizon,
        costs=CostModel(
            taker_fee=costs.taker_fee * 2,
            slippage=costs.slippage * 2,
            funding_rate=costs.funding_rate,
        ),
    )
    frictionless = backtest(
        window,
        signals,
        horizon=horizon,
        costs=CostModel(taker_fee=0.0, slippage=0.0, funding_rate=0.0),
    )

    return {
        "base": base,
        "doubled_costs": doubled,
        "frictionless": frictionless,
        "signals": signals,
        "two_sided": two_sided_verdict(
            _side_stats(base.trades, 1), _side_stats(base.trades, -1)
        ),
        "survives_doubled_costs": bool(doubled.summary.get("total_return", 0) > 0),
        "cost_drag": float(
            frictionless.summary.get("total_return", 0.0)
            - base.summary.get("total_return", 0.0)
        ),
    }


def strategy_verdict(result: dict[str, Any]) -> dict[str, str]:
    """Whether the strategy is fit to trade, which is a stricter question than
    whether the model predicts anything.

    A positive backtest at the quoted fee is not enough. Real fills are worse
    than quoted, and an edge that dies when costs double was never an edge — it
    was a fee rebate in disguise.
    """
    base = result["base"].summary
    doubled = result["doubled_costs"].summary

    total = base.get("total_return", 0.0)
    drawdown = base.get("max_drawdown", 0.0)

    if base.get("n_trades", 0) < 100:
        return {
            "decision": "INSUFFICIENT",
            "reason": f"only {base.get('n_trades', 0)} trades — too few to conclude anything",
        }
    if total <= 0:
        return {"decision": "NO-GO", "reason": f"loses money after costs ({total:.1%})"}
    if not result["survives_doubled_costs"]:
        return {
            "decision": "NO-GO",
            "reason": (
                f"profitable at quoted costs ({total:.1%}) but collapses to "
                f"{doubled.get('total_return', 0):.1%} when costs double — the edge is "
                "thinner than the fill quality it assumes"
            ),
        }
    if drawdown < -0.35:
        return {
            "decision": "NO-GO",
            "reason": f"max drawdown {drawdown:.1%} is not survivable in practice",
        }
    # A profit from one side only is a bet on the trend of the test window, and
    # the gate used to pass it: DOGEUSDT made +29.4% (+16.9% at doubled costs)
    # with its shorts losing money, and would have been saved.
    sides = result.get("two_sided")
    if sides is not None and sides["decision"] != "TWO-SIDED":
        return {
            "decision": "NO-GO",
            "reason": f"{sides['decision'].lower()}: {sides['reason']}",
        }
    return {
        "decision": "GO",
        "reason": f"profitable after costs ({total:.1%}) and survives doubled costs",
    }


def format_backtest(result: dict[str, Any], symbol: str, interval: str) -> str:
    base = result["base"]
    doubled = result["doubled_costs"]
    free = result["frictionless"]
    v = strategy_verdict(result)

    lines = [
        "=" * 68,
        f"BACKTEST — {symbol} {interval} (out-of-sample walk-forward predictions)",
        "=" * 68,
        f"Trades:            {base.summary['n_trades']:,}",
        f"Win rate:          {_pct(base.summary['win_rate'])}",
        f"Avg net return:    {_pct(base.summary['avg_return'], 4)} per trade",
        f"Profit factor:     {base.summary['profit_factor']:.3f}",
        "",
        "AFTER REAL COSTS (0.05% fee + 0.02% slippage per side, plus funding)",
        f"  total return:    {_pct(base.summary['total_return'])}",
        f"  max drawdown:    {_pct(base.summary['max_drawdown'])}",
        f"  Sharpe:          {base.summary['sharpe']:.2f}",
        f"  Sortino:         {base.summary['sortino']:.2f}",
        "",
        "ROBUSTNESS",
        f"  frictionless total return:  {_pct(free.summary['total_return'])}",
        f"  doubled-cost total return:  {_pct(doubled.summary['total_return'])}",
        f"  cost drag:                  {_pct(result['cost_drag'])}",
        f"  survives doubled costs:     {result['survives_doubled_costs']}",
        f"  sides:                      "
        f"{result['two_sided']['decision'] if result.get('two_sided') else 'n/a'}",
        "",
        "=" * 68,
        f"STRATEGY VERDICT: {v['decision']}",
        f"  {v['reason']}",
        "=" * 68,
    ]
    return "\n".join(lines)


def _side_stats(trades: pd.DataFrame, direction: int) -> SideStats:
    """Per-side record in return units; two_sided_verdict compares the sides
    with each other, so the unit only has to be the same for both."""
    if trades.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    side = trades[trades["direction"] == direction]
    if side.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    return SideStats(
        n=int(len(side)),
        win_rate=float((side["net_return"] > 0).mean()),
        total_pnl=float(side["net_return"].sum()),
        avg_return=float(side["net_return"].mean()),
    )


def _pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"
