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


def run_strategy_backtest(
    bars: pd.DataFrame,
    evaluation: dict[str, Any],
    test_index: pd.Index,
    horizon: int,
    threshold: float = 0.5,
    costs: CostModel | None = None,
) -> dict[str, Any]:
    """Backtest the out-of-sample predictions, plus robustness variants.

    The doubled-cost run is not decoration: an edge that only survives at the
    quoted fee is an edge that will not survive a bad fill.
    """
    costs = costs or CostModel()
    signals = probabilities_to_signals(evaluation["proba"], test_index, threshold)
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
        "",
        "=" * 68,
        f"STRATEGY VERDICT: {v['decision']}",
        f"  {v['reason']}",
        "=" * 68,
    ]
    return "\n".join(lines)


def _pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"
