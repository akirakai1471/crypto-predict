"""Compare execution styles on one set of signals.

The lower maker fee is the easy half. The hard half is that a limit order
declines to fill precisely when price runs away from it, which removes trades
that would mostly have won. Reporting the fee saving without the fill rate and
the resulting change in win rate would be the most flattering possible lie.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.backtest.execution import ExecutionModel
from cryptopred.paper.replay import SideStats, two_sided_verdict


def _side_stats(trades: pd.DataFrame, direction: int, notional: float) -> SideStats:
    if trades.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    side = trades[trades["direction"] == direction]
    if side.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    weighted = side["net_return"] * side.get("size", 1.0)
    return SideStats(
        n=int(len(side)),
        win_rate=float((side["net_return"] > 0).mean()),
        total_pnl=float(weighted.sum() * notional),
        avg_return=float(side["net_return"].mean()),
    )


def score_execution(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    horizon: int,
    label: str,
    costs: CostModel,
    execution: ExecutionModel | None,
    starting_capital: float = 10_000.0,
) -> dict[str, Any]:
    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals}, index=index)

    base = backtest(window, frame, horizon=horizon, costs=costs, execution=execution)

    # Robustness for a maker means a worse fee and a wider spread to cross when
    # the limit misses. The fill rate itself is a property of the market, not a
    # knob, so it is reported rather than stressed.
    if execution is None:
        stressed_exec = None
        stressed_costs = CostModel(
            taker_fee=costs.taker_fee * 2,
            slippage=costs.slippage * 2,
            funding_rate=costs.funding_rate,
        )
    else:
        stressed_exec = ExecutionModel(
            style=execution.style,
            taker_fee=execution.taker_fee * 2,
            maker_fee=execution.maker_fee * 2,
            slippage=execution.slippage * 2,
            limit_offset=execution.limit_offset,
            unfilled=execution.unfilled,
            # Carry the fill rule through, or the stress test would quietly
            # revert to the optimistic assumption it exists to challenge.
            fill_buffer=execution.fill_buffer,
        )
        stressed_costs = costs

    stressed = backtest(
        window, frame, horizon=horizon, costs=stressed_costs, execution=stressed_exec
    )

    notional = starting_capital / max(horizon, 1)
    long_s = _side_stats(base.trades, 1, notional)
    short_s = _side_stats(base.trades, -1, notional)

    return {
        "label": label,
        "n_signals": base.summary.get("n_signals", 0),
        "n_trades": base.summary.get("n_trades", 0),
        "fill_rate": base.summary.get("fill_rate", 1.0),
        "maker_leg_share": base.summary.get("maker_leg_share", 0.0),
        "win_rate": base.summary.get("win_rate"),
        "avg_return": base.summary.get("avg_return"),
        "total_return": base.summary.get("total_return", 0.0),
        "max_drawdown": base.summary.get("max_drawdown", 0.0),
        "sharpe": base.summary.get("sharpe", 0.0),
        "doubled_cost_return": stressed.summary.get("total_return", 0.0),
        "survives_doubled_costs": bool(stressed.summary.get("total_return", 0) > 0),
        "two_sided": two_sided_verdict(long_s, short_s)["decision"],
    }


def compare_execution(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    horizon: int,
    costs: CostModel,
    maker_fee: float = 0.0002,
    offsets: tuple[float, ...] = (0.0005, 0.001, 0.002),
    fill_buffers: tuple[float, ...] = (0.0, 0.0005),
) -> list[dict[str, Any]]:
    """Taker, then maker at several distances inside the market.

    `fill_buffers` is the honest stress test. A buffer of zero believes a fill
    whenever price touches the limit; a positive buffer requires price to trade
    through it, which is closer to what happens with a queue ahead of you. If
    the maker advantage disappears under the strict rule, it was an artefact of
    the fill assumption rather than a real edge.
    """
    results = [score_execution(bars, index, signals, horizon, "taker", costs, None)]

    for buffer in fill_buffers:
        for offset in offsets:
            for unfilled in ("skip", "chase"):
                model = ExecutionModel(
                    style="maker",
                    taker_fee=costs.taker_fee,
                    maker_fee=maker_fee,
                    slippage=costs.slippage,
                    limit_offset=offset,
                    unfilled=unfilled,
                    fill_buffer=buffer,
                )
                strict = " strict" if buffer > 0 else ""
                results.append(
                    score_execution(
                        bars,
                        index,
                        signals,
                        horizon,
                        f"maker {offset * 100:.2f}% {unfilled}{strict}",
                        costs,
                        model,
                    )
                )
    return results


def format_execution_comparison(
    results: list[dict[str, Any]], symbol: str, horizon: int
) -> str:
    lines = [
        "=" * 100,
        f"EXECUTION STYLE — {symbol}, horizon {horizon} bars",
        "=" * 100,
        "Same model, same signals. Only how the orders reach the market changes.",
        "A limit order that never fills is a trade that never happened, and it",
        "misses disproportionately when price runs the way you predicted.",
        "",
        f"{'style':>22} {'signals':>8} {'filled':>8} {'fill%':>7} {'win%':>7} "
        f"{'avg net':>9} {'return':>9} {'maxDD':>8} {'2x fee':>9} {'sides':>10}",
        "-" * 100,
    ]

    for r in results:
        win = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
        avg = f"{r['avg_return'] * 100:.3f}%" if r["avg_return"] is not None else "—"
        lines.append(
            f"{r['label']:>22} {r['n_signals']:>8,} {r['n_trades']:>8,} "
            f"{r['fill_rate']:>6.1%} {win:>7} {avg:>9} {r['total_return']:>+8.1%} "
            f"{r['max_drawdown']:>7.1%} {r['doubled_cost_return']:>+8.1%} "
            f"{r['two_sided']:>10}"
        )

    lines += ["", "=" * 100, f"VERDICT: {execution_verdict(results)}", "=" * 100]
    return "\n".join(lines)


def execution_verdict(results: list[dict[str, Any]]) -> str:
    taker = next((r for r in results if r["label"] == "taker"), None)
    if taker is None:
        return "no taker baseline to compare against"

    makers = [r for r in results if r["label"] != "taker"]
    if not makers:
        return "no maker configuration was evaluated"

    def passes(r: dict[str, Any]) -> bool:
        return (
            r["total_return"] > taker["total_return"]
            and r["survives_doubled_costs"]
            and r["two_sided"] == "TWO-SIDED"
        )

    # A maker result only counts if it also holds under the strict fill rule.
    # The optimistic rule believes a fill on a touch, which is the one assumption
    # OHLC data cannot check.
    strict = [r for r in makers if "strict" in r["label"]]
    strict_winners = [r for r in strict if passes(r)]
    loose_winners = [r for r in makers if "strict" not in r["label"] and passes(r)]

    if loose_winners and strict and not strict_winners:
        best_loose = max(loose_winners, key=lambda r: r["total_return"])
        return (
            f"FILL-ASSUMPTION ARTEFACT — {best_loose['label']} returns "
            f"{best_loose['total_return']:+.1%} when a touch counts as a fill, but no "
            "configuration survives requiring price to trade through the limit. The "
            "advantage is in the fill model, not in the market"
        )

    better = strict_winners or loose_winners
    if better:
        best = max(better, key=lambda r: r["total_return"])
        held = "and holds under the strict fill rule" if strict_winners else ""
        return (
            f"{best['label'].upper()} beats taker execution — "
            f"{best['total_return']:+.1%} against {taker['total_return']:+.1%}, "
            f"filling {best['fill_rate']:.0%} of signals {held}. Worth testing live, "
            "where real fill behaviour will differ from a bar-range approximation"
        )

    closest = max(makers, key=lambda r: r["total_return"])

    # Say which gate actually stopped it. Reporting "the fills killed it" when
    # returns in fact improved and the long/short split failed would send the
    # reader looking in the wrong place.
    beat_return = [r for r in makers if r["total_return"] > taker["total_return"]]
    if beat_return:
        best = max(beat_return, key=lambda r: r["total_return"])
        if best["two_sided"] != "TWO-SIDED":
            return (
                f"NO — execution is not the problem here. {best['label']} does improve "
                f"returns ({best['total_return']:+.1%} against {taker['total_return']:+.1%}) "
                f"and the doubled-fee cushion, but the result stays {best['two_sided']}: "
                "cheaper fees make a one-sided bet cheaper, they do not make it a "
                "prediction"
            )
        if not best["survives_doubled_costs"]:
            return (
                f"NO — {best['label']} improves the headline return "
                f"({best['total_return']:+.1%} against {taker['total_return']:+.1%}) but "
                "still does not survive doubled fees"
            )

    return (
        "NO BENEFIT — the cheaper fee does not survive the missed fills. The best "
        f"maker configuration ({closest['label']}) returns "
        f"{closest['total_return']:+.1%} against taker's {taker['total_return']:+.1%} "
        f"while filling only {closest['fill_rate']:.0%} of signals. The trades a "
        "limit order misses are the ones that were about to work"
    )
