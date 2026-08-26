"""Break-even accuracy: the arithmetic that decides whether a horizon is viable.

Run this BEFORE training anything on a new symbol, horizon or venue. A
directional bet with accuracy `p` on a move of size `m`, paying round-trip cost
`c`, breaks even when

    p·m − (1−p)·m = c    ⟹    p = (c/m + 1) / 2

Cost is fixed; move size grows with horizon; model accuracy is roughly flat
across horizons. So the question is never "can the model predict?" but "is the
move big enough to pay the toll?" — and that is answerable from price data
alone, with no model at all.

The 1-minute scalping horizon in this project needed 149.5% accuracy to break
even. No model could have rescued it, and a week of training was spent finding
that out the expensive way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakEven:
    horizon: int
    median_move: float
    mean_move: float
    round_trip_cost: float
    breakeven_accuracy: float

    @property
    def is_possible(self) -> bool:
        """False when break-even needs an accuracy above 100%: no model, however
        good, can trade this horizon at this cost."""
        return self.breakeven_accuracy <= 1.0

    def margin(self, accuracy: float) -> float:
        """Percentage points of accuracy to spare, negative if short."""
        return accuracy - self.breakeven_accuracy


def round_trip_cost(taker_fee: float = 0.0005, slippage: float = 0.0002) -> float:
    """Both legs pay fee and slippage."""
    return 2 * (taker_fee + slippage)


def breakeven_accuracy(move: float, cost: float) -> float:
    """Accuracy needed for a directional bet to break even.

    Returns infinity for a zero move, where no accuracy suffices.
    """
    if move <= 0:
        return float("inf")
    return (cost / move + 1) / 2


def analyse(
    bars: pd.DataFrame,
    horizons: tuple[int, ...],
    cost: float | None = None,
) -> list[BreakEven]:
    """Break-even accuracy for each horizon, measured on real price moves."""
    cost = round_trip_cost() if cost is None else cost
    close = bars["close"]
    results = []

    for horizon in horizons:
        forward = (close.shift(-horizon) / close - 1).dropna()
        if forward.empty:
            continue
        moves = forward.abs()
        median_move = float(moves.median())
        results.append(
            BreakEven(
                horizon=horizon,
                median_move=median_move,
                mean_move=float(moves.mean()),
                round_trip_cost=cost,
                breakeven_accuracy=breakeven_accuracy(median_move, cost),
            )
        )

    return results


def format_table(
    results: list[BreakEven],
    interval: str,
    symbol: str,
    measured_accuracy: float | None = None,
) -> str:
    if not results:
        return "No horizons could be measured."

    cost = results[0].round_trip_cost
    lines = [
        "=" * 74,
        f"BREAK-EVEN ANALYSIS — {symbol} {interval} bars",
        "=" * 74,
        f"Round-trip cost: {cost * 100:.3f}%  (fee and slippage, both legs)",
        "",
        f"{'horizon':>9} {'median move':>13} {'mean move':>11} {'break-even acc':>15}"
        + (f" {'margin':>9}" if measured_accuracy is not None else ""),
        "-" * 74,
    ]

    for r in results:
        acc = "impossible" if not r.is_possible else f"{r.breakeven_accuracy:.1%}"
        row = (
            f"{r.horizon:>9} {r.median_move * 100:>12.3f}% "
            f"{r.mean_move * 100:>10.3f}% {acc:>15}"
        )
        if measured_accuracy is not None:
            margin = r.margin(measured_accuracy)
            row += f" {margin * 100:>+8.1f}pp" if r.is_possible else f" {'—':>9}"
        lines.append(row)

    impossible = [r for r in results if not r.is_possible]
    if impossible:
        lines += [
            "",
            "Horizons marked impossible need accuracy above 100%: the cost of the",
            "trade exceeds the size of the move it is trying to capture. A perfect",
            "oracle loses money there. Do not train a model for them.",
        ]

    if measured_accuracy is not None:
        lines += [
            "",
            f"Margin assumes accuracy stays at {measured_accuracy:.1%} across every",
            "horizon. That figure was measured at one horizon only — treat the other",
            "rows as a guide to where to look, not as a result.",
        ]

    lines.append("=" * 74)
    return "\n".join(lines)


def summarise_moves(bars: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Distribution of absolute forward moves, for sizing decisions."""
    forward = (bars["close"].shift(-horizon) / bars["close"] - 1).dropna().abs()
    return {
        "p25": float(np.percentile(forward, 25)),
        "median": float(np.percentile(forward, 50)),
        "p75": float(np.percentile(forward, 75)),
        "p95": float(np.percentile(forward, 95)),
        "mean": float(forward.mean()),
    }
