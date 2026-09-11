"""Evaluate the meta-labelled stack against the same gates as everything else.

A new model does not get an easier exam. It faces the doubled-cost test, the
long/short split, and a direct comparison with the primary model it is supposed
to improve — because "beats the baseline" is not the relevant question here.
The question is whether the second model earns its keep over the first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.paper.replay import SideStats, two_sided_verdict


def _side_stats_from_trades(
    trades: pd.DataFrame, direction: int, notional: float
) -> SideStats:
    """Per-side stats with PnL in USDT.

    `notional` is the capital committed per trade. Summing raw returns instead
    would produce a number in no unit at all, which is how a report ends up
    labelling a dimensionless figure "USDT" and misleading its own author.
    """
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


def score_signals(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    horizon: int,
    forward_return: np.ndarray | None = None,
    costs: CostModel | None = None,
    starting_capital: float = 10_000.0,
) -> dict[str, Any]:
    """Backtest a signal array and split the result by side."""
    costs = costs or CostModel()
    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals}, index=index)

    base = backtest(window, frame, horizon=horizon, costs=costs)
    doubled = backtest(
        window,
        frame,
        horizon=horizon,
        costs=CostModel(
            taker_fee=costs.taker_fee * 2,
            slippage=costs.slippage * 2,
            funding_rate=costs.funding_rate,
        ),
    )

    # Overlapping positions share capital: each trade gets 1/horizon of it,
    # matching how the paper trader sizes real positions.
    notional = starting_capital / max(horizon, 1)
    long_s = _side_stats_from_trades(base.trades, 1, notional)
    short_s = _side_stats_from_trades(base.trades, -1, notional)

    sign_acc = None
    if forward_return is not None:
        taken = signals != 0
        if taken.any():
            predicted = np.sign(signals[taken])
            actual = np.sign(forward_return[taken])
            sign_acc = float((predicted == actual).mean())

    return {
        "n_signals": int((signals != 0).sum()),
        "coverage": float((signals != 0).mean()),
        "sign_accuracy": sign_acc,
        "total_return": base.summary.get("total_return", 0.0),
        "max_drawdown": base.summary.get("max_drawdown", 0.0),
        "sharpe": base.summary.get("sharpe", 0.0),
        "win_rate": base.summary.get("win_rate"),
        "n_trades": base.summary.get("n_trades", 0),
        "doubled_cost_return": doubled.summary.get("total_return", 0.0),
        "survives_doubled_costs": bool(doubled.summary.get("total_return", 0) > 0),
        "long": long_s,
        "short": short_s,
        "two_sided": two_sided_verdict(long_s, short_s),
    }


def format_meta_report(
    primary: dict[str, Any],
    meta: dict[str, Any],
    symbol: str,
    horizon: int,
    config: Any,
    folds: list[dict[str, Any]],
    benchmark: dict[str, Any] | None = None,
    benchmark_threshold: float | None = None,
) -> str:
    lines = [
        "=" * 74,
        f"META-LABELLED STACK — {symbol}, horizon {horizon} bars",
        "=" * 74,
        f"Primary coverage: {config.primary_coverage:.0%} of bars  "
        f"(loose on purpose — the secondary filters)",
        f"Meta threshold:   {config.meta_threshold:.2f}",
        "",
        f"{'':<24}{'PRIMARY ALONE':>18}{'WITH META FILTER':>20}",
        "-" * 74,
        _row("signals taken", primary["n_signals"], meta["n_signals"], fmt="{:,}"),
        _row("coverage of bars", primary["coverage"], meta["coverage"], fmt="{:.1%}"),
        _row("sign accuracy", primary["sign_accuracy"], meta["sign_accuracy"], fmt="{:.2%}"),
        _row("win rate", primary["win_rate"], meta["win_rate"], fmt="{:.2%}"),
        _row("total return", primary["total_return"], meta["total_return"], fmt="{:+.1%}"),
        _row("max drawdown", primary["max_drawdown"], meta["max_drawdown"], fmt="{:.1%}"),
        _row("Sharpe", primary["sharpe"], meta["sharpe"], fmt="{:.2f}"),
        _row(
            "return at 2x costs",
            primary["doubled_cost_return"],
            meta["doubled_cost_return"],
            fmt="{:+.1%}",
        ),
        _row(
            "survives 2x costs",
            primary["survives_doubled_costs"],
            meta["survives_doubled_costs"],
            fmt="{}",
        ),
        "",
        "BY SIDE, WITH THE META FILTER",
        f"{'side':>7} {'trades':>9} {'win rate':>10} {'avg return':>12}",
        "-" * 74,
        _side_row("LONG", meta["long"]),
        _side_row("SHORT", meta["short"]),
        "",
        f"DIRECTIONAL VERDICT: {meta['two_sided']['decision']}",
        f"  {meta['two_sided']['reason']}",
        "",
        "PER FOLD",
    ]

    for fold in folds:
        if fold.get("meta_trained"):
            lines.append(
                f"  fold {fold['fold']}: meta trained on {fold['n_meta_rows']:,} simulated "
                f"trades ({fold['meta_positive_rate']:.1%} profitable), "
                f"{fold['n_primary']:,} → {fold['n_final']:,} signals"
            )
        else:
            lines.append(
                f"  fold {fold['fold']}: meta NOT trained (too few simulated trades); "
                f"primary used alone with {fold['n_primary']:,} signals"
            )

    if benchmark is not None:
        lines += [
            "",
            "BENCHMARK — one model alone under the production rule, top "
            f"{benchmark_threshold:.0%} by rank",
            f"  signals {benchmark['n_signals']:,}   "
            f"return {benchmark['total_return']:+.1%}   "
            f"drawdown {benchmark['max_drawdown']:.1%}   "
            f"at 2x costs {benchmark['doubled_cost_return']:+.1%}   "
            f"{benchmark['two_sided']['decision']}",
        ]

    lines += ["", "=" * 74, f"VERDICT: {_verdict(primary, meta, benchmark)}", "=" * 74]
    return "\n".join(lines)


def _verdict(
    primary: dict[str, Any],
    meta: dict[str, Any],
    benchmark: dict[str, Any] | None = None,
) -> str:
    """Beating the primary it filters is necessary but not sufficient.

    A loose primary is easy to improve on. The question that decides whether the
    second model is worth its complexity is whether the stack beats the best
    *simple* configuration already available — otherwise the honest answer is to
    ship one model with a tighter threshold.
    """
    if not meta["survives_doubled_costs"]:
        return "NO-GO — the filtered strategy does not survive doubled costs"
    if meta["two_sided"]["decision"] == "ONE-SIDED":
        return "NO-GO — profitable on one side only, which is market drift, not prediction"
    if meta["total_return"] <= primary["total_return"]:
        return (
            "NO BENEFIT — the second model does not beat the primary it filters; "
            "the extra complexity buys nothing"
        )
    if benchmark is not None and meta["total_return"] <= benchmark["total_return"]:
        return (
            f"NO BENEFIT OVER THE SIMPLE MODEL — the stack returns "
            f"{meta['total_return']:+.1%} against {benchmark['total_return']:+.1%} for a "
            "single model at the production threshold. Two models are not worth it here"
        )
    if meta["max_drawdown"] < primary["max_drawdown"]:
        return (
            "MIXED — higher return but a deeper drawdown than the primary alone; "
            "the filter traded risk for return rather than removing bad trades"
        )
    return "IMPROVEMENT — beats both its own primary and the simple production model"


def _row(label: str, left: Any, right: Any, fmt: str) -> str:
    return f"{label:<24}{_fmt(left, fmt):>18}{_fmt(right, fmt):>20}"


def _fmt(value: Any, fmt: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    return fmt.format(value)


def _side_row(label: str, stats: SideStats) -> str:
    if stats.n == 0:
        return f"{label:>7} {0:>9} {'—':>10} {'—':>12}"
    return (
        f"{label:>7} {stats.n:>9,} {stats.win_rate:>9.2%} {stats.avg_return * 100:>11.3f}%"
    )
