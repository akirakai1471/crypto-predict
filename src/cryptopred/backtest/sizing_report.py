"""Compare position-sizing rules on one set of signals.

Every rule is scored on the same model, the same folds and the same signals, so
any difference comes from how much was bet rather than from what was predicted.

Fixed sizing is the benchmark. A rule that cannot beat betting the same amount
every time is not worth the extra machinery, and a rule that raises return only
by raising drawdown has not improved anything — it has just taken more risk,
which anyone can do without a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel, backtest
from cryptopred.backtest.sizing import SIZING_METHODS, describe_sizing, size_from_confidence
from cryptopred.paper.replay import SideStats, two_sided_verdict


def _side_stats(trades: pd.DataFrame, direction: int, notional: float) -> SideStats:
    if trades.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    side = trades[trades["direction"] == direction]
    if side.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    # Weight each trade's return by the capital it actually used.
    weighted = side["net_return"] * side.get("size", 1.0)
    return SideStats(
        n=int(len(side)),
        win_rate=float((side["net_return"] > 0).mean()),
        total_pnl=float(weighted.sum() * notional),
        avg_return=float(side["net_return"].mean()),
    )


def score_sizing(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    confidence: np.ndarray,
    horizon: int,
    method: str,
    threshold: float,
    median_move: float,
    costs: CostModel | None = None,
    kelly_scale: float = 0.5,
    starting_capital: float = 10_000.0,
) -> dict[str, Any]:
    """Backtest one sizing rule end to end."""
    costs = costs or CostModel()
    sizes = size_from_confidence(
        confidence,
        method=method,
        threshold=threshold,
        median_move=median_move,
        round_trip_cost=costs.round_trip_cost(),
        kelly_scale=kelly_scale,
    )
    sizes = np.where(signals != 0, sizes, 0.0)

    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals, "size": sizes}, index=index)

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

    notional = starting_capital / max(horizon, 1)
    long_s = _side_stats(base.trades, 1, notional)
    short_s = _side_stats(base.trades, -1, notional)

    return {
        "method": method,
        "n_trades": base.summary.get("n_trades", 0),
        "total_return": base.summary.get("total_return", 0.0),
        "max_drawdown": base.summary.get("max_drawdown", 0.0),
        "sharpe": base.summary.get("sharpe", 0.0),
        "sortino": base.summary.get("sortino", 0.0),
        "avg_exposure": base.summary.get("avg_exposure", 0.0),
        "doubled_cost_return": doubled.summary.get("total_return", 0.0),
        "survives_doubled_costs": bool(doubled.summary.get("total_return", 0) > 0),
        "two_sided": two_sided_verdict(long_s, short_s)["decision"],
        **describe_sizing(sizes, signals),
    }


def match_exposure(
    size_arrays: dict[str, np.ndarray], signals: np.ndarray
) -> dict[str, np.ndarray]:
    """Rescale every rule to deploy the same average capital.

    Without this the comparison is rigged. Kelly near break-even stakes a few
    percent per trade, so it deploys a fraction of the capital fixed sizing does
    and loses on total return for a reason that has nothing to do with how well
    it allocates. Matching average exposure isolates the only question worth
    asking: given the same capital, does betting more on stronger signals beat
    betting the same on all of them?

    Every rule is scaled to the *smallest* average stake among them, so no rule
    is scaled up past full size and none gains hidden leverage.
    """
    taken = signals != 0
    if not taken.any():
        return size_arrays

    means = {
        name: float(sizes[taken].mean())
        for name, sizes in size_arrays.items()
        if sizes[taken].mean() > 0
    }
    if not means:
        return size_arrays

    target = min(means.values())
    matched = {}
    for name, sizes in size_arrays.items():
        own = means.get(name, 0.0)
        scale = (target / own) if own > 0 else 0.0
        matched[name] = np.clip(sizes * scale, 0.0, 1.0)
    return matched


def compare_sizing(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    confidence: np.ndarray,
    horizon: int,
    threshold: float,
    median_move: float,
    costs: CostModel | None = None,
    kelly_scale: float = 0.5,
) -> list[dict[str, Any]]:
    return [
        score_sizing(
            bars, index, signals, confidence, horizon, method, threshold,
            median_move, costs, kelly_scale,
        )
        for method in SIZING_METHODS
    ]


def compare_sizing_matched(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    confidence: np.ndarray,
    horizon: int,
    threshold: float,
    median_move: float,
    costs: CostModel | None = None,
    kelly_scale: float = 0.5,
    starting_capital: float = 10_000.0,
) -> list[dict[str, Any]]:
    """Compare the rules with average exposure held equal across all of them."""
    costs = costs or CostModel()
    raw = {
        method: np.where(
            signals != 0,
            size_from_confidence(
                confidence, method=method, threshold=threshold,
                median_move=median_move, round_trip_cost=costs.round_trip_cost(),
                kelly_scale=kelly_scale,
            ),
            0.0,
        )
        for method in SIZING_METHODS
    }
    matched = match_exposure(raw, signals)

    window = bars.loc[index.min() : index.max()]
    notional = starting_capital / max(horizon, 1)
    results = []

    for method, sizes in matched.items():
        frame = pd.DataFrame({"signal": signals, "size": sizes}, index=index)
        base = backtest(window, frame, horizon=horizon, costs=costs)
        doubled = backtest(
            window, frame, horizon=horizon,
            costs=CostModel(
                taker_fee=costs.taker_fee * 2,
                slippage=costs.slippage * 2,
                funding_rate=costs.funding_rate,
            ),
        )
        long_s = _side_stats(base.trades, 1, notional)
        short_s = _side_stats(base.trades, -1, notional)
        results.append(
            {
                "method": method,
                "n_trades": base.summary.get("n_trades", 0),
                "total_return": base.summary.get("total_return", 0.0),
                "max_drawdown": base.summary.get("max_drawdown", 0.0),
                "sharpe": base.summary.get("sharpe", 0.0),
                "sortino": base.summary.get("sortino", 0.0),
                "avg_exposure": base.summary.get("avg_exposure", 0.0),
                "doubled_cost_return": doubled.summary.get("total_return", 0.0),
                "survives_doubled_costs": bool(doubled.summary.get("total_return", 0) > 0),
                "two_sided": two_sided_verdict(long_s, short_s)["decision"],
                **describe_sizing(sizes, signals),
            }
        )
    return results


def confidence_diagnostics(
    bars: pd.DataFrame,
    index: pd.Index,
    signals: np.ndarray,
    confidence: np.ndarray,
    horizon: int,
    costs: CostModel | None = None,
    buckets: tuple[float, ...] = (0.6, 0.65, 0.70, 0.80, 1.01),
) -> dict[str, Any]:
    """Does confidence predict profit, among signals that already cleared the threshold?

    This is the question sizing depends on, and it is separate from whether
    confidence predicts direction overall. A threshold already keeps only the
    confident signals; if the remaining variation carries no further information,
    staking more on the top of that range is noise with extra steps.
    """
    costs = costs or CostModel()
    window = bars.loc[index.min() : index.max()]
    frame = pd.DataFrame({"signal": signals}, index=index)
    result = backtest(window, frame, horizon=horizon, costs=costs)
    if result.trades.empty:
        return {"n": 0, "table": pd.DataFrame()}

    trades = result.trades.set_index("signal_time")
    conf = pd.Series(confidence, index=index).loc[trades.index]
    net = trades["net_return"]
    move = trades["gross_return"].abs()
    correct = (trades["gross_return"] > 0).astype(float)

    grouped = pd.cut(conf, list(buckets), right=False)
    table = pd.DataFrame(
        {
            "n": net.groupby(grouped, observed=False).size(),
            "hit_rate": correct.groupby(grouped, observed=False).mean(),
            "avg_abs_move": move.groupby(grouped, observed=False).mean(),
            "avg_net": net.groupby(grouped, observed=False).mean(),
        }
    ).dropna(subset=["avg_net"])

    return {
        "n": int(len(trades)),
        "corr_net": float(conf.corr(net)),
        "corr_move": float(conf.corr(move)),
        "corr_correct": float(conf.corr(correct)),
        "table": table,
    }


def format_confidence_diagnostics(diag: dict[str, Any]) -> list[str]:
    if diag.get("n", 0) == 0:
        return []

    lines = [
        "",
        "WHY — does confidence predict profit among signals that cleared the threshold?",
        f"  corr(confidence, net return)   {diag['corr_net']:+.4f}",
        f"  corr(confidence, |move|)       {diag['corr_move']:+.4f}",
        f"  corr(confidence, was correct)  {diag['corr_correct']:+.4f}",
        "",
        f"{'confidence':>16} {'trades':>8} {'hit rate':>10} {'avg |move|':>12} {'avg net':>10}",
        "-" * 90,
    ]
    for bucket, row in diag["table"].iterrows():
        lines.append(
            f"{str(bucket):>16} {int(row['n']):>8,} {row['hit_rate']:>9.2%} "
            f"{row['avg_abs_move'] * 100:>11.3f}% {row['avg_net'] * 100:>9.3f}%"
        )

    best = diag["table"]["avg_net"].idxmax()
    top = diag["table"].index[-1]
    if best != top:
        lines += [
            "",
            f"  The most profitable bucket is {best}, not the most confident one "
            f"({top}).",
            "  Above the threshold, confidence carries no further information about",
            "  profit — which is exactly why staking by confidence cannot help.",
        ]
    return lines


def format_sizing_comparison(
    results: list[dict[str, Any]],
    symbol: str,
    horizon: int,
    median_move: float,
    matched: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    lines = [
        "=" * 90,
        f"POSITION SIZING — {symbol}, horizon {horizon} bars",
        "=" * 90,
        f"Median absolute move at this horizon: {median_move * 100:.3f}%",
        "Same model, same signals, same folds. Only the stake changes.",
        "",
        f"{'method':>11} {'trades':>8} {'return':>9} {'maxDD':>9} {'Sharpe':>8} "
        f"{'exposure':>9} {'2x cost':>9} {'unfunded':>9} {'sides':>10}",
        "-" * 90,
    ]

    for r in results:
        lines.append(
            f"{r['method']:>11} {r['n_trades']:>8,} {r['total_return']:>+8.1%} "
            f"{r['max_drawdown']:>8.1%} {r['sharpe']:>8.2f} {r['avg_exposure']:>8.1%} "
            f"{r['doubled_cost_return']:>+8.1%} {r['zero_size_share']:>8.1%} "
            f"{r['two_sided']:>10}"
        )

    if matched is not None:
        lines += [
            "",
            "EXPOSURE-MATCHED — every rule scaled to deploy the same average capital.",
            "The totals above reward whichever rule happens to bet the most; this table",
            "asks the actual question: given equal capital, does allocation matter?",
            "",
            f"{'method':>11} {'trades':>8} {'return':>9} {'maxDD':>9} {'Sharpe':>8} "
            f"{'exposure':>9} {'2x cost':>9} {'unfunded':>9} {'sides':>10}",
            "-" * 90,
        ]
        for r in matched:
            lines.append(
                f"{r['method']:>11} {r['n_trades']:>8,} {r['total_return']:>+8.1%} "
                f"{r['max_drawdown']:>8.1%} {r['sharpe']:>8.2f} {r['avg_exposure']:>8.1%} "
                f"{r['doubled_cost_return']:>+8.1%} {r['zero_size_share']:>8.1%} "
                f"{r['two_sided']:>10}"
            )

    if diagnostics is not None:
        lines += format_confidence_diagnostics(diagnostics)

    judged = matched if matched is not None else results
    label = "at equal exposure" if matched is not None else "at raw stakes"
    lines += ["", "=" * 90, f"VERDICT ({label}): {sizing_verdict(judged)}", "=" * 90]
    return "\n".join(lines)


def sizing_verdict(results: list[dict[str, Any]]) -> str:
    by_method = {r["method"]: r for r in results}
    fixed = by_method.get("fixed")
    if fixed is None:
        return "no fixed-size baseline to compare against"

    contenders = [r for r in results if r["method"] != "fixed"]
    # Better means more return AND no worse drawdown. Buying return with
    # drawdown is not an improvement, it is leverage by another name.
    winners = [
        r
        for r in contenders
        if r["total_return"] > fixed["total_return"]
        and r["max_drawdown"] >= fixed["max_drawdown"]
        and r["survives_doubled_costs"]
    ]
    if winners:
        best = max(winners, key=lambda r: r["total_return"])
        return (
            f"{best['method'].upper()} beats fixed sizing — "
            f"{best['total_return']:+.1%} vs {fixed['total_return']:+.1%} with a "
            f"drawdown of {best['max_drawdown']:.1%} vs {fixed['max_drawdown']:.1%}"
        )

    risk_adjusted = [
        r for r in contenders if r["sharpe"] > fixed["sharpe"] and r["survives_doubled_costs"]
    ]
    if risk_adjusted:
        best = max(risk_adjusted, key=lambda r: r["sharpe"])
        return (
            f"MIXED — {best['method'].upper()} has a better Sharpe "
            f"({best['sharpe']:.2f} vs {fixed['sharpe']:.2f}) on a lower return "
            f"({best['total_return']:+.1%} vs {fixed['total_return']:+.1%}). Better per "
            "unit of risk, less money. Worth it only if the drawdown was the problem"
        )
    return (
        "NO BENEFIT — no sizing rule beats betting the same amount every time. "
        "The probabilities do not separate good trades from bad ones finely enough "
        "for stake size to add anything"
    )
