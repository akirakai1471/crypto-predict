"""Replay historical predictions through the real paper trader.

The backtest engine and the paper trader are separate implementations of the
same idea, and separate implementations drift. This module drives the *live*
paper-trading code path — the same `PaperTrader` and SQLite store the scheduler
uses — over historical out-of-sample predictions, so the thing that will run
forward is the thing that gets measured.

It also answers a question the aggregate return hides: does the strategy make
money on both sides, or is it a long-only bet wearing a directional costume? In
a market that rose over the sample, a strategy can look profitable while every
short it takes loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve.store import PredictionStore

DOWN, FLAT, UP = 0, 1, 2


@dataclass
class SideStats:
    n: int
    win_rate: float | None
    total_pnl: float
    avg_return: float | None


def replay_predictions(
    cfg: Config,
    bars: pd.DataFrame,
    proba: np.ndarray,
    index: pd.Index,
    symbol: str,
    interval: str,
    horizon: int,
    threshold: float,
    store: PredictionStore,
    model_version: str = "replay",
) -> dict[str, Any]:
    """Drive every out-of-sample prediction through the paper trader.

    Trades are opened at the close of the signal bar and closed `horizon` bars
    later by the trader's own logic — not by a parallel calculation here.
    """
    parquet = ParquetStore(cfg.data.root / "raw")
    execution = cfg.strategy.execution_model()
    trader = PaperTrader(
        cfg=cfg, store=store, parquet=parquet, execution=execution
    )

    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    opened = 0
    for position, ts in enumerate(index):
        cls, conf = predicted[position], confidence[position]
        if cls == FLAT or conf < threshold:
            continue
        if ts not in bars.index:
            continue

        signal = 1 if cls == UP else -1
        if execution is None:
            placed = trader.open_from_signal(
                symbol=symbol,
                interval=interval,
                signal=signal,
                entry_time=ts,
                entry_price=float(bars.loc[ts, "close"]),
                model_version=model_version,
                horizon=horizon,
            )
        else:
            # Post a limit quoted from the close the model actually saw, exactly
            # as the scheduler would, then let the trader's own fill logic decide.
            placed = trader.post_limit(
                symbol=symbol,
                interval=interval,
                signal=signal,
                signal_time=ts,
                signal_close=float(bars.loc[ts, "close"]),
                model_version=model_version,
                horizon=horizon,
            )
        if placed:
            opened += 1

    fills = trader.resolve_pending(symbol, interval)
    closed = trader.close_due_trades(symbol, interval, horizon=horizon)
    trades = store.closed_trades(symbol, limit=1_000_000)

    return {
        "opened": opened,
        "closed": closed,
        "fills": fills,
        "trades": trades,
        "summary": trader.summary(symbol),
        "long": _side_stats(trades, direction=1),
        "short": _side_stats(trades, direction=-1),
        "equity_curve": _equity_curve(trades, cfg.strategy.starting_capital),
    }


def two_sided_verdict(long: SideStats, short: SideStats) -> dict[str, str]:
    """Does the strategy predict direction, or just ride the trend?

    Aggregate return cannot answer this. Over a sample where the market tripled,
    a long-only bias produces a healthy-looking equity curve and a win rate above
    50%, and the shorts quietly bleed underneath. Splitting by side is the only
    way to see it, and a strategy that only works in one direction is a bet on
    that direction, not a model.
    """
    if long.n == 0 and short.n == 0:
        return {"decision": "NO TRADES", "reason": "nothing was traded"}
    if short.n < 100:
        return {
            "decision": "UNPROVEN",
            "reason": f"only {short.n} short trades — too few to judge the short side",
        }
    if short.total_pnl < 0:
        return {
            "decision": "ONE-SIDED",
            "reason": (
                f"shorts lost {abs(short.total_pnl):,.0f} USDT at a {short.win_rate:.1%} "
                "win rate; the profit comes from the long side alone, which in a rising "
                "market is a directional bet rather than a prediction"
            ),
        }
    # Check the win rate before the contribution: "below a coin flip" is the more
    # specific diagnosis, and reporting the vaguer one would hide it.
    if short.win_rate is not None and short.win_rate < 0.5:
        return {
            "decision": "ONE-SIDED",
            "reason": (
                f"shorts win only {short.win_rate:.1%} of the time — below a coin flip — "
                f"and contribute {short.total_pnl:,.0f} USDT, which is noise, not edge"
            ),
        }
    if long.total_pnl > 0 and short.total_pnl < 0.05 * long.total_pnl:
        # Winning often while earning nothing means small wins and large losses:
        # the short side is carried by the long side, not standing on its own.
        return {
            "decision": "ONE-SIDED",
            "reason": (
                f"shorts win {short.win_rate:.1%} but contribute only "
                f"{short.total_pnl:,.0f} USDT against the long side's "
                f"{long.total_pnl:,.0f}: small wins and large losses, not an edge"
            ),
        }
    return {
        "decision": "TWO-SIDED",
        "reason": (
            f"shorts win {short.win_rate:.1%} and earn {short.total_pnl:+,.0f} USDT "
            "alongside the longs: the edge survives in both directions"
        ),
    }


def _side_stats(trades: pd.DataFrame, direction: int) -> SideStats:
    if trades.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    side = trades[trades["direction"] == direction]
    if side.empty:
        return SideStats(n=0, win_rate=None, total_pnl=0.0, avg_return=None)
    return SideStats(
        n=int(len(side)),
        win_rate=float((side["net_return"] > 0).mean()),
        total_pnl=float(side["pnl_usd"].sum()),
        avg_return=float(side["net_return"].mean()),
    )


def _equity_curve(trades: pd.DataFrame, starting_capital: float) -> pd.Series:
    """Cumulative realised PnL, stamped at each trade's exit."""
    if trades.empty:
        return pd.Series(dtype="float64")
    ordered = trades.sort_values("exit_time")
    equity = starting_capital + ordered["pnl_usd"].cumsum()
    equity.index = pd.to_datetime(ordered["exit_time"])
    return equity


def format_replay(result: dict[str, Any], symbol: str, threshold: float) -> str:
    summary = result["summary"]
    long_s, short_s = result["long"], result["short"]
    equity = result["equity_curve"]

    lines = [
        "=" * 70,
        f"PAPER REPLAY — {symbol}, threshold {threshold:.2f}",
        "=" * 70,
        "Driven through the live PaperTrader, not the backtest engine.",
        "",
        f"Orders placed:   {result['opened']:,}",
        f"Trades closed:   {result['closed']:,}",
        f"Starting equity: {summary['equity'] - summary['total_pnl_usd']:,.0f} USDT",
        f"Final equity:    {summary['equity']:,.0f} USDT",
        f"Realised PnL:    {summary['total_pnl_usd']:+,.0f} USDT",
        f"Win rate:        {_pct(summary['win_rate'])}",
        "",
        "BY SIDE — a strategy that only works long is a bull-market bet, not a model",
        f"{'side':>7} {'trades':>8} {'win rate':>10} {'avg return':>12} {'PnL USDT':>12}",
        "-" * 70,
        _side_row("LONG", long_s),
        _side_row("SHORT", short_s),
    ]

    fills = result.get("fills")
    if fills and (fills["filled"] or fills["chased"] or fills["cancelled"]):
        total = fills["filled"] + fills["chased"] + fills["cancelled"]
        lines += [
            "",
            "EXECUTION — a limit that misses is not a free option; the misses cluster",
            "on the moves the model got right",
            f"  rested at the limit: {fills['filled']:,} ({fills['filled'] / total:.1%})",
            f"  chased at market:    {fills['chased']:,} ({fills['chased'] / total:.1%})",
            f"  cancelled unfilled:  {fills['cancelled']:,}",
        ]

    if not equity.empty:
        peak = equity.cummax()
        drawdown = (equity / peak - 1).min()
        lines += [
            "",
            f"Peak equity:     {equity.max():,.0f} USDT",
            f"Trough equity:   {equity.min():,.0f} USDT",
            f"Max drawdown:    {_pct(drawdown)}",
            f"Period:          {equity.index.min():%Y-%m-%d} → {equity.index.max():%Y-%m-%d}",
        ]

    v = two_sided_verdict(long_s, short_s)
    lines += [
        "",
        "=" * 70,
        f"DIRECTIONAL VERDICT: {v['decision']}",
        f"  {v['reason']}",
        "=" * 70,
    ]
    return "\n".join(lines)


def _side_row(label: str, stats: SideStats) -> str:
    if stats.n == 0:
        return f"{label:>7} {0:>8} {'—':>10} {'—':>12} {'—':>12}"
    return (
        f"{label:>7} {stats.n:>8,} {_pct(stats.win_rate):>10} "
        f"{stats.avg_return * 100:>11.3f}% {stats.total_pnl:>+12,.0f}"
    )


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"
