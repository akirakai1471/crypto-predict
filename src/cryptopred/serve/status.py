"""One command that answers: what has actually happened since it started running?

Written for the check-in after several days away. Everything here comes from the
prediction log — records written before their outcome existed — so it is the one
report in this project that hindsight cannot edit.

It reports how much evidence has accumulated as prominently as what the evidence
says, because at small sample sizes the second number is meaningless without the
first, and a dashboard that shows "60% accurate" over eleven predictions has
misled its reader.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve import heartbeat
from cryptopred.serve.store import PredictionStore

# Below this many scored signals, a hit rate is noise dressed as a result.
MIN_SIGNALS_FOR_A_CLAIM = 100


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion, valid at small n.

    The normal approximation gives nonsense near 0 and 1 and with few samples —
    exactly the situation a freshly started log is in.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def collect(cfg: Config, interval: str = "1h") -> dict[str, Any]:
    store = PredictionStore(cfg.data.root / "predictions.db")
    parquet = ParquetStore(cfg.data.root / "raw")
    trader = PaperTrader(
        cfg=cfg, store=store, parquet=parquet, execution=cfg.strategy.execution_model()
    )

    per_symbol = []
    for symbol in cfg.data.symbols:
        history = store.history(symbol, interval, limit=100_000)
        if history.empty:
            continue

        backfilled_flag = (
            history["was_backfilled"].fillna(0).astype(int)
            if "was_backfilled" in history.columns
            else pd.Series(0, index=history.index)
        )
        live = history[backfilled_flag == 0]
        backfilled = history[backfilled_flag == 1]

        scored = live[live["actual_return"].notna()]
        signals = live[live["signal"] != 0]
        scored_signals = scored[scored["signal"] != 0]
        correct_signals = int(scored_signals["is_correct"].sum()) if not scored_signals.empty else 0

        closed = store.closed_trades(symbol, limit=100_000)
        rested = (
            int(closed["entry_was_maker"].fillna(0).sum())
            if not closed.empty and "entry_was_maker" in closed.columns
            else 0
        )

        per_symbol.append(
            {
                "symbol": symbol,
                "n_predictions": int(len(history)),
                "n_live": int(len(live)),
                "n_backfilled": int(len(backfilled)),
                "n_scored": int(len(scored)),
                "n_signals": int(len(signals)),
                "n_scored_signals": int(len(scored_signals)),
                "correct_signals": correct_signals,
                "signal_accuracy": (
                    correct_signals / len(scored_signals) if len(scored_signals) else None
                ),
                "interval_95": wilson_interval(correct_signals, len(scored_signals)),
                "first_prediction": history["bar_close_time"].min(),
                "last_prediction": history["bar_close_time"].max(),
                "pending_orders": int(len(store.pending_orders(symbol))),
                "open_positions": int(len(store.open_trades(symbol))),
                "n_closed": int(len(closed)),
                "n_rested": rested,
                "paper": trader.summary(symbol),
            }
        )

    return {
        "interval": interval,
        "symbols": per_symbol,
        "heartbeat": heartbeat.status(cfg.data.root / "heartbeat.json"),
    }


def format_status(status: dict[str, Any], backtest_reference: float = 0.589) -> str:
    lines = [
        "=" * 78,
        "LIVE STATUS — from predictions written before their outcome existed",
        "=" * 78,
    ]

    # Liveness first. Every number below is meaningless if nothing is still
    # writing them, and a scheduler that died leaves no other visible trace.
    beat = status.get("heartbeat") or {"state": "unknown", "detail": ""}
    marker = {
        "alive": "RUNNING",
        "stale": "STOPPED",
        "never_started": "NEVER STARTED",
    }.get(beat["state"], "UNKNOWN")
    lines += ["", f"Scheduler: {marker} — {beat['detail']}"]
    if beat["state"] != "alive":
        lines.append(
            "  Nothing new is being recorded. Run run.bat and leave both windows open."
        )

    if not status["symbols"]:
        lines += ["", "Nothing recorded yet.", "=" * 78]
        return "\n".join(lines)

    for s in status["symbols"]:
        span = _span(s["first_prediction"], s["last_prediction"])
        lines += [
            "",
            f"{s['symbol']} — {s['n_predictions']:,} predictions over {span}",
            f"  scored so far:      {s['n_scored']:,} of {s['n_live']:,} live rows "
            "(the rest are waiting for their horizon)",
            f"  signals taken:      {s['n_signals']:,}"
            f"   scored: {s['n_scored_signals']:,}",
            f"  resting orders:     {s['pending_orders']:,}"
            f"   open positions: {s['open_positions']:,}",
        ]

        if s["n_backfilled"]:
            lines.append(
                f"  backfilled:         {s['n_backfilled']:,} rows written after their "
                "bar closed (machine was off)"
            )
            lines.append(
                "                      excluded from the result below — a row written "
                "after the answer existed"
            )
            lines.append(
                "                      cannot prove it was not influenced by it"
            )

        n = s["n_scored_signals"]
        if n == 0:
            lines.append("  directional result: nothing scored yet")
        else:
            low, high = s["interval_95"]
            lines.append(
                f"  directional result: {s['signal_accuracy']:.1%} "
                f"({s['correct_signals']}/{n}), 95% CI [{low:.1%}, {high:.1%}]"
            )
            if n < MIN_SIGNALS_FOR_A_CLAIM:
                lines.append(
                    f"  NOT YET MEANINGFUL — {n} scored signals. The interval above "
                    "spans most of the plausible range;"
                )
                lines.append(
                    f"  it will stay that wide until roughly "
                    f"{MIN_SIGNALS_FOR_A_CLAIM} signals have been scored."
                )
            elif low > 0.5:
                lines.append(
                    f"  The interval clears 50%. Backtest expected "
                    f"{backtest_reference:.1%}."
                )
            elif high < 0.5:
                lines.append("  The interval sits BELOW 50%. The live edge is negative.")
            else:
                lines.append(
                    "  The interval still straddles 50%: not distinguishable from chance."
                )

        paper = s["paper"]
        if paper["n_trades"]:
            lines.append(
                f"  paper: {paper['n_trades']:,} closed, equity {paper['equity']:,.0f} USDT "
                f"({paper['total_pnl_usd']:+,.0f}), win rate {paper['win_rate']:.1%}"
            )
            if s["n_closed"]:
                lines.append(
                    f"  execution: {s['n_rested']}/{s['n_closed']} entries rested at the "
                    f"limit ({s['n_rested'] / s['n_closed']:.0%}), the rest chased"
                )
        else:
            lines.append("  paper: no closed trades yet")

    lines += [
        "",
        "=" * 78,
        "Read the sample size before the percentage. A hit rate over a handful of",
        "signals is noise, and this strategy only commits on about 8% of bars —",
        "roughly two signals a day, each needing 24 hours before it can be scored.",
        "=" * 78,
    ]
    return "\n".join(lines)


def _span(first: str, last: str) -> str:
    try:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
    except (ValueError, TypeError):
        return "an unknown period"
    hours = (end - start).total_seconds() / 3600
    if hours < 48:
        return f"{hours:.0f} hours"
    return f"{hours / 24:.1f} days"
