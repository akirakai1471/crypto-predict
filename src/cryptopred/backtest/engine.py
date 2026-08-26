"""Cost-aware backtest.

Two rules make this backtest honest, and both are enforced in code rather than
left to discipline:

1. A signal produced at the close of bar `t` is filled at the OPEN of bar `t+1`.
   Filling at the close of `t` is impossible in reality and is the single most
   common way a backtest lies.
2. Costs are not optional. Every trade pays the taker fee and slippage on both
   legs, and funding while the position is held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LONG, SHORT, FLAT = 1, -1, 0


@dataclass(frozen=True)
class CostModel:
    """Binance USDT-perpetual defaults, deliberately pessimistic."""

    taker_fee: float = 0.0005          # 0.05% per side
    slippage: float = 0.0002           # 0.02% per side
    funding_rate: float = 0.0001       # per 8h funding window, charged pro rata
    funding_interval_hours: float = 8.0

    def round_trip_cost(self) -> float:
        return 2 * (self.taker_fee + self.slippage)

    def funding_cost(self, direction: int, hours_held: float) -> float:
        """Positive value means the position paid funding."""
        windows = hours_held / self.funding_interval_hours
        return direction * self.funding_rate * windows


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    summary: dict[str, Any] = field(default_factory=dict)


def backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int,
    costs: CostModel | None = None,
) -> BacktestResult:
    """Run every signal as a fixed-horizon trade and compound the results.

    `signals` must have a `signal` column of +1 (long), -1 (short) or 0 (stand
    aside), indexed like `bars`. A signal on bar `t` enters at the open of bar
    `t+1` and exits at the open of bar `t+1+horizon`.
    """
    costs = costs or CostModel()
    aligned = signals.reindex(bars.index)["signal"].fillna(0).astype(int)

    opens = bars["open"].to_numpy()
    index = bars.index
    n = len(bars)
    bar_hours = _bar_hours(index)

    rows = []
    for position, direction in enumerate(aligned.to_numpy()):
        if direction == FLAT:
            continue
        entry_pos = position + 1
        exit_pos = position + 1 + horizon
        # Without a bar to exit on, the trade could never have been closed.
        if exit_pos >= n:
            continue

        entry_price = opens[entry_pos]
        exit_price = opens[exit_pos]
        gross = direction * (exit_price / entry_price - 1)

        hours_held = horizon * bar_hours
        funding = costs.funding_cost(direction, hours_held)
        net = gross - costs.round_trip_cost() - funding

        rows.append(
            {
                "signal_time": index[position],
                "entry_time": index[entry_pos],
                "exit_time": index[exit_pos],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "cost": costs.round_trip_cost() + funding,
                "net_return": net,
            }
        )

    trades = pd.DataFrame(rows)
    equity = _build_equity(trades, index)
    summary = _summarise(trades, equity, index)
    return BacktestResult(trades=trades, equity=equity, summary=summary)


def _bar_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    return float((index[1] - index[0]).total_seconds() / 3600.0)


def _build_equity(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Equity curve stamped at each trade's exit, forward-filled over the index."""
    equity = pd.Series(1.0, index=index, dtype="float64")
    if trades.empty:
        return equity

    ordered = trades.sort_values("exit_time")
    curve = (1 + ordered["net_return"]).cumprod()
    stamped = pd.Series(curve.to_numpy(), index=ordered["exit_time"].to_numpy())
    # Several trades can close on the same bar; keep the last.
    stamped = stamped[~stamped.index.duplicated(keep="last")]
    return stamped.reindex(index).ffill().fillna(1.0)


def _summarise(
    trades: pd.DataFrame, equity: pd.Series, index: pd.DatetimeIndex
) -> dict[str, Any]:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_return": float("nan"),
            "profit_factor": float("nan"),
            "total_return": 0.0,
        }

    wins = trades[trades["net_return"] > 0]["net_return"]
    losses = trades[trades["net_return"] < 0]["net_return"]
    gross_loss = float(-losses.sum())

    bars_per_year = 8760 / max(_bar_hours(index), 1e-9)
    stats = equity_metrics(equity, bars_per_year=bars_per_year)

    return {
        "n_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "avg_return": float(trades["net_return"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "total_cost_paid": float(trades["cost"].sum()),
        **stats,
    }


def equity_metrics(equity: pd.Series, bars_per_year: float = 8760) -> dict[str, float]:
    """Return, drawdown, and risk-adjusted stats from an equity curve."""
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}

    returns = equity.pct_change().fillna(0.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    std = returns.std(ddof=1)
    sharpe = 0.0
    if np.isfinite(std) and std > 0:
        sharpe = float(returns.mean() / std * np.sqrt(bars_per_year))

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = 0.0
    if np.isfinite(downside_std) and downside_std > 0:
        sortino = float(returns.mean() / downside_std * np.sqrt(bars_per_year))

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "sortino": sortino,
    }
