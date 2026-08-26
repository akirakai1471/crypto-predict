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

    An optional `size` column in [0, 1] scales how much of the trade's capital
    slot is used, so a rule can bet less on weaker signals. It cannot exceed 1:
    variable sizing may reduce exposure but never introduce leverage.
    """
    costs = costs or CostModel()
    aligned = signals.reindex(bars.index)["signal"].fillna(0).astype(int)
    if "size" in signals.columns:
        sizes = signals.reindex(bars.index)["size"].fillna(0.0).clip(0.0, 1.0).to_numpy()
    else:
        sizes = np.ones(len(bars))

    opens = bars["open"].to_numpy()
    index = bars.index
    n = len(bars)
    bar_hours = _bar_hours(index)

    rows = []
    for position, direction in enumerate(aligned.to_numpy()):
        if direction == FLAT:
            continue
        size = float(sizes[position])
        if size <= 0:
            continue   # the sizing rule declined to fund this signal
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
                "size": size,
                "gross_return": gross,
                "cost": costs.round_trip_cost() + funding,
                "net_return": net,
            }
        )

    trades = pd.DataFrame(rows)
    equity, exposure = _build_equity(trades, index, opens, horizon, costs, bar_hours)
    summary = _summarise(trades, equity, index)
    summary["max_exposure"] = float(exposure.max()) if len(exposure) else 0.0
    summary["avg_exposure"] = float(exposure.mean()) if len(exposure) else 0.0
    return BacktestResult(trades=trades, equity=equity, summary=summary)


def _bar_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    return float((index[1] - index[0]).total_seconds() / 3600.0)


def _build_equity(
    trades: pd.DataFrame,
    index: pd.DatetimeIndex,
    opens: np.ndarray,
    horizon: int,
    costs: CostModel,
    bar_hours: float,
) -> tuple[pd.Series, pd.Series]:
    """Portfolio equity from per-bar exposure.

    Fixed-horizon signals overlap: with a 24-bar horizon and a signal every bar,
    24 positions are open at once. Compounding each trade's return serially, as
    if the next trade started only after the previous closed, silently multiplies
    the same capital many times over and produces impossible returns. Instead
    every trade takes a `1/horizon` slice of capital, so full exposure is at most
    100%, and the equity curve compounds the portfolio's return bar by bar.

    Returns (equity, exposure).
    """
    n = len(index)
    equity = pd.Series(1.0, index=index, dtype="float64")
    exposure = pd.Series(0.0, index=index, dtype="float64")
    if trades.empty or n < 2:
        return equity, exposure

    slot = 1.0 / max(horizon, 1)
    position = np.zeros(n)
    cash_flow = np.zeros(n)  # costs charged at entry, exit, and funding per bar

    per_side_cost = costs.taker_fee + costs.slippage
    funding_per_bar = costs.funding_rate * (bar_hours / costs.funding_interval_hours)

    entry_pos = index.get_indexer(trades["entry_time"])
    exit_pos = index.get_indexer(trades["exit_time"])
    directions = trades["direction"].to_numpy()
    # A trade at half size commits half the capital and pays half the fees. Using
    # the full slot here while the ledger records a smaller trade would make
    # variable sizing look free.
    trade_sizes = (
        trades["size"].to_numpy() if "size" in trades.columns else np.ones(len(trades))
    )

    for entry, exit_, direction, size in zip(
        entry_pos, exit_pos, directions, trade_sizes, strict=True
    ):
        if entry < 0 or exit_ < 0:
            continue
        weight = slot * float(size)
        position[entry:exit_] += direction * weight
        cash_flow[entry] += per_side_cost * weight
        cash_flow[exit_] += per_side_cost * weight
        # Longs pay funding, shorts receive it.
        cash_flow[entry:exit_] += direction * funding_per_bar * weight

    # Open-to-open returns: a position entered at the open of bar t is exposed to
    # the move from that open to the next one.
    bar_return = np.zeros(n)
    bar_return[:-1] = opens[1:] / opens[:-1] - 1.0

    portfolio_return = position * bar_return - cash_flow
    equity = pd.Series(np.cumprod(1.0 + portfolio_return), index=index)
    exposure = pd.Series(np.abs(position), index=index)
    return equity, exposure


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
