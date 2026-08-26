"""Paper trading: the strategy run forward on money that does not exist.

Nothing here touches an exchange account. There is no API key, no order
placement, and no code path that could place one. That is deliberate: the
backtest in docs/findings.md found one promising configuration out of 28
examined, which is roughly what chance produces. Paper trading is how that
result gets tested on bars nobody has seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.store import PredictionStore


@dataclass
class PaperTrader:
    cfg: Config
    store: PredictionStore
    parquet: ParquetStore

    @property
    def _side_cost(self) -> float:
        return self.cfg.strategy.taker_fee + self.cfg.strategy.slippage

    def position_size(self, symbol: str, interval: str, horizon: int) -> float:
        """Capital per trade.

        Positions overlap with a multi-bar horizon, so each trade may claim only
        `1/horizon` of capital. Sizing every trade at full capital would silently
        run leverage of `horizon`x — the same mistake that made the first
        backtest report billions of percent.
        """
        return self.cfg.strategy.starting_capital / max(horizon, 1)

    def open_from_signal(
        self,
        symbol: str,
        interval: str,
        signal: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        model_version: str,
        horizon: int,
    ) -> bool:
        if signal == 0:
            return False
        return self.store.open_trade(
            symbol=symbol,
            interval=interval,
            direction=signal,
            entry_time=entry_time,
            entry_price=entry_price,
            size_usd=self.position_size(symbol, interval, horizon),
            model_version=model_version,
        )

    def close_due_trades(
        self, symbol: str, interval: str, horizon: int, now: pd.Timestamp | None = None
    ) -> int:
        """Close every open trade that has reached its horizon."""
        open_trades = self.store.open_trades(symbol)
        if open_trades.empty:
            return 0

        bars = self.parquet.read("klines", symbol, interval)
        if bars.empty:
            return 0

        closed = 0
        for _, trade in open_trades.iterrows():
            entry_time = pd.Timestamp(trade["entry_time"])
            matches = bars.index[bars.index >= entry_time]
            if len(matches) == 0:
                continue

            entry_pos = bars.index.get_loc(matches[0])
            exit_pos = entry_pos + horizon
            if exit_pos >= len(bars):
                continue

            direction = int(trade["direction"])
            entry_price = float(trade["entry_price"])
            exit_price = float(bars["open"].iloc[exit_pos])

            gross = direction * (exit_price / entry_price - 1.0)
            hours_held = horizon * _bar_hours(bars.index)
            funding = direction * self.cfg.strategy.funding_rate * (hours_held / 8.0)
            cost = 2 * self._side_cost + funding
            net = gross - cost

            self.store.close_trade(
                trade_id=int(trade["id"]),
                exit_time=bars.index[exit_pos],
                exit_price=exit_price,
                gross_return=gross,
                cost=cost,
                net_return=net,
                pnl_usd=net * float(trade["size_usd"]),
            )
            closed += 1

        return closed

    def summary(self, symbol: str | None = None) -> dict[str, float | int | None]:
        """Realised paper performance. Open trades are excluded — an unrealised
        position is not a result."""
        closed = self.store.closed_trades(symbol, limit=100_000)
        if closed.empty:
            return {
                "n_trades": 0,
                "total_pnl_usd": 0.0,
                "win_rate": None,
                "avg_net_return": None,
                "equity": self.cfg.strategy.starting_capital,
            }

        wins = int((closed["net_return"] > 0).sum())
        total_pnl = float(closed["pnl_usd"].sum())
        return {
            "n_trades": int(len(closed)),
            "total_pnl_usd": total_pnl,
            "win_rate": wins / len(closed),
            "avg_net_return": float(closed["net_return"].mean()),
            "equity": self.cfg.strategy.starting_capital + total_pnl,
        }


def _bar_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    return float((index[1] - index[0]).total_seconds() / 3600.0)
