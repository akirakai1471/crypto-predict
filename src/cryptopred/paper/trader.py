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

from cryptopred.backtest.execution import ExecutionModel
from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.serve.store import PredictionStore


@dataclass
class PaperTrader:
    cfg: Config
    store: PredictionStore
    parquet: ParquetStore
    # None means market orders, matching the original behaviour. A maker model
    # posts limits that may never fill — see cryptopred.backtest.execution for
    # why the missed fills matter more than the cheaper fee does.
    execution: ExecutionModel | None = None

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

    # --- maker execution -------------------------------------------------

    def post_limit(
        self,
        symbol: str,
        interval: str,
        signal: int,
        signal_time: pd.Timestamp,
        signal_close: float,
        model_version: str,
        horizon: int,
    ) -> bool:
        """Queue a limit order instead of buying at market.

        The limit is quoted from the last close, which is the newest price
        available when the order is sent. Quoting it from the next bar's open —
        as a backtest can — would be using a price that does not exist yet.
        """
        if signal == 0 or self.execution is None:
            return False
        offset = self.execution.limit_offset
        limit_price = signal_close * (1 - offset) if signal > 0 else signal_close * (1 + offset)
        return self.store.post_limit_order(
            symbol=symbol,
            interval=interval,
            direction=signal,
            signal_time=signal_time,
            limit_price=limit_price,
            size_usd=self.position_size(symbol, interval, horizon),
            model_version=model_version,
        )

    def resolve_pending(
        self, symbol: str, interval: str
    ) -> dict[str, int]:
        """Decide what happened to each resting order on the bar after it was sent.

        Filled at the limit, chased at the bar's close, or cancelled — the three
        outcomes a real maker faces. An order whose bar has not closed yet is
        left alone.
        """
        counts = {"filled": 0, "chased": 0, "cancelled": 0, "waiting": 0}
        if self.execution is None:
            return counts

        pending = self.store.pending_orders(symbol)
        if pending.empty:
            return counts

        bars = self.parquet.read("klines", symbol, interval)
        if bars.empty:
            return counts

        model = self.execution
        for _, order in pending.iterrows():
            signal_time = pd.Timestamp(order["signal_time"])
            later = bars.index[bars.index > signal_time]
            if len(later) == 0:
                counts["waiting"] += 1
                continue

            bar_time = later[0]
            bar = bars.loc[bar_time]
            direction = int(order["direction"])
            limit_price = float(order["limit_price"])

            if direction > 0:
                hit = bar["low"] <= limit_price * (1 - model.fill_buffer)
            else:
                hit = bar["high"] >= limit_price * (1 + model.fill_buffer)

            if hit:
                self.store.fill_pending(
                    int(order["id"]), bar_time, limit_price, was_maker=True
                )
                counts["filled"] += 1
            elif model.unfilled == "chase":
                # Cross the spread rather than lose the trade: the orders that
                # miss are disproportionately the ones about to work.
                chased = float(bar["close"]) * (1 + direction * model.slippage)
                self.store.fill_pending(int(order["id"]), bar_time, chased, was_maker=False)
                counts["chased"] += 1
            else:
                self.store.cancel_pending(int(order["id"]))
                counts["cancelled"] += 1

        return counts

    def _maker_exit(
        self,
        bars: pd.DataFrame,
        exit_pos: int,
        direction: int,
        model: ExecutionModel,
    ) -> tuple[float, float, bool]:
        """Exit with a limit if the bar reaches it, otherwise at market.

        An exit cannot be skipped. The position exists and has to be closed, so
        an unfilled exit limit always crosses the spread — makers can decline to
        enter, never to leave.
        """
        bar = bars.iloc[exit_pos]
        # Quoted from the previous close, the newest price available when the
        # order would be sent.
        reference = float(bars["close"].iloc[exit_pos - 1]) if exit_pos > 0 else float(bar["open"])

        if direction > 0:
            limit_price = reference * (1 + model.limit_offset)
            hit = bar["high"] >= limit_price * (1 + model.fill_buffer)
        else:
            limit_price = reference * (1 - model.limit_offset)
            hit = bar["low"] <= limit_price * (1 - model.fill_buffer)

        if hit:
            return limit_price, model.maker_fee, True

        market_price = float(bar["close"]) * (1 - direction * model.slippage)
        return market_price, model.taker_fee + model.slippage, False

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

            if self.execution is None:
                exit_price = float(bars["open"].iloc[exit_pos])
                entry_cost = exit_cost = self._side_cost
            else:
                exit_price, exit_cost, _ = self._maker_exit(
                    bars, exit_pos, direction, self.execution
                )
                # Whether the entry actually rested or crossed decides what it
                # cost. Assuming the maker fee for a chased entry would make the
                # paper ledger cheaper than the trade really was.
                entry_was_maker = bool(trade.get("entry_was_maker") or 0)
                entry_cost = (
                    self.execution.maker_fee
                    if entry_was_maker
                    else self.execution.taker_fee + self.execution.slippage
                )

            gross = direction * (exit_price / entry_price - 1.0)
            hours_held = horizon * _bar_hours(bars.index)
            funding = direction * self.cfg.strategy.funding_rate * (hours_held / 8.0)
            cost = entry_cost + exit_cost + funding
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
