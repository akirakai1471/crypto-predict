"""How an order actually reaches the market.

Taker orders cross the spread: they always fill, at a worse price, for a higher
fee. Maker orders post and wait: cheaper fee, better price — *if* they fill.

The "if" is the whole point, and it is not random. A buy limit posted below the
market fills only when price comes down to it, which is disproportionately the
times price keeps going down. When price jumps straight up — the move a long
signal most wants — the order never fills and the trade never happens. Maker
execution therefore removes a biased sample of trades, skewed toward the ones
that would have won.

Modelling the lower fee without modelling that selection produces a backtest
that is wrong in the most flattering possible direction. This module exists so
that cannot happen silently.

Fill detection uses the entry bar's high and low. That is an approximation:
within a bar we cannot know the path, so a limit inside the bar's range is
treated as filled. It errs slightly optimistic, which is worth stating rather
than hiding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LONG, SHORT = 1, -1


@dataclass(frozen=True)
class ExecutionModel:
    """Fees and fill behaviour for one execution style."""

    style: str = "taker"                 # "taker" or "maker"
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    slippage: float = 0.0002             # crossing the spread; makers do not pay it
    # How far inside the market to post, as a fraction of price. Larger means a
    # better fill price and a lower chance of filling at all.
    limit_offset: float = 0.0005
    # What to do when a limit does not fill: "skip" the trade entirely, or
    # "chase" it with a market order at the bar's close.
    unfilled: str = "skip"
    # How far price must trade THROUGH the limit before the fill is believed.
    # Zero means a touch is enough, which is optimistic: at the exact extreme of
    # a bar there may be little volume and a long queue ahead of you. Raising
    # this is the honest stress test for a maker strategy, because it attacks the
    # assumption the backtest cannot verify from OHLC data.
    fill_buffer: float = 0.0
    # Which price the limit is quoted from. "entry_open" uses the open of the
    # bar the order works in, which a backtest knows but a live trader does not:
    # at the moment of posting, that bar has not started. "signal_close" uses
    # the last price actually observable when the order is sent. Crypto trades
    # continuously so the two are close, but only one of them is reproducible
    # live, and the paper trader uses that one.
    limit_reference: str = "entry_open"

    def taker_cost(self) -> float:
        return self.taker_fee + self.slippage

    def maker_cost(self) -> float:
        return self.maker_fee


@dataclass(frozen=True)
class Fill:
    filled: bool
    price: float
    cost: float          # fractional cost of this leg
    was_maker: bool


def taker_fill(price: float, direction: int, model: ExecutionModel) -> Fill:
    """A market order always fills, at a price worsened by slippage."""
    slipped = price * (1 + direction * model.slippage)
    return Fill(filled=True, price=slipped, cost=model.taker_fee, was_maker=False)


def maker_entry_fill(
    open_price: float,
    high: float,
    low: float,
    close: float,
    direction: int,
    model: ExecutionModel,
    reference_price: float | None = None,
) -> Fill:
    """Try to enter with a limit posted inside the market.

    Long posts below the reference and fills only if the bar trades down to it.
    Short posts above and fills only if the bar trades up to it.

    `reference_price` overrides the bar open, so a caller can quote the limit
    from the last price it could actually see when sending the order.
    """
    reference = open_price if reference_price is None else reference_price
    if direction == LONG:
        limit_price = reference * (1 - model.limit_offset)
        hit = low <= limit_price * (1 - model.fill_buffer)
    else:
        limit_price = reference * (1 + model.limit_offset)
        hit = high >= limit_price * (1 + model.fill_buffer)

    if hit:
        return Fill(filled=True, price=limit_price, cost=model.maker_fee, was_maker=True)

    if model.unfilled == "chase":
        # Give up on the better price and cross the spread at the bar's close.
        return taker_fill(close, direction, model)

    return Fill(filled=False, price=float("nan"), cost=0.0, was_maker=False)


def maker_exit_fill(
    open_price: float,
    high: float,
    low: float,
    close: float,
    direction: int,
    model: ExecutionModel,
) -> Fill:
    """Try to exit with a limit, falling back to a market order at the close.

    An exit cannot simply be skipped — the position exists and must be closed —
    so an unfilled exit limit always becomes a market order. That asymmetry is
    real: makers can decline to enter, but not to leave.
    """
    exit_direction = -direction
    if direction == LONG:
        limit_price = open_price * (1 + model.limit_offset)
        hit = high >= limit_price * (1 + model.fill_buffer)
    else:
        limit_price = open_price * (1 - model.limit_offset)
        hit = low <= limit_price * (1 - model.fill_buffer)

    if hit:
        return Fill(filled=True, price=limit_price, cost=model.maker_fee, was_maker=True)

    return taker_fill(close, exit_direction, model)


def entry_fill(
    open_price: float,
    high: float,
    low: float,
    close: float,
    direction: int,
    model: ExecutionModel,
    signal_close: float | None = None,
) -> Fill:
    if model.style == "taker":
        return taker_fill(open_price, direction, model)
    if model.style == "maker":
        reference = (
            signal_close
            if model.limit_reference == "signal_close" and signal_close is not None
            else None
        )
        return maker_entry_fill(
            open_price, high, low, close, direction, model, reference_price=reference
        )
    raise ValueError(f"Unknown execution style: {model.style!r}")


def exit_fill(
    open_price: float,
    high: float,
    low: float,
    close: float,
    direction: int,
    model: ExecutionModel,
) -> Fill:
    if model.style == "taker":
        return taker_fill(open_price, -direction, model)
    if model.style == "maker":
        return maker_exit_fill(open_price, high, low, close, direction, model)
    raise ValueError(f"Unknown execution style: {model.style!r}")


def fill_statistics(
    entry_filled: np.ndarray, entry_was_maker: np.ndarray, exit_was_maker: np.ndarray
) -> dict[str, float]:
    """How much of the intended maker behaviour actually happened."""
    n = len(entry_filled)
    if n == 0:
        return {"n_signals": 0, "entry_fill_rate": 0.0, "maker_leg_share": 0.0}
    filled = entry_filled.sum()
    maker_legs = entry_was_maker.sum() + exit_was_maker.sum()
    return {
        "n_signals": int(n),
        "n_filled": int(filled),
        "entry_fill_rate": float(filled / n),
        # Two legs per filled trade; how many were genuinely maker.
        "maker_leg_share": float(maker_legs / (2 * filled)) if filled else 0.0,
    }
