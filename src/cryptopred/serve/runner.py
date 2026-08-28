"""Scheduled job: sync data, predict, score, and run the paper trader.

Runs a few minutes after each bar closes rather than exactly on the boundary, so
Binance has published the finished bar before it is requested.
"""

from __future__ import annotations

import logging

import pandas as pd

from cryptopred.config import Config
from cryptopred.ingest.binance import BinanceClient
from cryptopred.ingest.cli import run_klines_ingest
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve import heartbeat
from cryptopred.serve.gapfill import fill_gaps
from cryptopred.serve.predictor import Predictor
from cryptopred.serve.scoring import score_pending
from cryptopred.serve.store import PredictionStore

logger = logging.getLogger(__name__)


def run_cycle(cfg: Config, interval: str = "1h") -> dict[str, int]:
    """One full pass: fetch new bars, predict, score old predictions, trade paper.

    Every step is idempotent, so a duplicate run wastes time but corrupts
    nothing.
    """
    parquet = ParquetStore(cfg.data.root / "raw")
    predictions = PredictionStore(cfg.data.root / "predictions.db")
    execution = cfg.strategy.execution_model()
    trader = PaperTrader(
        cfg=cfg, store=predictions, parquet=parquet, execution=execution
    )
    horizon = cfg.labels.horizon_bars.get(interval, 24)

    counts = {
        "bars": 0, "predictions": 0, "scored": 0, "opened": 0, "closed": 0,
        "filled": 0, "chased": 0, "cancelled": 0, "pending": 0,
        "backfilled": 0,
    }

    sync_cfg = cfg.model_copy(deep=True)
    sync_cfg.data.intervals = [interval]
    with BinanceClient() as client:
        counts["bars"] = run_klines_ingest(sync_cfg, client, parquet)

    for symbol in cfg.data.symbols:
        try:
            predictor = Predictor.from_registry(cfg, symbol, interval)
        except FileNotFoundError:
            logger.warning("no model for %s %s, skipping prediction", symbol, interval)
            continue

        # Bars that closed while the machine was off get a flagged row, so a
        # shutdown leaves a visible gap rather than an invisible one.
        gaps = fill_gaps(
            cfg, predictions, parquet, symbol, interval,
            horizon=horizon, predictor=predictor,
        )
        counts["backfilled"] += gaps["filled"]

        prediction = predictor.predict_latest(symbol, interval)
        if prediction is not None:
            recorded = predictions.record_prediction(
                symbol=symbol,
                interval=interval,
                bar_close_time=prediction.bar_close_time,
                proba=prediction.proba,
                signal=prediction.signal,
                close_price=prediction.close_price,
                model_version=prediction.model_version,
            )
            if recorded:
                counts["predictions"] += 1
                bars = parquet.read("klines", symbol, interval)
                entry_time = bars.index.max()
                last_close = float(bars["close"].iloc[-1])

                if execution is None:
                    if trader.open_from_signal(
                        symbol=symbol,
                        interval=interval,
                        signal=prediction.signal,
                        entry_time=entry_time,
                        entry_price=last_close,
                        model_version=prediction.model_version,
                        horizon=horizon,
                    ):
                        counts["opened"] += 1
                elif trader.post_limit(
                    symbol=symbol,
                    interval=interval,
                    signal=prediction.signal,
                    signal_time=entry_time,
                    signal_close=last_close,
                    model_version=prediction.model_version,
                    horizon=horizon,
                ):
                    counts["opened"] += 1

        # Resting orders are resolved before anything else uses positions, so
        # a limit that filled this bar is a position for the rest of the cycle.
        fills = trader.resolve_pending(symbol, interval)
        counts["filled"] += fills["filled"]
        counts["chased"] += fills["chased"]
        counts["cancelled"] += fills["cancelled"]
        counts["pending"] += fills["waiting"]

        counts["scored"] += score_pending(
            predictions,
            parquet,
            symbol,
            interval,
            horizon=horizon,
            atr_period=cfg.labels.atr_period,
            band_k=cfg.labels.band_k,
        )
        counts["closed"] += trader.close_due_trades(symbol, interval, horizon=horizon)

    logger.info("cycle complete at %s: %s", pd.Timestamp.now(tz="UTC"), counts)
    # Written last: a heartbeat should mean the cycle finished, not that it began.
    heartbeat.write(cfg.data.root / "heartbeat.json", interval, counts)
    return counts
