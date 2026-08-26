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
    trader = PaperTrader(cfg=cfg, store=predictions, parquet=parquet)
    horizon = cfg.labels.horizon_bars.get(interval, 24)

    counts = {"bars": 0, "predictions": 0, "scored": 0, "opened": 0, "closed": 0}

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
                if trader.open_from_signal(
                    symbol=symbol,
                    interval=interval,
                    signal=prediction.signal,
                    entry_time=entry_time,
                    entry_price=float(bars["close"].iloc[-1]),
                    model_version=prediction.model_version,
                    horizon=horizon,
                ):
                    counts["opened"] += 1

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
    return counts
