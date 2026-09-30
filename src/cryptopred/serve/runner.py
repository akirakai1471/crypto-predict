"""Scheduled job: sync data, predict, score, and run the paper trader.

Runs a few minutes after each bar closes rather than exactly on the boundary, so
Binance has published the finished bar before it is requested.
"""

from __future__ import annotations

import logging

import pandas as pd

from cryptopred.config import Config
from cryptopred.ingest.binance import BinanceClient
from cryptopred.ingest.cli import run_funding_ingest, run_klines_ingest
from cryptopred.ingest.storage import ParquetStore
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve import heartbeat
from cryptopred.serve.alerts import format_alert, notify, should_alert
from cryptopred.serve.gapfill import fill_gaps
from cryptopred.serve.predictor import Predictor
from cryptopred.serve.scoring import score_pending
from cryptopred.serve.store import PredictionStore
from cryptopred.timeframes import interval_to_timedelta

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
        "alerts": 0,
        "funding": 0,
    }

    sync_cfg = cfg.model_copy(deep=True)
    sync_cfg.data.intervals = [interval]
    with BinanceClient() as client:
        counts["bars"] = run_klines_ingest(sync_cfg, client, parquet)
        # The model is trained with funding features and the predictor reads
        # them from this store. Without a refresh here they froze at the last
        # manual `ingest funding`: funding_ma_21 - the top feature by gain -
        # stopped moving, and hours_since_funding climbed into the hundreds
        # against a training maximum of 8. One request per symbol per cycle.
        counts["funding"] = run_funding_ingest(sync_cfg, client, parquet)

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

                # This bar's row is already in the log by now. should_alert
                # looks only at signals before this bar, so a signal never
                # suppresses itself, and the row is unscored, so the record
                # quoted in the alert is the record as of before this bet.
                if prediction.signal != 0:
                    counts["alerts"] += _maybe_alert(
                        cfg=cfg,
                        symbol=symbol,
                        interval=interval,
                        prediction=prediction,
                        close_price=last_close,
                        horizon=horizon,
                        predictions=predictions,
                        trader=trader,
                        predictor=predictor,
                    )

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


def _maybe_alert(
    cfg: Config,
    symbol: str,
    interval: str,
    prediction,
    close_price: float,
    horizon: int,
    predictions: PredictionStore,
    trader: PaperTrader,
    predictor: Predictor,
) -> int:
    """Alert on a signal that starts a new episode. Returns 1 if it alerted.

    Wrapped so that nothing here can break the cycle. A failed notification
    should cost the user a message, not the bar's prediction and the paper trade
    that followed it.
    """
    try:
        history = predictions.history(symbol, interval, limit=100_000)
        # The horizon is counted in bars; the episode window is wall-clock time.
        # They coincide only on 1h bars.
        horizon_hours = horizon * interval_to_timedelta(interval) / pd.Timedelta(hours=1)
        if not should_alert(
            signal=prediction.signal,
            bar_close_time=pd.Timestamp(prediction.bar_close_time),
            history=history,
            horizon_hours=horizon_hours,
        ):
            return 0

        meta = predictor.bundle.metadata
        message = format_alert(
            symbol=symbol,
            signal=prediction.signal,
            bar_close_time=pd.Timestamp(prediction.bar_close_time),
            close_price=close_price,
            margin=abs(prediction.proba[2] - prediction.proba[0]),
            cutoff=float(meta.get("margin_cutoff") or 0.0),
            history=history,
            paper_equity=float(trader.summary(symbol)["equity"]),
            starting_capital=float(cfg.strategy.starting_capital),
        )
        notify(message, log_path=cfg.data.root / "signals.log")
        logger.info("%s %s: alerted on a %+d signal", symbol, interval, prediction.signal)
        return 1
    except Exception:  # noqa: BLE001 - a missed alert must not cost the cycle
        logger.exception("%s %s: could not send the signal alert", symbol, interval)
        return 0
