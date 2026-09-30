"""FastAPI application.

Every endpoint that reports an accuracy also reports the sample size it was
computed from. A rate without its denominator is the easiest way for a dashboard
to mislead its own author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from cryptopred.config import Config, load_config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.registry import ModelRegistry
from cryptopred.paper.trader import PaperTrader
from cryptopred.serve import heartbeat
from cryptopred.serve.alerts import read_log
from cryptopred.serve.drift import coverage_drift
from cryptopred.serve.predictor import Predictor
from cryptopred.serve.store import PredictionStore

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="cryptopred", version="0.1.0")

    parquet = ParquetStore(cfg.data.root / "raw")
    predictions = PredictionStore(cfg.data.root / "predictions.db")
    trader = PaperTrader(
        cfg=cfg,
        store=predictions,
        parquet=parquet,
        execution=cfg.strategy.execution_model(),
    )
    predictors: dict[tuple[str, str], Predictor] = {}

    def get_predictor(symbol: str, interval: str) -> Predictor:
        """Cached, but invalidated when a newer model is saved.

        The background runner reloads the registry every cycle. A cache that
        never expires makes the dashboard show a different model's probabilities
        from the ones being traded, with nothing on screen saying so — and the
        two disagree most right after a retrain, which is exactly when someone
        is looking.
        """
        key = (symbol, interval)
        registry = ModelRegistry(cfg.data.root / "models")
        current = registry.latest(symbol, interval)
        cached = predictors.get(key)
        if cached is not None and cached.bundle.metadata.get("version") == current:
            return cached
        try:
            predictors[key] = Predictor.from_registry(cfg, symbol, interval)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return predictors[key]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        status = []
        for symbol in cfg.data.symbols:
            for interval in cfg.data.intervals:
                bars = parquet.read("klines", symbol, interval)
                if bars.empty:
                    status.append({"symbol": symbol, "interval": interval, "bars": 0})
                    continue
                last = bars.index.max()
                # Age from the bar's CLOSE, not its open. Measuring from the open
                # makes every hourly bar look an hour staler than it is, which
                # turns a healthy feed into a permanent red light.
                last_close = (
                    bars["close_time"].max()
                    if "close_time" in bars.columns
                    else last
                )
                age = (pd.Timestamp.now(tz="UTC") - last_close).total_seconds() / 60
                status.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "bars": int(len(bars)),
                        "last_bar": last.isoformat(),
                        "last_close": last_close.isoformat(),
                        "minutes_behind": round(age, 1),
                    }
                )
        return {"ok": True, "data": status, "scheduler": _scheduler()}

    def _scheduler() -> dict[str, Any]:
        """Whether the scheduler is still completing cycles.

        Bar freshness above cannot say this on its own: the API process never
        syncs bars, so stale data looks the same whether the scheduler died or
        Binance is slow, and it takes three hours to turn red at all.
        """
        beat = heartbeat.status(cfg.data.root / "heartbeat.json")
        last = beat.get("last_cycle")
        minutes = beat.get("minutes_ago")
        return {
            "state": beat["state"],
            "detail": beat["detail"],
            "last_cycle": last.isoformat() if last is not None else None,
            "minutes_ago": round(minutes, 1) if minutes is not None else None,
        }

    def _rule(symbol: str, interval: str) -> dict[str, Any]:
        """The cutoff the live system is actually applying, if a model is loaded."""
        try:
            meta = get_predictor(symbol, interval).bundle.metadata
        except HTTPException:
            return {"margin_cutoff": None, "signal_coverage": None}
        return {
            "margin_cutoff": meta.get("margin_cutoff"),
            "signal_coverage": meta.get("signal_coverage"),
        }

    @app.get("/api/predict")
    def predict(symbol: str = "BTCUSDT", interval: str = "1h") -> dict[str, Any]:
        rule = _rule(symbol, interval)
        stored = predictions.latest_prediction(symbol, interval)
        if stored:
            # What decides the trade is the gap between the two directional
            # probabilities, not the largest of the three. Showing the maximum
            # would let a 0.45-versus-0.44 coin flip look like conviction.
            margin = abs(float(stored["prob_up"]) - float(stored["prob_down"]))
            return {"source": "log", "margin": margin, **rule, **stored}

        prediction = get_predictor(symbol, interval).predict_latest(symbol, interval)
        if prediction is None:
            raise HTTPException(status_code=503, detail="not enough bars to predict yet")
        margin = abs(prediction.proba[2] - prediction.proba[0])
        return {"source": "live", "margin": margin, **rule, **prediction.as_dict()}

    @app.get("/api/history")
    def history(
        symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 300
    ) -> dict[str, Any]:
        frame = predictions.history(symbol, interval, limit=limit)
        return {"rows": frame.to_dict(orient="records")}

    @app.get("/api/metrics")
    def metrics(symbol: str = "BTCUSDT", interval: str = "1h") -> dict[str, Any]:
        live = predictions.accuracy_summary(symbol, interval)
        paper = trader.summary(symbol)

        model_meta: dict[str, Any] = {}
        try:
            model_meta = get_predictor(symbol, interval).bundle.metadata
        except HTTPException:
            model_meta = {}

        model_metrics = model_meta.get("metrics", {}) or {}

        # Whether the model is trading at all, which zero signals cannot
        # distinguish from a quiet market. A rule that has stopped firing makes
        # every other number on this page a report about nothing.
        drift = coverage_drift(
            predictions.history(symbol, interval, limit=100_000),
            model_version=model_meta.get("version"),
            cutoff=model_meta.get("margin_cutoff"),
            target=model_meta.get("signal_coverage"),
        )
        return {
            "live": live,
            "paper": paper,
            "drift": drift,
            "model": {
                "version": model_meta.get("version"),
                "trained_at": model_meta.get("created_at"),
                "n_features": model_meta.get("n_features"),
                "backtest_metrics": model_metrics,
                "gate_decision": model_metrics.get("gate_decision"),
                "gate_reason": model_metrics.get("gate_reason"),
                # A model saved despite a failing gate says so everywhere it is
                # used. Burying that in a log file is how it gets forgotten.
                "override_reason": model_metrics.get("override_reason"),
            },
            # Repeated in the payload so no consumer can render the live numbers
            # without the caveat attached to them.
            "caveat": (
                "Chạy cấu hình cố định trên 20 coin: 6/20 đạt (30%) — mức đã ghi trước "
                "là KHÔNG KẾT LUẬN ĐƯỢC. Thêm 0.6% dữ liệu (14 ngày) làm ADA nhảy "
                "+99.8pp và FIL rơi -51.6pp, danh sách coin đạt đổi 3/8 — nên con số "
                "lợi nhuận của từng coin gần như không có ý nghĩa. Thứ vững nhất là độ "
                "chính xác hướng: trung bình 54.42%, 19/20 coin trên 50%, và nó mạnh "
                "lên khi có thêm dữ liệu. Đừng đọc bất kỳ con số lợi nhuận nào chính "
                "xác hơn mức sai số gấp đôi. Chi tiết: docs/findings.md."
            ),
        }

    @app.get("/api/candles")
    def candles(
        symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 300
    ) -> dict[str, Any]:
        bars = parquet.read("klines", symbol, interval)
        if bars.empty:
            raise HTTPException(status_code=404, detail=f"no bars for {symbol} {interval}")
        tail = bars.tail(limit)
        return {
            "rows": [
                {
                    "time": ts.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for ts, row in tail.iterrows()
            ]
        }

    @app.get("/api/paper/positions")
    def paper_positions(symbol: str = "BTCUSDT") -> dict[str, Any]:
        closed = predictions.closed_trades(symbol, limit=5_000)
        rested = 0
        if not closed.empty and "entry_was_maker" in closed.columns:
            rested = int(closed["entry_was_maker"].fillna(0).sum())
        return {
            # A resting limit is not a position yet, and showing it as one would
            # overstate what the strategy is actually holding.
            "pending": predictions.pending_orders(symbol).to_dict(orient="records"),
            "open": predictions.open_trades(symbol).to_dict(orient="records"),
            "closed": closed.head(200).to_dict(orient="records"),
            "summary": trader.summary(symbol),
            "execution": {
                "style": cfg.strategy.execution.style,
                "n_closed": int(len(closed)),
                "n_rested": rested,
                "rested_share": (rested / len(closed)) if len(closed) else None,
            },
        }

    @app.get("/api/alerts")
    def alerts(limit: int = 10) -> dict[str, Any]:
        """What the scheduler announced, read back from data/signals.log.

        The messages are served verbatim, track record and caveat included. A
        dashboard that showed only "LONG" would be the alert without the part
        that stops it reading as advice.
        """
        return {"alerts": read_log(cfg.data.root / "signals.log", limit=min(limit, 100))}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "symbols": cfg.data.symbols,
            "intervals": cfg.data.intervals,
            "horizon_bars": cfg.labels.horizon_bars,
            # The trading rule is a rank converted to a margin cutoff, stored
            # per model. The old probability threshold is no longer consulted.
            "signal_coverage": cfg.strategy.signal_coverage,
            "taker_fee": cfg.strategy.taker_fee,
            "slippage": cfg.strategy.slippage,
            "execution_style": cfg.strategy.execution.style,
            "maker_fee": cfg.strategy.execution.maker_fee,
            "limit_offset": cfg.strategy.execution.limit_offset,
            "unfilled": cfg.strategy.execution.unfilled,
        }

    @app.get("/")
    def index() -> FileResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="dashboard not built")
        return FileResponse(page)

    return app


app = create_app()
