"""Turn the latest closed bar into a prediction.

The feature pipeline used here is the same code the training dataset was built
from — not a reimplementation. A separate "live" feature path is how production
predictions quietly diverge from what the model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cryptopred.config import Config
from cryptopred.features.pipeline import build_features
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.registry import ModelBundle, ModelRegistry

DOWN, FLAT, UP = 0, 1, 2

# Enough history to satisfy the longest warm-up window in the feature set
# (the 720-bar volatility percentile), with room to spare.
MIN_WARMUP_BARS = 900


@dataclass
class Prediction:
    symbol: str
    interval: str
    bar_close_time: pd.Timestamp
    proba: tuple[float, float, float]
    signal: int
    confidence: float
    close_price: float
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "bar_close_time": self.bar_close_time.isoformat(),
            "prob_down": self.proba[0],
            "prob_flat": self.proba[1],
            "prob_up": self.proba[2],
            "signal": self.signal,
            "direction": {1: "UP", -1: "DOWN", 0: "NO SIGNAL"}[self.signal],
            "confidence": self.confidence,
            "close_price": self.close_price,
            "model_version": self.model_version,
        }


class Predictor:
    def __init__(self, cfg: Config, bundle: ModelBundle, store: ParquetStore) -> None:
        self.cfg = cfg
        self.bundle = bundle
        self.store = store

    @classmethod
    def from_registry(
        cls, cfg: Config, symbol: str, interval: str, version: str | None = None
    ) -> Predictor:
        registry = ModelRegistry(cfg.data.root / "models")
        version = version or registry.latest(symbol, interval)
        if version is None:
            raise FileNotFoundError(
                f"no model in the registry for {symbol} {interval}. "
                "Train one with `cryptopred-model train --save`."
            )
        return cls(cfg, registry.load(version), ParquetStore(cfg.data.root / "raw"))

    def predict_latest(self, symbol: str, interval: str) -> Prediction | None:
        """Predict from the most recently closed bar, or None if data is short."""
        return self.predict_at(symbol, interval, upto=None)

    def predict_at(
        self, symbol: str, interval: str, upto: pd.Timestamp | None
    ) -> Prediction | None:
        """Predict the last bar at or before `upto`, or the newest bar if None.

        Used to fill gaps left by downtime. Features at a bar use only data that
        closed at or before it, so the probability here is identical to the one
        the model would have produced live — but the caller must still mark such
        a row as backfilled, because the log's value rests on being able to prove
        a row predates its outcome, not merely on the number being right.
        """
        bars = self.store.read("klines", symbol, interval)
        if upto is not None:
            bars = bars[bars.index <= upto]
        if len(bars) < MIN_WARMUP_BARS:
            return None

        window = bars.tail(MIN_WARMUP_BARS * 2)
        funding = self.store.read("funding", symbol, "8h")
        features = build_features(
            window,
            interval=interval,
            funding=funding if not funding.empty else None,
            config=self.cfg.features,
        )

        usable = features.dropna()
        if usable.empty:
            return None

        row = usable.tail(1)
        proba = self.bundle.predict(row)[0]
        return self._to_prediction(symbol, interval, window.loc[row.index[0]], proba)

    def _to_prediction(
        self, symbol: str, interval: str, bar: pd.Series, proba: np.ndarray
    ) -> Prediction:
        threshold = self.cfg.strategy.signal_threshold
        predicted = int(np.argmax(proba))
        confidence = float(np.max(proba))

        signal = 0
        if confidence >= threshold and predicted != FLAT:
            signal = 1 if predicted == UP else -1

        return Prediction(
            symbol=symbol,
            interval=interval,
            bar_close_time=bar["close_time"],
            proba=(float(proba[0]), float(proba[1]), float(proba[2])),
            signal=signal,
            confidence=confidence,
            close_price=float(bar["close"]),
            model_version=self.bundle.metadata["version"],
        )
