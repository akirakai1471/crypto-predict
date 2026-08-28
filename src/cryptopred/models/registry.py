"""Versioned model storage.

Every trained model is saved with the exact feature list it expects, its
calibrators, its config, and the metrics it earned. The serving layer loads
from here and nowhere else, so a model can never be deployed without the
metadata that says whether it deserved to be.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression

    from cryptopred.models.train import FoldResult, TrainConfig

N_CLASSES = 3


@dataclass
class ModelBundle:
    """A model plus everything needed to reproduce its predictions."""

    booster: lgb.Booster
    calibrators: list[IsotonicRegression] | None
    features: list[str]
    metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise KeyError(f"input is missing model features: {', '.join(missing)}")

        raw = np.asarray(self.booster.predict(frame[self.features]))
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if not self.calibrators:
            return raw

        calibrated = np.column_stack(
            [self.calibrators[cls].predict(raw[:, cls]) for cls in range(N_CLASSES)]
        )
        total = calibrated.sum(axis=1, keepdims=True)
        return np.divide(
            calibrated,
            total,
            out=np.full_like(calibrated, 1 / N_CLASSES),
            where=total > 0,
        )


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(
        self,
        result: FoldResult,
        symbol: str,
        interval: str,
        metrics: dict[str, Any],
        config: TrainConfig,
        n_train_rows: int | None = None,
        margin_cutoff: float | None = None,
        signal_coverage: float | None = None,
    ) -> str:
        # Millisecond precision so two saves in the same second stay distinct.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")[:-3]
        version = f"{symbol}_{interval}_{stamp}"
        directory = self.root / version
        directory.mkdir(parents=True, exist_ok=True)

        result.booster.save_model(str(directory / "model.txt"))
        if result.calibrators is not None:
            with (directory / "calibrators.pkl").open("wb") as fh:
                pickle.dump(result.calibrators, fh)

        metadata = {
            "version": version,
            "symbol": symbol,
            "interval": interval,
            "created_at": datetime.now(UTC).isoformat(),
            "features": result.features,
            "n_features": len(result.features),
            "n_train_rows": n_train_rows,
            # The trading rule travels with the model. A live bar cannot be
            # ranked against a distribution it does not have, so the rank rule
            # has to arrive as a concrete number computed when the distribution
            # was available.
            "margin_cutoff": margin_cutoff,
            "signal_coverage": signal_coverage,
            "metrics": metrics,
            "config": asdict(config),
            "calibrated": result.calibrators is not None,
            "top_features": dict(
                sorted(result.importance.items(), key=lambda kv: kv[1], reverse=True)[:20]
            ),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        # Guarantee a distinct timestamp for a subsequent save in the same tick.
        time.sleep(0.002)
        return version

    def load(self, version: str) -> ModelBundle:
        directory = self.root / version
        if not directory.exists():
            raise FileNotFoundError(f"no model version {version!r} under {self.root}")

        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(directory / "model.txt"))

        calibrators = None
        calibrator_path = directory / "calibrators.pkl"
        if calibrator_path.exists():
            with calibrator_path.open("rb") as fh:
                calibrators = pickle.load(fh)

        return ModelBundle(
            booster=booster,
            calibrators=calibrators,
            features=metadata["features"],
            metadata=metadata,
        )

    def list_versions(self, symbol: str | None = None, interval: str | None = None) -> list[str]:
        if not self.root.exists():
            return []
        prefix = ""
        if symbol:
            prefix = f"{symbol}_"
            if interval:
                prefix = f"{symbol}_{interval}_"
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith(prefix)
        )

    def latest(self, symbol: str, interval: str) -> str | None:
        versions = self.list_versions(symbol, interval)
        return versions[-1] if versions else None

    def index(self) -> pd.DataFrame:
        """One row per stored model, newest last."""
        rows = []
        for version in self.list_versions():
            meta_path = self.root / version / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "version": version,
                    "symbol": meta.get("symbol"),
                    "interval": meta.get("interval"),
                    "created_at": meta.get("created_at"),
                    "n_features": meta.get("n_features"),
                    "calibrated": meta.get("calibrated"),
                    **{f"metric_{k}": v for k, v in (meta.get("metrics") or {}).items()},
                }
            )
        return pd.DataFrame(rows)
