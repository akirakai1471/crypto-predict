"""Configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class DataConfig(BaseModel):
    """Which markets to download and where to put the files."""

    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    intervals: list[str] = Field(default_factory=lambda: ["1h", "1m"])
    start: str = "2019-09-01"
    root: Path = PROJECT_ROOT / "data"


class LabelConfig(BaseModel):
    """How 'up' and 'down' are defined."""

    # 1h bars are labelled 24 bars ahead, not 4: see docs/findings.md. The 4-bar
    # horizon produces moves too small to clear a fixed 0.14% round-trip cost.
    horizon_bars: dict[str, int] = Field(default_factory=lambda: {"1h": 24, "1m": 5})
    atr_period: int = 14
    band_k: float = 0.5


class StrategyConfig(BaseModel):
    """Trading rules applied on top of model probabilities."""

    signal_threshold: float = 0.60
    taker_fee: float = 0.0005
    slippage: float = 0.0002
    funding_rate: float = 0.0001
    starting_capital: float = 10_000.0


class FeatureConfig(BaseModel):
    """Feature engine knobs."""

    zscore_window: int = 200
    mtf_rules: dict[str, list[str]] = Field(
        default_factory=lambda: {"1h": ["4h", "1D"], "1m": ["15min", "1h"]}
    )


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)

    def raw_dir(self, kind: str) -> Path:
        """Directory holding raw downloads of a given kind (klines, funding, ...)."""
        return (self.data.root / "raw" / kind).resolve()

    def dataset_dir(self) -> Path:
        return (self.data.root / "datasets").resolve()


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults for anything unspecified."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
