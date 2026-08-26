"""Join features and labels into a model-ready table.

Rows with any NaN are dropped rather than imputed. Imputing a warm-up window
invents data; dropping costs a few hundred bars out of tens of thousands.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.config import FeatureConfig
from cryptopred.features.pipeline import build_features
from cryptopred.labels.barrier import make_labels

TARGET_COLUMNS = ["label", "label_class", "forward_return", "band"]
METADATA_COLUMNS = ["symbol", "interval"]

# LightGBM wants contiguous class ids starting at zero.
_CLASS_MAP = {-1.0: 0, 0.0: 1, 1.0: 2}


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Every column a model is allowed to train on."""
    excluded = set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
    return [c for c in dataset.columns if c not in excluded]


def build_dataset(
    bars: pd.DataFrame,
    interval: str,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 0.5,
    funding: pd.DataFrame | None = None,
    symbol: str | None = None,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Produce a clean training table for one symbol and interval."""
    features = build_features(
        bars, interval=interval, funding=funding, config=feature_config
    )
    labels = make_labels(bars, horizon=horizon, atr_period=atr_period, band_k=band_k)

    dataset = features.join(labels)
    dataset = dataset.dropna()
    if dataset.empty:
        return dataset

    dataset["label_class"] = dataset["label"].map(_CLASS_MAP).astype("int8")

    # 1m datasets run to millions of rows. float32 halves memory and file size
    # with no meaningful precision loss for gradient-boosted trees.
    feature_cols = [c for c in features.columns if c in dataset.columns]
    dataset[feature_cols] = dataset[feature_cols].astype("float32")

    if symbol is not None:
        dataset["symbol"] = symbol
    dataset["interval"] = interval

    return dataset
