"""Assemble every feature group into one wide frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.config import FeatureConfig
from cryptopred.features.derivatives import funding_features
from cryptopred.features.momentum import momentum_features
from cryptopred.features.mtf import mtf_features
from cryptopred.features.regime import regime_features
from cryptopred.features.structure import structure_features
from cryptopred.features.timefeat import time_features
from cryptopred.features.volatility import volatility_features
from cryptopred.features.volume import volume_features

# Higher timeframes to attach, keyed by base interval. The prefix becomes the
# column name prefix, so it must be a valid identifier fragment.
_MTF_MAP: dict[str, list[tuple[str, str]]] = {
    "1m": [("15min", "m15"), ("1h", "h1")],
    "1h": [("4h", "h4"), ("1D", "d1")],
}


def build_features(
    bars: pd.DataFrame,
    interval: str,
    funding: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute every feature for a kline frame.

    The result is indexed exactly like `bars`. Early rows contain NaN where an
    indicator's warm-up window is not yet satisfied; the dataset builder drops
    those rows rather than imputing them.
    """
    config = config or FeatureConfig()

    parts = [
        momentum_features(bars),
        volatility_features(bars),
        volume_features(bars, zscore_window=config.zscore_window),
        structure_features(bars),
        regime_features(bars),
        time_features(bars),
    ]

    for rule, prefix in _MTF_MAP.get(interval, []):
        parts.append(mtf_features(bars, rule=rule, prefix=prefix))

    if funding is not None:
        parts.append(funding_features(bars, funding))

    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype("float64")
