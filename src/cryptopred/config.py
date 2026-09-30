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


class ExecutionConfig(BaseModel):
    """How the paper trader sends orders.

    Market orders always fill and cost more. Limit orders cost less but fill
    only when price comes to them, and they miss disproportionately when price
    runs the way the model predicted — see docs/findings.md.
    """

    style: str = "taker"              # "taker" or "maker"
    maker_fee: float = 0.0002
    limit_offset: float = 0.002       # how far inside the market to post
    unfilled: str = "chase"           # "chase" or "skip"; skip loses to taker
    fill_buffer: float = 0.0005       # price must trade through, not merely touch


class StrategyConfig(BaseModel):
    """Trading rules applied on top of model probabilities."""

    # Fraction of bars to trade, applied by rank. A fixed probability threshold
    # was used here until it turned out to select 26.6% of one fold and 0.03% of
    # another with the same model — see docs/findings.md. Rank is invariant to
    # the probability scale; a threshold is not.
    signal_coverage: float = 0.08
    # Kept only so older reports and the dashboard keep rendering. Nothing in
    # the trading path reads it any more.
    signal_threshold: float = 0.60
    taker_fee: float = 0.0005
    slippage: float = 0.0002
    funding_rate: float = 0.0001
    starting_capital: float = 10_000.0
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    def execution_model(self):
        """Build the backtest execution model this config describes.

        Returns None for taker, which keeps the original market-order path.
        """
        if self.execution.style != "maker":
            return None
        from cryptopred.backtest.execution import ExecutionModel

        return ExecutionModel(
            style="maker",
            taker_fee=self.taker_fee,
            maker_fee=self.execution.maker_fee,
            slippage=self.slippage,
            limit_offset=self.execution.limit_offset,
            unfilled=self.execution.unfilled,
            fill_buffer=self.execution.fill_buffer,
            limit_reference="signal_close",
        )


class FeatureConfig(BaseModel):
    """Feature engine knobs."""

    zscore_window: int = 200
    mtf_rules: dict[str, list[str]] = Field(
        default_factory=lambda: {"1h": ["4h", "1D"], "1m": ["15min", "1h"]}
    )


class FeedConfig(BaseModel):
    """One RSS or Atom feed: a display name and the URL to poll."""

    name: str
    url: str


# Every host here has to be reachable from the machine running the scheduler.
# A feed that redirects to another host needs that host allowed too. Two URLs
# are the ones their old addresses redirect to, as measured on 2026-09-30:
# CoinDesk's ".../rss/" answers 308 to ".../rss" (a wasted round trip every
# minute), and Bitcoin Magazine's "/.rss/full/" answers 301 to PLAIN HTTP
# "http://bitcoinmagazine.com/feed", which the fetcher refuses.
DEFAULT_FEEDS: list[dict[str, str]] = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss"},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
    {"name": "The Block", "url": "https://www.theblock.co/rss.xml"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/feed"},
]


class NewsConfig(BaseModel):
    """Headline collection.

    Collected and shown, not predicted from. Free RSS keeps a few days of items,
    so there is no history to test a news feature on yet; nothing here reaches
    the model until months of point-in-time headlines exist to measure it with.
    """

    feeds: list[FeedConfig] = Field(
        default_factory=lambda: [FeedConfig(**feed) for feed in DEFAULT_FEEDS]
    )
    # Conditional GET makes an unchanged feed cost a 304 and a few hundred
    # bytes. The floor keeps a typo from polling five publishers every second.
    poll_seconds: int = Field(default=60, ge=10)
    alert_high_impact: bool = True
    max_alerts_per_hour: int = Field(default=3, ge=0)


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)

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
