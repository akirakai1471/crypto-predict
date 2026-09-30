from pathlib import Path

import pydantic
import pytest

from cryptopred.config import Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert "BTCUSDT" in cfg.data.symbols
    assert "1h" in cfg.data.intervals
    assert cfg.labels.horizon_bars["1h"] == 24
    assert cfg.labels.horizon_bars["1m"] == 5


def test_config_paths_are_absolute(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path / "data"
    assert cfg.raw_dir("klines").is_absolute()
    assert cfg.raw_dir("klines").name == "klines"


def test_yaml_override(tmp_path: Path):
    p = tmp_path / "custom.yaml"
    p.write_text("data:\n  symbols: [SOLUSDT]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.symbols == ["SOLUSDT"]
    # unspecified fields keep their defaults
    assert cfg.labels.atr_period == 14


def test_taker_style_produces_no_execution_model():
    """The default keeps the original market-order path untouched."""
    cfg = Config()
    assert cfg.strategy.execution.style == "taker"
    assert cfg.strategy.execution_model() is None


def test_maker_style_builds_a_model_quoted_from_a_live_price():
    cfg = Config()
    cfg.strategy.execution.style = "maker"
    model = cfg.strategy.execution_model()

    assert model is not None
    assert model.style == "maker"
    # A backtest may quote from the next bar's open; a live trader cannot.
    assert model.limit_reference == "signal_close"
    assert model.maker_fee < model.taker_fee + model.slippage


def test_maker_defaults_chase_rather_than_skip():
    """Skipping unfilled limits loses to market orders — see docs/findings.md."""
    assert Config().strategy.execution.unfilled == "chase"


def test_maker_defaults_require_price_to_trade_through():
    assert Config().strategy.execution.fill_buffer > 0


# -- news ---------------------------------------------------------------------


def test_a_yaml_written_before_the_news_section_existed_still_loads(tmp_path: Path):
    """Every local override predates the news pipeline. If the new section were
    required, upgrading would refuse to start the scheduler over a missing key
    that has a perfectly good default."""
    p = tmp_path / "old.yaml"
    p.write_text(
        "data:\n  symbols: [BTCUSDT]\nstrategy:\n  signal_threshold: 0.60\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.news.poll_seconds == 60
    assert cfg.news.alert_high_impact is True
    assert cfg.news.max_alerts_per_hour == 3
    assert [f.name for f in cfg.news.feeds] == [
        "CoinDesk", "Cointelegraph", "Decrypt", "The Block", "Bitcoin Magazine",
    ]


def test_the_shipped_yaml_carries_the_news_section():
    cfg = load_config()
    assert len(cfg.news.feeds) == 5
    # Plain http would let anything on the path rewrite a headline in transit.
    assert all(f.url.startswith("https://") for f in cfg.news.feeds)


def test_a_partial_news_section_keeps_the_other_defaults(tmp_path: Path):
    p = tmp_path / "partial.yaml"
    p.write_text("news:\n  poll_seconds: 120\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.news.poll_seconds == 120
    assert len(cfg.news.feeds) == 5


def test_the_poll_interval_has_a_floor(tmp_path: Path):
    """One second between polls of five publishers is a typo, not a setting."""
    p = tmp_path / "hammer.yaml"
    p.write_text("news:\n  poll_seconds: 1\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_config(p)
