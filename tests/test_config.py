from pathlib import Path

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
