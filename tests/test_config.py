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
