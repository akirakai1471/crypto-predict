import pandas as pd
from typer.testing import CliRunner

from cryptopred.config import Config
from cryptopred.dataset.cli import app, run_build
from cryptopred.ingest.storage import ParquetStore
from tests.conftest import make_ohlcv

runner = CliRunner()


def test_run_build_writes_parquet(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]

    store = ParquetStore(tmp_path / "raw")
    store.write("klines", "BTCUSDT", "1h", make_ohlcv(n=2000, seed=41))

    paths = run_build(cfg, store=store, quiet=True)

    assert len(paths) == 1
    assert paths[0].exists()
    ds = pd.read_parquet(paths[0])
    assert "label_class" in ds.columns
    assert len(ds) > 1000


def test_run_build_skips_missing_symbols(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["NOPEUSDT"]
    cfg.data.intervals = ["1h"]

    store = ParquetStore(tmp_path / "raw")
    assert run_build(cfg, store=store, quiet=True) == []


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
