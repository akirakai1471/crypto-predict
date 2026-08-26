import pandas as pd
from typer.testing import CliRunner

from cryptopred.config import Config
from cryptopred.ingest.cli import app, run_klines_ingest
from cryptopred.ingest.storage import ParquetStore

runner = CliRunner()


class FakeClient:
    def __init__(self, n_bars: int = 5):
        start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        self.rows = [
            [
                start_ms + i * 3_600_000, "1", "2", "0.5", str(100 + i), "10",
                start_ms + (i + 1) * 3_600_000 - 1, "100", 5, "5", "50", "0",
            ]
            for i in range(n_bars)
        ]

    def fetch_klines(self, symbol, interval, start_ms, end_ms, limit=1500):
        window = [r for r in self.rows if start_ms <= r[0] <= end_ms]
        return window[:limit]


def test_run_klines_ingest_writes_store(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]
    cfg.data.start = "2024-01-01"

    store = ParquetStore(tmp_path / "raw")
    n = run_klines_ingest(
        cfg,
        client=FakeClient(),
        store=store,
        now=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
    )

    assert n == 5
    assert len(store.read("klines", "BTCUSDT", "1h")) == 5


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "klines" in result.output
