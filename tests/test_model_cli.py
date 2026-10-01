"""The save gate of `cryptopred-model train`."""

from typer.testing import CliRunner

from cryptopred.config import Config
from cryptopred.dataset.builder import dataset_path
from cryptopred.models import cli
from tests.test_train import _learnable_dataset


def test_no_backtest_is_not_a_strategy_pass(tmp_path, monkeypatch):
    """With no raw klines there is no backtest, and the strategy verdict
    defaulted to GO: the model was saved with `strategy_decision: GO` for a
    verdict that was never computed."""
    cfg = Config()
    cfg.data.root = tmp_path
    path = dataset_path(cfg, "BTCUSDT", "1h", 24)
    path.parent.mkdir(parents=True, exist_ok=True)
    _learnable_dataset(n=3000).to_parquet(path)
    monkeypatch.setattr(cli, "load_config", lambda _path: cfg)

    result = CliRunner().invoke(
        cli.app,
        ["train", "--save", "--rounds", "15", "--n-splits", "3", "--jobs", "1",
         "--horizon", "24"],
    )
    assert result.exit_code == 1, result.output
    assert "strategy verdict is INSUFFICIENT" in result.output
    assert not (tmp_path / "models").exists() or not any((tmp_path / "models").iterdir())


def test_a_horizon_separates_the_final_training_rows_from_the_rule_block():
    data = _learnable_dataset(n=3000)
    final_train, rule_block, verify_block = cli.final_split(data, horizon=24, holdout_bars=600)
    gap = data.index.get_loc(rule_block.index[0]) - data.index.get_loc(final_train.index[-1])
    assert gap > 24
    assert rule_block.index[-1] < verify_block.index[0]
