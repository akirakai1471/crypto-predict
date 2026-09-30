"""Training in parallel must change how long it takes and nothing else."""

import numpy as np

from cryptopred import parallel
from cryptopred.models.train import TrainConfig, walk_forward_evaluate
from tests.test_train import _learnable_dataset


def test_parallel_folds_report_exactly_what_serial_folds_do():
    """Every number downstream - the verdict, the cutoff, the backtest - is read
    off these probabilities, so "close" is not good enough."""
    df = _learnable_dataset(n=3000)
    config = TrainConfig(num_boost_round=15, calibration_splits=2)

    serial = walk_forward_evaluate(df, n_splits=3, horizon=4, config=config, n_jobs=1)
    spread = walk_forward_evaluate(df, n_splits=3, horizon=4, config=config, n_jobs=3)

    np.testing.assert_array_equal(serial["proba"], spread["proba"])
    np.testing.assert_array_equal(serial["y_true"], spread["y_true"])
    assert [f["n_train"] for f in serial["folds"]] == [f["n_train"] for f in spread["folds"]]
    assert serial["model"] == spread["model"]


def test_the_plan_never_runs_more_workers_than_tasks_or_cores(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 8)
    assert parallel.plan(0, 20) == (8, 1)  # every core, one thread each
    assert parallel.plan(0, 5) == (5, 1)  # five folds cannot use eight workers
    assert parallel.plan(2, 20) == (2, 4)  # two workers share the eight cores
    assert parallel.plan(64, 20) == (8, 1)  # asking for more than exists
    assert parallel.plan(1, 20) == (1, 8)


def test_results_come_back_in_input_order_whatever_order_they_finish_in():
    for workers in (1, 2):
        done: list[int] = []
        out = parallel.run(
            pow,
            [(2, 3), (3, 2), (10, 0)],
            workers=workers,
            on_done=lambda i, _r, d=done: d.append(i),
        )
        assert out == [8, 9, 1]
        assert sorted(done) == [0, 1, 2]
