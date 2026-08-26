import pandas as pd
import pytest

from cryptopred.models.splits import PurgedWalkForward


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=10_000, freq="1h", tz="UTC")


def test_produces_requested_number_of_folds(index):
    cv = PurgedWalkForward(n_splits=5, horizon=4, embargo_frac=0.01)
    folds = list(cv.split(index))
    assert len(folds) == 5


def test_train_always_precedes_test(index):
    cv = PurgedWalkForward(n_splits=5, horizon=4, embargo_frac=0.01)
    for train_idx, test_idx in cv.split(index):
        assert index[train_idx].max() < index[test_idx].min()


def test_purge_gap_is_at_least_horizon(index):
    """Labels span `horizon` bars, so the last `horizon` training bars overlap
    the test window in time and must be removed."""
    horizon = 8
    cv = PurgedWalkForward(n_splits=4, horizon=horizon, embargo_frac=0.0)
    step = index.freq * horizon
    for train_idx, test_idx in cv.split(index):
        gap = index[test_idx].min() - index[train_idx].max()
        assert gap >= step


def test_embargo_shrinks_training_set(index):
    small = PurgedWalkForward(n_splits=4, horizon=4, embargo_frac=0.0)
    large = PurgedWalkForward(n_splits=4, horizon=4, embargo_frac=0.05)
    small_sizes = [len(tr) for tr, _ in small.split(index)]
    large_sizes = [len(tr) for tr, _ in large.split(index)]
    assert sum(large_sizes) < sum(small_sizes)


def test_test_windows_do_not_overlap(index):
    cv = PurgedWalkForward(n_splits=5, horizon=4, embargo_frac=0.01)
    windows = [(index[te].min(), index[te].max()) for _, te in cv.split(index)]
    for earlier, later in zip(windows[:-1], windows[1:], strict=True):
        assert earlier[1] < later[0]


def test_training_set_grows_with_each_fold(index):
    """Walk-forward is expanding-window: later folds train on more history."""
    cv = PurgedWalkForward(n_splits=5, horizon=4, embargo_frac=0.01)
    sizes = [len(tr) for tr, _ in cv.split(index)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rejects_too_few_samples():
    tiny = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    cv = PurgedWalkForward(n_splits=5, horizon=4, embargo_frac=0.01)
    with pytest.raises(ValueError, match="too few samples"):
        list(cv.split(tiny))


def test_indices_are_positional_and_within_range(index):
    cv = PurgedWalkForward(n_splits=3, horizon=4, embargo_frac=0.01)
    for train_idx, test_idx in cv.split(index):
        assert train_idx.min() >= 0
        assert test_idx.max() < len(index)
        assert len(set(train_idx) & set(test_idx)) == 0
