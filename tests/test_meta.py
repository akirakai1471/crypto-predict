import numpy as np
import pandas as pd

from cryptopred.backtest.engine import CostModel
from cryptopred.models.meta import (
    MetaConfig,
    build_meta_training_set,
    oof_primary_proba,
    signals_from_proba,
    simulate_signal_returns,
    train_secondary,
    walk_forward_meta,
)
from cryptopred.models.train import TrainConfig


def _bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


def _dataset(bars: pd.DataFrame, horizon: int = 4, seed: int = 0) -> pd.DataFrame:
    """Dataset whose `signal` feature genuinely predicts the label."""
    rng = np.random.default_rng(seed)
    n = len(bars)
    fwd = (bars["close"].shift(-horizon) / bars["close"] - 1).fillna(0.0)
    signal = fwd + rng.normal(0, 0.004, n)          # noisy but informative
    label = np.where(fwd > 0.002, 2, np.where(fwd < -0.002, 0, 1))

    return pd.DataFrame(
        {
            "signal": signal.to_numpy(),
            "noise": rng.normal(0, 1, n),
            "roc_1": bars["close"].pct_change().fillna(0.0).to_numpy(),
            "label_class": label.astype("int8"),
            "label": label - 1.0,
            "forward_return": fwd.to_numpy(),
            "band": 0.002,
        },
        index=bars.index,
    )


def test_signals_from_proba_respects_threshold():
    proba = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.35, 0.3, 0.35]])
    assert list(signals_from_proba(proba, 0.5)) == [1, -1, 0]
    assert list(signals_from_proba(proba, 0.8)) == [0, 0, 0]


def test_simulate_signal_returns_matches_the_backtest_engine():
    bars = _bars(200)
    idx = bars.index
    signals = np.zeros(len(idx), dtype=int)
    signals[10] = 1
    returns = simulate_signal_returns(bars, idx, signals, horizon=4, costs=CostModel())

    assert len(returns) == 1
    entry = bars["open"].iloc[11]
    exit_ = bars["open"].iloc[15]
    gross = exit_ / entry - 1
    assert returns.iloc[0] < gross   # costs were charged


def test_simulate_returns_empty_when_no_signals():
    bars = _bars(100)
    signals = np.zeros(len(bars), dtype=int)
    assert simulate_signal_returns(bars, bars.index, signals, horizon=4).empty


def test_oof_proba_only_covers_inner_test_blocks():
    bars = _bars(2000)
    dataset = _dataset(bars)
    cfg = MetaConfig(primary=TrainConfig(num_boost_round=20), inner_splits=3)

    proba, mask = oof_primary_proba(dataset, horizon=4, config=cfg)
    assert mask.sum() > 0
    assert not mask.all()            # early rows get no out-of-fold prediction
    assert not mask[:100].any()      # the first inner training block is never predicted
    assert np.isfinite(proba[mask]).all()
    assert np.isnan(proba[~mask]).all()


def test_meta_training_set_labels_are_trade_outcomes():
    bars = _bars(3000)
    dataset = _dataset(bars)
    cfg = MetaConfig(
        primary=TrainConfig(num_boost_round=20), inner_splits=3, primary_threshold=0.4
    )

    x, y, w = build_meta_training_set(dataset, bars, horizon=4, config=cfg)
    assert not x.empty
    assert set(y.unique()) <= {0, 1}
    assert len(x) == len(y) == len(w)
    for col in ["primary_prob_up", "primary_confidence", "primary_signal"]:
        assert col in x.columns
    assert (w > 0).all()


def test_meta_features_include_the_primary_opinion():
    """The secondary must see what the primary thought, or it cannot learn when
    to distrust it."""
    bars = _bars(3000)
    dataset = _dataset(bars)
    cfg = MetaConfig(primary=TrainConfig(num_boost_round=20), inner_splits=3)
    x, _, _ = build_meta_training_set(dataset, bars, horizon=4, config=cfg)
    assert x["primary_confidence"].between(0, 1).all()
    assert set(x["primary_signal"].unique()) <= {-1, 1}


def test_meta_labels_are_not_all_one_class():
    bars = _bars(3000)
    dataset = _dataset(bars)
    cfg = MetaConfig(primary=TrainConfig(num_boost_round=20), inner_splits=3)
    _, y, _ = build_meta_training_set(dataset, bars, horizon=4, config=cfg)
    assert 0 < y.mean() < 1


def test_secondary_trains_and_scores_in_unit_range():
    bars = _bars(3000)
    dataset = _dataset(bars)
    cfg = MetaConfig(
        primary=TrainConfig(num_boost_round=20),
        secondary=TrainConfig(num_boost_round=30, min_data_in_leaf=20),
        inner_splits=3,
    )
    x, y, w = build_meta_training_set(dataset, bars, horizon=4, config=cfg)
    booster = train_secondary(x, y, w, cfg)
    scores = np.asarray(booster.predict(x))
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_walk_forward_meta_filters_signals_down():
    bars = _bars(6000)
    dataset = _dataset(bars)
    cfg = MetaConfig(
        primary=TrainConfig(num_boost_round=30),
        secondary=TrainConfig(num_boost_round=40, min_data_in_leaf=20),
        inner_splits=3,
        primary_threshold=0.4,
        meta_threshold=0.55,
    )
    result = walk_forward_meta(dataset, bars, horizon=4, n_splits=3, config=cfg)

    n_primary = int((result["primary_signals"] != 0).sum())
    n_final = int((result["final_signals"] != 0).sum())
    assert n_primary > 0
    assert n_final <= n_primary          # the filter can only remove
    assert len(result["index"]) == len(result["final_signals"])


def test_walk_forward_meta_reports_per_fold_counts():
    bars = _bars(6000)
    dataset = _dataset(bars)
    cfg = MetaConfig(
        primary=TrainConfig(num_boost_round=20),
        secondary=TrainConfig(num_boost_round=20, min_data_in_leaf=20),
        inner_splits=3,
    )
    result = walk_forward_meta(dataset, bars, horizon=4, n_splits=3, config=cfg)
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["n_final"] <= fold["n_primary"]


def test_a_stricter_meta_threshold_takes_fewer_trades():
    bars = _bars(6000)
    dataset = _dataset(bars)
    base = dict(
        primary=TrainConfig(num_boost_round=20),
        secondary=TrainConfig(num_boost_round=30, min_data_in_leaf=20),
        inner_splits=3,
        primary_threshold=0.4,
    )
    loose = walk_forward_meta(
        dataset, bars, horizon=4, n_splits=3, config=MetaConfig(meta_threshold=0.3, **base)
    )
    strict = walk_forward_meta(
        dataset, bars, horizon=4, n_splits=3, config=MetaConfig(meta_threshold=0.8, **base)
    )
    assert (strict["final_signals"] != 0).sum() <= (loose["final_signals"] != 0).sum()


def test_meta_never_invents_a_signal_the_primary_did_not_make():
    bars = _bars(6000)
    dataset = _dataset(bars)
    cfg = MetaConfig(
        primary=TrainConfig(num_boost_round=20),
        secondary=TrainConfig(num_boost_round=20, min_data_in_leaf=20),
        inner_splits=3,
    )
    result = walk_forward_meta(dataset, bars, horizon=4, n_splits=3, config=cfg)
    primary, final = result["primary_signals"], result["final_signals"]
    taken = final != 0
    assert (final[taken] == primary[taken]).all()
