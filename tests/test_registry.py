import json

import numpy as np
import pandas as pd

from cryptopred.models.registry import ModelRegistry
from cryptopred.models.train import TrainConfig, train_fold
from tests.test_train import _learnable_dataset


def _fold_result():
    df = _learnable_dataset(n=2000)
    return df, train_fold(df.iloc[:1500], df.iloc[1500:], TrainConfig(num_boost_round=20))


def test_save_and_load_roundtrip(tmp_path):
    df, result = _fold_result()
    registry = ModelRegistry(tmp_path)

    version = registry.save(
        result,
        symbol="BTCUSDT",
        interval="1h",
        metrics={"accuracy": 0.42},
        config=TrainConfig(num_boost_round=20),
    )
    bundle = registry.load(version)

    assert bundle.features == result.features
    fresh = bundle.predict(df.iloc[1500:])
    assert fresh.shape == result.proba.shape
    assert np.allclose(fresh, result.proba, atol=1e-9)


def test_metadata_is_written(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    version = registry.save(
        result,
        symbol="BTCUSDT",
        interval="1h",
        metrics={"accuracy": 0.42},
        config=TrainConfig(num_boost_round=20),
    )

    meta = json.loads((tmp_path / version / "metadata.json").read_text(encoding="utf-8"))
    assert meta["symbol"] == "BTCUSDT"
    assert meta["interval"] == "1h"
    assert meta["metrics"]["accuracy"] == 0.42
    assert meta["n_features"] == len(result.features)
    assert "created_at" in meta


def test_latest_returns_most_recent_version(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    first = registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())
    second = registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())

    assert first != second
    assert registry.latest("BTCUSDT", "1h") == second


def test_latest_returns_none_when_empty(tmp_path):
    assert ModelRegistry(tmp_path).latest("BTCUSDT", "1h") is None


def test_predict_rejects_missing_features(tmp_path):
    df, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    version = registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())
    bundle = registry.load(version)

    broken = df.iloc[1500:].drop(columns=["signal"])
    try:
        bundle.predict(broken)
    except KeyError as exc:
        assert "signal" in str(exc)
    else:
        raise AssertionError("expected KeyError for a missing feature")


def test_saved_bundle_survives_a_new_registry_instance(tmp_path):
    df, result = _fold_result()
    version = ModelRegistry(tmp_path).save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    reloaded = ModelRegistry(tmp_path).load(version)
    assert isinstance(reloaded.predict(df.iloc[1500:]), np.ndarray)


def test_version_name_encodes_symbol_and_interval(tmp_path):
    _, result = _fold_result()
    version = ModelRegistry(tmp_path).save(
        result, symbol="ETHUSDT", interval="1m", metrics={}, config=TrainConfig()
    )
    assert "ETHUSDT" in version
    assert "1m" in version


def test_load_missing_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path)
    try:
        registry.load("nope")
    except FileNotFoundError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_predict_is_deterministic(tmp_path):
    df, result = _fold_result()
    version = ModelRegistry(tmp_path).save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    bundle = ModelRegistry(tmp_path).load(version)
    test = df.iloc[1500:]
    assert np.array_equal(bundle.predict(test), bundle.predict(test))


def test_predict_accepts_a_single_row(tmp_path):
    df, result = _fold_result()
    version = ModelRegistry(tmp_path).save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    bundle = ModelRegistry(tmp_path).load(version)
    one = df.iloc[[1600]]
    proba = bundle.predict(one)
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0)


def test_metadata_records_row_count(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    version = registry.save(
        result,
        symbol="BTCUSDT",
        interval="1h",
        metrics={},
        config=TrainConfig(),
        n_train_rows=1500,
    )
    meta = json.loads((tmp_path / version / "metadata.json").read_text(encoding="utf-8"))
    assert meta["n_train_rows"] == 1500


def test_registry_creates_directory(tmp_path):
    target = tmp_path / "nested" / "registry"
    _, result = _fold_result()
    ModelRegistry(target).save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    assert target.exists()


def test_calibrators_are_persisted(tmp_path):
    df = _learnable_dataset(n=2000)
    result = train_fold(
        df.iloc[:1500], df.iloc[1500:], TrainConfig(num_boost_round=20, calibrate=True)
    )
    assert result.calibrators is not None

    registry = ModelRegistry(tmp_path)
    version = registry.save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    bundle = registry.load(version)
    assert bundle.calibrators is not None
    assert len(bundle.calibrators) == 3


def test_uncalibrated_model_roundtrips_too(tmp_path):
    df = _learnable_dataset(n=2000)
    result = train_fold(
        df.iloc[:1500], df.iloc[1500:], TrainConfig(num_boost_round=20, calibrate=False)
    )
    assert result.calibrators is None

    registry = ModelRegistry(tmp_path)
    version = registry.save(
        result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig()
    )
    bundle = registry.load(version)
    assert bundle.calibrators is None
    proba = bundle.predict(df.iloc[1500:])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_list_versions_is_sorted(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    a = registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())
    b = registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())
    versions = registry.list_versions("BTCUSDT", "1h")
    assert versions == sorted([a, b])


def test_metadata_json_is_utf8_readable(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    version = registry.save(
        result, symbol="BTCUSDT", interval="1h", metrics={"note": "kiểm tra"}, config=TrainConfig()
    )
    text = (tmp_path / version / "metadata.json").read_text(encoding="utf-8")
    assert "kiểm tra" in text


def test_index_frame_lists_all_models(tmp_path):
    _, result = _fold_result()
    registry = ModelRegistry(tmp_path)
    registry.save(result, symbol="BTCUSDT", interval="1h", metrics={}, config=TrainConfig())
    registry.save(result, symbol="ETHUSDT", interval="1h", metrics={}, config=TrainConfig())

    index = registry.index()
    assert isinstance(index, pd.DataFrame)
    assert set(index["symbol"]) == {"BTCUSDT", "ETHUSDT"}
