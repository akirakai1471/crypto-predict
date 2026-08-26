from cryptopred.dataset.builder import build_dataset, feature_columns
from tests.conftest import make_ohlcv


def test_build_dataset_joins_features_and_labels():
    bars = make_ohlcv(n=1500, seed=21)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert "label" in ds.columns
    assert "forward_return" in ds.columns
    assert "label_class" in ds.columns
    assert len(feature_columns(ds)) >= 60


def test_build_dataset_drops_warmup_and_tail_rows():
    bars = make_ohlcv(n=1500, seed=22)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert ds.notna().all().all()
    assert len(ds) < len(bars)
    assert ds.index[-1] <= bars.index[-5]


def test_label_class_maps_to_zero_one_two():
    bars = make_ohlcv(n=1500, seed=23)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert set(ds["label_class"].unique()) <= {0, 1, 2}
    down = ds[ds["label"] == -1.0]
    if not down.empty:
        assert (down["label_class"] == 0).all()
    flat = ds[ds["label"] == 0.0]
    if not flat.empty:
        assert (flat["label_class"] == 1).all()


def test_metadata_columns_present():
    bars = make_ohlcv(n=1500, seed=24)
    ds = build_dataset(
        bars, interval="1h", horizon=4, atr_period=14, band_k=0.5, symbol="BTCUSDT"
    )
    assert (ds["symbol"] == "BTCUSDT").all()
    assert ds.index.name == "open_time"


def test_feature_columns_excludes_targets_and_metadata():
    bars = make_ohlcv(n=1500, seed=25)
    ds = build_dataset(
        bars, interval="1h", horizon=4, atr_period=14, band_k=0.5, symbol="BTCUSDT"
    )
    cols = feature_columns(ds)
    for banned in ["label", "label_class", "forward_return", "band", "symbol"]:
        assert banned not in cols


def test_features_are_float32_to_save_memory():
    bars = make_ohlcv(n=1500, seed=27)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    for col in feature_columns(ds):
        assert ds[col].dtype == "float32", col


def test_empty_input_returns_empty_dataset():
    empty = make_ohlcv(n=10, seed=26)
    ds = build_dataset(empty, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    assert ds.empty
