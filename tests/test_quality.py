from cryptopred.dataset.builder import build_dataset
from cryptopred.dataset.quality import quality_report
from tests.conftest import make_ohlcv


def test_quality_report_fields():
    bars = make_ohlcv(n=2000, seed=31)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    report = quality_report(ds)

    assert report["rows"] == len(ds)
    assert report["n_features"] >= 60
    assert set(report["class_balance"]) <= {"down", "flat", "up"}
    assert abs(sum(report["class_balance"].values()) - 1.0) < 1e-9
    assert report["start"] == ds.index.min()
    assert report["end"] == ds.index.max()


def test_quality_report_flags_constant_features():
    bars = make_ohlcv(n=2000, seed=32)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    ds["always_seven"] = 7.0
    report = quality_report(ds)
    assert "always_seven" in report["constant_features"]


def test_quality_report_flags_extreme_correlation():
    bars = make_ohlcv(n=2000, seed=33)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    ds["cheating"] = ds["forward_return"] * 10.0
    report = quality_report(ds)
    assert "cheating" in report["suspicious_features"]


def test_format_report_is_readable():
    from cryptopred.dataset.quality import format_report

    bars = make_ohlcv(n=2000, seed=34)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    text = format_report(quality_report(ds))
    assert "rows" in text.lower()
    assert "class balance" in text.lower()
