"""Regime cells must be decided by the past only.

Computing tercile boundaries once over all history would let the 2026 volatility
distribution decide which bucket a 2020 bar belongs to. The effect is small; this
project has already paid for assuming a small leak is a harmless one.
"""

from cryptopred.briefing.regime import MIN_HISTORY_BARS, RegimeCell, classify_regimes
from tests.conftest import make_ohlcv


def test_terciles_split_roughly_evenly():
    bars = make_ohlcv(n=6000, seed=3)
    cells = classify_regimes(bars, min_history=500)
    vol = cells["vol_bucket"].dropna()
    counts = vol.value_counts(normalize=True)
    assert set(counts.index) == {0, 1, 2}
    assert counts.min() > 0.2


def test_early_bars_are_excluded_not_bucketed_on_thin_quantiles():
    bars = make_ohlcv(n=3000, seed=4)
    cells = classify_regimes(bars, min_history=500)
    assert cells["vol_bucket"].iloc[:490].isna().all()
    assert cells["vol_bucket"].iloc[600:].notna().any()


def test_classification_is_prefix_invariant():
    """The cell for bar t must not change when bars after t are appended.

    This is the test that catches full-history tercile boundaries, which are the
    tempting shortcut.
    """
    bars = make_ohlcv(n=4000, seed=5)
    full = classify_regimes(bars, min_history=500)
    prefix = classify_regimes(bars.iloc[:3000], min_history=500)

    a = full["vol_bucket"].iloc[:3000]
    b = prefix["vol_bucket"]
    assert a.equals(b)

    a_t = full["trend_bucket"].iloc[:3000]
    b_t = prefix["trend_bucket"]
    assert a_t.equals(b_t)


def test_cell_of_names_both_dimensions_in_vietnamese():
    cell = RegimeCell(vol_bucket=2, trend_bucket=0)
    assert cell.label == "biến động cao / xu hướng giảm"
    assert cell.key == (2, 0)


def test_min_history_constant_is_exported():
    assert MIN_HISTORY_BARS == 2000
