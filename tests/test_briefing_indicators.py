"""RSI is not useless. It is unmeasured, here, and must say so.

The answer that prompted this feature read "RSI-14 is 66.1, momentum remains
positive". The first half is a fact and the second is an inference this project
has never tested.
"""

from cryptopred.briefing.indicators import current_indicators
from tests.conftest import make_ohlcv


def test_every_indicator_is_marked_unvalidated():
    bars = make_ohlcv(n=1000, seed=31)
    out = current_indicators(bars)
    assert out
    for name, payload in out.items():
        assert payload["source"] == "convention", name
        assert payload["validated"] is False, name
        assert "warning" in payload, name


def test_the_expected_indicators_are_present():
    bars = make_ohlcv(n=1000, seed=32)
    out = current_indicators(bars)
    for key in ("rsi_7", "rsi_14", "macd_histogram", "atr_pct", "adx_14"):
        assert key in out


def test_each_carries_its_conventional_reading():
    bars = make_ohlcv(n=1000, seed=33)
    out = current_indicators(bars)
    assert "70" in out["rsi_14"]["reading"]


def test_an_indicator_that_has_not_warmed_up_is_absent_not_approximated():
    """Warm-up is per indicator, so the cut is per indicator too.

    At 20 bars RSI-7 is genuinely available and realized volatility over 168
    bars genuinely is not. Reporting the latter from a partial window would put
    a number next to a name that does not describe it.
    """
    bars = make_ohlcv(n=20, seed=34)
    out = current_indicators(bars)
    assert "rsi_7" in out
    assert "realized_vol_168" not in out
    assert "atr_percentile" not in out


def test_the_value_reported_is_the_final_bars_not_the_last_one_available():
    """A user asking what ATR percentile is *now* is asking about this bar.

    atr_percentile needs 720 valid ATR readings, which arrive around bar 732.
    At 700 bars every earlier value is still NaN, so there is nothing stale to
    fall back to; at 800 there is a real one. The boundary is what pins the
    semantics.
    """
    assert "atr_percentile" not in current_indicators(make_ohlcv(n=700, seed=35))
    assert "atr_percentile" in current_indicators(make_ohlcv(n=800, seed=35))
