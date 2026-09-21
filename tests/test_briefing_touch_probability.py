"""Overlapping forward windows are not independent trials.

Consecutive windows share almost all their bars, so the effective sample size is
far below the window count. Wilson - which this codebase uses correctly for
independent trials in serve/status.py - would report a confidence the data does
not support.
"""

import numpy as np

from cryptopred.briefing.provenance import Measured, Unavailable
from cryptopred.briefing.touch import (
    MIN_CELL_BARS,
    block_bootstrap_ci,
    touch_probability,
)
from cryptopred.serve.status import wilson_interval
from tests.conftest import make_ohlcv


def test_bootstrap_is_wider_than_wilson_on_autocorrelated_data():
    """The specific claim the design makes, so the specific claim pinned here."""
    # 1000 outcomes in runs of 100: heavily autocorrelated, overall rate 0.5
    touched = np.concatenate([np.full(100, i % 2 == 0) for i in range(10)])

    lo_b, hi_b = block_bootstrap_ci(touched, block=100, n_boot=400, seed=1)
    lo_w, hi_w = wilson_interval(int(touched.sum()), len(touched))

    assert (hi_b - lo_b) > (hi_w - lo_w) * 2


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    touched = rng.random(2000) < 0.3
    lo, hi = block_bootstrap_ci(touched, block=24, n_boot=400, seed=2)
    assert lo < touched.mean() < hi


def test_a_thin_cell_returns_unavailable_naming_the_count():
    """Conditioning on two dimensions multiplies cells; a thin one produces a
    confident-looking number from noise."""
    bars = make_ohlcv(n=2600, seed=11)
    result = touch_probability(
        bars, target_pct=-0.03, horizon=24, min_cell_bars=10_000
    )
    assert isinstance(result["conditional"], Unavailable)
    assert "10,000" in result["conditional"].reason or "10000" in result["conditional"].reason
    # the unconditional figure survives regardless
    assert isinstance(result["unconditional"], Measured)


def test_unconditional_is_always_returned_alongside():
    bars = make_ohlcv(n=6000, seed=12)
    result = touch_probability(bars, target_pct=-0.03, horizon=24)
    assert isinstance(result["unconditional"], Measured)
    assert result["unconditional"].n > 0
    assert 0.0 <= result["unconditional"].value <= 1.0


def test_wait_time_percentiles_are_reported_in_hours():
    """A probability alone does not answer 'when'. The distribution has a long
    right tail and the median alone hides it."""
    bars = make_ohlcv(n=6000, seed=13)
    result = touch_probability(bars, target_pct=-0.01, horizon=72)
    wait = result["wait_hours"]
    assert wait["median"] <= wait["p90"]
    assert wait["p25"] <= wait["median"] <= wait["p75"]


def test_the_cell_is_named_so_the_answer_can_quote_it():
    bars = make_ohlcv(n=6000, seed=14)
    result = touch_probability(bars, target_pct=-0.03, horizon=24, min_cell_bars=100)
    assert "biến động" in result["cell_label"]


def test_min_cell_bars_constant_is_five_hundred():
    assert MIN_CELL_BARS == 500
