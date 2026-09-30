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
    block_length,
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


def test_wait_source_names_where_the_wait_time_came_from():
    """A thin cell falls back to the unconditional wait-time sample, but the
    dict still carries the cell's label. Without an explicit flag a renderer
    doing the obvious thing would misattribute unconditional wait times to a
    named regime -- exactly the class of error provenance.py exists to catch.
    """
    bars = make_ohlcv(n=2600, seed=11)
    thin = touch_probability(bars, target_pct=-0.03, horizon=24, min_cell_bars=10_000)
    assert isinstance(thin["conditional"], Unavailable)
    assert thin["wait_source"] == "unconditional"

    healthy_bars = make_ohlcv(n=6000, seed=14)
    healthy = touch_probability(
        healthy_bars, target_pct=-0.03, horizon=24, min_cell_bars=100
    )
    assert isinstance(healthy["conditional"], Measured)
    assert healthy["wait_source"] == "cell"


def test_a_level_never_touched_still_gets_an_upper_bound():
    """Zero touches in 500 bars is not proof the rate is zero.

    The old interval returned (0, 0) here and covered a true 1.3% rate 47% of
    the time. With ~10 effective blocks, the rule of three says the rate could
    plausibly be as high as 3/10.
    """
    lo, hi = block_bootstrap_ci(np.zeros(500, dtype=bool), block=48)
    assert lo == 0.0
    assert hi == 3 / 11  # 500 bars in blocks of 48 is 11 blocks

    lo, hi = block_bootstrap_ci(np.ones(500, dtype=bool), block=48)
    assert (lo, hi) == (1 - 3 / 11, 1.0)


def test_one_block_says_nothing_about_its_own_spread():
    lo, hi = block_bootstrap_ci(np.array([True, False] * 20), block=48)
    assert (lo, hi) == (0.0, 1.0)


def test_the_interval_never_leaves_zero_to_one():
    rng = np.random.default_rng(3)
    touched = rng.random(600) < 0.03
    lo, hi = block_bootstrap_ci(touched, block=48, n_boot=400)
    assert 0.0 <= lo <= touched.mean() <= hi <= 1.0


def test_few_blocks_give_a_wider_interval_than_the_percentile_method_did():
    """The reason for studentizing. Ten block means say little about their own
    spread, and an interval that ignores that is too narrow - measured at 77-81%
    coverage for the percentile method at this sample size."""
    rng = np.random.default_rng(4)
    touched = np.repeat(rng.random(10) < 0.4, 50)  # 500 bars, 10 independent runs
    lo, hi = block_bootstrap_ci(touched, block=48, n_boot=400, seed=1)

    # The percentile version, as it was: resample circular blocks, take the
    # 2.5% and 97.5% quantiles of the resampled means.
    n_blocks = int(np.ceil(touched.size / 48))
    means = []
    for _ in range(400):
        starts = rng.integers(0, touched.size, size=n_blocks)
        idx = (starts[:, None] + np.arange(48)).ravel() % touched.size
        means.append(touched[idx].mean())
    p_lo, p_hi = np.quantile(means, [0.025, 0.975])
    assert (hi - lo) > (p_hi - p_lo)


def test_the_block_grows_with_the_sample_and_not_below_it():
    assert block_length(24, 500) == 48
    assert block_length(24, 100) == 48  # never shorter than the base
    assert block_length(72, 500) == 144
    assert block_length(24, 4000) == 96  # (4000/500)^(1/3) = 2
    assert block_length(24, 60_000) > block_length(24, 4000)


def test_a_full_history_sample_is_cheap_to_bootstrap():
    """The unconditional figure runs on ~60,000 bars. Resampling block means
    rather than copying the series keeps that to a small array per draw."""
    import time

    touched = np.random.default_rng(5).random(60_000) < 0.3
    start = time.perf_counter()
    block_bootstrap_ci(touched, block=block_length(72, touched.size), n_boot=1000)
    assert time.perf_counter() - start < 5.0
