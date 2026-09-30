"""How often does the touch-probability interval contain the truth?

This is the harness behind MEASURED_COVERAGE in briefing/touch.py. It simulates
series whose true touch rate is known, builds the interval on each, and counts
how often the truth lands inside. Run it before changing the interval or the
constant; the number the user reads is only as honest as this measurement.

    uv run python scripts/touch_interval_coverage.py            # 800 reps, a few minutes
    uv run python scripts/touch_interval_coverage.py --reps 200 # quick look

Two dependence models set the headline figure:

  markov  a two-state chain, true rate 0.4, mean runs of 40 and 60 bars. The
          model the first measurement used, kept so old and new are comparable.
  walk    real touch outcomes - did a -3% level trade within the horizon - on a
          fat-tailed (t4) random walk at roughly BTC's hourly volatility. The
          dependence here comes from overlapping forward windows, which is where
          it comes from in the real data.

Two more are measured and reported but kept out of the headline, because every
method tried fails them and averaging them in would hide that:

  regimes the same walk with volatility that switches between two levels and
          holds each for ~500 bars. Touch outcomes then depend on each other
          over weeks, far past any block.
  rare    a -8% target, touched in about 1% of windows. At 500 bars most
          samples hold no touch at all.

"old" is a frozen copy of the percentile block bootstrap used until 2026-09-30,
with the fixed block it used; "new" is block_bootstrap_ci with block_length, as
touch_probability calls them now. Both run on identical series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptopred.briefing.touch import block_bootstrap_ci, block_length  # noqa: E402

HOURLY_VOL = 0.007
T4_SCALE = 1 / np.sqrt(2.0)  # a t4 draw has variance 2


def old_percentile_ci(touched, block, n_boot, seed):
    """The interval as it was before 2026-09-30, for comparison only."""
    n = len(touched)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    means = touched[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# -- series with a known answer -----------------------------------------------


def markov(n, rng, leave_one=1 / 40, leave_zero=1 / 60):
    """Stationary P(touched) = leave_zero / (leave_zero + leave_one) = 0.4."""
    x = np.empty(n, dtype=bool)
    x[0] = rng.random() < 0.4
    u = rng.random(n)
    for i in range(1, n):
        x[i] = (u[i] >= leave_one) if x[i - 1] else (u[i] < leave_zero)
    return x


def _vol(n, rng, regime_bars):
    if regime_bars is None:
        return np.full(n, HOURLY_VOL)
    state = np.empty(n, dtype=bool)
    state[0] = rng.random() < 0.5
    flips = rng.random(n) < 1 / regime_bars
    for i in range(1, n):
        state[i] = state[i - 1] ^ flips[i]
    return np.where(state, 0.010, 0.004)


def walk_touches(n, horizon, target, rng, regime_bars=None):
    """Did a `target` move trade within `horizon` bars, for each of n bars?"""
    steps = _vol(n + horizon, rng, regime_bars) * rng.standard_t(4, n + horizon) * T4_SCALE
    path = np.concatenate([[0.0], np.cumsum(steps)])
    windows = sliding_window_view(path, horizon + 1)[:n]
    return (windows[:, 1:].min(axis=1) - windows[:, 0]) <= np.log1p(target)


def walk_truth(horizon, target, rng, regime_bars=None, m=400_000):
    """The true rate, from m independent windows."""
    if regime_bars is None:
        vol = np.full((m, horizon), HOURLY_VOL)
    else:
        path = _vol(m * 4 + horizon, rng, regime_bars)
        starts = rng.integers(0, len(path) - horizon, size=m)
        vol = path[starts[:, None] + np.arange(horizon)]
    steps = vol * rng.standard_t(4, size=(m, horizon)) * T4_SCALE
    return float((np.cumsum(steps, axis=1).min(axis=1) <= np.log1p(target)).mean())


# -- measurement --------------------------------------------------------------


def measure(label, headline, series, truth, horizon, n_boot):
    old_hits = new_hits = 0
    new_width = 0.0
    for rep, s in enumerate(series):
        # Old: the fixed block it shipped with. New: what touch_probability uses now.
        lo, hi = old_percentile_ci(s, max(2 * horizon, 48), n_boot, seed=rep)
        old_hits += lo <= truth <= hi
        block = block_length(horizon, len(s))
        lo, hi = block_bootstrap_ci(s, block=block, n_boot=n_boot, seed=rep)
        new_hits += lo <= truth <= hi
        new_width += hi - lo
    reps = len(series)
    return {
        "label": label,
        "headline": headline,
        "old": old_hits / reps,
        "new": new_hits / reps,
        "width": new_width / reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=800)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cases = []

    def add(model, truth, n, horizon, note, headline, make):
        block = block_length(horizon, n)
        label = f"{model:<7} p={truth:.3f}  n={n:<5} block={block:<3} {note}"
        cases.append((label, headline, make, truth, horizon))

    # The markov chain has no horizon of its own; 24 gives it the block a 24-hour
    # query gets, which is what the first measurement used.
    for n in (500, 3000):
        add("markov", 0.4, n, 24, "", True, lambda n=n: markov(n, rng))
    for horizon in (24, 72):
        note = f"-3% in {horizon}h"
        truth = walk_truth(horizon, -0.03, rng)
        # 20,000 is the scale of the unconditional figure on real data.
        for n in (500, 3000, 20_000):
            add("walk", truth, n, horizon, note, True,
                lambda n=n, h=horizon: walk_touches(n, h, -0.03, rng))
        truth = walk_truth(horizon, -0.03, rng, regime_bars=500)
        for n in (500, 3000):
            add("regimes", truth, n, horizon, note, False,
                lambda n=n, h=horizon: walk_touches(n, h, -0.03, rng, regime_bars=500))
    truth = walk_truth(24, -0.08, rng)
    for n in (500, 3000):
        add("rare", truth, n, 24, "-8% in 24h", False,
            lambda n=n: walk_touches(n, 24, -0.08, rng))

    print(f"{args.reps} series per case, n_boot={args.n_boot}. Coverage = share of "
          "intervals containing the true rate.\n")
    print(f"{'case':<48} {'old':>6} {'new':>6} {'width':>6}")
    results = []
    for label, headline, make, truth, horizon in cases:
        series = [make() for _ in range(args.reps)]
        r = measure(label, headline, series, truth, horizon, args.n_boot)
        results.append(r)
        flag = "" if headline else "   (stress; not in headline)"
        print(f"{label:<48} {r['old']:>6.1%} {r['new']:>6.1%} {r['width']:>6.3f}{flag}",
              flush=True)

    head = [r for r in results if r["headline"]]
    se = np.sqrt(0.9 * 0.1 / args.reps)
    print(
        f"\nHeadline worst case: old {min(r['old'] for r in head):.1%}, "
        f"new {min(r['new'] for r in head):.1%} "
        f"(each figure +/- about {2 * se:.1%} at {args.reps} reps)"
    )


if __name__ == "__main__":
    main()
