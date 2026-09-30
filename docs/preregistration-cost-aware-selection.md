# Pre-registration: choose trades by expected move, not by confidence alone

**Written and committed before the rule was run on real data.**

Date: 2026-09-30.

## Why this, after two rejections

`docs/findings.md` ("Two ways to follow the market faster") closed with the
baseline at 54.44% mean sign accuracy on traded bars and two attempts to raise
it both rejected. Accuracy is not where the money is lost. The break-even table
in the same document: a bet right with probability `p` on a move of size `m`,
paying round-trip cost `c`, breaks even at `p = (c/m + 1) / 2`.

| typical 24h move | accuracy needed |
|---|---|
| 1.37% (the median) | 55.1% |
| 2.75% | 52.5% |

A model right 54.4% of the time loses on median bars and could win on large
ones. The current rule picks the 8% of bars with the widest directional margin
and ignores how far price is likely to move. Volatility, unlike direction, is
strongly predictable from its own recent past.

## The rule under test

Both rules run on **the same trained model and the same out-of-sample
probabilities** for each symbol. Each symbol is trained once; the only
difference between the two arms is which bars are picked. That makes the
comparison about the selection rule and nothing else.

| arm | ranks bars within each walk-forward fold by | trades |
|---|---|---|
| `margin` (baseline) | `abs(P(up) - P(down))` | top 8% per fold |
| `margin_x_vol` | `abs(P(up) - P(down)) * vol72` | top 8% per fold |

`vol72` is the standard deviation of the last 72 hourly log returns of the
symbol's close, known at the bar's close. It is not fitted to anything; there
is no second model and no parameter chosen from data. 72 bars (three days) was
picked as three times the 24-bar horizon before running, and will not be varied.

The product is a proxy for expected return in the predicted direction: margin
is the probability edge, vol72 scales it to the size of move the next day is
likely to bring. Cost is the same for every bar, so it cannot change a ranking;
it enters through the backtest.

As before, a bar whose most likely class is FLAT is never traded, the 8% is of
all bars in the fold, and direction is the sign of `P(up) - P(down)`.

Everything else is the frozen configuration of the earlier experiments: 1h bars,
24-bar horizon, 5 purged folds, 400 rounds, 3-fold out-of-fold calibration,
maker execution at 0.20% with chase and strict fills. Data: the archive store
already downloaded (2020-01 to 2026-08), the same twenty symbols.

## The decision rule

One arm, so the full α = 0.05.

`margin_x_vol` is **ADOPTED** only if all four hold:

1. **It survives costs better on most symbols.** Total return at doubled costs
   higher than the baseline's on at least 75% of symbols tested, rounded up
   (15 of 20), with a one-sided sign test at p ≤ 0.05. This is the primary
   criterion because surviving costs is the whole point, and doubled costs is
   the stress test most symbols currently fail.
2. **It is not just bigger bets.** Mean Sharpe ratio across symbols not lower
   than the baseline's. Trading high-volatility bars with a fixed stake takes
   more risk per trade; a higher return bought only with proportionally more
   risk is not an improvement.
3. **It does not cost gate passes.** Symbols passing all three multi-symbol
   gates (positive return, survives doubled costs, two-sided) not fewer than the
   baseline's.
4. **It does not fall into the long-bias trap.** Symbols with a TWO-SIDED
   verdict not fewer than the baseline's. High volatility in crypto often means
   crashes and rallies, and a rule drawn to them could end up earning on one
   side only.

Anything else is **REJECTED**.

Reported but not judged: sign accuracy (expected to move either way — a rule
that trades bigger moves can win with lower accuracy), mean absolute 24h move
of the traded bars, and max drawdown.

## Caveats written down now

- Ranking within a fold by volatility may concentrate a fold's trades in its
  most turbulent weeks. That is part of what is being measured, not a defect to
  fix afterwards; Sharpe and drawdown are there to show it.
- Twenty correlated symbols over one period are not twenty independent tests.
- If adopted, the live system would need the score's cutoff computed the way
  `margin_cutoff` is now, out of sample. That is follow-up work, not part of
  this test.

## What happens to the result

Recorded in `docs/findings.md` either way, with the per-symbol table. Run with:

```
uv run cryptopred-model selection-experiment --config configs/twenty-symbols.yaml --jobs 0
```
