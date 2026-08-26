# Findings — Phase P2 / P3

Date: 2026-08-26. All numbers are out-of-sample, from purged walk-forward
cross-validation with a 4-fold-plus purge/embargo split, on Binance USDT
perpetual data from 2019 to 2026.

## Headline

The model has **real predictive signal**. Whether that signal is a **tradeable
edge after costs** is unproven, and the evidence is weaker than any single
number in this document makes it look.

## What the model actually knows

Directional calls, BTCUSDT 1h bars, 4-hour horizon, 50,245 out-of-sample bars:

| Confidence bucket | Bars | Accuracy |
|---|---|---|
| below 0.40 | 14,280 | 36.3% |
| 0.40 – 0.50 | 24,869 | 41.7% |
| 0.50 – 0.60 | 9,242 | 50.3% |
| 0.60 – 0.70 | 1,371 | 54.6% |
| above 0.70 | 483 | 63.2% |

Accuracy rises monotonically with the model's own stated confidence. That is
the single most convincing result here: the probabilities carry information,
they are not decoration. Sign accuracy — did price move the predicted way at
all — is 58.9% on committed BTC signals, and 54.7% on ETH.

It beats all four baselines with a bootstrap confidence interval that excludes
zero. As a classifier, the verdict is GO.

## Where it fails

A classifier that knows something is not yet a strategy. Costs decide.

**BTCUSDT, 1h bars, 4-hour horizon, threshold 0.50:**

| | |
|---|---|
| Trades | 1,509 over 5.7 years |
| Win rate | 52.6% |
| Average net return per trade | +0.089% |
| Total return after costs | +34.0% |
| Max drawdown | −22.6% |
| Sharpe | 0.46 |
| Total return at doubled costs | **−21.0%** |

Round-trip cost is 0.14% of notional. The gross edge per trade is roughly
0.23%. Around 60% of the gross edge is consumed by fees, and the strategy
inverts from profitable to loss-making when fill quality is merely mediocre.
**Strategy verdict: NO-GO.** An edge that requires perfect fills is not an edge.

## What fixed it, and why that is only partly reassuring

The failure mode is structural, not statistical: a 4-hour move on BTC is simply
not large enough relative to a fixed 0.14% toll. Lengthening the horizon to 24
hours makes each move bigger while the toll stays the same.

**BTCUSDT, 1h bars, 24-hour horizon**, across thresholds:

| Threshold | Signals | Sign accuracy | Net return | Max DD | At 2x costs | Robust |
|---|---|---|---|---|---|---|
| 0.40 | 46,070 | 51.9% | −82.1% | −94.1% | −98.8% | no |
| 0.45 | 24,065 | 53.5% | −34.4% | −75.0% | −83.9% | no |
| 0.50 | 14,488 | 55.1% | +24.8% | −52.6% | −46.4% | no |
| **0.55** | 7,362 | 58.4% | +144.9% | −23.0% | +59.4% | **yes** |
| **0.60** | 4,217 | 58.9% | +70.9% | −16.7% | +33.6% | **yes** |
| **0.65** | 3,101 | 59.6% | +49.8% | −16.5% | +25.0% | **yes** |
| **0.70** | 2,301 | 58.4% | +16.2% | −15.1% | +1.6% | **yes** |

Four adjacent thresholds are all robust, sign accuracy is stable across the
whole band, and returns fall off smoothly on both sides. A plateau like this is
far better evidence than a single winning cell would be — an isolated good
number surrounded by bad ones is noise, which is exactly what the 4-hour result
at threshold 0.50 turned out to be.

## Why this is still not proof

**ETHUSDT fails at every threshold on the same 24-hour horizon.** Sign accuracy
is only 51.8–53.5%, and no configuration survives doubled costs. The effect does
not generalise to the second symbol tested.

Counting honestly: 2 symbols × 2 horizons × 7 thresholds is 28 configurations
examined. One region passed. That is roughly the hit rate chance alone would
produce. This is textbook multiple-comparison territory, and the BTC plateau —
however clean it looks — was found by searching, not predicted in advance.

Two further caveats:

- Choosing a threshold from the sweep table fits it to the same out-of-sample
  data used to judge the model. The default was set to 0.60, the middle of the
  plateau rather than its peak at 0.55, because picking the maximum is the more
  overfitted choice.
- The whole 24-hour result rests on one asset. One asset is an anecdote.

## Decision

Defaults changed to the 24-hour horizon and a 0.60 threshold, as the only
configuration with any evidence behind it. **No live trading.** The next step is
paper trading, which measures the strategy forward on bars that did not exist
when any of these choices were made. That is the only test none of the above
can fake.

Expect the paper result to be worse than +70.9%. If it is not, re-read this
document before believing it.

## Bug found and fixed along the way

The first backtest reported +180.65% for the 4-hour strategy and, at the
24-hour horizon, returns in the billions of percent alongside a −99.6%
drawdown. Impossible numbers, and the cause was in the engine: fixed-horizon
signals overlap, and each trade's return was being compounded serially as if
positions were sequential, reusing the same capital many times over.

The engine now allocates each trade `1/horizon` of capital and compounds the
portfolio bar by bar, so total exposure never exceeds 100%. The corrected
4-hour figure is +34.0%, not +180.65%. Every number in this document comes from
the corrected engine. `tests/test_backtest.py` has a regression test that fails
if serial compounding ever returns.
