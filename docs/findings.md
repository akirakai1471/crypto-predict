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

## The break-even table, which explains everything above

Every result in this document follows from one inequality that could have been
computed on day one, before a single model was trained. A directional bet with
accuracy `p` on a move of size `m`, paying round-trip cost `c`, breaks even when

```
p·m − (1−p)·m = c      →      p = (c/m + 1) / 2
```

Cost is fixed. Move size grows with horizon. So the accuracy a strategy *needs*
falls as the horizon lengthens, while the accuracy a model *has* is roughly
flat. Measured on BTCUSDT, taker round trip 0.140%:

| Horizon | Median abs move | Break-even accuracy | Measured accuracy | |
|---|---|---|---|---|
| 1m → 5 min | 0.070% | **149.5%** | 61.4% | impossible |
| 1h → 4h | 0.464% | 65.1% | 58.9% | short by 6.2pp |
| 1h → 12h | 0.883% | 57.9% | — | |
| 1h → 24h | 1.373% | **55.1%** | 58.9% | clears by 3.8pp |
| 1h → 48h | 2.066% | 53.4% | — | |
| 1h → 168h | 4.092% | 51.7% | — | |

The 1m result is the sharpest illustration. That model is the **most accurate in
the entire project** — 61.4% sign accuracy at threshold 0.60, better than the
24-hour model — and it lost 45% of capital. A five-minute BTC move is 0.070%
while the toll to capture it is 0.140%, twice the size. Break-even would require
149.5% accuracy, which does not exist. A perfect oracle loses money scalping at
taker fees. Accuracy was never the binding constraint.

This reframes the earlier retune. Moving from 4h to 24h was not a lucky search
result; it moved the strategy from the wrong side of an arithmetic inequality to
the right side. The plateau across thresholds is what that looks like in data.

Two caveats on the table. The right-hand column holds accuracy constant at
58.9%, which was measured at the 24-hour horizon only — accuracy at 48h and 168h
is **assumed, not measured**. And median move understates the mean in a
fat-tailed market, so this is a rough guide, not a precise threshold.

**Maker fees change the arithmetic materially.** Limit orders on Binance pay
0.02% instead of 0.05% and cross less spread; a 0.060% round trip drops the 4h
break-even from 65.1% to 56.5%, which the measured 58.9% would clear. That is
the largest single lever available, and it is a change to execution, not to the
model. It also introduces fill risk the current backtest does not model: a limit
order that never fills is a trade that never happened, and the strategy's
realised accuracy is then measured on a different set of bars than the backtest
assumed.

## Longer horizons: the experiment that narrowed the evidence

The break-even table predicted, before any of these runs, that longer horizons
should make more configurations viable by lowering the bar rather than raising
accuracy. That prediction was testable, and the threshold sweeps appeared to
confirm it: at 48h and 72h, **ETHUSDT became robust for the first time**, and
BTCUSDT stayed robust. Two symbols working looked like the cross-symbol
confirmation the 24h result was missing.

It was not. Splitting the trades by side dissolves it.

| Configuration | Long PnL | Short PnL | Short win rate | Verdict |
|---|---|---|---|---|
| **BTC 1h, 24h horizon, t=0.60** | +4,701 | **+793** | **55.1%** | **two-sided** |
| BTC 1h, 48h horizon, t=0.65 | +4,029 | +58 | 43.6% | one-sided |
| ETH 1h, 48h horizon, t=0.65 | +4,544 | **−357** | 50.9% | one-sided |

At 48 hours both symbols make all their money on the long side. BTC's shorts win
43.6% of the time — worse than a coin flip — and contribute 58 USDT out of 4,087,
which is rounding. ETH's shorts lose outright. Over a sample where both assets
rose several-fold, a long-only strategy produces a rising equity curve and a win
rate above 50% while predicting nothing at all; that is what these two rows are.

Only the 24-hour BTC configuration earns on both sides, with shorts winning 55.1%
across 1,106 trades. That is the one result here that cannot be explained by
market drift.

Two things follow. First, the aggregate return in a threshold sweep is not
sufficient evidence — it hid a long-only bias in three of four configurations,
and would have hidden it in the headline 24h number too had the split not been
checked. Second, the 24h choice is now supported by a test it was not selected
on: it was chosen for surviving doubled costs, and it separately turns out to be
the only configuration with a genuine short side.

The classification gate agrees, for its own reasons: BTC's directional edge at
48h has a bootstrap CI of [−0.001, +0.029], which **includes zero**. At 24h the
same interval is [+0.056, +0.088]. The longer horizon is not significant.

**Configurations examined now total 63** (2 symbols × 4 horizons × 7 thresholds,
plus 7 on the 1m timeframe). The multiple-comparison caveat gets worse with every
one of them, which is precisely why the by-side split matters: it is a structural
test that noise cannot pass by luck, not another cell in a search grid.

## Meta-labelling: the technique works, the stack does not pay for itself

The idea: run the primary model loose so it produces many signals, simulate the
trade each one would have made, and train a second model on those outcomes to
decide which signals are worth taking. Two questions, two models — "which way?"
and "is this trade worth making after costs?" — because a bar can be genuinely
more likely to rise and still be a bad trade.

Every primary probability the secondary learns from comes from an inner purged
walk-forward split inside the training window. Skipping that would be fatal:
trained on in-sample primary output, the secondary learns to trust a confidence
that does not exist in production, concludes "always take the trade", and
becomes an expensive no-op that only reveals itself with real money.

**The technique demonstrably works.** Starting from a loose primary at threshold
0.50, adding the filter:

| | Primary alone | With meta filter |
|---|---|---|
| Signals | 14,320 | 3,789 |
| Sign accuracy | 55.22% | **57.30%** |
| Total return | +36.6% | **+48.2%** |
| Max drawdown | −48.4% | **−22.4%** |
| Sharpe | 0.33 | **0.53** |
| Return at 2x costs | **−40.8%** | **+18.8%** |
| Survives 2x costs | no | **yes** |

It converted a doubled-cost failure into a survivor and halved the drawdown.
That is real, and it is what learning from simulated outcomes is supposed to do.

**One failure mode, found and fixed.** With a single shared secondary, the filter
made the strategy *more* one-sided: shorts fell from 1,106 trades to 408 and lost
696 USDT. Long signals outnumber short ones several to one in a rising sample, so
the shared model simply learned "shorts do not work" — fitting the sample's drift
rather than any rule. Training a separate secondary per side fixed it: shorts
returned to 869 trades winning 58.2% and earning +465 USDT, and the directional
verdict flipped back to two-sided.

**And yet it does not earn its complexity.** Against the honest benchmark — one
model alone at the production threshold of 0.60, which is what would ship
otherwise:

| | Two-model stack | Single model at 0.60 |
|---|---|---|
| Signals | 3,789 | **4,295** |
| Total return | +48.2% | **+61.8%** |
| Max drawdown | −22.4% | **−19.1%** |
| Return at 2x costs | +18.8% | **+25.9%** |
| Directional | two-sided | two-sided |

The simple model wins on every axis, including signal count. The reason is
visible once stated: the secondary's job overlaps the threshold's job. Both
answer "is this signal confident enough to act on", and a scalar threshold does
it with one model and no extra fitting. The second model rediscovers the
primary's own confidence ranking and charges an extra layer of overfitting risk
for the privilege.

Note the gate that produced this answer was itself broken at first. The
benchmark was computed and then never reached the verdict function, so the
report printed "beats the simple production model" while comparing against
nothing. `tests/test_meta_report.py` now fails if the benchmark stops being
passed through.

**What would make a second model pay off** is giving it a job the threshold
cannot do — position sizing rather than filtering. A probability of profit is a
natural bet size, and varying size is something no threshold can express. That is
a different experiment and it has not been run.

## Decision

Defaults stay at the 24-hour horizon and a 0.60 threshold. That is now the only
configuration that is robust to doubled costs, statistically significant against
its baselines, and profitable on both sides of the market. The 48h and 72h
experiments are recorded above and rejected. **No live trading.** The next step is
paper trading, which measures the strategy forward on bars that did not exist
when any of these choices were made. That is the only test none of the above
can fake.

Expect the paper result to be worse than +70.9%. If it is not, re-read this
document before believing it.

**The 1m scalping horizon is closed, permanently.** Not because the model failed
— it is the best model here — but because the arithmetic forbids it at taker
fees. No amount of further modelling reopens it. Only an execution change
(maker-only fills) or a venue with materially lower fees would, and neither is
in scope.

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
