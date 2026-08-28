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

## Position sizing by probability: also no benefit, and now we know why

Meta-labelling failed because the secondary's job overlapped the threshold's.
Sizing is the job a threshold genuinely cannot do — a threshold says act or
don't, and cannot say *how much*. So the second model was given that job
instead: stake more when the probability is higher, using fractional Kelly.

The first comparison was rigged and had to be thrown out. Kelly near break-even
stakes a few percent per trade, so it deployed 1.7% average exposure against
fixed sizing's 8.1% and lost on total return for a reason that had nothing to do
with allocation skill: it simply bet less money. Matching average exposure
across every rule isolates the real question — given the same capital, does
betting more on stronger signals beat betting the same on all of them?

**BTCUSDT, 24h horizon, threshold 0.60, exposure held equal at ~1.6%:**

| Rule | Return | Max DD | Sharpe |
|---|---|---|---|
| **fixed** | **+12.8%** | **−3.6%** | **0.67** |
| sqrt_kelly | +11.6% | −4.3% | 0.57 |
| kelly | +10.6% | −5.0% | 0.47 |
| linear | +9.7% | −5.5% | 0.40 |

Fixed wins on return, drawdown and Sharpe simultaneously, and the ordering is
monotonic in how aggressively each rule varies the stake: `sqrt_kelly`, the
flattest rule, lands closest to fixed. **The more the stake varies, the worse
the result.**

The diagnosis is measurable rather than a shrug. Among signals that already
cleared the 0.60 threshold:

| Confidence | Trades | Hit rate | Avg abs move | Avg net |
|---|---|---|---|---|
| 0.60 – 0.65 | 1,116 | 56.99% | 2.632% | +0.315% |
| **0.65 – 0.70** | 800 | **63.00%** | 2.824% | **+0.794%** |
| 0.70 – 0.80 | 1,320 | 59.32% | 2.433% | +0.156% |
| 0.80 – 1.00 | 981 | 57.29% | 2.537% | +0.241% |

Correlation between confidence and net return is **−0.0223**; between confidence
and absolute move size, **−0.0263**. The most profitable bucket is the middle
one, not the most confident.

This does not contradict the earlier finding that accuracy rises monotonically
with confidence across all bars, from 36% to 63%. Both are true, and together
they say something precise: **the threshold extracts all the information
confidence contains, and there is nothing left over for stake size.** Below the
threshold confidence separates good from bad; above it, it does not.

That closes the question the meta-labelling section left open. A second model
was tried at filtering and at sizing. Neither pays, for the same underlying
reason in two guises.

## Limit orders: the first change that survives its own stress test

The break-even table named this as the largest lever available: maker fees are
roughly a third of the taker cost, and cost is the term that kills every
configuration in this document. Unlike the two failed second-model experiments,
this changes execution rather than prediction.

It also carries the easiest way to fool yourself in the whole project. A limit
order fills only when price comes to it, so it declines exactly the trades where
price ran the way the model predicted. Modelling the cheaper fee without that
selection produces a number that is wrong in the most flattering direction.

**BTCUSDT, 24h horizon, threshold 0.60, 4,217 signals:**

| Execution | Fill rate | Win rate | Return | Max DD | At 2x fees |
|---|---|---|---|---|---|
| taker | 100% | 55.7% | +70.9% | −16.7% | +33.6% |
| maker 0.20% chase, touch-fill | 100% | 57.4% | **+109.4%** | −16.4% | +85.0% |
| **maker 0.20% chase, strict fill** | 100% | 56.2% | **+84.7%** | −16.4% | **+61.4%** |
| maker 0.20% skip, strict fill | 63.0% | 57.4% | +67.0% | −15.3% | +57.3% |

Three things to read here.

**Two thirds of the apparent gain was the fill model, not the market.** Fill
detection from OHLC bars cannot be exact, so the assumption was stress-tested
rather than asserted: a strict mode requires price to trade *through* the limit
instead of merely touching it, which is closer to what happens with a queue
ahead of you. The advantage over taker falls from +38.5pp to +13.8pp. Without
that test this section would have claimed +109%.

**What survives is the fee, and it is structural.** The doubled-fee cushion
nearly doubles, from +33.6% to +61.4%. That is the same arithmetic as the
break-even table: a 0.060% round trip instead of 0.140% lowers the accuracy the
strategy must reach, and the margin it earns is not sensitive to any modelling
choice.

**Chasing unfilled limits is mandatory.** Every pure `skip` variant loses to
taker execution under the strict rule (+67.0%, +65.9%, +68.0% against +70.9%).
Skipping an unfilled limit means skipping the trades where price moved away
immediately — which are the winners. The adverse selection is real and it is
large enough to erase the fee saving on its own.

**It does not manufacture an edge that is not there.** On ETHUSDT every maker
variant improves returns (+21.1% to +32.1% under the strict rule) and turns the
doubled-fee result from −5.6% to +17.1% — and every one stays ONE-SIDED. Cheaper
fees make a one-sided bet cheaper; they do not make it a prediction.

**Caveats that no backtest can settle.** Bar-range fill detection ignores queue
position, order size and the intrabar path. The `chase` rule assumes a missed
fill is noticed and crossed at the bar's close, which on 1h bars means up to an
hour of drift. And the paper trader currently places market orders at the bar
close, so none of this is live yet: adopting maker execution is an
implementation change to the trading path, not a config switch.

## The paper trader now places the orders it was measuring

Backtesting maker execution is one thing; running it is another, and the gap
between them is where a strategy quietly stops being the one that was tested.
The paper trader posts real resting orders now, and two details had to be right.

**The limit is quoted from the signal bar's close, not the next bar's open.** A
backtest can use the open of the bar an order works in; a live trader cannot,
because that bar has not started when the order is sent. Binance klines are
continuous — `open[t+1]` equals `close[t]` exactly — so the two references
produce identical results here, which was verified rather than assumed. On a
market with gaps they would not.

**A resting order is not a position.** It sits in its own state until price
reaches it, and the dashboard shows it separately. Collapsing the two would
overstate what the strategy is holding at any moment.

The two implementations agree where it matters. Over the same 4,217 signals, the
backtest engine and the paper trader both report a **63.0% resting fill rate** —
independently computed, to the decimal. On the paper path, switching from market
to limit orders moves final equity from 15,493 to **16,419 USDT**, the win rate
from 54.90% to 55.89%, and short-side profit from +793 to +914 USDT, while the
directional verdict stays TWO-SIDED.

Two bugs surfaced while building it, both worth recording.

`paper_trades` was keyed on `(symbol, interval, entry_time)`. That works while
every order fills instantly, and breaks the moment orders rest: a filled limit's
`entry_time` moves onto the bar the next pending order already occupies, and the
insert fails. The key is now `signal_time`, which is what actually identifies an
order — one per signal. Existing databases are rebuilt in place.

The dashboard's freshness indicator measured a bar's age from its **open**, so
every hourly bar looked an hour staler than it was and the light sat red on a
healthy feed. It also took the worst age across all intervals, including the 1m
store that the scheduler deliberately does not sync. Both are fixed: age is
measured from the close, and only the traded interval is judged.

## Twenty symbols, one frozen configuration

The criteria for this were committed before it ran
(`docs/preregistration-multisymbol.md`). Nothing was tuned per symbol: same
horizon, same threshold, same execution, same folds, twenty USDT perpetuals.

**The headline, by the criteria written in advance: 7 of 20 pass, 35% — which
falls in the band pre-registered as INCONCLUSIVE.** Not the 40% that would have
supported a general effect, not the 10% that would have marked BTC as a lucky
draw. Reported as it stands.

| Passed all three gates | Failed |
|---|---|
| ADA, APT, ATOM, BTC, DOGE, DOT, LINK | ARB, AVAX, BCH, ETC, ETH, FIL, INJ, LTC, NEAR, OP, SOL, TRX, XRP |

Broken down by gate: 11/20 returned a profit after costs, 9/20 survived doubled
fees, 14/20 were two-sided.

### The stronger finding underneath

Pass rate answers "is this tradeable". It is the wrong instrument for asking "does
the model know anything", because a symbol can predict direction genuinely and
still lose to its own transaction costs. Directional accuracy answers that
question directly, and it is unambiguous:

| | |
|---|---|
| Mean sign accuracy across 20 symbols | **53.70%** |
| 95% confidence interval for the mean | **[51.81%, 55.58%]** — excludes 50% |
| Symbols above 50% | **17 of 20**, binomial p = 0.0013 |
| Excluding BTCUSDT, the symbol tuned on | **16 of 19**, p = 0.0022, mean 53.40% |

**The predictive signal generalises.** It survives removing the one symbol the
configuration was fitted to, which is the check that matters most — BTC's own
number here (59.3%, the highest in the set) is optimistically biased and cannot
be used as evidence for itself.

The two results together say something precise: the model really does predict
direction across the asset class, at roughly 53–54%, and that is **not enough to
trade most of them**. Symbols that passed averaged 57.4% sign accuracy; symbols
that failed averaged 51.7%. The gates are not rejecting a model that knows
nothing; they are rejecting an edge too thin to survive the toll, exactly as the
break-even table predicts.

### The caveat that keeps this from being conclusive

**These are not twenty independent tests.** Crypto assets are strongly
correlated and share one five-year period, so the effective sample is
considerably smaller than twenty and every p-value above is optimistic. If the
effective sample were closer to eight, the binomial result would sit near p =
0.04 rather than 0.001; nearer five, it would not be significant at all.

A market-wide regime could produce a correlated pass or a correlated failure,
and this experiment cannot distinguish "the model reads crypto" from "the model
reads this particular five years of crypto". Only forward time separates those,
which is what the live log is accumulating.

### What is notably absent

No symbol was excluded, none was skipped for short history, and the pass rate is
quoted over all twenty attempted. AVAXUSDT fired 8,448 signals at 49.7% accuracy
and lost 70.5% — the largest single failure, and it is in the table rather than
in a footnote.

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
