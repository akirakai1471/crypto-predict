# Findings — Phase P2 / P3

Date: 2026-08-26. All numbers are out-of-sample, from purged walk-forward
cross-validation with a 4-fold-plus purge/embargo split, on Binance USDT
perpetual data from 2019 to 2026.

## CORRECTION — the threshold rule was a calibration artefact

**Every return figure recorded below the "What the model actually knows" section
was inflated, and the cause was a mistake in this project, not in the market.**
The corrected numbers are here; the original text is left in place underneath so
the error stays visible rather than being quietly overwritten.

### What happened

Signals were selected by a fixed probability threshold of 0.60. That assumes the
probability scale means the same thing in every fold. It does not. Isotonic
calibration fitted on a small window overfits and emits extreme probabilities;
fitted on a large one it converges and rarely leaves the middle. Coverage across
the five walk-forward folds, same model, same threshold:

| Fold | Calibration rows | Bars traded |
|---|---|---|
| 0 | 1,412 | **26.6%** |
| 1 | 2,919 | 11.1% |
| 2 | 4,426 | 1.0% |
| 3 | 5,933 | **0.03%** |
| 4 | 7,439 | 3.3% |

The reported "8.4% coverage" was a pooled average dominated by fold 0, which
alone supplied **63% of all signals**. Fold 0 is 2020-12 to 2022-01 — the largest
bull run in the sample. The strategy was therefore concentrated in the best
period and nearly absent from the rest, not by design but by accident.

That is market timing produced by a calibration bug, and it inflated every
return figure derived from it.

### The corrected result

Selection is now by rank — trade the most confident 8% of bars *within each
fold* — which is invariant to the probability scale. Same model, same horizon,
same execution:

| | Threshold 0.60 (wrong) | Top 8% per fold (correct) |
|---|---|---|
| Signals | 4,217 | 4,020 |
| Sign accuracy | 58.90% | 58.36% |
| Coverage per fold | 26.6 / 11.1 / 1.0 / 0.03 / 3.3% | 8 / 8 / 8 / 8 / 8% |
| **Total return** | **+84.7%** | **+25.9%** |
| Max drawdown | −16.4% | −14.8% |
| Sharpe | — | 0.39 |
| At doubled fees | +61.4% | **+8.3%** |
| Long side | — | 3,165 trades, 53.6% win, +2,202 USDT |
| Short side | — | 855 trades, 54.0% win, **+535 USDT** |
| Directional verdict | two-sided | **two-sided** |

Nearly identical signal count and accuracy, a third of the return. The
difference was never skill.

### What survives, and what does not

**Survives.** Directional accuracy is unaffected — it sits at 56.5%, 58.4% and
57.0% at 4%, 8% and 15% coverage, stable because ranking does not change which
bars look most confident. The 20-symbol result (mean 53.70%, 17/20 above 50%)
measures accuracy, not selection, and stands. The short side still earns.

**Does not survive.** Every headline return: +70.9%, +84.7%, +109.4%, +118.2%.
The threshold plateau across 0.55–0.70 that looked like evidence of robustness
was measuring how much fold 0 each threshold happened to include. The doubled-fee
cushion is +8.3%, not +61.4%.

**Unclear.** The maker-versus-taker comparison and the sizing study were both run
under the broken selection rule. Their *relative* conclusions may hold, since
both arms used the same signals, but their absolute figures do not and they have
not been re-run.

*Both were re-run on 2026-09-11 under the rank rule; the quoted re-run notes
inside "Limit orders" and "Position sizing by probability" below have the
results.*

### Why this was missed for so long

Every gate in the project checked whether the *strategy* was sound. Nothing
checked whether the *rule that produced the signals* meant the same thing across
folds. It was only caught because the live system produced zero signals in two
days, which forced a comparison between the served model and the evaluated ones.
A coverage check now runs before any model is saved, and refuses to store a model
whose signal rate differs from its evaluation by more than 3x.

## SECOND CORRECTION — the coverage check was measured in-sample

Date: 2026-09-11. The check added above did not work, and it failed silently in
the way that matters: it passed a model that then traded nothing.

The saved model reported "fires on 12.87% of bars against an 8% target, ratio
1.6x" and was stored. It then produced **0 signals on 336 logged live bars**.

Two mistakes stacked:

1. **The cutoff came from the fold models, the model came from a different fit.**
   The final model trains on all the data, calibrates better, and so produces
   *narrower* directional margins than any fold model. A cutoff set on fold
   output is a bar the deployed model cannot clear.
2. **The check ran on the final model's own training rows.** Margins on data a
   model was fitted to are wide, because it has memorised those rows. The check
   therefore confirmed a rule against the one sample guaranteed to satisfy it.

### The fix

The final fit now holds back the last 3,000 bars. The older two-thirds set the
cutoff; the newest third verifies it. Neither block was in the training data, and
the verification block did not set the rule, so the check can fail — which is the
only property that makes a check worth having.

`COVERAGE_TOLERANCE` also dropped from 3.0 to 2.0. A regression test showed an
in-sample cutoff producing 3.2% coverage against an 8% target — a 2.5x shortfall
that the old tolerance waved through. Sampling noise on a 1,000-bar check is
about ±1.3x at two sigma, so 2.0 is comfortably outside noise.

### What changed in the numbers

The retrained model's honest cutoff is **0.0610**, not the 0.0795 taken from the
fold models. Verified coverage on unseen bars: **8.90% against an 8% target
(1.1x)**.

The training report was also scoring the wrong strategy. It backtested the
withdrawn 0.60 threshold while deploying the rank rule, so the verdict on screen
described trades nobody placed:

| rule | trades | win rate | net return | max DD | verdict |
|---|---|---|---|---|---|
| threshold 0.60 (withdrawn, still printed) | 7,215 | 52.50% | +70.4% | −42.3% | NO-GO |
| rank, top 8% (deployed) | 4,020 | 53.43% | +54.5% | −23.3% | GO |

The rank rule earns less and risks much less. The `sweep` command now sweeps
coverage rather than the threshold, because a threshold number no longer refers
to anything stable.

This does not change the project's conclusion. The edge remains unproven; see
"Twenty symbols, one frozen configuration". It changes which model is deployed
and makes the live experiment able to record anything at all.

## A gate that read the wrong verdict, and twelve days that went badly

Date: 2026-09-23. Two findings from one day, and the second is the more
uncomfortable.

### The save gate never asked whether the strategy made money

Retraining on data through 2026-09-23 put **ETHUSDT into the registry** at
−0.24% after costs, −30.5% maximum drawdown, and −20.7% at doubled costs. Its
own report printed `STRATEGY VERDICT: NO-GO` on the line above, and it was saved
anyway.

The gate read only the classification verdict — *does the model beat its
baselines?* — which ETH passes. The strategy verdict — *does that knowledge
survive fees?* — was computed for the report and never consulted. A model this
project had spent weeks correctly refusing walked in through a gate that was
asking the wrong question.

This is the fourth defect of the same shape recorded in this document:

| gate | what it measured | what it should have measured |
|---|---|---|
| coverage check | the model's own training rows | data the model had not seen |
| two-sided check | blocked ONE-SIDED, passed UNPROVEN | anything that is not TWO-SIDED |
| meta comparison | return at unequal exposure | return at matched exposure |
| **save gate** | **does it beat baselines** | **that, and does it survive fees** |

The recurring lesson is not "check more things". It is that **a gate must be
asked what it would reject**, not whether the current candidate passes. Every one
of these was written by someone who believed it worked, and every one was caught
by running a case it should have refused.

Both verdicts are now required, both failures are listed when either blocks, and
both are stored in the model metadata. The ETH model was deleted; ETHUSDT has no
model again, which is the correct output of its gates.

### The deployed model had a bad twelve days

The scheduler had been down since 2026-09-11. Backfilling to 2026-09-23 added
282 scored bars and the drift monitor fired immediately:

> `Signal rate: DRIFTED — fires on 3.9% of the last 284 bars against a 8% target`

The monitor was built on 2026-09-11 for exactly this, and it is worth recording
that it worked: the previous time a model went out of tune, it took **sixteen
days** to notice. This time the data had been back for minutes.

Accuracy over the new stretch, against the one baseline that matters:

| | accuracy | n | 95% CI |
|---|---|---|---|
| before 2026-09-11 | 50.7% | 383 | [45.7%, 55.6%] |
| **2026-09-11 to 09-23** | **41.0%** | 268 | [35.3%, 47.0%] |

The true labels over that stretch were UP 55.6%, DOWN 28.7%, FLAT 15.7%. So
**guessing "always UP" would have scored 55.6%** and the model scored 41.0%,
with the top of its interval still well below the naive baseline. It predicted
DOWN on 39.9% of bars in a market that rose.

**Four reasons not to over-read it**, all of which cut the same way — this is
suggestive, not settled. It is three-class accuracy over *every* bar, not the 8%
the strategy would actually trade, and the strategy traded almost none of them.
268 bars is twelve days and one regime. The rows are backfilled, so the model
used no future data but cannot prove it was written first. And the 55.6%
baseline is high *because this period rose* — the same long-bias trap this
document records elsewhere, now pointing the other way.

What is not ambiguous is the direction: drift and accuracy agreed, and both said
the model was out of tune with the present regime. BTCUSDT was retrained and
passed both gates (+50.1% after costs, −16.6% drawdown, +18.4% at doubled
costs, coverage 4.90% against an 8% target — still firing light, in the same
direction as the drift).

## The touch-probability interval, second pass: 85%, and where it still fails

Date: 2026-09-30. The section below ended with "closing the remaining gap —
studentized or bias-corrected bootstrap — was not attempted. It is open work."
This is that work. The short version: the interval is better everywhere it was
measured, the label moves from ≈80% to ≈85%, and one kind of market still
breaks it, along with every other method tried.

### What changed

Two things, each measured before it was kept.

**Studentized, not percentile.** The percentile bootstrap reads its interval
straight off the spread of resampled means. At 500 bars and a 48-bar block that
is the spread of eleven block means, treated as if it were known exactly. The
studentized version scales each resample by its own standard error and reads the
interval off those t-ratios, so a handful of blocks produces the heavier tails it
should. A resample whose blocks all agree has a standard error of zero; its ratio
is floored rather than dropped, because dropping those draws is what made rare
targets under-cover.

**The block grows with the sample**, as n^(1/3) above `MIN_CELL_BARS`. A fixed
48-bar block was the dominant error at large n: bias, not noise, and only a
longer block reduces bias. At 60,000 bars a 24-hour query now uses a 237-bar
block, a 72-hour one 710.

Plus one fix for an edge case the old code got badly wrong: **a level never
touched in the sample** now gets a rule-of-three upper bound (3 / number of
blocks) instead of the interval (0, 0). Zero touches in 500 bars is not
evidence the rate is zero; the old interval covered a true 1.3% rate 47% of the
time.

### The measurement

`scripts/touch_interval_coverage.py`, committed so it can be re-run, 800
series per case, fixed seed. "old" is a frozen copy of the previous interval
with its previous block; "new" is what `touch_probability` calls now.

| model | n | true rate | old | **new** | new width |
|---|---|---|---|---|---|
| markov | 500 | 0.400 | 81.1% | **93.4%** | 0.48 |
| markov | 3,000 | 0.400 | 85.0% | **91.1%** | 0.21 |
| walk, −3% in 24h | 500 | 0.293 | 87.9% | **94.6%** | 0.32 |
| walk, −3% in 24h | 3,000 | 0.293 | 90.6% | **92.6%** | 0.12 |
| walk, −3% in 24h | 20,000 | 0.293 | 90.1% | **92.5%** | 0.05 |
| walk, −3% in 72h | 500 | 0.543 | 80.4% | **90.2%** | 0.56 |
| walk, −3% in 72h | 3,000 | 0.543 | 91.5% | **94.5%** | 0.23 |
| walk, −3% in 72h | 20,000 | 0.543 | 91.4% | **94.0%** | 0.09 |
| *rare: −8% in 24h* | 500 | 0.013 | 47.1% | **91.2%** | 0.16 |
| *rare: −8% in 24h* | 3,000 | 0.013 | 83.0% | **96.0%** | 0.05 |
| *regimes: −3% in 24h* | 500 | 0.264 | 52.9% | **69.6%** | 0.32 |
| *regimes: −3% in 24h* | 3,000 | 0.264 | 58.6% | **68.5%** | 0.15 |
| *regimes: −3% in 72h* | 500 | 0.491 | 66.1% | **78.8%** | 0.55 |
| *regimes: −3% in 72h* | 3,000 | 0.491 | 78.2% | **85.1%** | 0.26 |

"markov" is the two-state chain from the first measurement, kept so the two are
comparable — and the old column reproduces it (81% and 85% here, 80% and 85%
then). "walk" is new: real touch outcomes, computed exactly as `touch_outcomes`
does, on a fat-tailed (t4) random walk at roughly BTC's hourly volatility. The
dependence there comes from overlapping forward windows, which is where it comes
from in the real data. Each figure is ±about 2pp at 800 reps.

**New beats old in all fourteen cases.** Every gain is paid for in width, which
is the point: the old interval was narrow because it was overconfident.

### The label is 85%, not 90%

The worst headline case in the table is 90.2%. The same case — walk, 72 hours,
500 bars — measured 86.5% and 86.8% in two earlier runs of the same method on
other seeds while this was being built. The label takes the lowest of those and
rounds down for the noise: **≈85%**. That is what `MEASURED_COVERAGE` holds and
what every string the user reads now says. Quoting the best run of three would
be the exact habit this document exists to catch.

The old label had the same problem in the other direction: its own method
measures **80.4%** on the 72-hour walk, and 77.1% on another seed. "≈80%" was
the Markov figure, and it was not the worst case.

### Where it still fails

**Volatility regimes that last for weeks** (rows in italics, "regimes": the walk
with volatility switching between 0.4% and 1.0% an hour, each level held ~500
bars). Touch outcomes then depend on each other over a far longer span than any
block, and coverage falls to **69–85%**. The old interval did worse (53–78%) and
nothing tried reached 90% without widening the interval to most of [0, 1]. A
longer block helps a little, and so did fixed-b HAC and batch-means variants
that were tried and not kept, but with six regime switches in 3,000 bars the
sample simply does not contain the information. Crypto volatility does cluster for weeks, so this is
not a corner case. Two things limit the damage, and neither removes it:

- The **conditional** figure is less exposed than the unconditional one. Its
  cell fixes the volatility bucket, so within a cell the regime that breaks the
  interval is mostly held constant.
- The rows are kept out of the 85% headline **because every method fails them**,
  not because they do not matter. Averaging them in would produce one number
  that describes neither case.

So the honest reading of "≈85%" is: that is what the interval delivers when
volatility is not in a long-lived regime shift. When it is, the interval is too
narrow and there is currently no method here that fixes it.

**Rare targets** are no longer a failure — 91% and 96% — but the reason is worth
knowing. At 500 bars most samples of a 1.3% event contain no touch at all, and
the rule-of-three bound is doing the work, not the bootstrap.

## The touch-probability interval covers 80%, not 95%

*Superseded 2026-09-30 by the second pass above; kept as it was written.*

Date: 2026-09-11 work, measured 2026-09-21. Recorded here because it is a
property of a number this project reports, and because the honest version is
less flattering than the label would have been.

`briefing/touch.py` reports how often price reached a level within a horizon.
The obvious interval for a proportion is Wilson, which this project already uses
correctly in `serve/status.py` — but that is for scored signals, which are
independent trials. Touch outcomes are not: bar *t* and bar *t+1* look forward
over windows sharing `horizon − 1` of their `horizon` bars, so at a 72-hour
horizon consecutive observations overlap 98.6%.

A moving-block bootstrap resamples contiguous blocks so that dependence survives
resampling. It is a large improvement. It is not a 95% interval.

**Measured coverage.** A two-state Markov chain with a known long-run rate of
0.4 and a mean run length of 50, 800 realisations, checking how often the true
value falls inside the interval:

| block | coverage at n=3,000 | coverage at n=500 |
|---|---|---|
| 1 | 22.9% | — |
| 24 | 75.2% | 73.1% |
| 48 | 85.0% | 79.5% |
| 72 | 88.6% | 79.6% |
| 200 | 91.1% | — |

Wilson on the same data covers 24.7%. `block=1` covers 22.9%, which is the
sanity check: with no blocking the bootstrap should behave like the naive
interval, and it does.

**Three things follow, and two of them are unflattering.**

The block length now defaults to `max(2 × horizon, 48)` rather than
`max(horizon, 24)`. A block equal to the dependence length under-covers; the
sweep above is why, and the change lifts a 24-hour query from 75.2% to 85.0%.

Switching to a *circular* block bootstrap — wrapping block starts modulo n so
edge positions are drawn as often as interior ones — was tried against the
review's prediction that it would help. Measured on identical draws it gains
0.3–1.0pp at n=3,000 and 3–4pp at n=500. Real, in the predicted direction, and
nowhere near enough to reach 95%. It was kept because it is free and correct,
not because it solved the problem.

**The interval is therefore labelled with what it delivers.** `MEASURED_COVERAGE
= 0.80` — the worst case actually measured, at `MIN_CELL_BARS` scale — is
appended to the `method` string that reaches the user: *"bootstrap khối 48 nến
(độ phủ đo được ≈80%, không phải 95%)"*. The `Measured` field that holds it is
named `interval`, not `ci95`, because the same field also holds genuine Wilson
95% intervals elsewhere and a name claiming 95% for both would be the quiet kind
of overstatement this document exists to record.

Closing the remaining gap — studentized or bias-corrected bootstrap — was not
attempted. It is open work, not a solved problem.

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

> **Re-run 2026-09-11 under the rank rule. Conclusion unchanged; the reasons
> changed entirely, and finding that out required fixing the gate.**
>
> Meta-labelling also selected its primary signals by a fixed probability
> threshold (0.40), across inner folds that each fit their own calibrator. Same
> defect as everywhere else, so the original "no benefit" result was measured on
> a lopsided sample. `MetaConfig.primary_coverage` now ranks within each inner
> fold. Re-run on BTCUSDT, 24h horizon:
>
> | | primary alone (20%) | with meta filter | benchmark (top 8%) |
> |---|---|---|---|
> | signals | 10,046 | 6,774 | 4,020 |
> | coverage | 20.0% | 13.5% | 8.0% |
> | sign accuracy | 57.16% | 58.53% | — |
> | total return | +93.6% | +120.2% | +79.5% |
> | max drawdown | −29.0% | −27.9% | **−17.5%** |
> | at 2x costs | +7.7% | +48.3% | +42.0% |
>
> The report initially printed **IMPROVEMENT**. It was wrong, and two gates had
> to be tightened before it stopped being wrong:
>
> **The stack is 99.4% long.** 6,732 long trades against 42 short. The two-sided
> check returned UNPROVEN — too few shorts to judge — and only ONE-SIDED was
> being gated, so "we cannot tell" passed as though it were "it works". Being
> unable to measure the short side is not evidence for a strategy; a book that
> is 99% long is a bet on the market rising, which this sample does on its own.
> UNPROVEN now fails.
>
> **The comparison was not exposure-matched.** 13.5% of bars against the
> benchmark's 8.0% is 69% more trades and therefore 69% more capital at risk.
> More money earns more money in a rising sample without predicting anything.
> This is the same error that invalidated the first sizing comparison, made
> again in a different file. The verdict now refuses to compare a stack that
> trades more than 1.25x the benchmark's coverage.
>
> Under the corrected gates the run reads **NO-GO**. And the drawdown column
> settles it independently of any gate: the stack draws down 27.9% where the
> plain production rule draws down 17.5%, for a return advantage that comes from
> carrying more risk in one direction.
>
> What the re-run *does* confirm is the sizing finding: the secondary genuinely
> improves sign accuracy, 57.16% → 58.53%. The information is there. Neither
> way of spending it — filtering or sizing — survives contact with the gates.

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

> **REVERSED by the 2026-09-11 re-run.** This section concluded that confidence
> carries no usable information above the cutoff and that flat betting wins.
> Re-run under the rank rule, the ordering flips and the diagnosis with it.
>
> **BTCUSDT, 24h horizon, top 8% by margin, exposure held equal at 0.3%:**
>
> | Rule | Return | Max DD | Sharpe | trades |
> |---|---|---|---|---|
> | fixed | +1.7% | −0.9% | 0.67 | 4,020 |
> | linear | +3.3% | −0.8% | 0.96 | 4,016 |
> | sqrt_kelly | +5.8% | −1.2% | 1.08 | 1,673 |
> | **kelly** | **+6.2%** | −1.5% | 1.05 | 1,673 |
>
> Every confidence-weighted rule now beats flat betting at equal capital, and
> Sharpe rises monotonically with how much the stake varies — the exact opposite
> of the ordering recorded below.
>
> The reason is the same defect that inflated everything else. Under a fixed
> threshold, "above 0.60" pooled folds whose probability scales differed by
> orders of magnitude, so confidence within the selected set was mostly a fold
> label, not a strength. Ranking within each fold makes confidence comparable,
> and the gradient appears:
>
> | confidence | trades | hit rate | avg net |
> |---|---|---|---|
> | [0.60, 0.65) | 1,169 | 65.5% | 0.856% |
> | [0.65, 0.70) | 124 | 67.7% | 1.220% |
> | [0.70, 0.80) | 89 | 71.9% | 1.353% |
> | [0.80, 1.00] | 16 | 75.0% | 1.557% |
>
> corr(confidence, net return) = +0.17, against the ~0 reported below.
>
> **Two reasons not to bank this.** The Kelly rules decline 58.4% of signals for
> want of funding, so their advantage mixes sizing with extra selection and is
> not a clean sizing result. And the top confidence buckets hold 89 and 16
> trades — the monotone tail is four points fitted on almost nothing. What is
> solid is the direction and the `linear` row, which trades the same 4,016
> signals as fixed and still wins on return, drawdown and Sharpe at once.
>
> ### And then it did not survive twenty symbols
>
> Criteria were committed first (`docs/preregistration-sizing.md`) precisely
> because the result above was found while fixing a bug. Running the same frozen
> configuration across all 20 symbols, at matched exposure:
>
> **5 of 20 favour linear = 25%, at or below the 35% registered as "noise".
> VERDICT: KEEP FIXED.** The live paper trader was not changed.
>
> BTCUSDT is the outlier that started this, and it is extreme: a +14.5pp return
> gap against +6.0pp for the next best. Drop it and the mean gap across the
> other 19 symbols is **−0.75pp** — linear is worse on average. Keeping it, the
> mean across all 20 is +0.015pp, which is zero.
>
> **The drawdown result is the more useful finding, because it is nearly
> unanimous.** Linear sizing produced a deeper maximum drawdown than fixed on
> **19 of 20 symbols** (the exception is INJUSDT, −1.7% against −1.8%). Several
> are not close: SOLUSDT −12.0% → −21.0%, BCHUSDT −8.2% → −22.4%, LTCUSDT
> −5.1% → −9.8%. At matched average exposure, staking by confidence concentrates
> the same capital into fewer, larger positions, so less of it is diversified
> across trades at any moment. That mechanism does not depend on whether
> confidence predicts anything, which is why it shows up on almost every symbol
> while the return effect does not.
>
> So the corrected picture is narrower than either earlier claim. Confidence
> **does** carry information the fixed threshold was hiding — the +0.17
> correlation and the monotone buckets are real, and the meta-labelling section
> below deserves the same suspicion. But acting on it through position size
> makes returns no better on average and drawdowns reliably worse.
>
> **The general lesson is worth more than the result.** Two separate conclusions
> in this document — meta-labelling and sizing — were "no signal here" findings
> that turned out to be measurements of a broken selection rule. A null result
> is only as trustworthy as the rule that produced the sample.

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

> **Re-run 2026-09-11 under the rank rule.** The table below was produced with
> the withdrawn 0.60 threshold. Re-running it with the corrected rule
> (BTCUSDT, 24h horizon, top 8% by margin, 4,020 signals) keeps the conclusion
> and changes the size of the effect:
>
> | Execution | Fill rate | Win rate | Return | Max DD | At 2x fees |
> |---|---|---|---|---|---|
> | taker | 100% | 53.4% | +54.5% | −23.3% | +22.2% |
> | maker 0.05% chase, touch-fill | 100% | 56.2% | +78.9% | −20.9% | +63.3% |
> | **maker 0.05% chase, strict fill** | 100% | 53.8% | **+56.9%** | −24.2% | **+40.6%** |
> | maker 0.05% skip, strict fill | 74.4% | 56.6% | +56.6% | −18.6% | +47.0% |
>
> **The maker advantage on returns is now +2.4pp, not +13.8pp.** Under strict
> fills every maker variant lands between +54.9% and +59.3% against taker's
> +54.5% — inside the noise of a single backtest. The paragraph below calling
> two thirds of the gain a fill-model artefact was right about the direction and
> understated it: under the corrected selection rule, *almost all* of the
> return advantage is the fill assumption.
>
> **The doubled-fee cushion survives and is the real reason to keep maker
> execution.** +40.6% against +22.2% is a near-doubling, and it comes from
> arithmetic rather than from any fill assumption: the round trip costs less, so
> the break-even accuracy is lower. That is what the live paper trader is
> buying — resilience to worse fills, not extra return.
>
> **`skip` is no longer clearly worse than taker** under the corrected rule
> (+56.6% against +54.5%), unlike under the threshold. The chase-versus-skip
> conclusion below is therefore weaker than stated: `chase` remains the default
> because skipping still declines the trades where price ran, but the evidence
> for it is now one backtest inside the noise band, not a clear ordering.

**BTCUSDT, 24h horizon, threshold 0.60, 4,217 signals (WITHDRAWN RULE):**

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

> **Re-run 2026-09-11 with fourteen more days of data. Verdict unchanged, and
> the re-run is more informative than the verdict.**
>
> All twenty symbols were refreshed to 2026-09-11 — 339 new hourly bars each,
> which is **0.6% more data** — and the frozen configuration re-run unchanged.
>
> **6 of 20 pass, 30%. Still INCONCLUSIVE** (previously 7 of 20, 35%).
>
> What 0.6% more data did to the individual results:
>
> | symbol | before | after | change |
> |---|---|---|---|
> | ADAUSDT | +2.9% | **+102.7%** | +99.8pp |
> | FILUSDT | +29.9% | −21.7% | −51.6pp |
> | TRXUSDT | +2.2% | −37.2% | −39.4pp |
> | DOTUSDT | +45.3% | +7.1% | −38.2pp |
> | ETCUSDT | +17.0% | +52.1% | +35.1pp |
>
> Adding bars moves every walk-forward fold boundary, so all five models refit on
> shifted windows and every test block changes. The mechanism is ordinary; the
> magnitude is the finding. **A per-symbol return figure in this document can
> move by 100 percentage points on 0.6% more data.** Nothing here should be read
> to better than a factor of two, and this is the direct evidence for that,
> replacing the earlier inference from disagreeing selection rules.
>
> The pass list moved with it. Kept: ATOM, BTC, DOGE, ETC, OP. Lost: DOT, FIL.
> Gained: ADA. Jaccard 0.62 — better than the 0.40 recorded between two different
> selection rules, but this is the *same* rule on *almost the same* data, so 0.62
> is the more damning of the two numbers. Which symbols pass is close to a
> coin flip.
>
> **The directional result went the other way and got stronger.** Mean sign
> accuracy across all twenty rose from 53.98% to **54.42%**, and symbols above
> 50% from 18/20 to **19/20** (only INJUSDT at 49.6% is below). Binomial p on
> 19/20 is 4e-05, against 4e-04 before — though crypto symbols move together, so
> both p-values overstate the evidence and neither is twenty independent tests.
>
> That split is the whole project in one table. The thing that survives more data
> is the claim that the model knows something about direction. The thing that
> does not survive is any particular statement about money.

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

### Re-run under the corrected rule

The table above was produced with the broken threshold selection. Re-running the
whole experiment with rank selection and matched out-of-fold calibration gives
the **same headline and a different cast**:

| | Threshold rule | Rank rule |
|---|---|---|
| Passed all three | 7/20 (35%) | **7/20 (35%)** |
| Positive return | 11/20 | 13/20 |
| Survives doubled fees | 9/20 | 8/20 |
| Two-sided | 14/20 | 12/20 |
| Passing symbols | ADA APT ATOM BTC DOGE DOT LINK | **ATOM BTC DOGE DOT ETC FIL OP** |

Only four of seven symbols pass under both rules — a Jaccard overlap of 0.40.
**Which symbols pass is not stable**, even though how many pass is. That is
itself a result: at this effect size, whether an individual symbol clears three
gates is close to a coin flip, and any story about *why* ADA passed and SOL
failed would be a story about noise.

Rank selection is far more robust than a threshold but it is not fully
calibration-invariant either. Per-class isotonic maps are monotone individually,
yet the margin |P(up) − P(down)| combines two of them, so a different calibration
can reorder margins. BTCUSDT returns +49.6% here against +25.9% measured with
mismatched calibration, on the same rule. The correlation between per-symbol sign
accuracy across the two runs is only 0.355.

**None of the return figures in this project should be read to better than about
a factor of two.**

### The stronger finding underneath

Pass rate answers "is this tradeable". It is the wrong instrument for asking "does
the model know anything", because a symbol can predict direction genuinely and
still lose to its own transaction costs. Directional accuracy answers that
question directly, and it is unambiguous:

Measured under the corrected rule, with the earlier run's figures beside them:

| | Rank rule | Threshold rule |
|---|---|---|
| Mean sign accuracy across 20 symbols | **53.98%** | 53.70% |
| 95% confidence interval for the mean | **[52.78%, 55.19%]** — excludes 50% | [51.81%, 55.58%] |
| Symbols above 50% | **18 of 20**, p = 0.0002 | 17 of 20, p = 0.0013 |
| Excluding BTCUSDT, the symbol tuned on | **17 of 19**, p = 0.0004, mean 53.75% | 16 of 19, p = 0.0022 |

The two runs agree closely on the aggregate even though they disagree on which
individual symbols pass. That is the signature of a real but small effect: the
average is measurable, each individual draw is not.

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
