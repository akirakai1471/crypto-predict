# Pre-registration: does the BTC result generalise?

**Written and committed before the experiment was run.** The git history is the
point: criteria chosen after seeing results are not criteria, they are
rationalisations, and this document exists so that cannot happen here.

Date: 2026-08-26.

## The problem this addresses

`docs/findings.md` records one configuration that passes every gate — BTCUSDT,
1h bars, 24-hour horizon, threshold 0.60 — and states plainly that it was found
by searching 63 configurations, which is roughly the hit rate chance produces.
ETHUSDT failed. The document's own words: *"The whole 24-hour result rests on one
asset. One asset is an anecdote."*

That is the weakest claim in the project and it is testable.

## What is being tested

The configuration is **frozen**. Nothing below will be tuned per symbol:

| Setting | Value | Fixed by |
|---|---|---|
| Bar interval | 1h | spec |
| Label horizon | 24 bars | break-even arithmetic |
| Signal threshold | 0.60 | middle of the BTC plateau |
| Label band | 0.5 × ATR(14) | spec |
| Walk-forward folds | 5, purged, 1% embargo | spec |
| Boosting rounds | 400 | BTC run |
| Execution | maker 0.20%, chase, strict fill | execution study |
| Costs | 0.05% taker / 0.02% maker / 0.02% slippage | Binance |

Because no parameter varies across symbols, this is **out-of-sample validation
of a fixed hypothesis**, not another sweep. The result is the *distribution*
across symbols, and every symbol is reported whether it passes or fails.

## Symbols

Every USDT perpetual below, subject only to having at least 15,000 hourly bars
(roughly two years) so five folds are possible. The list was chosen for
liquidity before any of them was tested, and no symbol will be dropped after
seeing its result.

SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, DOTUSDT, LTCUSDT,
BCHUSDT, ATOMUSDT, NEARUSDT, APTUSDT, ARBUSDT, OPUSDT, FILUSDT, INJUSDT,
TRXUSDT, ETCUSDT

Plus BTCUSDT and ETHUSDT, already measured, for reference.

## The criterion, and why it is not total return

In a sample where most of crypto rose several-fold, a long-biased strategy
produces a positive return on almost any symbol while predicting nothing. Total
return therefore cannot discriminate, and using it would guarantee a flattering
answer.

The criterion is the **two-sided verdict**: the strategy must earn on the short
side as well as the long side, with at least 100 short trades and a short win
rate at or above 50%. Market drift cannot produce that.

A symbol **passes** when all three hold:

1. Positive total return after costs
2. Survives doubled fees
3. Two-sided (as defined in `paper/replay.py`, unchanged)

## Predictions, stated in advance

- **If the effect is real and general:** at least **40%** of symbols pass.
- **If BTC was a lucky draw:** roughly **10% or fewer** pass, consistent with
  the base rate of noise clearing three gates at once.
- **Between 10% and 40%** is the awkward middle: some real effect concentrated
  in a subset, or a weak effect near the noise floor. It will be reported as
  inconclusive rather than argued into either camp.

## What will not be done afterwards

- No symbol will be excluded for a reason discovered after its result.
- No threshold, horizon or execution setting will be re-tuned to improve the
  pass rate. If a different setting looks better in hindsight, that observation
  belongs in a new pre-registration, not in this one's conclusion.
- The pass count will be reported as a fraction of **all** symbols attempted,
  including any that fail for boring reasons like insufficient history — those
  are listed separately so the denominator stays honest.

## Why this can still be wrong

All symbols share the same period and the same market. Crypto assets are highly
correlated, so twenty symbols are not twenty independent tests; the effective
sample size is smaller than the count suggests, and a market-wide regime could
produce a correlated pass or a correlated failure. A pass rate above 40% would
be encouraging, not conclusive.

The only test that escapes this is the forward one already running.
