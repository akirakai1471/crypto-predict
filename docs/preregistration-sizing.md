# Pre-registration — does confidence-weighted sizing beat flat betting?

Written and committed **before** the multi-symbol run, on 2026-09-11. Nothing
below may be edited after the numbers exist; corrections go in a dated appendix.

## Why this needs pre-registering

A single BTCUSDT backtest just reversed a conclusion this project had already
written down. Under the withdrawn probability threshold, flat betting beat every
confidence-weighted rule on return, drawdown and Sharpe at once. Under the
corrected rank rule, the ordering flipped completely:

| Rule | Return | Max DD | Sharpe |
|---|---|---|---|
| fixed | +1.7% | −0.9% | 0.67 |
| linear | +3.3% | −0.8% | 0.96 |

That is one symbol, one horizon, one backtest, arrived at while fixing a bug.
Every previous time this project acted on a number of that shape, the number
shrank or vanished. The live paper trader currently stakes a fixed amount per
trade; changing that on this evidence would be the same mistake in a new costume.

## The question

Among trades selected by the rank rule, does scaling the stake with the model's
confidence beat staking the same amount every time, **at equal average
exposure**, across symbols?

Exposure matching is not optional. A rule that bets less money earns less money
for reasons that have nothing to do with allocation skill; the first sizing
comparison in this project had to be thrown out for exactly that.

## Configuration — frozen before running

Identical to `FrozenConfig` in `src/cryptopred/models/validation.py`: 24-bar
horizon, top 8% by rank per fold, 5 folds, 400 rounds, 3 inner calibration
folds, maker 0.20% chase with strict fill, doubled-cost stress. Same 20 symbols
as `docs/preregistration-multisymbol.md`, same order, none added or dropped
after seeing a result.

Only `linear` is tested against `fixed`. The Kelly variants are excluded on
purpose: they decline 58.4% of signals for want of funding, so their advantage
mixes sizing with extra selection and cannot answer the question asked here.
`linear` trades the same signals as `fixed`, which makes the comparison clean.

## Criteria — fixed in advance

A symbol **favours linear** if, at matched exposure, linear's total return
exceeds fixed's *and* linear's max drawdown is no worse than fixed's by more
than 20% relative. Both conditions, because a return improvement bought with
proportionally more risk is not an improvement.

Read the pass rate over all 20 attempted symbols:

- **≥ 65% favour linear** → adopt linear sizing in the live paper trader.
- **≤ 35%** → keep fixed sizing. The BTCUSDT result was noise.
- **between** → keep fixed sizing and record the result as inconclusive. Ties go
  to the incumbent, because switching on a coin flip adds a change to explain
  later with nothing bought for it.

The asymmetry is deliberate: 65% to adopt versus 35% to reject is not a
symmetric test. Changing the live configuration should require more evidence
than leaving it alone.

## What this cannot settle

The 20 symbols are crypto perpetuals over one period and move together, so these
are not 20 independent tests. A 70% pass rate here is weaker than 70% would be
across uncorrelated markets. Whatever comes out, it is evidence about sizing
under this one selection rule at this one horizon, and it does not make the
underlying edge any more proven than the multi-symbol run left it.
