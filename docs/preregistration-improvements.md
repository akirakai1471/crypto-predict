# Pre-registration: two changes meant to make the model follow the market faster

**Written and committed before either change was run on real data.** As with
`preregistration-multisymbol.md`, the git history is the point: a criterion
chosen after seeing results is a rationalisation.

Date: 2026-09-30.

## The request, and the honest version of it

The request was a model that is "more sensitive and more accurate", close to
perfect. This project's own rule 5 says an out-of-sample hourly accuracy above
60% is to be treated as a bug until proven otherwise, and its measured edge is
53–58% directional accuracy on the bars it trades. Nothing here aims at
perfection; anything that appeared to reach it would be investigated as a leak.

"More sensitive" also cannot mean "trades more". The break-even arithmetic in
`docs/findings.md` shows shorter horizons and more signals lose to fees. The
version that is testable and not already refuted: **adapt faster when the market
changes regime, while trading the same 8% of bars.** Two changes aim at that.
Each is fixed below, with one value, chosen before running.

## The arms

Everything not listed is the frozen configuration of the multi-symbol run —
1h bars, 24-bar horizon, top 8% of bars by directional margin per fold, 5
purged walk-forward folds, 400 rounds, 3-fold out-of-fold calibration, maker
execution at 0.20% with chase and strict fills. Every arm is run in the same
invocation on the same data, and the baseline is re-run there rather than
compared with any number reported earlier.

| arm | change | fixed value |
|---|---|---|
| `baseline` | none | — |
| `recency` | training rows weighted by age, `0.5 ** (age / half_life)`, age counted in bars back from the newest row of each training window; calibrators fitted unweighted | half-life **8,760 bars** (one year) |
| `market_context` | five features from a context symbol at the same bar close: its 1-, 4- and 24-bar log returns, its 24-bar realised volatility, and this symbol's 24-bar return minus the context's | context **BTCUSDT** for every symbol except BTCUSDT, whose context is **ETHUSDT** |

Why these two. Recency weighting is the direct form of "follow the regime you
are in": the model still sees every year, but last month counts roughly twice
as much as last year. The half-life is one year because shorter ones leave five
folds with too little effective data in the early folds, and a sweep over
half-lives is exactly the search this document exists to prevent. Market
context is the most common piece of information a single-symbol model lacks:
most coins move with BTC, and a move in BTC over the last hours is known at the
bar close without looking ahead.

Missing context values (a context symbol with shorter history) are left as NaN
rather than dropped, so every arm trains and is tested on exactly the same rows.

## Symbols

The twenty from `preregistration-multisymbol.md`, unchanged: BTCUSDT, ETHUSDT,
SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, DOTUSDT, LTCUSDT,
BCHUSDT, ATOMUSDT, NEARUSDT, APTUSDT, ARBUSDT, OPUSDT, FILUSDT, INJUSDT,
TRXUSDT, ETCUSDT. The existing 15,000-bar minimum applies. No symbol is dropped
after seeing its result.

## The decision rule

Each arm is judged against the baseline on its own. Two arms are tested, so
each gets half the usual false-positive budget: α = 0.025.

An arm is **ADOPTED** only if all four hold:

1. **It wins on most symbols.** Higher sign accuracy on the traded bars than the
   baseline on at least 75% of the symbols tested, rounded up — 15 of 20 when
   all twenty are tested. Under no effect, 15 or more of 20 has a one-sided
   probability of 0.021.
2. **It wins by enough to matter.** Mean paired improvement in sign accuracy of
   at least **+0.5 percentage points.** Smaller than that is inside what a
   single re-run moves.
3. **It does not cost gates.** The number of symbols passing all three
   multi-symbol gates (positive return, survives doubled costs, two-sided) is
   not lower than the baseline's.
4. **It does not buy the 8% with worse probabilities everywhere else.** Mean
   log loss across symbols is not higher than the baseline's.

Anything else is **REJECTED** and the default configuration does not change.

A symbol skipped for short history counts toward neither side. Ties count as
not winning.

If both arms are adopted, the combination is **not** assumed to help. It gets
its own pre-registration and its own run.

## Caveats written down now

- Twenty crypto symbols over one five-year period are not twenty independent
  tests. The sign-test probability above is optimistic; `docs/findings.md`
  estimates the effective sample could be as small as five to eight.
- Sign accuracy on 8% of bars is the metric the strategy trades on, which is
  why it is primary, and also why it is noisy: about 1,300–4,000 trades per
  symbol.
- A pass here earns a place in the default configuration and nothing more. It
  does not make the edge proven; only the forward log can do that.

## What happens to the result

Recorded in `docs/findings.md` whichever way it goes, including a rejection of
both, with the per-symbol table.

Run with:

```
uv run cryptopred-model experiment --arms baseline,recency,market_context --jobs 0
```
