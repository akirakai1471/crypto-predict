# Market briefing and question answering — design

Date: 2026-09-21. Status: approved for planning.

## Why this exists

The user was shown a market-analysis answer from another assistant and asked for
the same capability here. That answer is worth reading closely, because it
contains two kinds of sentence that look alike and are not alike.

Checkable: *"RSI-14 is 66.1"*, *"funding averages +0.0058%"*, *"price is $2,669"*.

Not checkable: *"risk of falling to $2,400 **increases**"* — from what to what?
*"the scenario with **more reasonable probability**"* — more than what? These read
as quantitative and contain no measured quantity at all.

This project exists because of how easily the second kind survives inspection.
`docs/findings.md` is a record of confident figures evaporating: a +24 billion
percent return from a compounding bug, a threshold that was a calibration
artefact, a coverage gate that passed a model firing on nothing, an
"IMPROVEMENT" verdict on a book that was 99.4% long. Every one of those looked
like a number and behaved like a sentence.

So the goal is not to reproduce that answer. It is to answer the same question
with the parts that can be measured, measured — and the parts that cannot,
labelled.

**The question "when will ETH drop so I can buy" is answerable from data.** ETH
has 59,523 hourly bars in the store. Instead of "risk increases", the system can
say: in the 4,112 historical hours that resembled this one, price touched −3%
within 72 hours 58.4% of the time, and the median wait was 38 hours.

## Scope

**In:** BTCUSDT and ETHUSDT, 1h bars. A measurement layer callable without any
API key, and a natural-language question-answering layer over it using Claude
Opus 5 with tool calling. A CLI entry point.

**Out, deliberately:**

- **News and citations.** There is no news source in this project and inventing
  one is the failure mode being designed against.
- **Buy/sell recommendations.** The system reports measured probabilities; the
  user decides. This project has never issued a recommendation and adding one
  would change what it is.
- **The other 18 symbols.** Their bars are refreshed manually; only BTC and ETH
  update on the scheduler. Answering about a symbol whose data is two weeks old
  invites exactly the staleness problem this design guards against.
- **Open interest beyond 30 days.** Binance retention, already recorded in the
  2026-08-26 spec.
- **A dashboard chat box.** CLI first. The HTTP layer comes after the tools are
  proven, and reuses them unchanged.

## Architecture

Two layers, and the lower one does not know the upper one exists.

```
src/cryptopred/briefing/     measurement. pure functions over stored data.
  snapshot.py                price, changes, volume, funding, freshness
  indicators.py              RSI / MACD / ATR / ADX, each tagged unvalidated
  levels.py                  pivots, swing highs and lows, consolidation range
  touch.py                   P(touch level within H hours), conditioned
  regime.py                  the volatility x trend bucketing shared by touch.py
  provenance.py              the Measured / Convention / Unavailable wrapper
  report.py                  assembles the table
  cli.py                     cryptopred-brief

src/cryptopred/ask/          the mouth.
  tools.py                   the six tool definitions over briefing/
  prompt.py                  system prompt, including the findings.md summary
  audit.py                   post-check: every number traced to a tool result
  session.py                 the Claude Opus 5 tool-calling loop
  cli.py                     cryptopred-ask
```

`briefing/` has no dependency on `anthropic` and no network access beyond the
existing Binance ingest. `cryptopred-brief BTCUSDT` prints the whole table with
no API key present. The question-answering layer is a presentation choice on top
of it, and if it is removed the measurements remain.

This split is not tidiness. It is what makes the numbers testable: every figure
the assistant can utter is produced by a pure function with a unit test, and the
model's only job is to select and phrase them.

## The provenance wrapper

Every value leaving `briefing/` carries its own epistemic status. There is no
path that returns a bare float.

```python
@dataclass(frozen=True)
class Measured:
    value: float
    n: int                      # observations behind it
    ci95: tuple[float, float] | None
    method: str                 # how it was computed, in one phrase

@dataclass(frozen=True)
class Convention:
    value: float
    reading: str                # what the convention says, e.g. ">70 is overbought"
    validated: bool = False     # never True until someone measures it here
    warning: str = "Chưa đo chỉ báo này có giá trị dự báo trên dữ liệu này."

@dataclass(frozen=True)
class Unavailable:
    reason: str                 # why there is no number, in plain language
```

RSI cannot be returned as `Measured`. Not because RSI is useless, but because
this project has not measured whether RSI predicts anything on this data, and
the type system should say so rather than a comment nobody reads. If someone
later runs that measurement, the type changes and the warning goes away — that
is the intended path, and `validated=True` should require a committed result.

`Unavailable` is a first-class return, not an exception. "ETHUSDT has no model
because it fails the gates at −6.83% after costs" is an answer, and a more
useful one than silence.

## The six tools

Each is a thin wrapper over `briefing/`, with a JSON schema, `strict: true`, and
`additionalProperties: false`.

### 1. `market_snapshot(symbol, interval="1h")`

Last close, changes over 24h / 7d / 30d, quote volume, current funding and its
trailing mean, and **data freshness in hours**. Freshness is not a footnote: it
is a top-level field, because an answer computed on ten-day-old bars is wrong in
a way the prose will not reveal.

### 2. `indicators(symbol, interval="1h")`

RSI-7, RSI-14, MACD line / signal / histogram, ATR and its percentile rank, ADX,
realized volatility, distance from EMA. All `Convention`. The tool's description
says so, so the model reads it before it reads the values.

### 3. `levels(symbol, interval="1h")`

Classic daily pivots (P, R1-R3, S1-S3), swing highs and lows from fractal
detection over a configurable lookback, and the most recent consolidation range.
`Convention` — these are conventional constructions, not measured attractors.
Their value is that they name the prices a user is likely to ask about, which
feeds tool 4, where the actual measurement happens.

### 4. `touch_probability(symbol, target, horizon_hours, interval="1h")`

The one that answers the user's question.

`target` is a single required object with a discriminator, so a caller cannot
pass `-3` and leave the tool guessing whether that means 3% down or a price of
minus three: `{"kind": "pct", "value": -0.03}` or
`{"kind": "price", "value": 2500.0}`. A percentage is signed relative to the
current close; a price is compared to it directly, and the direction of the test
follows from whether it sits above or below.

Method:

1. Classify the current bar into a **regime cell**: volatility tercile
   (ATR-14 / close) crossed with trend tercile (close / EMA-168 − 1). Nine cells.

   **Tercile boundaries come from an expanding window, not from the full
   history.** The boundary for bar *t* uses bars up to *t* only. Computing them
   once over all data would let the 2026 volatility distribution decide which
   bucket a 2020 bar belongs to — future information deciding a past
   classification. The effect is small and the fix is cheap, and this project has
   already paid for assuming that a small leak is a harmless one. Bars before the
   window has enough history to form stable boundaries (`MIN_HISTORY_BARS = 2000`)
   are excluded from the sample rather than bucketed on thin quantiles.
2. Collect every historical bar in the same cell.
3. For each, look forward `horizon_hours`. A downward target is touched if any
   bar's `low` reaches it; an upward target if any bar's `high` does. Intrabar
   extremes, not closes — a level that was traded was touched.
4. Point estimate = touched / total across all overlapping windows in the cell.
5. **Confidence interval by moving-block bootstrap**, block length
   `max(horizon_hours, 24)` bars.
6. Among the windows that touched, report the time-to-touch distribution:
   median, p25, p75, p90.

Step 5 is the part that matters and the part most likely to be got wrong.
Overlapping forward windows are heavily autocorrelated — consecutive windows
share almost all their bars — so the effective sample size is far below the
window count. A Wilson interval, which this codebase already uses correctly
elsewhere for independent trials, would report a confidence here that the data
does not support. The block bootstrap resamples contiguous blocks and so
preserves the autocorrelation the naive interval ignores.

Step 6 is what actually answers "when". A probability alone does not: the
time-to-touch distribution has a long right tail, and a median of 38 hours
against a p90 of 214 hours is the honest shape of the answer.

**Sample floor.** Conditioning on two dimensions multiplies cells, and a thin
cell produces a confident-looking number from noise. Below `MIN_CELL_BARS = 500`
observations the tool returns `Unavailable` naming the cell and its count, rather
than a probability. The trade is deliberate: conditioning makes the estimate more
relevant and noisier, and the floor is where relevance stops being worth the
noise. Terciles rather than quintiles for the same reason.

The tool also reports the **unconditional** probability alongside the conditional
one. When they disagree sharply that is information; when the conditional is
unavailable the unconditional is still an answer.

### 5. `model_signal(symbol, interval="1h")`

Current prediction, the directional margin, the saved cutoff, and whether the
margin clears it. For ETHUSDT this returns `Unavailable` with the reason: no
saved model, because it fails the gates at −6.83% after costs and negative at
doubled costs. Also surfaces the drift state from `serve/drift.py` — a model that
has stopped firing must not be quoted as though it were trading.

### 6. `track_record(symbol, interval="1h")`

Live accuracy with n and Wilson interval (independent trials here, so Wilson is
correct), backfilled accuracy reported separately, the walk-forward reference,
and the **break-even accuracy** the strategy must exceed given current costs.
Break-even belongs in the same breath as accuracy: 54% sounds good until it is
placed next to the 56% the fees require.

## The honesty mechanism

Three layers, in increasing order of how much they actually bind.

**Layer 1 — the types.** A conventional indicator cannot leave `briefing/`
without its warning attached, because there is no constructor that omits it.

**Layer 2 — the system prompt.** Contains the rules (never state a number that
did not come from a tool result; never convert a conventional reading into a
prediction; say "không đo được" when that is the answer) and a short summary of
`docs/findings.md`, so the model knows this project's specific history of
overconfidence rather than being told to be careful in the abstract.

Prompt caching applies here: tools and system prompt are stable across questions
and form the cached prefix, with the question after the last breakpoint.

**Layer 3 — the audit, which is the one with teeth.** After the answer is
produced, `ask/audit.py` extracts every numeric token from the text and matches
it against the set of numbers returned by tool calls in that session, allowing
for rounding and formatting. Unmatched numbers are flagged, and the answer is
followed by an auto-generated source table mapping each figure to the tool that
produced it.

This catches invented price levels and invented percentages — precisely the
failure mode in the answer that prompted this work. It does **not** catch
qualitative claims like "động lượng còn tích cực", and that limit is accepted
rather than papered over: the audit reports what it checked, so its silence on
prose is visible rather than implied.

## Data flow

```
question → refresh bars (Binance public, existing ingest, no key)
         → Claude Opus 5 + 6 tools (tool_runner loop)
         → answer text
         → audit: numbers vs tool results
         → stdout: answer + source table + any flags
```

Bars refresh before answering. If the refresh fails the system answers from
stored data and states the staleness in the answer itself, not only in a log —
a silent answer on stale data is the same class of error as a silent model that
has stopped firing, which cost this project sixteen days.

## Model configuration

- `claude-opus-5`, chosen by the user.
- `thinking: {"type": "adaptive"}` — on by default for this model.
- `output_config: {"effort": "medium"}` as the default. The reasoning load is
  light: select tools, read numbers, write carefully. Overridable by flag.
- Streaming, with `.get_final_message()`; `eager_input_streaming: true` on the
  client tools, with input validated against each schema before execution.
- `client.beta.messages.tool_runner` rather than a hand-written loop. The tools
  are where the risk lives and they are tested directly; the loop is not worth
  owning.
- Errors handled most-specific-first: `NotFoundError`, `RateLimitError`,
  `APIStatusError`, `APIConnectionError`. `stop_reason == "refusal"` is checked
  before reading content.

**Cost.** Opus 5 is $5.00 / $25.00 per million tokens. A question costs roughly
$0.03–0.06 before caching; the stable prefix should cut repeat questions
substantially. The CLI prints token usage and estimated cost per answer, because
a per-question cost the user cannot see is a cost they will find out about later.

## CLI

```bash
cryptopred-ask "khi nào ETH rớt về 2500?"
cryptopred-brief ETHUSDT          # no API key needed
```

`cryptopred-ask` flags: `--symbol` (default inferred from the question, BTCUSDT
if ambiguous), `--effort`, `--no-fetch`, `--json`, `--model`.

With no credentials available, `cryptopred-ask` explains how to authenticate and
points at `cryptopred-brief`, which answers the same questions in table form. It
does not fail with a stack trace, and it does not pretend to answer.

## Error handling

| Condition | Behaviour |
|---|---|
| No API credentials | Explain `ant auth login` or `ANTHROPIC_API_KEY`; point at `cryptopred-brief` |
| Binance refresh fails | Answer from stored data, state staleness in the answer |
| Regime cell below floor | `Unavailable` naming the cell and count; unconditional figure still returned |
| No saved model (ETH) | `Unavailable` with the gate result as the reason |
| Model drifted or silent | Surfaced by `model_signal`; answer must not quote a dead model |
| Tool raises | `tool_result` with `is_error: true`, so the model sees and reports it |
| `stop_reason: "refusal"` | Surfaced to the user, not swallowed |
| Audit finds unmatched number | Flagged in the source table, answer still shown |

## Testing

`briefing/` is pure, so it is tested directly and exhaustively:

- **Touch probability against known answers.** A synthetic series that always
  falls 3% within 24 hours must return 1.0; one that never does must return 0.0.
  A series with a known touch rate must return it within tolerance.
- **Intrabar extremes.** A bar whose `low` pierces the target but whose `close`
  does not must count as touched. This is the easiest detail to get wrong.
- **The bootstrap interval must be wider than Wilson on autocorrelated data.**
  This is the specific claim the design makes, so it is the specific claim the
  test pins.
- **The sample floor fires.** A cell below `MIN_CELL_BARS` returns `Unavailable`,
  and the message names the count.
- **Regime classification is point-in-time.** The cell assigned to bar *t* must
  not change when bars after *t* are appended — the prefix-invariance test from
  the existing leakage suite, applied to the bucketing. This is the test that
  catches full-history tercile boundaries, which are the tempting shortcut.
- **Provenance.** No code path returns a conventional indicator without its
  warning; asserted by construction and by test.

`ask/` is tested without network:

- **The audit catches an invented number.** A fabricated answer containing a
  price no tool returned must be flagged. This test is the reason the audit
  exists, so it is written first.
- **The audit tolerates rounding.** A tool returning 58.4231 and an answer saying
  58.4% must not be flagged.
- **Staleness reaches the answer.** Stored data ten days old must produce an
  answer containing the staleness statement.
- **The loop** runs against a fake client; no test makes a paid API call.

## What this does not change

The standing constraint holds and is worth restating because this design
introduces an API key: **an Anthropic key is not an exchange key.** There is
still no exchange API key anywhere in this project, no order-placement code path,
and no way for any of this to place a real trade. Paper trading only. Live
trading remains a separate spec that has not been written.
