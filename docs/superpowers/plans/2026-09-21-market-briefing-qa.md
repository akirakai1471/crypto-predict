# Market Briefing and Question Answering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer questions like "khi nào ETH rớt về 2500?" with measured probabilities from 59,523 stored bars instead of unfalsifiable narrative.

**Architecture:** Two layers. `src/cryptopred/briefing/` is pure measurement — no `anthropic` import, runs with no API key, every value carries its epistemic status as a type. `src/cryptopred/ask/` wraps it in a Claude Opus 5 tool-calling loop. Removing the upper layer leaves the numbers intact.

**Tech Stack:** Python 3.12, pandas, numpy, typer, pytest, `anthropic` SDK (Tool Runner beta), existing `cryptopred` feature/ingest/serve modules.

**Spec:** `docs/superpowers/specs/2026-09-21-market-briefing-qa-design.md`

**Standing rules for every task:**
- Run `uv run --no-sync pytest -q` and `uv run --no-sync ruff check src tests` before each commit. The suite is at 451 tests and must stay green.
- End every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Commands run from `C:\Users\akira\Desktop\crypto-predict`.
- Use the Edit tool for multi-line Python edits. Bash heredocs mangle `\n` inside f-strings — this has cost this project four separate debugging rounds.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/cryptopred/briefing/__init__.py` | Package marker |
| `src/cryptopred/briefing/provenance.py` | `Measured` / `Convention` / `Unavailable` types |
| `src/cryptopred/briefing/regime.py` | Expanding-window volatility × trend bucketing |
| `src/cryptopred/briefing/touch.py` | P(touch level within H hours) + time-to-touch |
| `src/cryptopred/briefing/snapshot.py` | Price, changes, funding, data freshness |
| `src/cryptopred/briefing/indicators.py` | RSI / MACD / ATR / ADX, all `Convention` |
| `src/cryptopred/briefing/levels.py` | Daily pivots, swing highs and lows |
| `src/cryptopred/briefing/report.py` | Assembles the Vietnamese table |
| `src/cryptopred/briefing/cli.py` | `cryptopred-brief` |
| `src/cryptopred/ask/__init__.py` | Package marker |
| `src/cryptopred/ask/tools.py` | Six tool functions + JSON schemas |
| `src/cryptopred/ask/prompt.py` | System prompt with the findings.md summary |
| `src/cryptopred/ask/audit.py` | Traces every number in the answer to a tool result |
| `src/cryptopred/ask/session.py` | Claude Opus 5 tool-calling loop |
| `src/cryptopred/ask/cli.py` | `cryptopred-ask` |

Tests are flat under `tests/`, matching the existing convention.

---

# PHASE 1 — the measurement layer (no API key required)

## Task 1: Provenance types

**Files:**
- Create: `src/cryptopred/briefing/__init__.py`
- Create: `src/cryptopred/briefing/provenance.py`
- Test: `tests/test_briefing_provenance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_provenance.py`:

```python
"""A conventional indicator must not be able to escape without its warning.

RSI is not useless. But this project has never measured whether RSI predicts
anything on this data, and the type system should say so rather than a comment
nobody reads.
"""

import pytest

from cryptopred.briefing.provenance import Convention, Measured, Unavailable


def test_measured_carries_its_sample_size():
    m = Measured(value=0.584, n=2401, ci95=(0.569, 0.599), method="block bootstrap")
    assert m.to_dict() == {
        "source": "measured",
        "value": 0.584,
        "n": 2401,
        "ci95": [0.569, 0.599],
        "method": "block bootstrap",
    }


def test_convention_is_unvalidated_by_default_and_warns():
    c = Convention(value=66.1, reading="quy ước >70 là quá mua")
    d = c.to_dict()
    assert d["source"] == "convention"
    assert d["validated"] is False
    assert "chưa đo" in d["warning"].lower()


def test_convention_cannot_claim_validation_without_evidence():
    """validated=True is a claim about measurement, so it needs a citation."""
    with pytest.raises(ValueError, match="evidence"):
        Convention(value=66.1, reading="x", validated=True)

    ok = Convention(value=66.1, reading="x", validated=True, evidence="docs/findings.md#rsi")
    assert ok.to_dict()["validated"] is True
    assert "warning" not in ok.to_dict()


def test_unavailable_gives_a_reason_not_a_number():
    u = Unavailable(reason="ETHUSDT trượt cổng kiểm (−6.83% sau phí) nên không có model")
    assert u.to_dict() == {"source": "unavailable", "reason": u.reason}
    assert not hasattr(u, "value")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_provenance.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/__init__.py`:

```python
"""Measurement layer. No LLM, no API key, no network beyond the existing ingest."""
```

Create `src/cryptopred/briefing/provenance.py`:

```python
"""Every value leaving this package carries its own epistemic status.

There is no path that returns a bare float. This project's history is a record
of confident figures evaporating — a compounding bug, a calibration artefact, a
gate that passed a model firing on nothing — and each one looked like a number
while behaving like a sentence. The distinction is enforced by the type rather
than by a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONVENTION_WARNING = (
    "Chưa đo chỉ báo này có giá trị dự báo trên dữ liệu này. Đừng đặt lệnh dựa vào nó."
)


@dataclass(frozen=True)
class Measured:
    """A number computed from data, with the sample behind it."""

    value: float
    n: int
    ci95: tuple[float, float] | None = None
    method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "measured",
            "value": self.value,
            "n": self.n,
            "ci95": list(self.ci95) if self.ci95 is not None else None,
            "method": self.method,
        }


@dataclass(frozen=True)
class Convention:
    """A number computed by a conventional formula whose predictive value this
    project has not measured. RSI, MACD, pivot points.

    `validated` requires `evidence`: a pointer to where the measurement lives.
    Without it the flag would be a claim nobody has to back up, which is the
    exact failure this class exists to prevent.
    """

    value: float
    reading: str
    validated: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.validated and not self.evidence:
            raise ValueError(
                "validated=True needs `evidence` naming where the measurement lives"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": "convention",
            "value": self.value,
            "reading": self.reading,
            "validated": self.validated,
        }
        if self.validated:
            out["evidence"] = self.evidence
        else:
            out["warning"] = CONVENTION_WARNING
        return out


@dataclass(frozen=True)
class Unavailable:
    """No number, and why. A first-class answer, not an exception.

    "ETHUSDT has no model because it fails the gates" is more useful than
    silence and more honest than a fallback figure.
    """

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": "unavailable", "reason": self.reason}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_provenance.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/ tests/test_briefing_provenance.py
git commit -m "$(cat <<'EOF'
Provenance types for the briefing layer

Every value leaving briefing/ carries its epistemic status as a type: Measured
with its sample size, Convention with a warning attached by construction, or
Unavailable with a reason. There is no constructor that returns a bare float.

Convention.validated=True requires evidence naming where the measurement lives.
Without that, the flag is a claim nobody has to back up - which is the failure
the class exists to prevent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Regime bucketing, point-in-time

**Files:**
- Create: `src/cryptopred/briefing/regime.py`
- Test: `tests/test_briefing_regime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_regime.py`:

```python
"""Regime cells must be decided by the past only.

Computing tercile boundaries once over all history would let the 2026 volatility
distribution decide which bucket a 2020 bar belongs to. The effect is small; this
project has already paid for assuming a small leak is a harmless one.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.regime import MIN_HISTORY_BARS, RegimeCell, classify_regimes
from tests.conftest import make_ohlcv


def test_terciles_split_roughly_evenly():
    bars = make_ohlcv(n=6000, seed=3)
    cells = classify_regimes(bars, min_history=500)
    vol = cells["vol_bucket"].dropna()
    counts = vol.value_counts(normalize=True)
    assert set(counts.index) == {0, 1, 2}
    assert counts.min() > 0.2


def test_early_bars_are_excluded_not_bucketed_on_thin_quantiles():
    bars = make_ohlcv(n=3000, seed=4)
    cells = classify_regimes(bars, min_history=500)
    assert cells["vol_bucket"].iloc[:490].isna().all()
    assert cells["vol_bucket"].iloc[600:].notna().any()


def test_classification_is_prefix_invariant():
    """The cell for bar t must not change when bars after t are appended.

    This is the test that catches full-history tercile boundaries, which are the
    tempting shortcut.
    """
    bars = make_ohlcv(n=4000, seed=5)
    full = classify_regimes(bars, min_history=500)
    prefix = classify_regimes(bars.iloc[:3000], min_history=500)

    a = full["vol_bucket"].iloc[:3000]
    b = prefix["vol_bucket"]
    assert a.equals(b)

    a_t = full["trend_bucket"].iloc[:3000]
    b_t = prefix["trend_bucket"]
    assert a_t.equals(b_t)


def test_cell_of_names_both_dimensions_in_vietnamese():
    cell = RegimeCell(vol_bucket=2, trend_bucket=0)
    assert cell.label == "biến động cao / xu hướng giảm"
    assert cell.key == (2, 0)


def test_min_history_constant_is_exported():
    assert MIN_HISTORY_BARS == 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_regime.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.regime'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/regime.py`:

```python
"""Which kind of market hour is this one like?

Touch probabilities conditioned on nothing answer a question nobody asked: the
unconditional chance of a 3% drop mixes calm weeks with crashes. Conditioning on
volatility and trend makes the comparison set resemble now.

Both boundaries come from an expanding window. The tercile edges for bar t use
bars up to t only, so a classification made in 2020 cannot be revised by what
volatility did in 2026.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptopred.features.base import ewma
from cryptopred.features.volatility import atr

# Bars needed before an expanding quantile is stable enough to bucket on. Below
# this the boundaries move with almost every new bar and the label is noise.
MIN_HISTORY_BARS = 2000

VOL_LABELS = {0: "biến động thấp", 1: "biến động vừa", 2: "biến động cao"}
TREND_LABELS = {0: "xu hướng giảm", 1: "đi ngang", 2: "xu hướng tăng"}


@dataclass(frozen=True)
class RegimeCell:
    vol_bucket: int
    trend_bucket: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.vol_bucket, self.trend_bucket)

    @property
    def label(self) -> str:
        return f"{VOL_LABELS[self.vol_bucket]} / {TREND_LABELS[self.trend_bucket]}"


def expanding_tercile(series: pd.Series, min_history: int) -> pd.Series:
    """Which third of its own past does each value sit in?

    `expanding().rank(pct=True)` ranks the window's final value against
    everything before it, which is exactly the point-in-time question. The same
    property is already relied on by `features.base.pct_rank`.
    """
    ranks = series.expanding(min_periods=min_history).rank(pct=True)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[ranks <= 1 / 3] = 0.0
    out[(ranks > 1 / 3) & (ranks <= 2 / 3)] = 1.0
    out[ranks > 2 / 3] = 2.0
    return out


def classify_regimes(
    bars: pd.DataFrame, min_history: int = MIN_HISTORY_BARS
) -> pd.DataFrame:
    """Per-bar volatility and trend buckets. NaN where history is too thin."""
    vol = atr(bars, 14) / bars["close"]
    trend = bars["close"] / ewma(bars["close"], 168) - 1.0
    return pd.DataFrame(
        {
            "vol_bucket": expanding_tercile(vol, min_history),
            "trend_bucket": expanding_tercile(trend, min_history),
        },
        index=bars.index,
    )


def current_cell(cells: pd.DataFrame) -> RegimeCell | None:
    """The cell of the most recent fully classified bar, or None."""
    valid = cells.dropna()
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return RegimeCell(int(last["vol_bucket"]), int(last["trend_bucket"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_regime.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/regime.py tests/test_briefing_regime.py
git commit -m "$(cat <<'EOF'
Point-in-time regime bucketing for touch probabilities

Volatility tercile crossed with trend tercile, nine cells. Conditioning makes the
comparison set resemble the current hour - the unconditional chance of a 3% drop
mixes calm weeks with crashes.

Tercile boundaries come from an expanding window, not from the full history.
Computing them once over all data would let the 2026 volatility distribution
decide which bucket a 2020 bar belongs to. The prefix-invariance test from the
leakage suite is extended to catch exactly that shortcut.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Touch outcomes — did price reach the level, and when

**Files:**
- Create: `src/cryptopred/briefing/touch.py`
- Test: `tests/test_briefing_touch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_touch.py`:

```python
"""A level that was traded was touched.

Using closes instead of intrabar extremes is the easiest detail to get wrong
here, and it biases every probability downward.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.touch import touch_outcomes


def _bars(close, low=None, high=None):
    n = len(close)
    close = np.asarray(close, dtype=float)
    low = np.asarray(low, dtype=float) if low is not None else close
    high = np.asarray(high, dtype=float) if high is not None else close
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_a_series_that_always_falls_enough_returns_one():
    close = np.full(50, 100.0)
    low = np.full(50, 90.0)  # every bar dips 10%
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=5)
    assert touched.mean() == 1.0


def test_a_series_that_never_falls_enough_returns_zero():
    close = np.full(50, 100.0)
    low = np.full(50, 99.0)  # never more than 1% down
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=5)
    assert touched.mean() == 0.0


def test_an_intrabar_wick_counts_even_when_the_close_does_not():
    """The bar that pierces the level and recovers still traded there."""
    close = np.full(20, 100.0)
    low = np.full(20, 100.0)
    low[5] = 96.0  # a wick, close stays at 100
    bars = _bars(close, low=low)
    touched, _ = touch_outcomes(bars, target_pct=-0.03, horizon=5)
    # bars 0..4 look forward far enough to see bar 5
    assert touched[0] and touched[4]
    assert not touched[5]


def test_time_to_touch_is_the_first_bar_that_reached_it():
    close = np.full(20, 100.0)
    low = np.full(20, 100.0)
    low[3] = 90.0
    touched, bars_to = touch_outcomes(_bars(close, low=low), target_pct=-0.03, horizon=10)
    assert touched[0]
    assert bars_to[0] == 3  # three bars ahead
    assert bars_to[2] == 1


def test_an_upward_target_uses_highs():
    close = np.full(20, 100.0)
    high = np.full(20, 100.0)
    high[4] = 110.0
    touched, _ = touch_outcomes(_bars(close, high=high), target_pct=0.05, horizon=6)
    assert touched[0]


def test_bars_without_a_full_forward_window_are_dropped():
    """A bar whose horizon runs past the end of the data has no outcome, and
    counting it as 'not touched' would bias every probability downward."""
    close = np.full(20, 100.0)
    touched, _ = touch_outcomes(_bars(close), target_pct=-0.03, horizon=5)
    assert len(touched) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_touch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.touch'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/touch.py`:

```python
"""Did price reach a level within a horizon, and how long did it take?

This is the measurement that replaces "risk of falling to $2,400 increases".
It is descriptive, not predictive: it reports what price did in the past, and
makes no claim that the future resembles it beyond the regime conditioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def touch_outcomes(
    bars: pd.DataFrame, target_pct: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each bar, did the target get touched in the next `horizon` bars?

    Returns (touched, bars_to_touch), both length `len(bars) - horizon`. Bars
    whose forward window runs past the end of the data are dropped rather than
    counted as misses, which would bias every probability downward.

    `bars_to_touch` is the number of bars ahead of the first touch, and is
    meaningless (0) where `touched` is False.

    A downward target is tested against `low` and an upward one against `high`:
    a level that was traded was touched, whatever the bar closed at. Testing
    closes instead is the easiest error to make here and it biases the answer.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 bar")
    n = len(bars)
    if n <= horizon:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    close = bars["close"].to_numpy(dtype=float)
    extreme = (bars["low"] if target_pct < 0 else bars["high"]).to_numpy(dtype=float)

    # windows[i] == extreme[i : i + horizon]; the window starting at i + 1 is the
    # forward window for bar i, so bar i itself is never its own outcome.
    windows = sliding_window_view(extreme, horizon)[1:]
    targets = close[: len(windows)] * (1.0 + target_pct)

    hit = (
        windows <= targets[:, None] if target_pct < 0 else windows >= targets[:, None]
    )
    touched = hit.any(axis=1)
    bars_to_touch = np.where(touched, hit.argmax(axis=1) + 1, 0)
    return touched, bars_to_touch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_touch.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/touch.py tests/test_briefing_touch.py
git commit -m "$(cat <<'EOF'
Touch outcomes: did price reach the level, and when

For each bar, whether the target was touched within the horizon and how many
bars it took. Downward targets test intrabar lows and upward ones test highs - a
level that was traded was touched, whatever the bar closed at. Testing closes is
the easiest error to make here and it biases every probability downward.

Bars whose forward window runs past the end of the data are dropped rather than
counted as misses, for the same reason.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Conditional touch probability with a block-bootstrap interval

**Files:**
- Modify: `src/cryptopred/briefing/touch.py`
- Test: `tests/test_briefing_touch_probability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_touch_probability.py`:

```python
"""Overlapping forward windows are not independent trials.

Consecutive windows share almost all their bars, so the effective sample size is
far below the window count. Wilson - which this codebase uses correctly for
independent trials in serve/status.py - would report a confidence the data does
not support.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.provenance import Measured, Unavailable
from cryptopred.briefing.touch import (
    MIN_CELL_BARS,
    block_bootstrap_ci,
    touch_probability,
)
from cryptopred.serve.status import wilson_interval
from tests.conftest import make_ohlcv


def test_bootstrap_is_wider_than_wilson_on_autocorrelated_data():
    """The specific claim the design makes, so the specific claim pinned here."""
    # 1000 outcomes in runs of 100: heavily autocorrelated, overall rate 0.5
    touched = np.concatenate([np.full(100, i % 2 == 0) for i in range(10)])

    lo_b, hi_b = block_bootstrap_ci(touched, block=100, n_boot=400, seed=1)
    lo_w, hi_w = wilson_interval(int(touched.sum()), len(touched))

    assert (hi_b - lo_b) > (hi_w - lo_w) * 2


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    touched = rng.random(2000) < 0.3
    lo, hi = block_bootstrap_ci(touched, block=24, n_boot=400, seed=2)
    assert lo < touched.mean() < hi


def test_a_thin_cell_returns_unavailable_naming_the_count():
    """Conditioning on two dimensions multiplies cells; a thin one produces a
    confident-looking number from noise."""
    bars = make_ohlcv(n=2600, seed=11)
    result = touch_probability(
        bars, target_pct=-0.03, horizon=24, min_cell_bars=10_000
    )
    assert isinstance(result["conditional"], Unavailable)
    assert "10,000" in result["conditional"].reason or "10000" in result["conditional"].reason
    # the unconditional figure survives regardless
    assert isinstance(result["unconditional"], Measured)


def test_unconditional_is_always_returned_alongside():
    bars = make_ohlcv(n=6000, seed=12)
    result = touch_probability(bars, target_pct=-0.03, horizon=24)
    assert isinstance(result["unconditional"], Measured)
    assert result["unconditional"].n > 0
    assert 0.0 <= result["unconditional"].value <= 1.0


def test_wait_time_percentiles_are_reported_in_hours():
    """A probability alone does not answer 'when'. The distribution has a long
    right tail and the median alone hides it."""
    bars = make_ohlcv(n=6000, seed=13)
    result = touch_probability(bars, target_pct=-0.01, horizon=72)
    wait = result["wait_hours"]
    assert wait["median"] <= wait["p90"]
    assert wait["p25"] <= wait["median"] <= wait["p75"]


def test_the_cell_is_named_so_the_answer_can_quote_it():
    bars = make_ohlcv(n=6000, seed=14)
    result = touch_probability(bars, target_pct=-0.03, horizon=24, min_cell_bars=100)
    assert "biến động" in result["cell_label"]


def test_min_cell_bars_constant_is_five_hundred():
    assert MIN_CELL_BARS == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_touch_probability.py -q`
Expected: FAIL with `ImportError: cannot import name 'MIN_CELL_BARS'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/cryptopred/briefing/touch.py`:

```python
# Below this many observations a regime cell produces a confident-looking number
# out of noise. The trade is deliberate: conditioning makes the estimate more
# relevant and noisier, and this is where relevance stops being worth the noise.
MIN_CELL_BARS = 500


def block_bootstrap_ci(
    touched: np.ndarray,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval that survives autocorrelation.

    Resamples contiguous blocks rather than individual outcomes, so the
    dependence between neighbouring windows is preserved instead of being
    assumed away. A Wilson interval on the same data reports roughly the width
    it would have if every window were an independent trial, which is wrong by
    a factor of several here.
    """
    n = len(touched)
    if n == 0:
        return (0.0, 1.0)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)

    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n]
        means[b] = touched[idx].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def touch_probability(
    bars: pd.DataFrame,
    target_pct: float,
    horizon: int,
    min_cell_bars: int = MIN_CELL_BARS,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """How often price reached this level, in hours that resembled this one.

    Returns both the conditional and unconditional figures. When they disagree
    sharply that is information; when the cell is too thin the unconditional one
    is still an answer.
    """
    from cryptopred.briefing.regime import classify_regimes, current_cell

    touched, bars_to = touch_outcomes(bars, target_pct, horizon)
    if touched.size == 0:
        reason = f"chỉ có {len(bars):,} nến, cần hơn {horizon:,} nến để nhìn tới đích"
        return {
            "conditional": Unavailable(reason=reason),
            "unconditional": Unavailable(reason=reason),
            "cell_label": "không xác định",
            "wait_hours": {},
        }

    block = max(horizon, 24)
    unconditional = Measured(
        value=float(touched.mean()),
        n=int(touched.size),
        ci95=block_bootstrap_ci(touched, block=block, n_boot=n_boot),
        method=f"mọi nến lịch sử, bootstrap khối {block} nến",
    )

    cells = classify_regimes(bars)
    cell = current_cell(cells)
    if cell is None:
        return {
            "conditional": Unavailable(
                reason="chưa đủ lịch sử để xếp nến hiện tại vào chế độ nào"
            ),
            "unconditional": unconditional,
            "cell_label": "không xác định",
            "wait_hours": _wait_percentiles(touched, bars_to, bars),
        }

    aligned = cells.iloc[: touched.size]
    in_cell = (
        (aligned["vol_bucket"] == cell.vol_bucket)
        & (aligned["trend_bucket"] == cell.trend_bucket)
    ).to_numpy()
    cell_touched = touched[in_cell]
    cell_bars_to = bars_to[in_cell]

    if cell_touched.size < min_cell_bars:
        conditional: Measured | Unavailable = Unavailable(
            reason=(
                f"ô '{cell.label}' chỉ có {cell_touched.size:,} quan sát, "
                f"cần ít nhất {min_cell_bars:,} — số sẽ là nhiễu"
            )
        )
        wait_source = (touched, bars_to)
    else:
        conditional = Measured(
            value=float(cell_touched.mean()),
            n=int(cell_touched.size),
            ci95=block_bootstrap_ci(cell_touched, block=block, n_boot=n_boot),
            method=f"nến cùng chế độ '{cell.label}', bootstrap khối {block} nến",
        )
        wait_source = (cell_touched, cell_bars_to)

    return {
        "conditional": conditional,
        "unconditional": unconditional,
        "cell_label": cell.label,
        "wait_hours": _wait_percentiles(*wait_source, bars),
    }


def _wait_percentiles(
    touched: np.ndarray, bars_to: np.ndarray, bars: pd.DataFrame
) -> dict[str, float]:
    """How long the touches that happened took, in hours.

    Only the windows that touched contribute: the ones that did not have no wait
    time, and filling them with the horizon would invent data.
    """
    hit = bars_to[touched]
    if hit.size == 0:
        return {}
    hours_per_bar = _bar_hours(bars)
    q = np.quantile(hit, [0.25, 0.5, 0.75, 0.90]) * hours_per_bar
    return {
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p90": float(q[3]),
        "n": int(hit.size),
    }


def _bar_hours(bars: pd.DataFrame) -> float:
    if len(bars.index) < 2:
        return 1.0
    delta = bars.index[1] - bars.index[0]
    return float(delta.total_seconds() / 3600.0)
```

Add to the imports at the top of `src/cryptopred/briefing/touch.py`:

```python
from typing import Any

from cryptopred.briefing.provenance import Measured, Unavailable
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_touch_probability.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/touch.py tests/test_briefing_touch_probability.py
git commit -m "$(cat <<'EOF'
Conditional touch probability with a block-bootstrap interval

The measurement that replaces "risk of falling to $2,400 increases": how often
price actually reached a level within a horizon, in the historical hours that
resembled this one, with the time-to-touch distribution.

The interval is a moving-block bootstrap, not Wilson. Overlapping forward windows
share almost all their bars, so the effective sample size is far below the window
count; a regression test pins the bootstrap at more than twice Wilson's width on
autocorrelated outcomes. Wilson remains correct where this project uses it today,
in serve/status.py, where the trials are independent.

Conditioning on volatility crossed with trend gives nine cells and a thin cell
produces a confident-looking number from noise, so below 500 observations the
conditional figure is Unavailable naming the count. The unconditional figure is
returned alongside regardless - when they disagree sharply that is information,
and when the cell is thin it is still an answer.

Wait-time percentiles come only from the windows that touched. Filling the rest
with the horizon would invent data, and the p90 is what shows the long tail that
a median alone hides.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Market snapshot with data freshness

**Files:**
- Create: `src/cryptopred/briefing/snapshot.py`
- Test: `tests/test_briefing_snapshot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_snapshot.py`:

```python
"""Freshness is a top-level field, not a footnote.

An answer computed on ten-day-old bars is wrong in a way the prose will not
reveal. The live experiment lost sixteen days to a model that had silently
stopped firing; stale data is the same class of failure.
"""

import pandas as pd

from cryptopred.briefing.snapshot import STALE_AFTER_HOURS, market_snapshot
from tests.conftest import make_ohlcv


def test_snapshot_reports_price_and_changes():
    bars = make_ohlcv(n=1000, seed=21)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=bars["close_time"].max())
    assert snap["last_close"]["value"] == bars["close"].iloc[-1]
    assert "change_24h" in snap
    assert "change_7d" in snap


def test_freshness_is_top_level_and_flags_stale_data():
    bars = make_ohlcv(n=1000, seed=22)
    late = bars["close_time"].max() + pd.Timedelta(days=10)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=late)
    assert snap["data_age_hours"] > 200
    assert snap["is_stale"] is True
    assert "cũ" in snap["staleness_note"]


def test_fresh_data_says_so_without_a_warning():
    bars = make_ohlcv(n=1000, seed=23)
    snap = market_snapshot(
        bars, funding=pd.DataFrame(), now=bars["close_time"].max() + pd.Timedelta(minutes=30)
    )
    assert snap["is_stale"] is False
    assert snap["staleness_note"] == ""


def test_missing_funding_is_unavailable_not_zero():
    bars = make_ohlcv(n=1000, seed=24)
    snap = market_snapshot(bars, funding=pd.DataFrame(), now=bars["close_time"].max())
    assert snap["funding_now"]["source"] == "unavailable"


def test_funding_present_is_measured_with_its_sample():
    bars = make_ohlcv(n=1000, seed=25)
    idx = pd.date_range("2024-01-01", periods=120, freq="8h", tz="UTC", name="funding_time")
    funding = pd.DataFrame({"funding_rate": [0.0001] * 120}, index=idx)
    snap = market_snapshot(bars, funding=funding, now=bars["close_time"].max())
    assert snap["funding_now"]["source"] == "measured"
    assert snap["funding_mean_30d"]["n"] > 0


def test_stale_threshold_is_three_hours():
    assert STALE_AFTER_HOURS == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_snapshot.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.snapshot'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/snapshot.py`:

```python
"""Where the market is, and how old that statement is.

Freshness sits at the top of the payload rather than in a footnote. An answer
computed on ten-day-old bars reads exactly like one computed on current bars,
and this project has already lost sixteen days to a failure whose only symptom
was silence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Measured, Unavailable

# An hourly feed more than three hours behind has missed at least two bars,
# which is past the point where a single slow fetch explains it.
STALE_AFTER_HOURS = 3.0


def _change(close: pd.Series, bars_back: int) -> Measured | Unavailable:
    if len(close) <= bars_back:
        return Unavailable(reason=f"chưa có đủ {bars_back:,} nến lịch sử")
    now, then = float(close.iloc[-1]), float(close.iloc[-1 - bars_back])
    return Measured(
        value=now / then - 1.0,
        n=bars_back,
        method=f"so với {bars_back:,} nến trước",
    )


def market_snapshot(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Price, recent changes, funding, and how stale the data is."""
    if bars.empty:
        return {
            "last_close": Unavailable(reason="chưa có nến nào trong kho").to_dict(),
            "data_age_hours": None,
            "is_stale": True,
            "staleness_note": "Không có dữ liệu.",
        }

    now = now or pd.Timestamp.now(tz="UTC")
    close = bars["close"]
    last_close_time = (
        bars["close_time"].max() if "close_time" in bars.columns else bars.index.max()
    )
    age_hours = float((now - last_close_time).total_seconds() / 3600.0)
    is_stale = age_hours > STALE_AFTER_HOURS

    if funding.empty:
        funding_now: Measured | Unavailable = Unavailable(
            reason="chưa có lịch sử funding trong kho"
        )
        funding_mean: Measured | Unavailable = Unavailable(
            reason="chưa có lịch sử funding trong kho"
        )
    else:
        recent = funding["funding_rate"].tail(90)  # 30 days at 8h intervals
        funding_now = Measured(
            value=float(funding["funding_rate"].iloc[-1]),
            n=1,
            method="lần funding gần nhất",
        )
        funding_mean = Measured(
            value=float(recent.mean()),
            n=int(recent.size),
            method="trung bình 30 ngày",
        )

    return {
        "last_close": Measured(
            value=float(close.iloc[-1]), n=1, method="nến đóng gần nhất"
        ).to_dict(),
        "change_24h": _change(close, 24).to_dict(),
        "change_7d": _change(close, 168).to_dict(),
        "change_30d": _change(close, 720).to_dict(),
        "quote_volume_24h": Measured(
            value=float(bars.get("quote_volume", pd.Series(dtype=float)).tail(24).sum()),
            n=24,
            method="tổng 24 nến gần nhất",
        ).to_dict(),
        "funding_now": funding_now.to_dict(),
        "funding_mean_30d": funding_mean.to_dict(),
        "last_bar_close_time": str(last_close_time),
        "data_age_hours": round(age_hours, 1),
        "is_stale": is_stale,
        "staleness_note": (
            f"Dữ liệu cũ {age_hours:.0f} giờ. Mọi con số dưới đây mô tả thời điểm đó, "
            "không phải bây giờ."
            if is_stale
            else ""
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_snapshot.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/snapshot.py tests/test_briefing_snapshot.py
git commit -m "$(cat <<'EOF'
Market snapshot with freshness as a top-level field

Price, changes over 24h/7d/30d, volume, funding. Freshness is not a footnote: an
answer computed on ten-day-old bars reads exactly like one computed on current
bars, and this project has already lost sixteen days to a failure whose only
symptom was silence.

Missing funding returns Unavailable rather than zero. A zero funding rate is a
claim about the market; no stored funding history is a claim about the store.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Indicators, all Convention

**Files:**
- Create: `src/cryptopred/briefing/indicators.py`
- Test: `tests/test_briefing_indicators.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_indicators.py`:

```python
"""RSI is not useless. It is unmeasured, here, and must say so.

The answer that prompted this feature read "RSI-14 is 66.1, momentum remains
positive". The first half is a fact and the second is an inference this project
has never tested.
"""

from cryptopred.briefing.indicators import current_indicators
from tests.conftest import make_ohlcv


def test_every_indicator_is_marked_unvalidated():
    bars = make_ohlcv(n=1000, seed=31)
    out = current_indicators(bars)
    assert out
    for name, payload in out.items():
        assert payload["source"] == "convention", name
        assert payload["validated"] is False, name
        assert "warning" in payload, name


def test_the_expected_indicators_are_present():
    bars = make_ohlcv(n=1000, seed=32)
    out = current_indicators(bars)
    for key in ("rsi_7", "rsi_14", "macd_histogram", "atr_pct", "adx_14"):
        assert key in out


def test_each_carries_its_conventional_reading():
    bars = make_ohlcv(n=1000, seed=33)
    out = current_indicators(bars)
    assert "70" in out["rsi_14"]["reading"]


def test_too_little_history_yields_no_indicator_rather_than_a_wrong_one():
    bars = make_ohlcv(n=20, seed=34)
    out = current_indicators(bars)
    assert "rsi_14" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_indicators.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.indicators'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/indicators.py`:

```python
"""Conventional indicators, reported as conventions.

Every value here is computed by a formula the industry agrees on and whose
predictive value this project has not measured on this data. That is not a
reason to hide them — they are what a user will ask about — but it is a reason
that none of them can leave as a `Measured`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Convention
from cryptopred.features.base import pct_rank
from cryptopred.features.momentum import macd, rsi
from cryptopred.features.regime import adx
from cryptopred.features.volatility import atr, realized_vol

READINGS = {
    "rsi_7": "quy ước: >70 quá mua, <30 quá bán",
    "rsi_14": "quy ước: >70 quá mua, <30 quá bán",
    "macd_line": "quy ước: dương là đà tăng",
    "macd_signal": "quy ước: đường tín hiệu của MACD",
    "macd_histogram": "quy ước: histogram dương và mở rộng là đà tăng mạnh lên",
    "atr_pct": "biên độ thật trung bình, tính theo % giá",
    "atr_percentile": "ATR hiện tại đứng ở đâu so với 720 nến gần nhất, 0–1",
    "adx_14": "quy ước: >25 là có xu hướng rõ",
    "realized_vol_168": "độ lệch chuẩn lợi suất 168 nến, chưa quy năm",
}


def _last(series: pd.Series) -> float | None:
    """The final value, or None if the indicator has not warmed up."""
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


def current_indicators(bars: pd.DataFrame) -> dict[str, Any]:
    """Every indicator that has enough history, each as a Convention."""
    if bars.empty:
        return {}

    close = bars["close"]
    line, signal, hist = macd(close)
    values = {
        "rsi_7": _last(rsi(close, 7)),
        "rsi_14": _last(rsi(close, 14)),
        "macd_line": _last(line),
        "macd_signal": _last(signal),
        "macd_histogram": _last(hist),
        "atr_pct": _last(atr(bars, 14) / close),
        "atr_percentile": _last(pct_rank(atr(bars, 14) / close, 720)),
        "adx_14": _last(adx(bars, 14)),
        "realized_vol_168": _last(realized_vol(close, 168)),
    }
    return {
        name: Convention(value=value, reading=READINGS[name]).to_dict()
        for name, value in values.items()
        if value is not None
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_indicators.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/indicators.py tests/test_briefing_indicators.py
git commit -m "$(cat <<'EOF'
Conventional indicators, reported as conventions

RSI, MACD, ATR, ADX, realized volatility - every one computed by a formula the
industry agrees on and whose predictive value this project has not measured here.
They are what a user will ask about, so they are reported; none of them can leave
as Measured.

An indicator without enough history is omitted rather than returned from a
partial window.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Levels — pivots and swings

**Files:**
- Create: `src/cryptopred/briefing/levels.py`
- Test: `tests/test_briefing_levels.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_levels.py`:

```python
"""Levels name the prices a user will ask about.

Their value is not that price respects them - untested - but that they turn
"where might it go" into a specific number that touch_probability can measure.
"""

import numpy as np
import pandas as pd

from cryptopred.briefing.levels import daily_pivots, swing_levels
from tests.conftest import make_ohlcv


def test_pivots_follow_the_classic_formula():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": np.concatenate([np.full(24, 110.0), np.full(24, 105.0)]),
            "low": np.concatenate([np.full(24, 90.0), np.full(24, 95.0)]),
            "close": np.concatenate([np.full(24, 100.0), np.full(24, 102.0)]),
        },
        index=idx,
    )
    piv = daily_pivots(bars)
    # previous day: H=110, L=90, C=100 -> P = 100
    assert piv["P"]["value"] == 100.0
    assert piv["R1"]["value"] == 110.0   # 2P - L
    assert piv["S1"]["value"] == 90.0    # 2P - H
    assert piv["R2"]["value"] == 120.0   # P + (H - L)
    assert piv["S2"]["value"] == 80.0    # P - (H - L)


def test_pivots_are_conventions():
    bars = make_ohlcv(n=200, seed=41)
    piv = daily_pivots(bars)
    assert all(v["source"] == "convention" for v in piv.values())


def test_a_single_day_of_data_gives_no_pivots():
    bars = make_ohlcv(n=10, seed=42)
    assert daily_pivots(bars) == {}


def test_swing_high_is_the_max_of_its_window():
    close = np.full(41, 100.0)
    high = np.full(41, 100.0)
    high[20] = 130.0
    idx = pd.date_range("2024-01-01", periods=41, freq="1h", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {"open": close, "high": high, "low": close, "close": close}, index=idx
    )
    out = swing_levels(bars, k=5, lookback=41)
    assert 130.0 in [lvl["value"] for lvl in out["resistance"]]


def test_recent_unconfirmed_bars_produce_no_swing():
    """A swing needs k bars after it to be confirmed. The last k bars cannot
    have one yet, and inventing one would be a claim about bars that have not
    happened."""
    bars = make_ohlcv(n=200, seed=43)
    out = swing_levels(bars, k=5, lookback=200)
    last_five = bars.index[-5:]
    assert all(pd.Timestamp(lvl["at"]) not in last_five for lvl in out["resistance"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_levels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.levels'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/levels.py`:

```python
"""Prices worth asking about.

Pivots and swing points are conventional constructions, and this project has not
measured whether price respects them. Their job here is narrower and defensible:
they turn "where might it go" into specific numbers that `touch_probability` can
then measure honestly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.provenance import Convention

PIVOT_READING = "pivot cổ điển tính từ H/L/C của ngày hôm trước"


def daily_pivots(bars: pd.DataFrame) -> dict[str, Any]:
    """Classic floor-trader pivots from the previous completed day."""
    if bars.empty:
        return {}
    daily = bars.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()
    if len(daily) < 2:
        return {}

    prev = daily.iloc[-2]
    high, low, close = float(prev["high"]), float(prev["low"]), float(prev["close"])
    p = (high + low + close) / 3.0
    span = high - low

    values = {
        "P": p,
        "R1": 2 * p - low,
        "S1": 2 * p - high,
        "R2": p + span,
        "S2": p - span,
        "R3": high + 2 * (p - low),
        "S3": low - 2 * (high - p),
    }
    return {
        name: Convention(value=float(value), reading=PIVOT_READING).to_dict()
        for name, value in values.items()
    }


def swing_levels(
    bars: pd.DataFrame, k: int = 5, lookback: int = 720
) -> dict[str, list[dict[str, Any]]]:
    """Fractal swing highs and lows within the lookback window.

    A swing at bar i requires i to be the extreme of [i-k, i+k], so the last k
    bars cannot hold a confirmed swing. They are excluded rather than guessed:
    a swing that needs future bars to confirm is not a level yet.
    """
    if bars.empty:
        return {"resistance": [], "support": []}

    window = bars.tail(lookback)
    high, low = window["high"], window["low"]
    width = 2 * k + 1

    is_high = high == high.rolling(width, center=True).max()
    is_low = low == low.rolling(width, center=True).min()

    def pack(mask: pd.Series, series: pd.Series) -> list[dict[str, Any]]:
        picked = series[mask.fillna(False)]
        return [
            {
                "value": float(value),
                "at": str(at),
                "source": "convention",
                "reading": f"đỉnh/đáy xoay, xác nhận bằng {k} nến hai bên",
                "validated": False,
            }
            for at, value in picked.items()
        ]

    return {
        "resistance": pack(is_high, high)[-8:],
        "support": pack(is_low, low)[-8:],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_levels.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/levels.py tests/test_briefing_levels.py
git commit -m "$(cat <<'EOF'
Levels: daily pivots and confirmed swing points

Conventional constructions, and this project has not measured whether price
respects them. Their job here is narrower: they turn "where might it go" into
specific numbers that touch_probability can measure honestly.

A fractal swing needs k bars on each side to confirm, so the last k bars hold no
swing. They are excluded rather than guessed - a level that needs future bars to
confirm is not a level yet.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: The report and `cryptopred-brief`

**Files:**
- Create: `src/cryptopred/briefing/report.py`
- Create: `src/cryptopred/briefing/cli.py`
- Modify: `pyproject.toml` (add the script entry)
- Test: `tests/test_briefing_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_report.py`:

```python
"""The table must be readable without the LLM layer, and must not quietly
present a convention as a measurement."""

import pandas as pd

from cryptopred.briefing.report import format_brief
from tests.conftest import make_ohlcv


def _brief_input(stale: bool = False):
    bars = make_ohlcv(n=3000, seed=51)
    now = bars["close_time"].max() + (
        pd.Timedelta(days=10) if stale else pd.Timedelta(minutes=10)
    )
    return bars, now


def test_the_table_separates_measured_from_convention():
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "ĐO ĐƯỢC" in text
    assert "QUY ƯỚC" in text


def test_staleness_appears_at_the_top_when_data_is_old():
    bars, now = _brief_input(stale=True)
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    head = text.split("QUY ƯỚC")[0]
    assert "cũ" in head


def test_touch_probabilities_are_shown_with_their_sample_size():
    bars, now = _brief_input()
    text = format_brief("BTCUSDT", "1h", bars, pd.DataFrame(), now=now)
    assert "n=" in text


def test_it_runs_on_an_empty_store_without_raising():
    text = format_brief("BTCUSDT", "1h", pd.DataFrame(), pd.DataFrame())
    assert "Không có dữ liệu" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_briefing_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.briefing.report'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/briefing/report.py`:

```python
"""The briefing as text, readable with no API key and no model.

Measured figures and conventional ones are in separate sections with a heading
between them. Interleaving would let a reader's eye carry the authority of the
first into the second, which is the whole failure being designed against.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.briefing.indicators import current_indicators
from cryptopred.briefing.levels import daily_pivots
from cryptopred.briefing.snapshot import market_snapshot
from cryptopred.briefing.touch import touch_probability

# The drops a buyer actually asks about, and the horizons they wait.
DEFAULT_TARGETS = (-0.03, -0.05, -0.10)
DEFAULT_HORIZONS = (24, 72, 168)


def format_brief(
    symbol: str,
    interval: str,
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> str:
    if bars.empty:
        return f"{symbol} {interval}: Không có dữ liệu trong kho."

    snap = market_snapshot(bars, funding, now=now)
    lines = [
        "=" * 72,
        f"{symbol} {interval} — bảng số liệu",
        "=" * 72,
    ]
    if snap["staleness_note"]:
        lines += ["", f"!! {snap['staleness_note']}"]

    lines += [
        "",
        "ĐO ĐƯỢC",
        f"  giá đóng gần nhất: {snap['last_close']['value']:,.2f}",
        f"  đổi 24h: {_pct(snap['change_24h'])}   "
        f"7d: {_pct(snap['change_7d'])}   30d: {_pct(snap['change_30d'])}",
        f"  dữ liệu cũ: {snap['data_age_hours']} giờ",
        "",
        "  xác suất chạm mức (đo từ lịch sử, không phải dự báo):",
    ]

    for target in DEFAULT_TARGETS:
        for horizon in DEFAULT_HORIZONS:
            result = touch_probability(bars, target_pct=target, horizon=horizon)
            lines.append(
                f"    {target:+.0%} trong {horizon:>3}h: "
                f"{_prob(result['conditional'])}  |  "
                f"mọi chế độ: {_prob(result['unconditional'])}"
            )

    sample = touch_probability(bars, target_pct=DEFAULT_TARGETS[0], horizon=72)
    lines.append(f"    chế độ hiện tại: {sample['cell_label']}")
    wait = sample["wait_hours"]
    if wait:
        lines.append(
            f"    thời gian chờ khi có chạm: trung vị {wait['median']:.0f}h, "
            f"p90 {wait['p90']:.0f}h (n={wait['n']:,})"
        )

    lines += ["", "QUY ƯỚC — CHƯA KIỂM CHỨNG", ""]
    for name, payload in current_indicators(bars).items():
        lines.append(f"  {name:<18} {payload['value']:>10.4f}   {payload['reading']}")

    pivots = daily_pivots(bars)
    if pivots:
        lines.append("")
        lines.append(
            "  pivot ngày: "
            + "  ".join(f"{k}={v['value']:,.0f}" for k, v in pivots.items())
        )

    lines += [
        "",
        "  Không có dòng nào trong mục này được đo là có giá trị dự báo trên dữ liệu",
        "  này. Chúng có mặt vì bạn sẽ hỏi tới, không phải vì chúng đã được chứng minh.",
        "=" * 72,
    ]
    return "\n".join(lines)


def _pct(payload: dict) -> str:
    if payload.get("source") != "measured":
        return "n/a"
    return f"{payload['value']:+.2%}"


def _prob(payload) -> str:
    d = payload.to_dict() if hasattr(payload, "to_dict") else payload
    if d.get("source") != "measured":
        return "không đủ mẫu"
    lo, hi = d["ci95"]
    return f"{d['value']:.1%} n={d['n']:,} CI[{lo:.1%},{hi:.1%}]"
```

Create `src/cryptopred/briefing/cli.py`:

```python
"""`cryptopred-brief` — the numbers, with no model and no API key."""

from __future__ import annotations

from pathlib import Path

import typer

from cryptopred.briefing.report import format_brief
from cryptopred.config import load_config
from cryptopred.ingest.storage import ParquetStore

app = typer.Typer(help="Print the measured market briefing for a symbol.")


@app.command()
def show(
    symbol: str = typer.Argument("BTCUSDT", help="Symbol to describe."),
    interval: str = typer.Option("1h", help="Bar interval."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    cfg = load_config(config)
    parquet = ParquetStore(cfg.data.root / "raw")
    bars = parquet.read("klines", symbol, interval)
    funding = parquet.read("funding", symbol, "8h")
    typer.echo(format_brief(symbol, interval, bars, funding))


if __name__ == "__main__":
    app()
```

Modify `pyproject.toml`, in `[project.scripts]`, after the `cryptopred-meta` line:

```toml
cryptopred-brief = "cryptopred.briefing.cli:app"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_briefing_report.py -q`
Expected: PASS, 4 passed

Then run it for real:

Run: `uv run --no-sync python -m cryptopred.briefing.cli show ETHUSDT`
Expected: the table, with ETH's measured touch probabilities and a QUY ƯỚC section.

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/briefing/report.py src/cryptopred/briefing/cli.py pyproject.toml tests/test_briefing_report.py
git commit -m "$(cat <<'EOF'
cryptopred-brief: the numbers without the model

Measured figures and conventional ones sit in separate sections with a heading
between them. Interleaving would let a reader's eye carry the authority of the
first into the second, which is the failure this whole feature is designed
against.

The briefing runs with no API key. If the question-answering layer is never
built, or is removed, this still answers the question in table form.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

# PHASE 2 — the question-answering layer

## Task 9: Add the SDK and the tool schemas

**Files:**
- Modify: `pyproject.toml` (optional dependency group)
- Create: `src/cryptopred/ask/__init__.py`
- Create: `src/cryptopred/ask/tools.py`
- Test: `tests/test_ask_tools.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ask_tools.py`:

```python
"""The tools are the surface the model sees. Their schemas must be strict, and
their descriptions must carry the warnings, because the model reads the
description before it reads the values."""

import pandas as pd
import pytest

from cryptopred.ask.tools import TOOL_SCHEMAS, BriefingTools
from tests.conftest import make_ohlcv


@pytest.fixture
def tools(tmp_path):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore

    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT", "ETHUSDT"]
    ParquetStore(tmp_path / "raw").write(
        "klines", "BTCUSDT", "1h", make_ohlcv(n=3000, seed=61)
    )
    return BriefingTools(cfg)


def test_all_six_tools_are_declared():
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "market_snapshot",
        "indicators",
        "levels",
        "touch_probability",
        "model_signal",
        "track_record",
    }


def test_every_schema_is_strict_and_closed():
    for schema in TOOL_SCHEMAS:
        assert schema["strict"] is True, schema["name"]
        assert schema["input_schema"]["additionalProperties"] is False, schema["name"]
        assert "required" in schema["input_schema"], schema["name"]


def test_the_indicator_tool_description_carries_the_warning():
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "indicators")
    assert "chưa đo" in schema["description"].lower()


def test_target_is_a_discriminated_object_not_a_bare_number():
    """Passing -3 must not leave the tool guessing between 3% down and a price
    of minus three."""
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "touch_probability")
    target = schema["input_schema"]["properties"]["target"]
    assert target["properties"]["kind"]["enum"] == ["pct", "price"]


def test_touch_probability_accepts_an_absolute_price(tools):
    out = tools.touch_probability(
        symbol="BTCUSDT", target={"kind": "price", "value": 20_000.0}, horizon_hours=24
    )
    assert "conditional" in out and "unconditional" in out


def test_a_symbol_outside_the_allowed_pair_is_refused(tools):
    out = tools.market_snapshot(symbol="SOLUSDT")
    assert out["source"] == "unavailable"
    assert "BTCUSDT" in out["reason"]


def test_model_signal_for_a_symbol_with_no_model_explains_why(tools):
    out = tools.model_signal(symbol="ETHUSDT")
    assert out["source"] == "unavailable"
    assert "model" in out["reason"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_ask_tools.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ask'`

- [ ] **Step 3: Write minimal implementation**

Modify `pyproject.toml`, adding a group under `[project.optional-dependencies]` alongside `dev`:

```toml
ask = [
    "anthropic>=1.0",
]
```

Install it:

```bash
uv pip install -e ".[dev,ask]"
```

Create `src/cryptopred/ask/__init__.py`:

```python
"""Natural-language question answering over the briefing layer."""
```

Create `src/cryptopred/ask/tools.py`:

```python
"""The six tools the model may call.

Descriptions matter as much as schemas: the model reads the description before
it reads the values, so the warning about conventional indicators lives there
rather than only in the payload.

Only BTCUSDT and ETHUSDT are allowed. The other eighteen symbols are refreshed
by hand and answering about two-week-old bars is the staleness failure this
design exists to avoid.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.indicators import current_indicators
from cryptopred.briefing.levels import daily_pivots, swing_levels
from cryptopred.briefing.provenance import Measured, Unavailable
from cryptopred.briefing.snapshot import market_snapshot
from cryptopred.briefing.touch import touch_probability
from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.registry import ModelRegistry
from cryptopred.serve.drift import coverage_drift
from cryptopred.serve.status import wilson_interval
from cryptopred.serve.store import PredictionStore

ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "market_snapshot",
        "description": (
            "Giá đóng gần nhất, thay đổi 24h/7d/30d, khối lượng, funding, và "
            "ĐỘ CŨ CỦA DỮ LIỆU. Luôn gọi tool này trước; nếu is_stale là true thì "
            "phải nói rõ trong câu trả lời rằng số liệu mô tả thời điểm cũ."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "indicators",
        "description": (
            "RSI, MACD, ATR, ADX, realized vol. CẢNH BÁO: dự án này CHƯA ĐO các "
            "chỉ báo đó có giá trị dự báo trên dữ liệu này hay không. Được phép "
            "đọc số ra, KHÔNG được suy ra dự báo từ chúng."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "levels",
        "description": (
            "Pivot ngày và đỉnh/đáy xoay đã xác nhận. Là quy ước, chưa kiểm chứng. "
            "Dùng để lấy ra các mức giá cụ thể rồi đưa vào touch_probability — "
            "đó mới là chỗ có số đo thật."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "touch_probability",
        "description": (
            "SỐ ĐO THẬT. Trong lịch sử, giá chạm mức này trong bao nhiêu phần trăm "
            "số lần, tính trên những giờ có cùng chế độ biến động và xu hướng, kèm "
            "cỡ mẫu, khoảng tin cậy bootstrap, và phân phối thời gian chờ. Đây là "
            "tool trả lời câu 'khi nào'. Nếu conditional là unavailable thì nói rõ "
            "là ô chế độ không đủ mẫu và dùng con số unconditional."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
                "target": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["pct", "price"]},
                        "value": {"type": "number"},
                    },
                    "required": ["kind", "value"],
                    "additionalProperties": False,
                },
                "horizon_hours": {"type": "integer", "minimum": 1, "maximum": 720},
            },
            "required": ["symbol", "target", "horizon_hours"],
            "additionalProperties": False,
        },
    },
    {
        "name": "model_signal",
        "description": (
            "Dự báo hướng hiện tại của model đã lưu, margin so với cutoff, và "
            "trạng thái drift. ETHUSDT KHÔNG có model vì trượt cổng kiểm — tool sẽ "
            "trả unavailable kèm lý do, và đó là câu trả lời đúng, không phải lỗi."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "track_record",
        "description": (
            "Độ chính xác thật đã ghi nhận (kèm n và khoảng tin cậy), và ĐỘ CHÍNH "
            "XÁC HOÀ VỐN mà chiến lược phải vượt sau phí. Luôn đọc hai số này cạnh "
            "nhau: 54% nghe hay cho tới khi đặt cạnh mức 56% mà phí đòi hỏi."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
]


class BriefingTools:
    """Executes the tools. Pure reads; nothing here writes or trades."""

    def __init__(self, cfg: Config, interval: str = "1h") -> None:
        self.cfg = cfg
        self.interval = interval
        self.parquet = ParquetStore(cfg.data.root / "raw")
        self.registry = ModelRegistry(cfg.data.root / "models")

    # -- helpers -----------------------------------------------------------

    def _bars(self, symbol: str) -> pd.DataFrame | Unavailable:
        if symbol not in ALLOWED_SYMBOLS:
            return Unavailable(
                reason=(
                    f"{symbol} không nằm trong phạm vi. Chỉ trả lời về "
                    f"{' và '.join(ALLOWED_SYMBOLS)}, vì các coin khác không được "
                    "cập nhật tự động và dữ liệu sẽ cũ."
                )
            )
        bars = self.parquet.read("klines", symbol, self.interval)
        if bars.empty:
            return Unavailable(reason=f"không có nến {symbol} {self.interval} trong kho")
        return bars

    # -- tools -------------------------------------------------------------

    def market_snapshot(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        funding = self.parquet.read("funding", symbol, "8h")
        return market_snapshot(bars, funding)

    def indicators(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        return current_indicators(bars)

    def levels(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        return {"pivots": daily_pivots(bars), **swing_levels(bars)}

    def touch_probability(
        self, symbol: str, target: dict[str, Any], horizon_hours: int
    ) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()

        last = float(bars["close"].iloc[-1])
        if target["kind"] == "price":
            target_pct = float(target["value"]) / last - 1.0
        else:
            target_pct = float(target["value"])

        result = touch_probability(bars, target_pct=target_pct, horizon=horizon_hours)
        return {
            "target_pct": target_pct,
            "target_price": last * (1.0 + target_pct),
            "reference_close": last,
            "horizon_hours": horizon_hours,
            "cell_label": result["cell_label"],
            "conditional": result["conditional"].to_dict(),
            "unconditional": result["unconditional"].to_dict(),
            "wait_hours": result["wait_hours"],
        }

    def model_signal(self, symbol: str) -> dict[str, Any]:
        if symbol not in ALLOWED_SYMBOLS:
            return self._bars(symbol).to_dict()
        version = self.registry.latest(symbol, self.interval)
        if version is None:
            return Unavailable(
                reason=(
                    f"{symbol} không có model nào được lưu. Model chỉ được lưu khi "
                    "vượt cổng kiểm; không có model là kết quả đúng của cổng, "
                    "không phải lỗi."
                )
            ).to_dict()

        meta = self.registry.load(version).metadata
        store = PredictionStore(self.cfg.data.root / "predictions.db")
        latest = store.latest_prediction(symbol, self.interval)
        drift = coverage_drift(
            store.history(symbol, self.interval, limit=100_000),
            model_version=version,
            cutoff=meta.get("margin_cutoff"),
            target=meta.get("signal_coverage"),
        )
        if latest is None:
            return Unavailable(reason="chưa ghi nhận dự báo nào cho model này").to_dict()

        margin = abs(float(latest["prob_up"]) - float(latest["prob_down"]))
        return {
            "model_version": version,
            "bar_close_time": latest["bar_close_time"],
            "prob_down": latest["prob_down"],
            "prob_flat": latest["prob_flat"],
            "prob_up": latest["prob_up"],
            "margin": margin,
            "margin_cutoff": meta.get("margin_cutoff"),
            "signal": latest["signal"],
            "drift": drift,
        }

    def track_record(self, symbol: str) -> dict[str, Any]:
        if symbol not in ALLOWED_SYMBOLS:
            return self._bars(symbol).to_dict()
        store = PredictionStore(self.cfg.data.root / "predictions.db")
        history = store.history(symbol, self.interval, limit=100_000)
        if history.empty:
            return Unavailable(reason="chưa có dự báo nào được ghi").to_dict()

        scored = history[history["actual_return"].notna()]
        correct = int(scored["is_correct"].sum()) if not scored.empty else 0
        n = int(len(scored))
        cost = self.cfg.strategy.taker_fee * 2 + self.cfg.strategy.slippage * 2
        return {
            "n_scored": n,
            "accuracy": Measured(
                value=(correct / n if n else 0.0),
                n=n,
                ci95=wilson_interval(correct, n),
                method="mọi nến đã chấm điểm, Wilson (các lần thử độc lập)",
            ).to_dict(),
            "round_trip_cost": cost,
            "note": (
                "Độ chính xác hoà vốn phụ thuộc biên độ di chuyển trung vị; "
                "xem `cryptopred-model breakeven` để có con số cho từng khung."
            ),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_ask_tools.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cryptopred/ask/ tests/test_ask_tools.py
git commit -m "$(cat <<'EOF'
Six tools over the briefing layer

Descriptions carry the warnings, not just the payloads - the model reads the
description before it reads the values, so the fact that RSI is unmeasured here
belongs in the text it reads first.

touch_probability takes a discriminated target object rather than a bare number,
so passing -3 cannot leave the tool guessing between three percent down and a
price of minus three.

Only BTCUSDT and ETHUSDT are allowed. The other eighteen symbols are refreshed by
hand, and answering about two-week-old bars is the staleness failure this design
exists to avoid.

anthropic is an optional dependency. The briefing layer must keep working for
anyone who never installs it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: The audit — trace every number to a tool result

**Files:**
- Create: `src/cryptopred/ask/audit.py`
- Test: `tests/test_ask_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ask_audit.py`:

```python
"""The audit is the layer with teeth, so its test is written first.

It catches invented price levels and invented percentages - the exact failure in
the answer that prompted this feature. It does not catch qualitative claims, and
that limit is reported rather than implied.
"""

from cryptopred.ask.audit import audit_answer, extract_numbers


def test_vietnamese_decimals_and_thousands_are_both_parsed():
    """2.500 means two thousand five hundred; 58,4 means fifty-eight point four."""
    assert 2500.0 in extract_numbers("giá về 2.500 USD")
    assert 58.4 in extract_numbers("xác suất 58,4%")
    assert 2500.0 in extract_numbers("giá về 2,500 USD")
    assert 58.4 in extract_numbers("xác suất 58.4%")


def test_an_invented_number_is_flagged():
    report = audit_answer(
        answer="Giá sẽ về 2.500 rồi bật lên 2.900.",
        tool_numbers=[2500.0],
    )
    assert report["unmatched"] == [2900.0]
    assert report["ok"] is False


def test_rounding_is_tolerated():
    report = audit_answer(answer="xác suất 58,4%", tool_numbers=[0.584231])
    assert report["unmatched"] == []
    assert report["ok"] is True


def test_percentages_match_their_fractional_tool_value():
    report = audit_answer(answer="chạm 31,2% số lần", tool_numbers=[0.312])
    assert report["ok"] is True


def test_small_ordinals_are_not_treated_as_claims():
    """'trong 24 giờ' and 'top 3' are structure, not measurements."""
    report = audit_answer(answer="trong 24 giờ, xem 3 mức", tool_numbers=[])
    assert report["ok"] is True


def test_the_report_states_what_it_did_not_check():
    report = audit_answer(answer="động lượng còn tích cực", tool_numbers=[])
    assert "định tính" in report["limitation"]


def test_the_source_table_lists_matched_figures():
    report = audit_answer(answer="xác suất 58,4%", tool_numbers=[0.584231])
    assert report["matched"] == [58.4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_ask_audit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ask.audit'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/ask/audit.py`:

```python
"""Did every number in the answer come from a tool?

This is the layer that actually binds. The types make it hard to lose a warning
and the system prompt asks for care, but neither can stop a model from writing a
price nobody measured. This can.

What it cannot do is check "động lượng còn tích cực". That limit is reported in
every audit rather than left implied, so the audit's silence on prose is visible.
"""

from __future__ import annotations

import re
from typing import Any

# Numbers below this are structure - "trong 24 giờ", "3 mức" - not measurements.
TRIVIAL_BELOW = 100.0

# A figure matches a tool value if it is within this relative distance, which
# covers rounding and unit-of-percent differences.
REL_TOLERANCE = 0.02

LIMITATION = (
    "Chỉ kiểm được con số. Không kiểm được câu định tính như "
    "'động lượng còn tích cực' — loại câu đó không có gì để đối chiếu."
)

_NUMBER = re.compile(r"-?\d[\d.,]*\d|-?\d")


def _parse(token: str) -> list[float]:
    """Both conventions, because the answer is Vietnamese and the tools are not.

    "2.500" is two thousand five hundred to a Vietnamese reader and two point
    five to a parser that assumes English. Rather than guess, return every
    plausible reading and let the match decide.

    Ordered largest first and deduplicated, deliberately: an unmatched figure is
    reported by its first reading, and a set here would make that report vary
    between runs. When nothing matches, the Vietnamese reading is the one to
    show the user.
    """
    candidates = [
        token.replace(".", "").replace(",", "."),  # Vietnamese: . groups, , decimal
        token.replace(",", ""),                    # English: , groups, . decimal
    ]
    out: list[float] = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return sorted(out, key=abs, reverse=True)


def extract_numbers(text: str) -> list[float]:
    """Every plausible numeric reading in the text."""
    values: list[float] = []
    for match in _NUMBER.finditer(text):
        values.extend(_parse(match.group()))
    return values


def _matches(figure: float, tool_numbers: list[float]) -> bool:
    for value in tool_numbers:
        for scaled in (value, value * 100.0, value / 100.0):
            if scaled == 0:
                if figure == 0:
                    return True
                continue
            if abs(figure - scaled) / abs(scaled) <= REL_TOLERANCE:
                return True
    return False


def audit_answer(answer: str, tool_numbers: list[float]) -> dict[str, Any]:
    """Match every figure in the answer against the numbers tools returned."""
    matched: list[float] = []
    unmatched: list[float] = []

    for match in _NUMBER.finditer(answer):
        readings = _parse(match.group())
        if not readings:
            continue
        if all(abs(r) < TRIVIAL_BELOW for r in readings) and not any(
            _matches(r, tool_numbers) for r in readings
        ):
            # Small bare integers are structure unless a tool actually produced
            # them; percentages above 1 are caught by the scaling in _matches.
            if all(float(r).is_integer() for r in readings):
                continue
        hit = next((r for r in readings if _matches(r, tool_numbers)), None)
        if hit is not None:
            matched.append(hit)
        else:
            unmatched.append(readings[0])

    return {
        "ok": not unmatched,
        "matched": matched,
        "unmatched": unmatched,
        "limitation": LIMITATION,
    }


def collect_tool_numbers(payload: Any) -> list[float]:
    """Every numeric leaf in a tool result, at any depth."""
    out: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            out.extend(collect_tool_numbers(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            out.extend(collect_tool_numbers(value))
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        out.append(float(payload))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_ask_audit.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ask/audit.py tests/test_ask_audit.py
git commit -m "$(cat <<'EOF'
The audit: every number in the answer traced to a tool result

This is the layer that binds. Types make it hard to lose a warning and the system
prompt asks for care, but neither stops a model writing a price nobody measured.

Both numeric conventions are parsed, because the answer is Vietnamese and the
tools are not: "2.500" is two thousand five hundred to one reader and two point
five to the other. Rather than guess, every plausible reading is kept and the
match decides.

What it cannot check is "động lượng còn tích cực". That limit is stated in every
audit rather than left implied, so the audit's silence on prose is visible rather
than mistaken for approval.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: The system prompt

**Files:**
- Create: `src/cryptopred/ask/prompt.py`
- Test: `tests/test_ask_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ask_prompt.py`:

```python
"""The prompt tells the model this project's specific history of overconfidence,
not a generic instruction to be careful."""

from cryptopred.ask.prompt import SYSTEM_PROMPT


def test_it_forbids_numbers_that_did_not_come_from_a_tool():
    assert "không được" in SYSTEM_PROMPT.lower()
    assert "tool" in SYSTEM_PROMPT.lower()


def test_it_names_the_projects_own_failures_concretely():
    """Generic caution does not survive contact with a confident-sounding table.
    Specific history does."""
    for marker in ("0 tín hiệu", "99.4%", "findings.md"):
        assert marker in SYSTEM_PROMPT


def test_it_forbids_recommendations():
    assert "khuyến nghị" in SYSTEM_PROMPT


def test_it_requires_saying_khong_do_duoc():
    assert "không đo được" in SYSTEM_PROMPT


def test_it_is_not_so_long_it_dominates_the_cache():
    assert len(SYSTEM_PROMPT) < 6000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_ask_prompt.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ask.prompt'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/ask/prompt.py`:

```python
"""What the model is told before it sees a single number.

Generic caution does not survive contact with a confident-sounding table. This
project's actual failures do, so they are named.
"""

SYSTEM_PROMPT = """\
Bạn trả lời câu hỏi về thị trường crypto cho một hệ thống đo lường, bằng tiếng Việt.

QUY TẮC CỨNG

1. KHÔNG được viết ra bất kỳ con số nào không đến từ kết quả tool. Không ước
   lượng, không nội suy, không làm tròn từ trí nhớ. Mọi câu trả lời đều bị một
   lớp hậu kiểm đối chiếu từng con số với kết quả tool.

2. Chỉ số quy ước (RSI, MACD, pivot, ADX) được phép ĐỌC RA, KHÔNG được suy ra dự
   báo. "RSI-14 là 66,1" thì được. "RSI cao nên sắp giảm" thì không — dự án này
   chưa đo điều đó.

3. Khi không có số đo, nói thẳng "không đo được" và nói tại sao. Đó là câu trả
   lời hợp lệ, không phải thất bại.

4. KHÔNG đưa khuyến nghị mua bán. Bạn báo xác suất đo được; người dùng tự quyết.

5. Nếu market_snapshot báo is_stale, câu trả lời phải nói rõ dữ liệu cũ bao
   nhiêu, ngay ở đầu.

CÁCH TRẢ LỜI

Tách hai mục rõ ràng: "ĐO ĐƯỢC" trước, "QUY ƯỚC — CHƯA KIỂM CHỨNG" sau. Mỗi xác
suất phải đi kèm cỡ mẫu và khoảng tin cậy. Câu hỏi "khi nào" trả lời bằng phân
phối thời gian chờ (trung vị và p90), không phải bằng một mốc thời gian.

LỊCH SỬ CỦA CHÍNH DỰ ÁN NÀY

Đây không phải lời khuyên chung chung về sự cẩn thận. Đây là những lần dự án này
đã tự lừa mình, ghi trong docs/findings.md:

- Một lỗi gộp lãi báo lợi nhuận +24 tỉ %, drawdown −99,6%.
- Một ngưỡng xác suất cố định hoá ra là hiện vật hiệu chỉnh: cùng model, cùng
  ngưỡng, chọn 26,6% số nến ở fold này và 0,03% ở fold khác.
- Một cổng kiểm báo model bắn 12,87% số nến, rồi model bắn 0 tín hiệu trên 336
  nến thật suốt 16 ngày. Cổng đo trên chính dữ liệu model đã học thuộc.
- Một báo cáo in "IMPROVEMENT" cho chiến lược 99.4% long trong thị trường đang
  lên, vì cổng chỉ chặn "một chiều" mà để lọt "chưa kết luận được".
- Thêm 0,6% dữ liệu làm lợi nhuận một coin nhảy +99,8 điểm phần trăm.

Mỗi con số đó trông như số và hành xử như câu văn. Công việc của bạn là không
thêm cái thứ sáu vào danh sách.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_ask_prompt.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ask/prompt.py tests/test_ask_prompt.py
git commit -m "$(cat <<'EOF'
System prompt: this project's failures, not generic caution

Generic caution does not survive contact with a confident-sounding table.
Specific history does, so the prompt names five real ones from findings.md - the
compounding bug, the calibration artefact, the gate that passed a model firing on
nothing, the IMPROVEMENT verdict on a 99.4% long book, and the hundred-point
swing from 0.6% more data.

Each of those looked like a number and behaved like a sentence. The prompt's
closing instruction is not to add a sixth.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: The tool-calling session

**Files:**
- Create: `src/cryptopred/ask/session.py`
- Test: `tests/test_ask_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ask_session.py`:

```python
"""No test here makes a paid API call.

The tools are where the risk lives and they are tested directly. This layer is
tested for wiring: does the audit run, does staleness reach the answer, are
errors surfaced rather than swallowed.
"""

import pandas as pd
import pytest

from cryptopred.ask.session import AskResult, answer_question
from tests.conftest import make_ohlcv


class FakeMessage:
    def __init__(self, text, input_tokens=1000, output_tokens=200):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.stop_details = None
        self.usage = type(
            "U",
            (),
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )()


class FakeRunner:
    """Stands in for client.beta.messages.tool_runner."""

    def __init__(self, text, tool_payloads=()):
        self._text = text
        self._payloads = list(tool_payloads)

    def until_done(self):
        return FakeMessage(self._text)

    @property
    def tool_results(self):
        return self._payloads


@pytest.fixture
def cfg(tmp_path):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore

    c = Config()
    c.data.root = tmp_path
    c.data.symbols = ["BTCUSDT", "ETHUSDT"]
    ParquetStore(tmp_path / "raw").write(
        "klines", "BTCUSDT", "1h", make_ohlcv(n=3000, seed=71)
    )
    return c


def test_an_invented_number_in_the_answer_is_flagged(cfg):
    runner = FakeRunner("Giá về 2.500 rồi lên 2.900.", tool_payloads=[{"x": 2500.0}])
    result = answer_question("test", cfg=cfg, runner_factory=lambda **_: runner)
    assert isinstance(result, AskResult)
    assert result.audit["ok"] is False
    assert 2900.0 in result.audit["unmatched"]


def test_a_clean_answer_passes_the_audit(cfg):
    runner = FakeRunner("Xác suất 58,4%.", tool_payloads=[{"p": 0.584231}])
    result = answer_question("test", cfg=cfg, runner_factory=lambda **_: runner)
    assert result.audit["ok"] is True


def test_cost_is_reported(cfg):
    runner = FakeRunner("ok", tool_payloads=[])
    result = answer_question("test", cfg=cfg, runner_factory=lambda **_: runner)
    assert result.cost_usd > 0
    assert result.usage["input_tokens"] == 1000


def test_a_refusal_is_surfaced_not_swallowed(cfg):
    class Refusing(FakeRunner):
        def until_done(self):
            msg = FakeMessage("")
            msg.stop_reason = "refusal"
            msg.stop_details = type("D", (), {"category": "test", "explanation": "no"})()
            return msg

    result = answer_question(
        "test", cfg=cfg, runner_factory=lambda **_: Refusing("", [])
    )
    assert "từ chối" in result.answer.lower()


def test_the_model_id_is_opus_5_by_default(cfg):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeRunner("ok")

    answer_question("test", cfg=cfg, runner_factory=factory)
    assert captured["model"] == "claude-opus-5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_ask_session.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ask.session'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/ask/session.py`:

```python
"""The Claude Opus 5 tool-calling loop.

Uses the SDK's tool runner rather than a hand-written loop: the tools are where
the risk lives and they are tested directly, so the loop is not worth owning.

`runner_factory` is injectable so every test in this file runs without a network
call or a cent of spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cryptopred.ask.audit import audit_answer, collect_tool_numbers
from cryptopred.ask.prompt import SYSTEM_PROMPT
from cryptopred.ask.tools import TOOL_SCHEMAS, BriefingTools
from cryptopred.config import Config

MODEL = "claude-opus-5"

# Anthropic list price, dollars per million tokens, cached 2026-09-21.
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 25.00


@dataclass
class AskResult:
    answer: str
    audit: dict[str, Any]
    usage: dict[str, int]
    cost_usd: float
    tool_calls: list[str] = field(default_factory=list)


def _default_runner_factory(**kwargs: Any) -> Any:
    import anthropic

    client = anthropic.Anthropic()
    return client.beta.messages.tool_runner(**kwargs)


def answer_question(
    question: str,
    cfg: Config,
    interval: str = "1h",
    effort: str = "medium",
    model: str = MODEL,
    runner_factory: Callable[..., Any] = _default_runner_factory,
) -> AskResult:
    """Answer one question, then audit the answer against what the tools said."""
    tools = BriefingTools(cfg, interval=interval)

    runner = runner_factory(
        model=model,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=TOOL_SCHEMAS,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": question}],
        tool_executor=tools,
    )

    message = runner.until_done()

    if getattr(message, "stop_reason", None) == "refusal":
        detail = getattr(message, "stop_details", None)
        reason = getattr(detail, "explanation", "") if detail else ""
        return AskResult(
            answer=f"Model từ chối trả lời câu này. Lý do hệ thống báo: {reason}",
            audit={"ok": True, "matched": [], "unmatched": [], "limitation": ""},
            usage=_usage(message),
            cost_usd=_cost(message),
        )

    text = "\n".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    tool_numbers = collect_tool_numbers(list(getattr(runner, "tool_results", [])))

    return AskResult(
        answer=text,
        audit=audit_answer(text, tool_numbers),
        usage=_usage(message),
        cost_usd=_cost(message),
    )


def _usage(message: Any) -> dict[str, int]:
    u = message.usage
    return {
        "input_tokens": int(u.input_tokens),
        "output_tokens": int(u.output_tokens),
        "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
    }


def _cost(message: Any) -> float:
    u = _usage(message)
    return (
        u["input_tokens"] / 1e6 * INPUT_PER_MTOK
        + u["output_tokens"] / 1e6 * OUTPUT_PER_MTOK
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_ask_session.py -q`
Expected: PASS, 5 passed

**Two things to settle from the SDK docs before writing this file.** Invoke the
`claude-api` skill and read `python/claude-api/tool-use.md`; do not guess either
of them, because guessed SDK bindings are the one failure mode that skill calls
out by name.

1. **How tools bind to the runner.** `tool_executor=tools` above is this plan's
   placeholder for whatever that file documents — the `@beta_tool` decorator
   registration or an explicit mapping. The tests pin behaviour rather than the
   keyword, and the fake runner accepts any keyword, so they pass either way.

2. **Streaming.** The spec calls for the streaming runner with
   `.get_final_message()`. Use it if `tool-use.md` shows a streaming tool-runner
   path; the non-streaming `.until_done()` above is the fallback and is adequate
   here, since answers run one to two thousand output tokens, far from the
   timeout territory that makes streaming mandatory. Do **not** set
   `eager_input_streaming`: it exists for large tool inputs like file contents,
   and every input here is a symbol, a number and an horizon. Whichever path is
   taken, record it in the commit message so the spec and the code agree.

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ask/session.py tests/test_ask_session.py
git commit -m "$(cat <<'EOF'
The Claude Opus 5 tool-calling session

Uses the SDK tool runner rather than a hand-written loop. The tools are where the
risk lives and they are tested directly, so the loop is not worth owning.

runner_factory is injectable, so every test in this layer runs with no network
call and no spend. Refusals are surfaced rather than swallowed, and token usage
and cost come back with the answer - a per-question cost the user cannot see is
one they find out about later.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `cryptopred-ask`

**Files:**
- Create: `src/cryptopred/ask/cli.py`
- Modify: `pyproject.toml` (add the script entry)
- Test: `tests/test_ask_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ask_cli.py`:

```python
"""With no credentials the CLI must explain itself, not raise."""

from typer.testing import CliRunner

from cryptopred.ask.cli import app

runner = CliRunner()


def test_missing_credentials_explains_and_points_at_the_brief(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "cryptopred.ask.cli._credentials_available", lambda: False
    )
    result = runner.invoke(app, ["hỏi gì đó"])
    assert result.exit_code == 1
    assert "cryptopred-brief" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_the_source_table_and_flags_are_printed(monkeypatch, tmp_path):
    from cryptopred.ask.session import AskResult

    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question",
        lambda *a, **k: AskResult(
            answer="Giá về 2.900.",
            audit={
                "ok": False,
                "matched": [],
                "unmatched": [2900.0],
                "limitation": "chỉ kiểm được con số",
            },
            usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
            cost_usd=0.0038,
        ),
    )
    result = runner.invoke(app, ["hỏi gì đó"])
    assert result.exit_code == 0
    assert "2900" in result.output or "2,900" in result.output
    assert "KHÔNG TRUY ĐƯỢC NGUỒN" in result.output
    assert "$0.0038" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_ask_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ask.cli'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cryptopred/ask/cli.py`:

```python
"""`cryptopred-ask` — ask in Vietnamese, get measured numbers back."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from cryptopred.ask.session import MODEL, answer_question
from cryptopred.config import load_config

app = typer.Typer(help="Hỏi về thị trường, trả lời bằng số đo được.")


def _credentials_available() -> bool:
    """An unset ANTHROPIC_API_KEY does not mean there are no credentials: the SDK
    also reads ANTHROPIC_AUTH_TOKEN and an `ant auth login` profile."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Câu hỏi, bằng tiếng Việt."),
    symbol: str = typer.Option(
        None,
        help=(
            "Ép về một coin. Bỏ trống thì model tự suy từ câu hỏi — nó có tham số "
            "symbol trên mọi tool."
        ),
    ),
    interval: str = typer.Option("1h", help="Khung nến."),
    effort: str = typer.Option("medium", help="low | medium | high | xhigh | max"),
    model: str = typer.Option(MODEL, help="Model ID."),
    fetch: bool = typer.Option(True, help="Nạp nến mới trước khi trả lời."),
    as_json: bool = typer.Option(
        False, "--json", help="In câu trả lời, kết quả hậu kiểm và chi phí dạng JSON."
    ),
    config: Path = typer.Option(None, help="Đường dẫn file config YAML."),
) -> None:
    if not _credentials_available():
        typer.echo(
            "Chưa có thông tin đăng nhập Anthropic.\n"
            "  Cách 1: ant auth login\n"
            "  Cách 2: đặt biến môi trường ANTHROPIC_API_KEY\n"
            "\n"
            "Trong lúc đó, bảng số liệu vẫn chạy không cần key:\n"
            "  cryptopred-brief ETHUSDT"
        )
        raise typer.Exit(code=1)

    cfg = load_config(config)

    if fetch:
        try:
            from cryptopred.ingest.binance import BinanceClient
            from cryptopred.ingest.storage import ParquetStore
            from cryptopred.ingest.runner import run_klines_ingest

            sync_cfg = cfg.model_copy(deep=True)
            sync_cfg.data.intervals = [interval]
            with BinanceClient() as client:
                run_klines_ingest(sync_cfg, client, ParquetStore(cfg.data.root / "raw"))
        except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
            typer.echo(f"(Không nạp được nến mới: {exc}. Trả lời trên dữ liệu đã lưu.)")

    if symbol:
        question = f"[Chỉ hỏi về {symbol}] {question}"

    result = answer_question(
        question, cfg=cfg, interval=interval, effort=effort, model=model
    )

    if as_json:
        import json
        from dataclasses import asdict

        typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    typer.echo("")
    typer.echo(result.answer)
    typer.echo("")
    typer.echo("-" * 72)

    if result.audit["unmatched"]:
        typer.echo("KHÔNG TRUY ĐƯỢC NGUỒN — các số sau không đến từ tool nào:")
        for value in result.audit["unmatched"]:
            typer.echo(f"  {value:,}")
        typer.echo("  Đừng tin những con số này.")
    else:
        typer.echo("Mọi con số trong câu trả lời đều truy được về kết quả tool.")

    if result.audit.get("limitation"):
        typer.echo(f"Giới hạn hậu kiểm: {result.audit['limitation']}")

    typer.echo(
        f"Token: {result.usage['input_tokens']:,} vào / "
        f"{result.usage['output_tokens']:,} ra   "
        f"Chi phí: ${result.cost_usd:.4f}"
    )


if __name__ == "__main__":
    app()
```

Modify `pyproject.toml`, in `[project.scripts]`, after the `cryptopred-brief` line:

```toml
cryptopred-ask = "cryptopred.ask.cli:app"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_ask_cli.py -q`
Expected: PASS, 2 passed

Then the whole suite:

Run: `uv run --no-sync pytest -q && uv run --no-sync ruff check src tests`
Expected: 451 + ~55 new tests passing, lint clean.

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ask/cli.py pyproject.toml tests/test_ask_cli.py
git commit -m "$(cat <<'EOF'
cryptopred-ask: the CLI

Refreshes bars before answering; a failed refresh is non-fatal and says so, so
the answer is never silently computed on stale data.

With no credentials it explains how to authenticate and points at
cryptopred-brief, which answers the same questions in table form. It does not
raise a stack trace and it does not pretend to answer.

Every answer is followed by the audit result and the cost. Unmatched numbers are
printed with an instruction not to trust them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: End-to-end check against real stored data

**Files:**
- Test: `tests/test_briefing_real_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_real_data.py`:

```python
"""Run the measurement layer over the actual store, if it is present.

Synthetic bars cannot catch a column that the real parquet files name
differently, or a regime cell that turns out to be empty on seven years of ETH.
Skipped in a clean checkout so the suite stays portable.
"""

import pytest

from cryptopred.briefing.report import format_brief
from cryptopred.briefing.touch import touch_probability
from cryptopred.config import load_config
from cryptopred.ingest.storage import ParquetStore


@pytest.fixture
def eth_bars():
    cfg = load_config(None)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", "ETHUSDT", "1h")
    if bars.empty:
        pytest.skip("no stored ETHUSDT bars in this checkout")
    return bars


def test_touch_probability_runs_on_seven_years_of_eth(eth_bars):
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    assert result["unconditional"].n > 10_000
    assert 0.0 <= result["unconditional"].value <= 1.0


def test_the_conditional_cell_is_populated_on_real_data(eth_bars):
    """If nine cells leave the current one below the floor on 59,000 bars, the
    bucketing is too fine and the floor needs revisiting."""
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    assert result["conditional"].to_dict()["source"] == "measured"


def test_the_brief_renders_for_eth(eth_bars):
    text = format_brief("ETHUSDT", "1h", eth_bars, __import__("pandas").DataFrame())
    assert "ĐO ĐƯỢC" in text
    assert "QUY ƯỚC" in text
```

- [ ] **Step 2: Run test to verify it fails or skips**

Run: `uv run --no-sync pytest tests/test_briefing_real_data.py -q`
Expected: PASS (the store has ETHUSDT bars), or SKIP in a clean checkout.

If `test_the_conditional_cell_is_populated_on_real_data` fails, the nine-cell
split is leaving the current cell below 500 observations on 59,000 bars. Do not
lower `MIN_CELL_BARS` to make it pass — report the actual cell count and discuss
coarsening the trend dimension to two buckets instead.

- [ ] **Step 3: Run the brief for both symbols and read the output**

```bash
uv run --no-sync python -m cryptopred.briefing.cli show BTCUSDT
uv run --no-sync python -m cryptopred.briefing.cli show ETHUSDT
```

Check by eye: the ĐO ĐƯỢC section carries sample sizes and intervals, the QUY ƯỚC
section carries the warning, and ETH's numbers differ from BTC's.

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run --no-sync pytest -q && uv run --no-sync ruff check src tests`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_briefing_real_data.py
git commit -m "$(cat <<'EOF'
End-to-end check against the real store

Synthetic bars cannot catch a regime cell that turns out empty on seven years of
ETH, or a column the real parquet files name differently. Skipped in a clean
checkout so the suite stays portable.

If the conditional cell falls below the floor on 59,000 bars, the bucketing is
too fine - the fix is coarsening the trend dimension, not lowering the floor.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes for the implementer

**Spec coverage.** Every spec section maps to a task: provenance (1), regime with
expanding terciles (2), touch outcomes with intrabar extremes (3), block bootstrap
and the sample floor (4), snapshot with freshness (5), indicators (6), levels (7),
report and `cryptopred-brief` (8), the six tools (9), the audit (10), the prompt
(11), the session with Opus 5 config (12), `cryptopred-ask` (13), real-data check
(14). The dashboard chat box is explicitly out of scope in the spec and has no
task.

**Known soft spot.** Task 12 has two things this plan deliberately does not
specify — how the SDK binds tools to the runner, and whether the streaming runner
is used. Both come from `python/claude-api/tool-use.md`, read at implementation
time. Guessing an SDK binding is the failure mode that skill names explicitly,
and a plan that guesses confidently is worse than one that says where to look.

**Numeric parsing is order-sensitive.** `_parse` in Task 10 returns readings
sorted by magnitude, not a set. An unmatched figure is reported by its first
reading, so a set would make the audit's output differ between runs on the same
answer — a flaky report about flakiness.

**Do not weaken to make green.** Three constants encode judgements, not defaults:
`MIN_CELL_BARS = 500`, `MIN_HISTORY_BARS = 2000`, `STALE_AFTER_HOURS = 3.0`. If a
test fails against them, the finding is about the data, and it belongs in
`docs/findings.md` — not in a lowered constant.
