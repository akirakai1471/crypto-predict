# Data Foundation Implementation Plan (Phase P0 + P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data layer of the crypto direction predictor — download Binance USDT-perpetual data reliably, turn it into leakage-free point-in-time features and 3-class labels, and produce a training dataset with a single command.

**Architecture:** Four independent packages under `src/cryptopred/`. `ingest/` talks to Binance and owns all network + parquet storage. `features/` is pure functions over a DataFrame — no I/O, no network, backward-looking windows only. `labels/` computes forward returns and ATR-banded 3-class labels. `dataset/` joins the two and writes a training table. Each layer is tested independently against synthetic fixtures, so the whole suite runs offline.

**Tech Stack:** Python 3.12 (via `uv`), pandas, numpy, pyarrow, httpx, pydantic v2, PyYAML, typer (CLI), pytest, ruff.

**Prerequisite reading:** `docs/superpowers/specs/2026-08-26-crypto-direction-prediction-design.md` — especially section 5 (leakage prevention). Every feature in this plan must obey the point-in-time rule: **a feature at bar `t` may only use data from bars that closed at or before `t`.**

---

## Data Conventions (memorise these — every task depends on them)

A "kline DataFrame" always looks like this:

- **Index:** `open_time`, a `DatetimeIndex` in UTC, sorted ascending, unique, no duplicates.
- **Columns:** `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote` (all `float64`, `trades` is `int64`), plus `close_time` (`datetime64[ns, UTC]`).
- **Semantics:** Row `t` describes a bar that OPENED at `open_time[t]` and CLOSED at `close_time[t]`. A prediction for row `t` is made at `close_time[t]` — the instant the bar closes. So a feature at row `t` may use `close[t]`, but must never use `open[t+1]` or anything later.
- **Never store unclosed bars.** The most recent bar from Binance may still be forming. It is dropped at ingest time.

---

## File Structure

```
crypto-predict/
├── pyproject.toml                    # deps, ruff + pytest config
├── config/
│   └── default.yaml                  # symbols, intervals, paths, label params
├── src/cryptopred/
│   ├── __init__.py
│   ├── config.py                     # pydantic models + YAML loader
│   ├── timeframes.py                 # interval math, bar-grid helpers
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── binance.py                # HTTP client: klines, funding, OI, L/S ratio
│   │   ├── storage.py                # parquet read/write/merge, dedupe
│   │   ├── backfill.py               # pagination, gap detection, repair
│   │   └── cli.py                    # `cryptopred-ingest` command
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                   # rolling z-score, safe_divide, pct_rank
│   │   ├── momentum.py               # RSI, MACD, ROC, EMA distance
│   │   ├── volatility.py             # true range, ATR, BB width, realized vol
│   │   ├── volume.py                 # OBV slope, volume z-score, taker imbalance
│   │   ├── structure.py              # range position, distance to N-bar high/low, streaks
│   │   ├── derivatives.py            # funding rate features
│   │   ├── regime.py                 # ADX, volatility percentile, trend/range flag
│   │   ├── timefeat.py               # hour/day cyclical encoding, trading session
│   │   ├── mtf.py                    # higher-timeframe features via merge_asof
│   │   └── pipeline.py               # build_features() — assembles all of the above
│   ├── labels/
│   │   ├── __init__.py
│   │   └── barrier.py                # forward return, ATR band, 3-class label
│   └── dataset/
│       ├── __init__.py
│       ├── builder.py                # features + labels -> training table
│       ├── quality.py                # data quality report
│       └── cli.py                    # `cryptopred-dataset` command
└── tests/
    ├── conftest.py                   # synthetic OHLCV fixtures
    ├── test_timeframes.py
    ├── test_binance.py               # httpx.MockTransport, no real network
    ├── test_storage.py
    ├── test_backfill.py
    ├── test_features_momentum.py
    ├── test_features_volatility.py
    ├── test_features_volume.py
    ├── test_features_structure.py
    ├── test_features_derivatives.py
    ├── test_features_regime_time.py
    ├── test_features_mtf.py
    ├── test_leakage.py               # THE critical suite
    ├── test_labels.py
    └── test_dataset.py
```

Why this split: network code, pure math, and orchestration have different failure modes and different test strategies. Keeping `features/` free of I/O is what makes the leakage tests possible — you can call any feature function on an arbitrary DataFrame slice and compare results.

---

## Task 1: Project scaffold and toolchain

**Files:**
- Create: `pyproject.toml`
- Create: `src/cryptopred/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create the virtualenv with Python 3.12**

The machine has Python 3.14 and 3.10 installed. LightGBM (needed in Plan 2) may not have wheels for 3.14, so pin 3.12. `uv` downloads it automatically.

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv venv --python 3.12
```
Expected: output ends with `Activate with: source .venv/Scripts/activate`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "cryptopred"
version = "0.1.0"
description = "Crypto price direction prediction system"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=16.0",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "typer>=0.12",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[project.scripts]
cryptopred-ingest = "cryptopred.ingest.cli:app"
cryptopred-dataset = "cryptopred.dataset.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cryptopred"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]          # lets tests do `from tests.conftest import make_ohlcv`
markers = [
    "network: test performs real network calls (deselect with '-m \"not network\"')",
]
addopts = "-m 'not network'"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

Note the `addopts`: network tests are excluded by default. Run them explicitly with `-m network`.

- [ ] **Step 3: Create the package marker**

Create `src/cryptopred/__init__.py`:
```python
"""Crypto price direction prediction system."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write a smoke test**

Create `tests/test_smoke.py`:
```python
import cryptopred


def test_package_imports():
    assert cryptopred.__version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the smoke test**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv pip install -e ".[dev]" && uv run pytest -v
```
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src tests
git commit -m "chore: scaffold cryptopred package with Python 3.12 toolchain"
```

---

## Task 2: Configuration

**Files:**
- Create: `src/cryptopred/config.py`
- Create: `config/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:
```python
from pathlib import Path

from cryptopred.config import Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert "BTCUSDT" in cfg.data.symbols
    assert "1h" in cfg.data.intervals
    assert cfg.labels.horizon_bars["1h"] == 4
    assert cfg.labels.horizon_bars["1m"] == 5


def test_config_paths_are_absolute(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path / "data"
    assert cfg.raw_dir("klines").is_absolute()
    assert cfg.raw_dir("klines").name == "klines"


def test_yaml_override(tmp_path: Path):
    p = tmp_path / "custom.yaml"
    p.write_text("data:\n  symbols: [SOLUSDT]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.symbols == ["SOLUSDT"]
    # unspecified fields keep their defaults
    assert cfg.labels.atr_period == 14
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.config'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/config.py`:
```python
"""Configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class DataConfig(BaseModel):
    """Which markets to download and where to put the files."""

    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    intervals: list[str] = Field(default_factory=lambda: ["1h", "1m"])
    start: str = "2019-09-01"
    root: Path = PROJECT_ROOT / "data"


class LabelConfig(BaseModel):
    """How 'up' and 'down' are defined."""

    horizon_bars: dict[str, int] = Field(default_factory=lambda: {"1h": 4, "1m": 5})
    atr_period: int = 14
    band_k: float = 0.5


class FeatureConfig(BaseModel):
    """Feature engine knobs."""

    zscore_window: int = 200
    mtf_rules: dict[str, list[str]] = Field(
        default_factory=lambda: {"1h": ["4h", "1D"], "1m": ["15min", "1h"]}
    )


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)

    def raw_dir(self, kind: str) -> Path:
        """Directory holding raw downloads of a given kind (klines, funding, ...)."""
        return (self.data.root / "raw" / kind).resolve()

    def dataset_dir(self) -> Path:
        return (self.data.root / "datasets").resolve()


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults for anything unspecified."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
```

- [ ] **Step 4: Write the default YAML**

Create `config/default.yaml`:
```yaml
# Default configuration. Copy to config/local.yaml to override without touching git.
data:
  symbols: [BTCUSDT, ETHUSDT]
  intervals: ["1h", "1m"]
  start: "2019-09-01"

labels:
  horizon_bars:
    "1h": 4    # predict direction 4 hours ahead
    "1m": 5    # predict direction 5 minutes ahead
  atr_period: 14
  band_k: 0.5  # dead zone = 0.5 * ATR; moves smaller than this are labelled FLAT

features:
  zscore_window: 200
  mtf_rules:
    "1h": ["4h", "1D"]
    "1m": ["15min", "1h"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/cryptopred/config.py config tests/test_config.py
git commit -m "feat(config): add pydantic config models with YAML override"
```

---

## Task 3: Timeframe arithmetic

Bar-grid math is used by gap detection, resampling, and the scheduler later. Getting it wrong silently corrupts everything downstream, so it gets its own tested module.

**Files:**
- Create: `src/cryptopred/timeframes.py`
- Test: `tests/test_timeframes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_timeframes.py`:
```python
import pandas as pd
import pytest

from cryptopred.timeframes import (
    align_down,
    expected_open_times,
    interval_to_timedelta,
    to_ms,
)


def test_interval_to_timedelta():
    assert interval_to_timedelta("1m") == pd.Timedelta(minutes=1)
    assert interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert interval_to_timedelta("4h") == pd.Timedelta(hours=4)
    assert interval_to_timedelta("1d") == pd.Timedelta(days=1)


def test_unknown_interval_raises():
    with pytest.raises(ValueError, match="Unsupported interval"):
        interval_to_timedelta("7s")


def test_to_ms():
    ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert to_ms(ts) == 1704067200000


def test_align_down_snaps_to_bar_open():
    ts = pd.Timestamp("2024-01-01 03:47:31", tz="UTC")
    assert align_down(ts, "1h") == pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    assert align_down(ts, "4h") == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert align_down(ts, "1m") == pd.Timestamp("2024-01-01 03:47:00", tz="UTC")


def test_align_down_is_idempotent():
    ts = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    assert align_down(align_down(ts, "1h"), "1h") == ts


def test_expected_open_times_is_half_open():
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-01 05:00:00", tz="UTC")
    grid = expected_open_times(start, end, "1h")
    assert len(grid) == 5
    assert grid[0] == start
    assert grid[-1] == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_timeframes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.timeframes'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/timeframes.py`:
```python
"""Bar-grid arithmetic. Every timestamp in this project is UTC."""

from __future__ import annotations

import pandas as pd

_INTERVALS: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    try:
        return _INTERVALS[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval!r}") from exc


def to_ms(ts: pd.Timestamp) -> int:
    """Binance speaks epoch milliseconds."""
    return int(ts.timestamp() * 1000)


def from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def align_down(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Snap a timestamp back to the open of the bar containing it."""
    delta = interval_to_timedelta(interval)
    return ts.floor(delta)


def expected_open_times(
    start: pd.Timestamp, end: pd.Timestamp, interval: str
) -> pd.DatetimeIndex:
    """Every bar open time in [start, end). Used to detect missing bars."""
    delta = interval_to_timedelta(interval)
    return pd.date_range(
        start=align_down(start, interval),
        end=align_down(end, interval) - delta,
        freq=delta,
        tz="UTC",
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_timeframes.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/timeframes.py tests/test_timeframes.py
git commit -m "feat(timeframes): add bar-grid arithmetic helpers"
```

---

## Task 4: Binance HTTP client

Tested entirely against `httpx.MockTransport` — no real network in the default test run.

**Files:**
- Create: `src/cryptopred/ingest/__init__.py`
- Create: `src/cryptopred/ingest/binance.py`
- Test: `tests/test_binance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_binance.py`:
```python
import httpx
import pandas as pd
import pytest

from cryptopred.ingest.binance import BinanceClient, parse_klines

RAW_ROW = [
    1704067200000,       # open time
    "42000.10",          # open
    "42500.00",          # high
    "41900.00",          # low
    "42300.50",          # close
    "1234.567",          # volume
    1704070799999,       # close time
    "52000000.0",        # quote volume
    9876,                # trades
    "600.0",             # taker buy base
    "25000000.0",        # taker buy quote
    "0",                 # ignore
]


def test_parse_klines_builds_typed_frame():
    df = parse_klines([RAW_ROW])
    assert df.index.name == "open_time"
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df["close"].iloc[0] == pytest.approx(42300.50)
    assert df["trades"].iloc[0] == 9876
    assert df["close_time"].iloc[0] == pd.Timestamp("2024-01-01 00:59:59.999", tz="UTC")
    assert df["close"].dtype == "float64"


def test_parse_klines_empty_returns_empty_frame_with_schema():
    df = parse_klines([])
    assert df.empty
    assert list(df.columns) == [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "close_time",
    ]


def test_fetch_klines_sends_expected_params():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=[RAW_ROW])

    client = BinanceClient(transport=httpx.MockTransport(handler))
    rows = client.fetch_klines("BTCUSDT", "1h", start_ms=1704067200000, end_ms=1704070800000)

    assert seen["symbol"] == "BTCUSDT"
    assert seen["interval"] == "1h"
    assert seen["startTime"] == "1704067200000"
    assert seen["limit"] == "1500"
    assert len(rows) == 1


def test_fetch_klines_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"code": -1003})
        return httpx.Response(200, json=[RAW_ROW])

    client = BinanceClient(transport=httpx.MockTransport(handler), max_retries=3, backoff_base=0.0)
    rows = client.fetch_klines("BTCUSDT", "1h", start_ms=0, end_ms=1)

    assert calls["n"] == 2
    assert len(rows) == 1


def test_fetch_klines_gives_up_after_max_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = BinanceClient(transport=httpx.MockTransport(handler), max_retries=2, backoff_base=0.0)
    with pytest.raises(httpx.HTTPStatusError):
        client.fetch_klines("BTCUSDT", "1h", start_ms=0, end_ms=1)


def test_parse_funding():
    from cryptopred.ingest.binance import parse_funding

    raw = [{"symbol": "BTCUSDT", "fundingTime": 1704067200000, "fundingRate": "0.0001"}]
    df = parse_funding(raw)
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_binance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ingest'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/ingest/__init__.py`:
```python
"""Data acquisition: Binance HTTP access and parquet storage."""
```

Create `src/cryptopred/ingest/binance.py`:
```python
"""Binance USDT-margined futures public API client.

No API key is required for any endpoint used here. All endpoints are read-only
market data. Rate limits are respected via exponential backoff on 429/418.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.binance.com"

KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "close_time",
]
_FLOAT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]

# Binance caps klines at 1500 rows per request on the futures API.
KLINE_LIMIT = 1500
# The /futures/data/* endpoints cap at 500 and only retain ~30 days of history.
STATS_LIMIT = 500


class BinanceClient:
    """Thin synchronous wrapper. Sync is deliberate: backfill is a batch job and
    async adds failure modes without changing the wall-clock budget much."""

    def __init__(
        self,
        base_url: str = FUTURES_BASE,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout)
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            response = self._client.get(path, params=params)
            if response.status_code in (429, 418):
                wait = float(response.headers.get("Retry-After", self._backoff_base * 2**attempt))
                logger.warning("Rate limited on %s, sleeping %.1fs", path, wait)
                time.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                )
                continue
            if response.status_code >= 500:
                wait = self._backoff_base * 2**attempt
                logger.warning("Server error %s on %s, retrying in %.1fs",
                               response.status_code, path, wait)
                time.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    "server error", request=response.request, response=response
                )
                continue
            response.raise_for_status()
            return response.json()
        assert last_exc is not None
        raise last_exc

    def fetch_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = KLINE_LIMIT
    ) -> list[list[Any]]:
        """One page of klines. Binance returns bars with open_time in [start, end]."""
        return self._get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            },
        )

    def fetch_funding(
        self, symbol: str, start_ms: int, end_ms: int, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Funding rate history. Full history is available for perpetuals."""
        return self._get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        )

    def fetch_open_interest(
        self, symbol: str, period: str = "1h", limit: int = STATS_LIMIT
    ) -> list[dict[str, Any]]:
        """Open interest history. WARNING: Binance retains only ~30 days here, so
        this can never be a training feature — it is collected for display and for
        slow accumulation over time."""
        return self._get(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )

    def fetch_long_short_ratio(
        self, symbol: str, period: str = "1h", limit: int = STATS_LIMIT
    ) -> list[dict[str, Any]]:
        """Global long/short account ratio. Same ~30 day retention caveat as OI."""
        return self._get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )


def parse_klines(rows: list[list[Any]]) -> pd.DataFrame:
    """Turn raw Binance kline arrays into the canonical DataFrame shape."""
    if not rows:
        empty = pd.DataFrame(columns=KLINE_COLUMNS)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="open_time")
        return empty

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[_FLOAT_COLUMNS] = df[_FLOAT_COLUMNS].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    df = df.drop(columns=["ignore"]).set_index("open_time").sort_index()
    return df[KLINE_COLUMNS]


def parse_funding(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Funding rate history indexed by funding time."""
    if not rows:
        empty = pd.DataFrame(columns=["funding_rate"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="funding_time")
        return empty
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype("float64")
    return df.set_index("funding_time").sort_index()[["funding_rate"]]


def parse_stats(rows: list[dict[str, Any]], value_key: str, out_name: str) -> pd.DataFrame:
    """Parse /futures/data/* rows, which use `timestamp` instead of a bar open time."""
    if not rows:
        empty = pd.DataFrame(columns=[out_name])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return empty
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[out_name] = df[value_key].astype("float64")
    return df.set_index("timestamp").sort_index()[[out_name]]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_binance.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ingest tests/test_binance.py
git commit -m "feat(ingest): add Binance futures client with retry and parsers"
```

---

## Task 5: Parquet storage

**Files:**
- Create: `src/cryptopred/ingest/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_storage.py`:
```python
import pandas as pd

from cryptopred.ingest.storage import ParquetStore, merge_frames


def _frame(start: str, periods: int, close_start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(periods),
            "low": range(periods),
            "close": [close_start + i for i in range(periods)],
            "volume": [1.0] * periods,
            "quote_volume": [1.0] * periods,
            "trades": [1] * periods,
            "taker_buy_base": [1.0] * periods,
            "taker_buy_quote": [1.0] * periods,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    ).astype({"open": "float64", "high": "float64", "low": "float64"})


def test_merge_frames_dedupes_keeping_newest():
    old = _frame("2024-01-01", 3, close_start=100.0)
    new = _frame("2024-01-01 02:00", 3, close_start=999.0)
    merged = merge_frames(old, new)

    assert len(merged) == 5
    assert merged.index.is_monotonic_increasing
    assert not merged.index.has_duplicates
    # overlapping bar takes the value from the newer frame
    assert merged.loc[pd.Timestamp("2024-01-01 02:00", tz="UTC"), "close"] == 999.0


def test_merge_frames_with_empty_old():
    new = _frame("2024-01-01", 3)
    assert len(merge_frames(pd.DataFrame(), new)) == 3


def test_store_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    df = _frame("2024-01-01", 5)
    store.write("klines", "BTCUSDT", "1h", df)

    loaded = store.read("klines", "BTCUSDT", "1h")
    pd.testing.assert_frame_equal(loaded, df)


def test_store_append_merges_and_persists(tmp_path):
    store = ParquetStore(tmp_path)
    store.write("klines", "BTCUSDT", "1h", _frame("2024-01-01", 3))
    store.append("klines", "BTCUSDT", "1h", _frame("2024-01-01 02:00", 3))

    loaded = store.read("klines", "BTCUSDT", "1h")
    assert len(loaded) == 5
    assert not loaded.index.has_duplicates


def test_store_splits_by_year(tmp_path):
    store = ParquetStore(tmp_path)
    idx = pd.date_range("2023-12-31 22:00", periods=4, freq="1h", tz="UTC", name="open_time")
    df = _frame("2023-12-31 22:00", 4)
    df.index = idx
    store.write("klines", "BTCUSDT", "1h", df)

    assert (tmp_path / "klines" / "BTCUSDT" / "1h" / "2023.parquet").exists()
    assert (tmp_path / "klines" / "BTCUSDT" / "1h" / "2024.parquet").exists()
    assert len(store.read("klines", "BTCUSDT", "1h")) == 4


def test_read_missing_returns_empty(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.read("klines", "NOPE", "1h").empty


def test_last_open_time(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.last_open_time("klines", "BTCUSDT", "1h") is None
    store.write("klines", "BTCUSDT", "1h", _frame("2024-01-01", 3))
    assert store.last_open_time("klines", "BTCUSDT", "1h") == pd.Timestamp(
        "2024-01-01 02:00", tz="UTC"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ingest.storage'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/ingest/storage.py`:
```python
"""Parquet-backed storage, partitioned by kind/symbol/interval/year.

Year partitioning keeps individual files small enough to rewrite cheaply when
appending, while still allowing a full multi-year read in one call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two indexed frames, dropping duplicate index entries and
    keeping the row from `new` (freshly downloaded data wins over stale)."""
    if old.empty:
        return new.sort_index()
    if new.empty:
        return old.sort_index()
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, kind: str, symbol: str, interval: str) -> Path:
        return self.root / kind / symbol / interval

    def read(self, kind: str, symbol: str, interval: str) -> pd.DataFrame:
        directory = self._dir(kind, symbol, interval)
        if not directory.exists():
            return pd.DataFrame()
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(f) for f in files]
        return pd.concat(frames).sort_index()

    def write(self, kind: str, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Replace stored data for every year present in `df`."""
        if df.empty:
            return
        directory = self._dir(kind, symbol, interval)
        directory.mkdir(parents=True, exist_ok=True)
        for year, chunk in df.groupby(df.index.year):
            chunk.to_parquet(directory / f"{year}.parquet", engine="pyarrow", index=True)

    def append(self, kind: str, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Merge `df` into stored data, rewriting only the affected year files."""
        if df.empty:
            return
        directory = self._dir(kind, symbol, interval)
        directory.mkdir(parents=True, exist_ok=True)
        for year, chunk in df.groupby(df.index.year):
            path = directory / f"{year}.parquet"
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            merge_frames(existing, chunk).to_parquet(path, engine="pyarrow", index=True)

    def last_open_time(self, kind: str, symbol: str, interval: str) -> pd.Timestamp | None:
        """Newest stored index value, used to resume an interrupted backfill."""
        directory = self._dir(kind, symbol, interval)
        if not directory.exists():
            return None
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return None
        newest = pd.read_parquet(files[-1])
        if newest.empty:
            return None
        return newest.index.max()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_storage.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ingest/storage.py tests/test_storage.py
git commit -m "feat(ingest): add year-partitioned parquet store with dedupe"
```

---

## Task 6: Backfill with pagination, unclosed-bar guard, and gap detection

**Files:**
- Create: `src/cryptopred/ingest/backfill.py`
- Test: `tests/test_backfill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill.py`:
```python
import pandas as pd

from cryptopred.ingest.backfill import backfill_klines, drop_unclosed, find_gaps


def _kline_row(open_ms: int, interval_ms: int, close: float) -> list:
    return [
        open_ms, "1", "2", "0.5", str(close), "10",
        open_ms + interval_ms - 1, "100", 5, "5", "50", "0",
    ]


class FakeClient:
    """Serves klines from an in-memory grid, mimicking Binance pagination."""

    def __init__(self, start_ms: int, n_bars: int, interval_ms: int, limit: int = 3):
        self.rows = [
            _kline_row(start_ms + i * interval_ms, interval_ms, 100.0 + i) for i in range(n_bars)
        ]
        self.limit = limit
        self.calls = 0

    def fetch_klines(self, symbol, interval, start_ms, end_ms, limit=1500):
        self.calls += 1
        window = [r for r in self.rows if start_ms <= r[0] <= end_ms]
        return window[: self.limit]


def test_drop_unclosed_removes_forming_bar():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC", name="open_time")
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0], "close_time": idx + pd.Timedelta(hours=1)}, index=idx
    )
    now = pd.Timestamp("2024-01-01 02:30", tz="UTC")
    out = drop_unclosed(df, now=now)
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_find_gaps_returns_missing_bar_ranges():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            # 02:00 and 03:00 missing
            pd.Timestamp("2024-01-01 04:00", tz="UTC"),
        ],
        name="open_time",
    )
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    gaps = find_gaps(df, "1h")
    assert gaps == [
        (pd.Timestamp("2024-01-01 02:00", tz="UTC"), pd.Timestamp("2024-01-01 03:00", tz="UTC"))
    ]


def test_find_gaps_empty_when_contiguous():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC", name="open_time")
    df = pd.DataFrame({"close": range(10)}, index=idx)
    assert find_gaps(df, "1h") == []


def test_backfill_paginates_until_end():
    interval_ms = 3_600_000
    start = pd.Timestamp("2024-01-01", tz="UTC")
    client = FakeClient(int(start.timestamp() * 1000), n_bars=10, interval_ms=interval_ms, limit=3)

    df = backfill_klines(
        client,
        "BTCUSDT",
        "1h",
        start=start,
        end=start + pd.Timedelta(hours=10),
        now=start + pd.Timedelta(hours=10),
    )

    assert len(df) == 10
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert client.calls >= 4  # 10 bars at 3 per page


def test_backfill_stops_when_page_empty():
    interval_ms = 3_600_000
    start = pd.Timestamp("2024-01-01", tz="UTC")
    client = FakeClient(int(start.timestamp() * 1000), n_bars=0, interval_ms=interval_ms)

    df = backfill_klines(
        client, "BTCUSDT", "1h", start=start, end=start + pd.Timedelta(hours=5), now=start
    )
    assert df.empty
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ingest.backfill'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/ingest/backfill.py`:
```python
"""Pagination, unclosed-bar filtering, and gap detection for kline downloads."""

from __future__ import annotations

import logging

import pandas as pd

from cryptopred.ingest.binance import KLINE_LIMIT, parse_klines
from cryptopred.timeframes import expected_open_times, interval_to_timedelta, to_ms

logger = logging.getLogger(__name__)


def drop_unclosed(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove bars that have not finished forming.

    A bar is usable only once its close_time has passed. Keeping a forming bar
    would leak partial future information into features.
    """
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] <= now]


def find_gaps(df: pd.DataFrame, interval: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of missing bar open times, as (first_missing, last_missing)."""
    if df.empty or len(df) < 2:
        return []
    delta = interval_to_timedelta(interval)
    expected = expected_open_times(df.index.min(), df.index.max() + delta, interval)
    missing = expected.difference(df.index)
    if missing.empty:
        return []

    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev != delta:
            gaps.append((run_start, prev))
            run_start = ts
        prev = ts
    gaps.append((run_start, prev))
    return gaps


def backfill_klines(
    client,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    now: pd.Timestamp | None = None,
    limit: int = KLINE_LIMIT,
) -> pd.DataFrame:
    """Download every closed bar with open_time in [start, end).

    Pagination advances from the last returned bar. If a page comes back empty
    the download stops — Binance has no data before a contract's listing date.
    """
    delta = interval_to_timedelta(interval)
    frames: list[pd.DataFrame] = []
    cursor = start

    while cursor < end:
        rows = client.fetch_klines(
            symbol, interval, start_ms=to_ms(cursor), end_ms=to_ms(end), limit=limit
        )
        if not rows:
            break
        page = parse_klines(rows)
        frames.append(page)
        next_cursor = page.index.max() + delta
        if next_cursor <= cursor:  # defensive: never loop forever
            break
        cursor = next_cursor

    if not frames:
        return parse_klines([])

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= start) & (df.index < end)]
    return drop_unclosed(df, now=now)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_backfill.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/ingest/backfill.py tests/test_backfill.py
git commit -m "feat(ingest): add paginated backfill with gap detection"
```

---

## Task 7: Ingest CLI and first real download

**Files:**
- Create: `src/cryptopred/ingest/cli.py`
- Test: `tests/test_ingest_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_cli.py`:
```python
import pandas as pd
from typer.testing import CliRunner

from cryptopred.config import Config
from cryptopred.ingest.cli import app, run_klines_ingest
from cryptopred.ingest.storage import ParquetStore

runner = CliRunner()


class FakeClient:
    def __init__(self, n_bars: int = 5):
        start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        self.rows = [
            [
                start_ms + i * 3_600_000, "1", "2", "0.5", str(100 + i), "10",
                start_ms + (i + 1) * 3_600_000 - 1, "100", 5, "5", "50", "0",
            ]
            for i in range(n_bars)
        ]

    def fetch_klines(self, symbol, interval, start_ms, end_ms, limit=1500):
        window = [r for r in self.rows if start_ms <= r[0] <= end_ms]
        return window[:limit]


def test_run_klines_ingest_writes_store(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]
    cfg.data.start = "2024-01-01"

    store = ParquetStore(tmp_path / "raw")
    n = run_klines_ingest(
        cfg,
        client=FakeClient(),
        store=store,
        now=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
    )

    assert n == 5
    assert len(store.read("klines", "BTCUSDT", "1h")) == 5


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "klines" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ingest_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.ingest.cli'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/ingest/cli.py`:
```python
"""Command line entry point for data ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from cryptopred.config import Config, load_config
from cryptopred.ingest.backfill import backfill_klines, find_gaps
from cryptopred.ingest.binance import BinanceClient, parse_funding, parse_stats
from cryptopred.ingest.storage import ParquetStore
from cryptopred.timeframes import interval_to_timedelta, to_ms

app = typer.Typer(help="Download Binance market data into the local parquet store.")
logger = logging.getLogger(__name__)


def run_klines_ingest(
    cfg: Config,
    client,
    store: ParquetStore,
    now: pd.Timestamp | None = None,
) -> int:
    """Backfill or resume every configured symbol/interval. Returns bars written."""
    now = now or pd.Timestamp.now(tz="UTC")
    total = 0
    for symbol in cfg.data.symbols:
        for interval in cfg.data.intervals:
            last = store.last_open_time("klines", symbol, interval)
            delta = interval_to_timedelta(interval)
            start = last + delta if last is not None else pd.Timestamp(cfg.data.start, tz="UTC")
            if start >= now:
                logger.info("%s %s already current", symbol, interval)
                continue

            typer.echo(f"Downloading {symbol} {interval} from {start} ...")
            df = backfill_klines(client, symbol, interval, start=start, end=now, now=now)
            if df.empty:
                typer.echo(f"  no new bars for {symbol} {interval}")
                continue

            store.append("klines", symbol, interval, df)
            total += len(df)
            typer.echo(f"  wrote {len(df)} bars ({df.index.min()} -> {df.index.max()})")

            stored = store.read("klines", symbol, interval)
            gaps = find_gaps(stored, interval)
            if gaps:
                typer.echo(f"  WARNING: {len(gaps)} gap(s) remain, first at {gaps[0][0]}")
    return total


def run_funding_ingest(cfg: Config, client, store: ParquetStore) -> int:
    """Funding history has full retention, so this backfills from cfg.data.start."""
    now = pd.Timestamp.now(tz="UTC")
    total = 0
    for symbol in cfg.data.symbols:
        last = store.last_open_time("funding", symbol, "8h")
        start = last + pd.Timedelta(hours=1) if last is not None else pd.Timestamp(
            cfg.data.start, tz="UTC"
        )
        cursor = start
        frames = []
        while cursor < now:
            rows = client.fetch_funding(symbol, to_ms(cursor), to_ms(now), limit=1000)
            if not rows:
                break
            page = parse_funding(rows)
            frames.append(page)
            next_cursor = page.index.max() + pd.Timedelta(minutes=1)
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        if frames:
            df = pd.concat(frames)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            store.append("funding", symbol, "8h", df)
            total += len(df)
            typer.echo(f"  {symbol}: wrote {len(df)} funding rows")
    return total


def run_stats_ingest(cfg: Config, client, store: ParquetStore) -> int:
    """Open interest and long/short ratio. Binance keeps only ~30 days, so this
    is an accumulation job: run it regularly and history builds up locally.
    These are display-only features — never used for training in v1."""
    total = 0
    for symbol in cfg.data.symbols:
        oi = parse_stats(
            client.fetch_open_interest(symbol, period="1h"),
            value_key="sumOpenInterest",
            out_name="open_interest",
        )
        if not oi.empty:
            store.append("open_interest", symbol, "1h", oi)
            total += len(oi)

        ls = parse_stats(
            client.fetch_long_short_ratio(symbol, period="1h"),
            value_key="longShortRatio",
            out_name="long_short_ratio",
        )
        if not ls.empty:
            store.append("long_short_ratio", symbol, "1h", ls)
            total += len(ls)
    return total


@app.command()
def klines(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Backfill or update OHLCV klines."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    with BinanceClient() as client:
        n = run_klines_ingest(cfg, client, store)
    typer.echo(f"Done. {n} bars written.")


@app.command()
def funding(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Backfill or update funding rate history."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    with BinanceClient() as client:
        n = run_funding_ingest(cfg, client, store)
    typer.echo(f"Done. {n} funding rows written.")


@app.command()
def stats(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Collect open interest and long/short ratio (30-day retention, display only)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    with BinanceClient() as client:
        n = run_stats_ingest(cfg, client, store)
    typer.echo(f"Done. {n} stat rows written.")


@app.command()
def report(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Print what is currently stored and any gaps."""
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    for symbol in cfg.data.symbols:
        for interval in cfg.data.intervals:
            df = store.read("klines", symbol, interval)
            if df.empty:
                typer.echo(f"{symbol} {interval}: EMPTY")
                continue
            gaps = find_gaps(df, interval)
            typer.echo(
                f"{symbol} {interval}: {len(df):,} bars "
                f"{df.index.min()} -> {df.index.max()}, gaps={len(gaps)}"
            )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ingest_cli.py -v`
Expected: `2 passed`

- [ ] **Step 5: Download real 1h data (network — this takes a few minutes)**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run cryptopred-ingest klines
```
Expected: progress lines per symbol, then `Done. N bars written.` with N in the tens of thousands. 1h bars from 2019-09 to now is roughly 60,000 per symbol; 1m bars are roughly 3.5 million per symbol and this step will take 10–20 minutes.

- [ ] **Step 6: Download funding history**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run cryptopred-ingest funding
```
Expected: `Done. N funding rows written.` with N around 7,000 per symbol.

- [ ] **Step 7: Verify coverage and gaps**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run cryptopred-ingest report
```
Expected: each symbol/interval shows a start near 2019-09, an end within one bar of now, and `gaps=0`.

**If gaps > 0:** re-run `cryptopred-ingest klines` — it resumes from the last stored bar and will not re-download what exists. If a gap persists across two runs, Binance genuinely has no data there (exchange downtime). Record the gap in a note at `docs/data-notes.md` and continue; the feature layer handles missing bars.

- [ ] **Step 8: Commit**

```bash
git add src/cryptopred/ingest/cli.py tests/test_ingest_cli.py
git commit -m "feat(ingest): add CLI for klines, funding, stats, and coverage report"
```

---

## Task 8: Feature helpers

Every feature module builds on these. They are separated because rolling normalisation is the single most common place a lookahead bug hides.

**Files:**
- Create: `src/cryptopred/features/__init__.py`
- Create: `src/cryptopred/features/base.py`
- Create: `tests/conftest.py`
- Test: `tests/test_features_base.py`

- [ ] **Step 1: Write the shared fixture**

Create `tests/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1000, seed: int = 42, freq: str = "1h") -> pd.DataFrame:
    """Synthetic but well-formed OHLCV: geometric random walk with valid highs/lows."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.004, n)
    close = 30_000 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(1000, 300, n))
    taker_buy = volume * rng.uniform(0.3, 0.7, n)

    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="open_time")
    delta = idx.freq.delta
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(100, 5000, n),
            "taker_buy_base": taker_buy,
            "taker_buy_quote": taker_buy * close,
            "close_time": idx + delta - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def ohlcv_1m() -> pd.DataFrame:
    return make_ohlcv(n=2000, seed=7, freq="1min")


@pytest.fixture
def rising() -> pd.DataFrame:
    """Strictly increasing closes — useful for asserting bounded indicators."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(np.linspace(100, 300, n), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
            "quote_volume": 1000.0 * close,
            "trades": 100,
            "taker_buy_base": 500.0,
            "taker_buy_quote": 500.0 * close,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_features_base.py`:
```python
import numpy as np
import pandas as pd
import pytest

from cryptopred.features.base import pct_rank, rolling_zscore, safe_divide


def test_rolling_zscore_uses_only_past():
    s = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
    z = rolling_zscore(s, window=3, min_periods=3)
    # index 2 sees [1,2,3] only; the 100 spike at index 3 must not affect it
    expected = (3.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1)
    assert z.iloc[2] == pytest.approx(expected)
    assert pd.isna(z.iloc[0])
    assert pd.isna(z.iloc[1])


def test_rolling_zscore_constant_series_is_zero_not_inf():
    s = pd.Series([5.0] * 10)
    z = rolling_zscore(s, window=5, min_periods=5)
    assert (z.dropna() == 0).all()
    assert np.isfinite(z.dropna()).all()


def test_safe_divide_handles_zero_denominator():
    num = pd.Series([1.0, 2.0, 3.0])
    den = pd.Series([1.0, 0.0, 3.0])
    out = safe_divide(num, den)
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert np.isfinite(out).all()


def test_pct_rank_is_backward_looking():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pct_rank(s, window=3)
    # at index 4 the window is [3,4,5]; 5 is the max -> rank 1.0
    assert r.iloc[4] == pytest.approx(1.0)
    assert pd.isna(r.iloc[0])
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_features_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features'`

- [ ] **Step 4: Write the implementation**

Create `src/cryptopred/features/__init__.py`:
```python
"""Pure feature functions. No I/O, no network, backward-looking windows only."""
```

Create `src/cryptopred/features/base.py`:
```python
"""Shared numeric helpers.

Every windowed operation here is strictly backward-looking. Never introduce a
function that uses `center=True` or a negative `shift` — that is future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill: float = 0.0
) -> pd.Series:
    """Element-wise division that yields `fill` instead of inf/NaN on zero."""
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def rolling_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Standardise against a trailing window.

    Using a trailing window (rather than whole-series mean/std) is mandatory:
    whole-series statistics encode information from the future into every row.
    """
    min_periods = min_periods or window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=1)
    z = (series - mean) / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    # A perfectly flat window has zero deviation; its correct z-score is 0, not NaN.
    z[(std == 0) & mean.notna()] = 0.0
    return z


def pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile of the current value within its trailing window, in [0, 1].

    Uses pandas' native rolling rank, which ranks the window's final value.
    A `.rolling().apply()` version is ~200x slower and would make the 1m
    dataset build take hours.
    """
    return series.rolling(window, min_periods=window).rank(pct=True)


def ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, used by RSI, ATR and ADX."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_features_base.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/cryptopred/features tests/conftest.py tests/test_features_base.py
git commit -m "feat(features): add backward-looking numeric helpers"
```

---

## Task 9: Momentum features

**Files:**
- Create: `src/cryptopred/features/momentum.py`
- Test: `tests/test_features_momentum.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_momentum.py`:
```python
import pandas as pd

from cryptopred.features.momentum import momentum_features, rsi


def test_rsi_is_100_on_monotonic_rise(rising):
    r = rsi(rising["close"], period=14).dropna()
    assert (r > 99.9).all()


def test_rsi_is_bounded(ohlcv):
    r = rsi(ohlcv["close"], period=14).dropna()
    assert r.min() >= 0.0
    assert r.max() <= 100.0


def test_momentum_features_shape_and_names(ohlcv):
    feats = momentum_features(ohlcv)
    assert len(feats) == len(ohlcv)
    assert feats.index.equals(ohlcv.index)
    for name in ["rsi_14", "macd_hist", "roc_12", "ema_dist_50"]:
        assert name in feats.columns
    assert all(c.startswith(("rsi", "macd", "roc", "ema")) for c in feats.columns)


def test_momentum_features_have_no_infinities(ohlcv):
    feats = momentum_features(ohlcv)
    assert feats.replace([float("inf"), float("-inf")], pd.NA).notna().sum().sum() > 0
    assert not feats.isin([float("inf"), float("-inf")]).any().any()


def test_ema_distance_positive_in_uptrend(rising):
    feats = momentum_features(rising)
    assert feats["ema_dist_50"].dropna().iloc[-1] > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_momentum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.momentum'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/momentum.py`:
```python
"""Momentum indicators: how fast and in which direction price has been moving."""

from __future__ import annotations

import pandas as pd

from cryptopred.features.base import ewma, safe_divide, wilder_smooth


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 0 = only losses in the window, 100 = only gains."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = safe_divide(avg_gain, avg_loss, fill=0.0)
    out = 100.0 - 100.0 / (1.0 + rs)
    # when there are no losses at all, rs is 0 by safe_divide's fill; fix to 100
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram), normalised by price so the
    values are comparable across symbols with very different price levels."""
    line = (ewma(close, fast) - ewma(close, slow)) / close
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    for period in (7, 14, 28):
        out[f"rsi_{period}"] = rsi(close, period)

    macd_line, macd_signal, macd_hist = macd(close)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    for period in (1, 3, 6, 12, 24, 72):
        out[f"roc_{period}"] = close.pct_change(period)

    for span in (20, 50, 200):
        ema = ewma(close, span)
        out[f"ema_dist_{span}"] = safe_divide(close - ema, ema)

    out["ema_20_50_spread"] = safe_divide(ewma(close, 20) - ewma(close, 50), close)

    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_momentum.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/momentum.py tests/test_features_momentum.py
git commit -m "feat(features): add momentum indicators"
```

---

## Task 10: Volatility features

**Files:**
- Create: `src/cryptopred/features/volatility.py`
- Test: `tests/test_features_volatility.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_volatility.py`:
```python
import numpy as np
import pandas as pd
import pytest

from cryptopred.features.volatility import atr, true_range, volatility_features


def test_true_range_uses_previous_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"high": [10.0, 12.0], "low": [8.0, 11.0], "close": [9.0, 11.5]}, index=idx
    )
    tr = true_range(df)
    # bar 1: max(12-11, |12-9|, |11-9|) = 3
    assert tr.iloc[1] == pytest.approx(3.0)


def test_atr_is_positive(ohlcv):
    a = atr(ohlcv, period=14).dropna()
    assert (a > 0).all()


def test_atr_rises_with_volatility():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    spread = pd.Series([1.0] * 100 + [10.0] * 100, index=idx)
    df = pd.DataFrame({"high": close + spread, "low": close - spread, "close": close})
    a = atr(df, period=14)
    assert a.iloc[-1] > a.iloc[99] * 3


def test_volatility_features_names(ohlcv):
    feats = volatility_features(ohlcv)
    for name in ["atr_14_norm", "bb_width_20", "realized_vol_24", "vol_ratio_24_72"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_volatility_features_finite(ohlcv):
    feats = volatility_features(ohlcv)
    assert not np.isinf(feats.to_numpy(dtype="float64", na_value=0.0)).any()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_volatility.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.volatility'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/volatility.py`:
```python
"""Volatility measures. ATR is also used by the label engine, so it lives here
and is imported rather than duplicated."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import pct_rank, safe_divide, wilder_smooth


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    return wilder_smooth(true_range(df), period)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Standard deviation of log returns over a trailing window."""
    log_ret = np.log(close).diff()
    return log_ret.rolling(window, min_periods=window).std(ddof=1)


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    for period in (7, 14, 28):
        out[f"atr_{period}_norm"] = safe_divide(atr(df, period), close)

    for window in (20, 50):
        mid = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=1)
        out[f"bb_width_{window}"] = safe_divide(2 * std, mid)
        out[f"bb_position_{window}"] = safe_divide(close - mid, 2 * std)

    for window in (12, 24, 72, 168):
        out[f"realized_vol_{window}"] = realized_vol(close, window)

    out["vol_ratio_24_72"] = safe_divide(
        out["realized_vol_24"], out["realized_vol_72"], fill=1.0
    )
    out["vol_pct_rank_720"] = pct_rank(out["realized_vol_24"], window=720)

    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_volatility.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/volatility.py tests/test_features_volatility.py
git commit -m "feat(features): add volatility indicators"
```

---

## Task 11: Volume features

**Files:**
- Create: `src/cryptopred/features/volume.py`
- Test: `tests/test_features_volume.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_volume.py`:
```python
import pandas as pd
import pytest

from cryptopred.features.volume import obv, volume_features


def test_obv_accumulates_on_up_bars():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 100.0, 102.0], "volume": [10.0, 20.0, 30.0, 40.0]}, index=idx
    )
    o = obv(df)
    assert o.iloc[0] == 0.0
    assert o.iloc[1] == pytest.approx(20.0)      # up bar: +20
    assert o.iloc[2] == pytest.approx(-10.0)     # down bar: -30
    assert o.iloc[3] == pytest.approx(30.0)      # up bar: +40


def test_taker_imbalance_is_bounded(ohlcv):
    feats = volume_features(ohlcv)
    imb = feats["taker_imbalance"].dropna()
    assert imb.min() >= -1.0
    assert imb.max() <= 1.0


def test_volume_features_names(ohlcv):
    feats = volume_features(ohlcv)
    for name in ["volume_z_200", "obv_slope_24", "taker_imbalance", "trades_z_200"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_volume_features_survive_zero_volume_bar(ohlcv):
    df = ohlcv.copy()
    df.iloc[10, df.columns.get_loc("volume")] = 0.0
    df.iloc[10, df.columns.get_loc("taker_buy_base")] = 0.0
    feats = volume_features(df)
    assert feats.loc[df.index[10], "taker_imbalance"] == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_volume.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.volume'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/volume.py`:
```python
"""Volume and order-flow proxies derived from kline aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import rolling_zscore, safe_divide


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative signed volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def volume_features(df: pd.DataFrame, zscore_window: int = 200) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out[f"volume_z_{zscore_window}"] = rolling_zscore(df["volume"], zscore_window)
    out[f"trades_z_{zscore_window}"] = rolling_zscore(
        df["trades"].astype("float64"), zscore_window
    )
    out["quote_volume_z_200"] = rolling_zscore(df["quote_volume"], 200)

    obv_series = obv(df)
    for window in (12, 24, 72):
        out[f"obv_slope_{window}"] = safe_divide(
            obv_series.diff(window), df["volume"].rolling(window, min_periods=window).sum()
        )

    # taker_buy_base is the portion of volume that hit the ask (aggressive buying).
    # Imbalance in [-1, 1]: +1 = all aggressive buying, -1 = all aggressive selling.
    taker_sell = df["volume"] - df["taker_buy_base"]
    out["taker_imbalance"] = safe_divide(df["taker_buy_base"] - taker_sell, df["volume"])
    out["taker_imbalance_ma_24"] = out["taker_imbalance"].rolling(24, min_periods=24).mean()

    out["avg_trade_size_z_200"] = rolling_zscore(
        safe_divide(df["volume"], df["trades"].astype("float64")), 200
    )

    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_volume.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/volume.py tests/test_features_volume.py
git commit -m "feat(features): add volume and order-flow features"
```

---

## Task 12: Price structure features

**Files:**
- Create: `src/cryptopred/features/structure.py`
- Test: `tests/test_features_structure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_structure.py`:
```python
import pandas as pd
import pytest

from cryptopred.features.structure import consecutive_streak, structure_features


def test_consecutive_streak_counts_direction():
    s = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
    streak = consecutive_streak(s)
    assert streak.iloc[2] == 2      # two consecutive rises
    assert streak.iloc[4] == -2     # two consecutive falls
    assert streak.iloc[5] == 1


def test_close_position_in_range_is_bounded(ohlcv):
    feats = structure_features(ohlcv)
    pos = feats["close_pos_in_bar"].dropna()
    assert pos.min() >= 0.0
    assert pos.max() <= 1.0


def test_distance_to_high_is_non_positive(ohlcv):
    """Close can never exceed the rolling max that includes the current bar."""
    feats = structure_features(ohlcv)
    assert feats["dist_to_high_24"].dropna().max() <= 1e-9


def test_structure_features_names(ohlcv):
    feats = structure_features(ohlcv)
    for name in ["close_pos_in_bar", "dist_to_high_24", "dist_to_low_24", "streak"]:
        assert name in feats.columns
    assert feats.index.equals(ohlcv.index)


def test_doji_bar_does_not_produce_nan():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    feats = structure_features(df)
    assert feats["close_pos_in_bar"].iloc[0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_structure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.structure'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/structure.py`:
```python
"""Where price sits relative to its own recent range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import safe_divide


def consecutive_streak(close: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) or down (-) closes ending at each bar."""
    direction = np.sign(close.diff()).fillna(0.0)
    streak = np.zeros(len(direction), dtype="float64")
    for i in range(1, len(direction)):
        d = direction.iloc[i]
        if d == 0:
            streak[i] = 0.0
        elif np.sign(streak[i - 1]) == d:
            streak[i] = streak[i - 1] + d
        else:
            streak[i] = d
    return pd.Series(streak, index=close.index)


def structure_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    bar_range = df["high"] - df["low"]
    # A doji (high == low) has no meaningful position; 0.5 is the neutral answer.
    out["close_pos_in_bar"] = safe_divide(close - df["low"], bar_range, fill=0.5)
    out["bar_range_norm"] = safe_divide(bar_range, close)
    if "open" in df.columns:
        out["body_ratio"] = safe_divide((close - df["open"]).abs(), bar_range, fill=0.0)
        out["upper_wick"] = safe_divide(df["high"] - np.maximum(close, df["open"]), bar_range)
        out["lower_wick"] = safe_divide(np.minimum(close, df["open"]) - df["low"], bar_range)

    for window in (24, 72, 168):
        roll_high = df["high"].rolling(window, min_periods=window).max()
        roll_low = df["low"].rolling(window, min_periods=window).min()
        out[f"dist_to_high_{window}"] = safe_divide(close - roll_high, roll_high)
        out[f"dist_to_low_{window}"] = safe_divide(close - roll_low, roll_low)
        out[f"range_pos_{window}"] = safe_divide(
            close - roll_low, roll_high - roll_low, fill=0.5
        )

    out["streak"] = consecutive_streak(close)

    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_structure.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/structure.py tests/test_features_structure.py
git commit -m "feat(features): add price structure features"
```

---

## Task 13: Funding rate features

Funding is the only derivatives series with full history, so it is the only one that becomes a training feature. See spec section 2 for why open interest and long/short ratio are excluded.

**Files:**
- Create: `src/cryptopred/features/derivatives.py`
- Test: `tests/test_features_derivatives.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_derivatives.py`:
```python
import pandas as pd
import pytest

from cryptopred.features.derivatives import funding_features


@pytest.fixture
def funding() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="8h", tz="UTC", name="funding_time")
    return pd.DataFrame({"funding_rate": [0.0001] * 50 + [0.0005] * 50}, index=idx)


def test_funding_features_align_to_bar_index(ohlcv, funding):
    feats = funding_features(ohlcv, funding)
    assert feats.index.equals(ohlcv.index)
    assert "funding_rate" in feats.columns


def test_funding_uses_only_already_published_values(ohlcv, funding):
    """A funding value stamped at 08:00 must not appear on bars before 08:00."""
    feats = funding_features(ohlcv, funding)
    first_bar = ohlcv.index[0]
    first_funding = funding.index[0]
    if first_bar < first_funding:
        assert pd.isna(feats.loc[first_bar, "funding_rate"])


def test_funding_features_when_empty(ohlcv):
    empty = pd.DataFrame(columns=["funding_rate"])
    empty.index = pd.DatetimeIndex([], tz="UTC", name="funding_time")
    feats = funding_features(ohlcv, empty)
    assert feats.index.equals(ohlcv.index)
    assert feats["funding_rate"].isna().all()


def test_funding_ma_reflects_regime_change(ohlcv, funding):
    feats = funding_features(ohlcv, funding)
    tail = feats["funding_rate"].dropna()
    assert tail.iloc[-1] == pytest.approx(0.0005)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_derivatives.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.derivatives'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/derivatives.py`:
```python
"""Perpetual-futures funding features.

Funding is published every 8 hours. A value stamped at time T only becomes known
at T, so it is joined onto bars with `merge_asof(direction="backward")` against
each bar's close_time — never forward-filled from the future.
"""

from __future__ import annotations

import pandas as pd

FUNDING_COLUMNS = [
    "funding_rate",
    "funding_ma_3",
    "funding_ma_21",
    "funding_deviation",
    "hours_since_funding",
]


def funding_features(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Attach funding features to a kline frame, indexed like `bars`."""
    out = pd.DataFrame(index=bars.index, columns=FUNDING_COLUMNS, dtype="float64")
    if funding.empty:
        return out

    enriched = funding.copy()
    enriched["funding_ma_3"] = enriched["funding_rate"].rolling(3, min_periods=3).mean()
    enriched["funding_ma_21"] = enriched["funding_rate"].rolling(21, min_periods=21).mean()
    enriched["funding_deviation"] = enriched["funding_rate"] - enriched["funding_ma_21"]
    enriched = enriched.reset_index().rename(columns={"funding_time": "event_time"})

    decision_time = bars["close_time"] if "close_time" in bars.columns else bars.index
    left = pd.DataFrame({"decision_time": pd.Series(decision_time).to_numpy()}, index=bars.index)
    left = left.reset_index().sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        enriched.sort_values("event_time"),
        left_on="decision_time",
        right_on="event_time",
        direction="backward",
    ).set_index("open_time")

    out["funding_rate"] = merged["funding_rate"]
    out["funding_ma_3"] = merged["funding_ma_3"]
    out["funding_ma_21"] = merged["funding_ma_21"]
    out["funding_deviation"] = merged["funding_deviation"]
    out["hours_since_funding"] = (
        merged["decision_time"] - merged["event_time"]
    ).dt.total_seconds() / 3600.0

    return out.reindex(bars.index)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_derivatives.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/derivatives.py tests/test_features_derivatives.py
git commit -m "feat(features): add funding rate features with point-in-time join"
```

---

## Task 14: Regime and calendar features

**Files:**
- Create: `src/cryptopred/features/regime.py`
- Create: `src/cryptopred/features/timefeat.py`
- Test: `tests/test_features_regime_time.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_regime_time.py`:
```python
import numpy as np

from cryptopred.features.regime import adx, regime_features
from cryptopred.features.timefeat import time_features


def test_adx_is_bounded(ohlcv):
    a = adx(ohlcv, period=14).dropna()
    assert a.min() >= 0.0
    assert a.max() <= 100.0


def test_adx_high_in_strong_trend(rising):
    a = adx(rising, period=14).dropna()
    assert a.iloc[-1] > 40.0


def test_regime_features_names(ohlcv):
    feats = regime_features(ohlcv)
    for name in ["adx_14", "trend_strength", "vol_regime"]:
        assert name in feats.columns


def test_time_features_are_cyclical(ohlcv):
    feats = time_features(ohlcv)
    assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos"} <= set(feats.columns)
    # sin^2 + cos^2 == 1 for a proper cyclical encoding
    total = feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2
    assert np.allclose(total, 1.0)


def test_session_flags_are_mutually_exhaustive(ohlcv):
    feats = time_features(ohlcv)
    total = feats["session_asia"] + feats["session_europe"] + feats["session_us"]
    assert total.min() >= 1
    assert feats["session_asia"].isin([0, 1]).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_regime_time.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.regime'`

- [ ] **Step 3: Write the regime implementation**

Create `src/cryptopred/features/regime.py`:
```python
"""Market regime: is price trending or ranging, is volatility high or low."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.base import pct_rank, safe_divide, wilder_smooth
from cryptopred.features.volatility import atr, realized_vol


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index: trend strength regardless of direction."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    atr_series = atr(df, period)
    plus_di = 100.0 * safe_divide(wilder_smooth(plus_dm, period), atr_series)
    minus_di = 100.0 * safe_divide(wilder_smooth(minus_dm, period), atr_series)

    dx = 100.0 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return wilder_smooth(dx, period)


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["adx_14"] = adx(df, 14)
    out["adx_28"] = adx(df, 28)
    out["trend_strength"] = out["adx_14"] / 100.0

    rv = realized_vol(df["close"], 24)
    out["vol_regime"] = pct_rank(rv, window=720)
    out["is_high_vol"] = (out["vol_regime"] > 0.8).astype("float64")
    out["is_low_vol"] = (out["vol_regime"] < 0.2).astype("float64")
    out["is_trending"] = (out["adx_14"] > 25).astype("float64")

    return out
```

- [ ] **Step 4: Write the calendar implementation**

Create `src/cryptopred/features/timefeat.py`:
```python
"""Calendar features. Crypto trades 24/7 but liquidity is not uniform: Asian,
European and US hours have distinct volatility signatures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)

    hour = idx.hour.to_numpy(dtype="float64")
    dow = idx.dayofweek.to_numpy(dtype="float64")

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # UTC session windows, deliberately overlapping at the handovers
    out["session_asia"] = ((hour >= 0) & (hour < 9)).astype("float64")
    out["session_europe"] = ((hour >= 7) & (hour < 16)).astype("float64")
    out["session_us"] = ((hour >= 13) & (hour < 22)).astype("float64")
    # 22:00-00:00 UTC belongs to no major session; assign it to Asia's run-up
    out.loc[(hour >= 22), "session_asia"] = 1.0

    out["is_weekend"] = (dow >= 5).astype("float64")

    return out
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_features_regime_time.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/cryptopred/features/regime.py src/cryptopred/features/timefeat.py tests/test_features_regime_time.py
git commit -m "feat(features): add regime and calendar features"
```

---

## Task 15: Multi-timeframe features

This is the highest-risk module for leakage. A 4h bar that closes at 04:00 must not influence a 1h bar at 01:00. `merge_asof(direction="backward")` on close times enforces that.

**Files:**
- Create: `src/cryptopred/features/mtf.py`
- Test: `tests/test_features_mtf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_mtf.py`:
```python
import pandas as pd
import pytest

from cryptopred.features.mtf import mtf_features, resample_ohlcv


def test_resample_indexes_by_close_time(ohlcv):
    htf = resample_ohlcv(ohlcv, "4h")
    # right-labelled: the bar covering 00:00-04:00 is stamped 04:00
    assert htf.index[0] == pd.Timestamp("2024-01-01 04:00", tz="UTC")
    assert htf["high"].iloc[0] == pytest.approx(ohlcv["high"].iloc[0:4].max())
    assert htf["low"].iloc[0] == pytest.approx(ohlcv["low"].iloc[0:4].min())
    assert htf["close"].iloc[0] == pytest.approx(ohlcv["close"].iloc[3])


def test_mtf_features_do_not_use_unclosed_higher_bars(ohlcv):
    feats = mtf_features(ohlcv, rule="4h", prefix="h4")
    # bars before the first 4h close have nothing to inherit
    first_valid = feats["h4_rsi_14"].first_valid_index()
    assert first_valid is not None
    assert first_valid >= ohlcv.index[3]


def test_mtf_values_are_constant_within_a_higher_bar(ohlcv):
    feats = mtf_features(ohlcv, rule="4h", prefix="h4")
    window = feats["h4_close_ret_1"].iloc[100:104]
    assert window.nunique(dropna=True) <= 1


def test_mtf_prefixes_all_columns(ohlcv):
    feats = mtf_features(ohlcv, rule="1D", prefix="d1")
    assert all(c.startswith("d1_") for c in feats.columns)
    assert feats.index.equals(ohlcv.index)


def test_mtf_is_prefix_invariant(ohlcv):
    """Truncating the input must not change already-computed values."""
    cut = 500
    full = mtf_features(ohlcv, rule="4h", prefix="h4").iloc[:cut]
    partial = mtf_features(ohlcv.iloc[:cut], rule="4h", prefix="h4")
    pd.testing.assert_frame_equal(full.tail(50), partial.tail(50), check_freq=False)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_mtf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.mtf'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/mtf.py`:
```python
"""Higher-timeframe context, joined point-in-time.

The resampled frame is indexed by the higher-timeframe bar's CLOSE time, and the
join is `direction="backward"` against each base bar's close time. That means a
base bar can only see higher-timeframe bars that had already closed when the
prediction would have been made.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.features.base import ewma, safe_divide
from cryptopred.features.momentum import rsi
from cryptopred.features.volatility import atr

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a higher timeframe, indexed by each higher bar's close time."""
    cols = {k: v for k, v in _AGG.items() if k in df.columns}
    out = df.resample(rule, label="right", closed="left").agg(cols)
    return out.dropna(subset=["close"])


def _core_features(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=htf.index)
    close = htf["close"]

    out[f"{prefix}_rsi_14"] = rsi(close, 14)
    out[f"{prefix}_close_ret_1"] = close.pct_change(1)
    out[f"{prefix}_close_ret_3"] = close.pct_change(3)
    out[f"{prefix}_atr_norm"] = safe_divide(atr(htf, 14), close)
    ema20 = ewma(close, 20)
    out[f"{prefix}_ema_dist_20"] = safe_divide(close - ema20, ema20)
    roll_high = htf["high"].rolling(20, min_periods=20).max()
    roll_low = htf["low"].rolling(20, min_periods=20).min()
    out[f"{prefix}_range_pos_20"] = safe_divide(
        close - roll_low, roll_high - roll_low, fill=0.5
    )
    return out


def mtf_features(bars: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    """Higher-timeframe features aligned onto the base bar index."""
    htf = resample_ohlcv(bars, rule)
    if htf.empty:
        return pd.DataFrame(index=bars.index)

    feats = _core_features(htf, prefix).reset_index()
    feats = feats.rename(columns={feats.columns[0]: "htf_close_time"})

    decision_time = bars["close_time"] if "close_time" in bars.columns else bars.index
    left = pd.DataFrame(
        {"decision_time": pd.Series(decision_time).to_numpy()}, index=bars.index
    ).reset_index()
    left = left.sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        feats.sort_values("htf_close_time"),
        left_on="decision_time",
        right_on="htf_close_time",
        direction="backward",
    ).set_index("open_time")

    keep = [c for c in merged.columns if c.startswith(f"{prefix}_")]
    return merged[keep].reindex(bars.index)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_mtf.py -v`
Expected: `5 passed`

If `test_resample_indexes_by_close_time` fails on the first index value, check the `closed`/`label` pair: `closed="left", label="right"` means the bin covers `[00:00, 04:00)` and is stamped `04:00`, which is exactly when that bar closes.

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/mtf.py tests/test_features_mtf.py
git commit -m "feat(features): add point-in-time multi-timeframe features"
```

---

## Task 16: Feature pipeline

**Files:**
- Create: `src/cryptopred/features/pipeline.py`
- Test: `tests/test_features_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_pipeline.py`:
```python
import pandas as pd

from cryptopred.features.pipeline import build_features


def test_build_features_returns_wide_frame(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert feats.index.equals(ohlcv.index)
    assert len(feats.columns) >= 60


def test_build_features_column_names_are_unique(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert len(feats.columns) == len(set(feats.columns))


def test_build_features_all_float64(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    non_float = [c for c in feats.columns if feats[c].dtype != "float64"]
    assert non_float == []


def test_build_features_contains_no_raw_price_columns(ohlcv):
    """Raw price would let a model memorise date ranges instead of learning."""
    feats = build_features(ohlcv, interval="1h")
    for banned in ["open", "high", "low", "close", "volume", "close_time"]:
        assert banned not in feats.columns


def test_build_features_with_funding(ohlcv):
    idx = pd.date_range("2024-01-01", periods=200, freq="8h", tz="UTC", name="funding_time")
    funding = pd.DataFrame({"funding_rate": [0.0001] * 200}, index=idx)
    feats = build_features(ohlcv, interval="1h", funding=funding)
    assert "funding_rate" in feats.columns


def test_build_features_1m_uses_1m_mtf_rules(ohlcv_1m):
    feats = build_features(ohlcv_1m, interval="1m")
    assert any(c.startswith("m15_") for c in feats.columns)
    assert any(c.startswith("h1_") for c in feats.columns)


def test_no_infinite_values(ohlcv):
    feats = build_features(ohlcv, interval="1h")
    assert not feats.isin([float("inf"), float("-inf")]).any().any()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.features.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/features/pipeline.py`:
```python
"""Assemble every feature group into one wide frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.config import FeatureConfig
from cryptopred.features.derivatives import funding_features
from cryptopred.features.momentum import momentum_features
from cryptopred.features.mtf import mtf_features
from cryptopred.features.regime import regime_features
from cryptopred.features.structure import structure_features
from cryptopred.features.timefeat import time_features
from cryptopred.features.volatility import volatility_features
from cryptopred.features.volume import volume_features

# Higher timeframes to attach, keyed by base interval. The prefix becomes the
# column name prefix, so it must be a valid identifier fragment.
_MTF_MAP: dict[str, list[tuple[str, str]]] = {
    "1m": [("15min", "m15"), ("1h", "h1")],
    "1h": [("4h", "h4"), ("1D", "d1")],
}


def build_features(
    bars: pd.DataFrame,
    interval: str,
    funding: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute every feature for a kline frame.

    The result is indexed exactly like `bars`. Early rows contain NaN where an
    indicator's warm-up window is not yet satisfied; the dataset builder drops
    those rows rather than imputing them.
    """
    config = config or FeatureConfig()

    parts = [
        momentum_features(bars),
        volatility_features(bars),
        volume_features(bars, zscore_window=config.zscore_window),
        structure_features(bars),
        regime_features(bars),
        time_features(bars),
    ]

    for rule, prefix in _MTF_MAP.get(interval, []):
        parts.append(mtf_features(bars, rule=rule, prefix=prefix))

    if funding is not None:
        parts.append(funding_features(bars, funding))

    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype("float64")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_pipeline.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/features/pipeline.py tests/test_features_pipeline.py
git commit -m "feat(features): assemble full feature pipeline"
```

---

## Task 17: Leakage test suite

This is the most important task in the plan. If these tests are weak, every metric produced in Plan 2 is worthless.

**Files:**
- Create: `tests/test_leakage.py`

- [ ] **Step 1: Write the prefix-invariance test**

Create `tests/test_leakage.py`:
```python
"""Leakage detection.

The point-in-time rule: a feature at bar t may only use data from bars that
closed at or before t. These tests are the enforcement mechanism. If any of them
fails, STOP — do not train a model, do not report a metric. Find the leak first.
"""

import numpy as np
import pandas as pd
import pytest

from cryptopred.features.pipeline import build_features
from cryptopred.labels.barrier import make_labels
from tests.conftest import make_ohlcv


def test_features_are_prefix_invariant():
    """Computing features on a truncated history must give identical values for
    the rows both runs share. If truncation changes a value, that value depended
    on data from the future."""
    bars = make_ohlcv(n=1200, seed=1)
    cut = 800

    full = build_features(bars, interval="1h").iloc[:cut]
    partial = build_features(bars.iloc[:cut], interval="1h")

    # Compare the last 100 shared rows — enough to catch window bugs, cheap to run.
    pd.testing.assert_frame_equal(
        full.tail(100), partial.tail(100), check_freq=False, rtol=1e-9, atol=1e-12
    )


def test_future_bars_cannot_change_past_features():
    """Replace every bar after the cut with an extreme outlier. Features at and
    before the cut must be bit-identical."""
    bars = make_ohlcv(n=1000, seed=2)
    cut = 700

    poisoned = bars.copy()
    for col in ["open", "high", "low", "close"]:
        poisoned.iloc[cut:, poisoned.columns.get_loc(col)] = 1e9
    poisoned.iloc[cut:, poisoned.columns.get_loc("volume")] = 1e12

    clean_feats = build_features(bars, interval="1h").iloc[:cut]
    poisoned_feats = build_features(poisoned, interval="1h").iloc[:cut]

    pd.testing.assert_frame_equal(
        clean_feats.tail(100), poisoned_feats.tail(100), check_freq=False,
        rtol=1e-9, atol=1e-12,
    )


def test_no_feature_is_suspiciously_correlated_with_the_label():
    """A feature correlating above 0.30 with the forward return is a red flag.
    Real technical indicators land far below that on hourly crypto data."""
    bars = make_ohlcv(n=3000, seed=3)
    feats = build_features(bars, interval="1h")
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)

    joined = feats.join(labels[["forward_return"]]).dropna()
    correlations = joined.corr()["forward_return"].drop("forward_return").abs()

    offenders = correlations[correlations > 0.30].sort_values(ascending=False)
    assert offenders.empty, f"Suspicious correlation with the future:\n{offenders}"


def test_label_uses_only_future_and_feature_only_past():
    """Explicit check of the contract: forward_return at t is derived from
    close[t+H], and is NaN for the final H bars where the future is unknown."""
    bars = make_ohlcv(n=100, seed=4)
    horizon = 4
    labels = make_labels(bars, horizon=horizon, atr_period=14, band_k=0.5)

    expected = bars["close"].iloc[10 + horizon] / bars["close"].iloc[10] - 1
    assert labels["forward_return"].iloc[10] == pytest.approx(expected)
    assert labels["forward_return"].iloc[-horizon:].isna().all()
    assert labels["label"].iloc[-horizon:].isna().all()


def test_shuffled_labels_destroy_all_signal():
    """Sanity check on the joining logic: if labels are shuffled, correlation
    with every feature must collapse. If it does not, features and labels are
    being aligned by position somewhere instead of by index."""
    bars = make_ohlcv(n=3000, seed=5)
    feats = build_features(bars, interval="1h")
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)

    joined = feats.join(labels[["forward_return"]]).dropna()
    rng = np.random.default_rng(0)
    joined["forward_return"] = rng.permutation(joined["forward_return"].to_numpy())

    correlations = joined.corr()["forward_return"].drop("forward_return").abs()
    assert correlations.max() < 0.15


def test_features_never_reference_the_current_bar_open_of_the_next_bar():
    """Regression guard: append one extra bar and confirm the previously-final
    row's features do not change."""
    bars = make_ohlcv(n=600, seed=6)
    shorter = bars.iloc[:-1]

    long_feats = build_features(bars, interval="1h").iloc[:-1]
    short_feats = build_features(shorter, interval="1h")

    pd.testing.assert_frame_equal(
        long_feats.tail(5), short_feats.tail(5), check_freq=False, rtol=1e-9, atol=1e-12
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.labels'` — the labels module is Task 18. Write it next, then return here.

- [ ] **Step 3: Commit the test file now (it will go green after Task 18)**

```bash
git add tests/test_leakage.py
git commit -m "test: add leakage detection suite (red until labels land)"
```

---

## Task 18: Label engine

**Files:**
- Create: `src/cryptopred/labels/__init__.py`
- Create: `src/cryptopred/labels/barrier.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_labels.py`:
```python
import numpy as np
import pandas as pd
import pytest

from cryptopred.labels.barrier import LABEL_DOWN, LABEL_FLAT, LABEL_UP, make_labels

from tests.conftest import make_ohlcv


def _flat_market(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": 1.0,
            "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=idx,
    )


def test_forward_return_is_computed_from_horizon_bars_ahead():
    bars = make_ohlcv(n=50, seed=11)
    labels = make_labels(bars, horizon=3, atr_period=14, band_k=0.5)
    expected = bars["close"].iloc[20 + 3] / bars["close"].iloc[20] - 1
    assert labels["forward_return"].iloc[20] == pytest.approx(expected)


def test_last_horizon_rows_are_nan():
    bars = make_ohlcv(n=50, seed=12)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert labels["forward_return"].iloc[-4:].isna().all()
    assert labels["label"].iloc[-4:].isna().all()


def test_flat_market_is_all_flat():
    bars = _flat_market()
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    valid = labels["label"].dropna()
    assert (valid == LABEL_FLAT).all()


def test_strong_rise_is_labelled_up():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
            "close_time": idx + pd.Timedelta(hours=1),
        },
        index=idx,
    )
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert (labels["label"].dropna() == LABEL_UP).all()


def test_band_scales_with_k():
    bars = make_ohlcv(n=500, seed=13)
    narrow = make_labels(bars, horizon=4, atr_period=14, band_k=0.1)
    wide = make_labels(bars, horizon=4, atr_period=14, band_k=2.0)
    assert (wide["label"] == LABEL_FLAT).sum() > (narrow["label"] == LABEL_FLAT).sum()


def test_label_values_are_only_minus_one_zero_one():
    bars = make_ohlcv(n=500, seed=14)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert set(labels["label"].dropna().unique()) <= {LABEL_DOWN, LABEL_FLAT, LABEL_UP}


def test_columns_present():
    bars = make_ohlcv(n=100, seed=15)
    labels = make_labels(bars, horizon=4, atr_period=14, band_k=0.5)
    assert list(labels.columns) == ["forward_return", "band", "label"]
    assert labels.index.equals(bars.index)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.labels'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/labels/__init__.py`:
```python
"""Label definitions: what counts as 'up', 'down', and 'not worth trading'."""
```

Create `src/cryptopred/labels/barrier.py`:
```python
"""Three-class labels with a volatility-scaled dead zone.

Forcing a binary up/down decision on a bar where price barely moved teaches the
model to fit noise. The dead zone is proportional to ATR so it adapts: in a calm
market a 0.2% move is meaningful, in a violent one it is nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopred.features.volatility import atr

LABEL_DOWN = -1.0
LABEL_FLAT = 0.0
LABEL_UP = 1.0


def make_labels(
    bars: pd.DataFrame,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 0.5,
) -> pd.DataFrame:
    """Label each bar by where price sits `horizon` bars later.

    Returns a frame indexed like `bars` with:
      - forward_return: (close[t+H] / close[t]) - 1
      - band: the dead-zone half-width at t, as a fraction of price
      - label: -1 down, 0 flat, +1 up; NaN for the last H bars
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = bars["close"]
    forward_return = close.shift(-horizon) / close - 1.0
    band = band_k * atr(bars, atr_period) / close

    label = pd.Series(np.nan, index=bars.index, dtype="float64")
    known = forward_return.notna() & band.notna()
    label[known & (forward_return > band)] = LABEL_UP
    label[known & (forward_return < -band)] = LABEL_DOWN
    label[known & (forward_return.abs() <= band)] = LABEL_FLAT

    return pd.DataFrame(
        {"forward_return": forward_return, "band": band, "label": label}
    )


def make_triple_barrier_labels(
    bars: pd.DataFrame,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 1.0,
) -> pd.DataFrame:
    """Alternative labelling: whichever barrier price touches first wins.

    Used by the backtest in Plan 2 to check that results are not an artefact of
    the fixed-horizon definition. Slower (a Python loop) but only run offline.
    """
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    band = (band_k * atr(bars, atr_period) / bars["close"]).to_numpy()

    n = len(bars)
    label = np.full(n, np.nan)
    for i in range(n - horizon):
        if not np.isfinite(band[i]):
            continue
        upper = close[i] * (1 + band[i])
        lower = close[i] * (1 - band[i])
        outcome = LABEL_FLAT
        for j in range(i + 1, i + horizon + 1):
            if high[j] >= upper:
                outcome = LABEL_UP
                break
            if low[j] <= lower:
                outcome = LABEL_DOWN
                break
        label[i] = outcome

    forward_return = pd.Series(close, index=bars.index).shift(-horizon) / bars["close"] - 1.0
    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "band": pd.Series(band, index=bars.index),
            "label": pd.Series(label, index=bars.index),
        }
    )
```

- [ ] **Step 4: Run the label tests**

Run: `uv run pytest tests/test_labels.py -v`
Expected: `7 passed`

- [ ] **Step 5: Run the leakage suite — this is the gate**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: `6 passed`

**If any leakage test fails, do not continue to Task 19.** Debug it: the failing assertion names the feature column. Common causes are a `center=True` rolling window, a negative `shift`, a whole-series `mean()`/`std()`, or an `merge_asof` with `direction="forward"` or `"nearest"`.

- [ ] **Step 6: Commit**

```bash
git add src/cryptopred/labels tests/test_labels.py
git commit -m "feat(labels): add ATR-banded three-class labels"
```

---

## Task 19: Dataset builder

**Files:**
- Create: `src/cryptopred/dataset/__init__.py`
- Create: `src/cryptopred/dataset/builder.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:
```python
from cryptopred.dataset.builder import build_dataset, feature_columns
from tests.conftest import make_ohlcv


def test_build_dataset_joins_features_and_labels():
    bars = make_ohlcv(n=1500, seed=21)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert "label" in ds.columns
    assert "forward_return" in ds.columns
    assert "label_class" in ds.columns
    assert len(feature_columns(ds)) >= 60


def test_build_dataset_drops_warmup_and_tail_rows():
    bars = make_ohlcv(n=1500, seed=22)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert ds.notna().all().all()
    assert len(ds) < len(bars)
    assert ds.index[-1] <= bars.index[-5]


def test_label_class_maps_to_zero_one_two():
    bars = make_ohlcv(n=1500, seed=23)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)

    assert set(ds["label_class"].unique()) <= {0, 1, 2}
    down = ds[ds["label"] == -1.0]
    if not down.empty:
        assert (down["label_class"] == 0).all()
    flat = ds[ds["label"] == 0.0]
    if not flat.empty:
        assert (flat["label_class"] == 1).all()


def test_metadata_columns_present():
    bars = make_ohlcv(n=1500, seed=24)
    ds = build_dataset(
        bars, interval="1h", horizon=4, atr_period=14, band_k=0.5, symbol="BTCUSDT"
    )
    assert (ds["symbol"] == "BTCUSDT").all()
    assert ds.index.name == "open_time"


def test_feature_columns_excludes_targets_and_metadata():
    bars = make_ohlcv(n=1500, seed=25)
    ds = build_dataset(
        bars, interval="1h", horizon=4, atr_period=14, band_k=0.5, symbol="BTCUSDT"
    )
    cols = feature_columns(ds)
    for banned in ["label", "label_class", "forward_return", "band", "symbol"]:
        assert banned not in cols


def test_features_are_float32_to_save_memory():
    bars = make_ohlcv(n=1500, seed=27)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    for col in feature_columns(ds):
        assert ds[col].dtype == "float32", col


def test_empty_input_returns_empty_dataset():
    empty = make_ohlcv(n=10, seed=26)
    ds = build_dataset(empty, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    assert ds.empty
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.dataset'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/dataset/__init__.py`:
```python
"""Training dataset assembly."""
```

Create `src/cryptopred/dataset/builder.py`:
```python
"""Join features and labels into a model-ready table.

Rows with any NaN are dropped rather than imputed. Imputing a warm-up window
invents data; dropping costs a few hundred bars out of tens of thousands.
"""

from __future__ import annotations

import pandas as pd

from cryptopred.config import FeatureConfig
from cryptopred.features.pipeline import build_features
from cryptopred.labels.barrier import make_labels

TARGET_COLUMNS = ["label", "label_class", "forward_return", "band"]
METADATA_COLUMNS = ["symbol", "interval"]

# LightGBM wants contiguous class ids starting at zero.
_CLASS_MAP = {-1.0: 0, 0.0: 1, 1.0: 2}


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Every column a model is allowed to train on."""
    excluded = set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
    return [c for c in dataset.columns if c not in excluded]


def build_dataset(
    bars: pd.DataFrame,
    interval: str,
    horizon: int,
    atr_period: int = 14,
    band_k: float = 0.5,
    funding: pd.DataFrame | None = None,
    symbol: str | None = None,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Produce a clean training table for one symbol and interval."""
    features = build_features(
        bars, interval=interval, funding=funding, config=feature_config
    )
    labels = make_labels(bars, horizon=horizon, atr_period=atr_period, band_k=band_k)

    dataset = features.join(labels)
    dataset = dataset.dropna()
    if dataset.empty:
        return dataset

    dataset["label_class"] = dataset["label"].map(_CLASS_MAP).astype("int8")

    # 1m datasets run to millions of rows. float32 halves memory and file size
    # with no meaningful precision loss for gradient-boosted trees.
    feature_cols = [c for c in features.columns if c in dataset.columns]
    dataset[feature_cols] = dataset[feature_cols].astype("float32")

    if symbol is not None:
        dataset["symbol"] = symbol
    dataset["interval"] = interval

    return dataset
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/dataset tests/test_dataset.py
git commit -m "feat(dataset): join features and labels into a training table"
```

---

## Task 20: Data quality report

Before training anything, you need to know the class balance and whether the data is sane. A 90/5/5 class split changes how Plan 2 must be built.

**Files:**
- Create: `src/cryptopred/dataset/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_quality.py`:
```python
from cryptopred.dataset.builder import build_dataset
from cryptopred.dataset.quality import quality_report
from tests.conftest import make_ohlcv


def test_quality_report_fields():
    bars = make_ohlcv(n=2000, seed=31)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    report = quality_report(ds)

    assert report["rows"] == len(ds)
    assert report["n_features"] >= 60
    assert set(report["class_balance"]) <= {"down", "flat", "up"}
    assert abs(sum(report["class_balance"].values()) - 1.0) < 1e-9
    assert report["start"] == ds.index.min()
    assert report["end"] == ds.index.max()


def test_quality_report_flags_constant_features():
    bars = make_ohlcv(n=2000, seed=32)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    ds["always_seven"] = 7.0
    report = quality_report(ds)
    assert "always_seven" in report["constant_features"]


def test_quality_report_flags_extreme_correlation():
    bars = make_ohlcv(n=2000, seed=33)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    ds["cheating"] = ds["forward_return"] * 10.0
    report = quality_report(ds)
    assert "cheating" in report["suspicious_features"]


def test_format_report_is_readable():
    from cryptopred.dataset.quality import format_report

    bars = make_ohlcv(n=2000, seed=34)
    ds = build_dataset(bars, interval="1h", horizon=4, atr_period=14, band_k=0.5)
    text = format_report(quality_report(ds))
    assert "rows" in text.lower()
    assert "class balance" in text.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.dataset.quality'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/dataset/quality.py`:
```python
"""Dataset sanity checks, run before any model sees the data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.dataset.builder import feature_columns

# Above this absolute correlation with the forward return, a feature is almost
# certainly leaking. See spec section 5.
SUSPICIOUS_CORRELATION = 0.30

_CLASS_NAMES = {-1.0: "down", 0.0: "flat", 1.0: "up"}


def quality_report(dataset: pd.DataFrame) -> dict[str, Any]:
    features = feature_columns(dataset)

    balance = (
        dataset["label"].map(_CLASS_NAMES).value_counts(normalize=True).to_dict()
    )

    stds = dataset[features].std()
    constant = sorted(stds[stds == 0].index.tolist())

    correlations = (
        dataset[features + ["forward_return"]]
        .corr()["forward_return"]
        .drop("forward_return")
        .abs()
    )
    suspicious = sorted(correlations[correlations > SUSPICIOUS_CORRELATION].index.tolist())

    return {
        "rows": len(dataset),
        "n_features": len(features),
        "start": dataset.index.min(),
        "end": dataset.index.max(),
        "class_balance": balance,
        "constant_features": constant,
        "suspicious_features": suspicious,
        "max_abs_correlation": float(correlations.max()),
        "top_correlations": correlations.sort_values(ascending=False).head(10).to_dict(),
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "DATASET QUALITY REPORT",
        "=" * 60,
        f"Rows:       {report['rows']:,}",
        f"Features:   {report['n_features']}",
        f"Period:     {report['start']} -> {report['end']}",
        "",
        "Class balance:",
    ]
    for name, share in sorted(report["class_balance"].items()):
        lines.append(f"  {name:<6} {share:6.2%}")

    lines.append("")
    lines.append(f"Max |correlation| with forward return: {report['max_abs_correlation']:.4f}")
    lines.append("Top correlations:")
    for name, value in report["top_correlations"].items():
        lines.append(f"  {name:<28} {value:.4f}")

    if report["constant_features"]:
        lines.append("")
        lines.append(f"WARNING constant features: {', '.join(report['constant_features'])}")

    if report["suspicious_features"]:
        lines.append("")
        lines.append("*** LEAKAGE SUSPECTED ***")
        lines.append(f"Features correlating >{SUSPICIOUS_CORRELATION} with the future:")
        for name in report["suspicious_features"]:
            lines.append(f"  {name}")
        lines.append("Do not train until this is explained.")

    lines.append("=" * 60)
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_quality.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cryptopred/dataset/quality.py tests/test_quality.py
git commit -m "feat(dataset): add quality report with leakage screening"
```

---

## Task 21: Dataset CLI and first real build

**Files:**
- Create: `src/cryptopred/dataset/cli.py`
- Test: `tests/test_dataset_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_cli.py`:
```python
import pandas as pd
from typer.testing import CliRunner

from cryptopred.config import Config
from cryptopred.dataset.cli import app, run_build
from cryptopred.ingest.storage import ParquetStore
from tests.conftest import make_ohlcv

runner = CliRunner()


def test_run_build_writes_parquet(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]

    store = ParquetStore(tmp_path / "raw")
    store.write("klines", "BTCUSDT", "1h", make_ohlcv(n=2000, seed=41))

    paths = run_build(cfg, store=store, quiet=True)

    assert len(paths) == 1
    assert paths[0].exists()
    ds = pd.read_parquet(paths[0])
    assert "label_class" in ds.columns
    assert len(ds) > 1000


def test_run_build_skips_missing_symbols(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["NOPEUSDT"]
    cfg.data.intervals = ["1h"]

    store = ParquetStore(tmp_path / "raw")
    assert run_build(cfg, store=store, quiet=True) == []


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_dataset_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cryptopred.dataset.cli'`

- [ ] **Step 3: Write the implementation**

Create `src/cryptopred/dataset/cli.py`:
```python
"""Command line entry point for dataset construction."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from cryptopred.config import Config, load_config
from cryptopred.dataset.builder import build_dataset
from cryptopred.dataset.quality import format_report, quality_report
from cryptopred.ingest.storage import ParquetStore

app = typer.Typer(help="Build model-ready datasets from stored raw data.")
logger = logging.getLogger(__name__)


def run_build(cfg: Config, store: ParquetStore, quiet: bool = False) -> list[Path]:
    """Build one dataset per symbol/interval. Returns the written file paths."""
    out_dir = cfg.dataset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for symbol in cfg.data.symbols:
        for interval in cfg.data.intervals:
            bars = store.read("klines", symbol, interval)
            if bars.empty:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: no raw data, skipping")
                continue

            funding = store.read("funding", symbol, "8h")
            horizon = cfg.labels.horizon_bars.get(interval)
            if horizon is None:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: no horizon configured, skipping")
                continue

            dataset = build_dataset(
                bars,
                interval=interval,
                horizon=horizon,
                atr_period=cfg.labels.atr_period,
                band_k=cfg.labels.band_k,
                funding=funding if not funding.empty else None,
                symbol=symbol,
                feature_config=cfg.features,
            )
            if dataset.empty:
                if not quiet:
                    typer.echo(f"{symbol} {interval}: dataset empty after cleaning")
                continue

            path = out_dir / f"{symbol}_{interval}.parquet"
            dataset.to_parquet(path, engine="pyarrow", index=True)
            written.append(path)

            if not quiet:
                typer.echo(f"\n{symbol} {interval} -> {path}")
                typer.echo(format_report(quality_report(dataset)))

    return written


@app.command()
def build(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Build datasets for every configured symbol and interval."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    store = ParquetStore(cfg.data.root / "raw")
    paths = run_build(cfg, store)
    typer.echo(f"\nWrote {len(paths)} dataset file(s).")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_dataset_cli.py -v`
Expected: `3 passed`

- [ ] **Step 5: Run the full suite**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run pytest -v
```
Expected: every test passes. Note the count — it should be around 80.

- [ ] **Step 6: Lint**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run ruff check src tests && uv run ruff format --check src tests
```
Expected: `All checks passed!`. If format check fails, run `uv run ruff format src tests` and re-run.

- [ ] **Step 7: Build the real dataset**

Run:
```bash
cd /c/Users/akira/Desktop/crypto-predict && uv run cryptopred-dataset build
```
Expected: for each symbol/interval, a quality report followed by `Wrote 4 dataset file(s).`

**Read the report carefully. Three things to check:**

1. **Row count** — 1h datasets should have roughly 50,000+ rows per symbol. Far fewer means ingest is incomplete; re-run `cryptopred-ingest klines`.
2. **Class balance** — expect roughly 30/40/30 (down/flat/up) with `band_k=0.5`. If `flat` exceeds 70%, lower `band_k` in `config/default.yaml` and rebuild; if `flat` is under 10%, raise it.
3. **Max |correlation|** — must be below 0.30. If the report prints `*** LEAKAGE SUSPECTED ***`, **stop**. Do not proceed to Plan 2. The named feature is reading the future; find and fix it, then rebuild.

- [ ] **Step 8: Record the outcome**

Create `docs/data-notes.md` with the actual numbers from Step 7:
```markdown
# Data Notes

## Dataset build YYYY-MM-DD

| Symbol | Interval | Rows | Period | down/flat/up | Max abs corr |
|---|---|---|---|---|---|
| BTCUSDT | 1h | ... | ... -> ... | ... | ... |
| ETHUSDT | 1h | ... | ... -> ... | ... | ... |
| BTCUSDT | 1m | ... | ... -> ... | ... | ... |
| ETHUSDT | 1m | ... | ... -> ... | ... | ... |

## Known gaps
(list any gaps reported by `cryptopred-ingest report`, with the reason if known)

## Decisions
- band_k = <value>, chosen because <reason from class balance>
```

- [ ] **Step 9: Commit**

```bash
git add src/cryptopred/dataset/cli.py tests/test_dataset_cli.py docs/data-notes.md
git commit -m "feat(dataset): add build CLI and record first real dataset build"
```

---

## Definition of Done for this plan

All of the following must be true before Plan 2 starts:

- [ ] `uv run pytest` passes with zero failures, including all six leakage tests.
- [ ] `uv run ruff check src tests` passes.
- [ ] `uv run cryptopred-ingest report` shows full coverage with zero unexplained gaps.
- [ ] `uv run cryptopred-dataset build` produces four parquet files.
- [ ] Every quality report shows max absolute correlation with the forward return below 0.30.
- [ ] Class balance is recorded in `docs/data-notes.md` and is not more extreme than 70% in any single class.

**If the leakage screen fires, this plan is not done.** A dataset that leaks makes every number in Plan 2 a lie, and the lie is expensive because it looks like success.
