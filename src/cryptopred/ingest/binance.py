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
                logger.warning(
                    "Server error %s on %s, retrying in %.1fs", response.status_code, path, wait
                )
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
