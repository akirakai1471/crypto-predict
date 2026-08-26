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
