"""Bulk history from Binance's public archive, data.binance.vision.

The REST API pages 1,500 bars at a time and refuses some regions outright (it
answers HTTP 451 from US cloud hosts). The archive serves the same futures
klines and funding as one zip per month, with a SHA-256 checksum beside each,
from a static host that is not region-locked. Five years of hourly bars for one
symbol is about 70 files instead of 30 paged requests - and it works where the
API does not.

Two properties of the archive shape what this module does:

- Monthly files exist only for finished months. Klines also have daily files
  for the current month; funding does not. Fetching the current month's bars
  would therefore give them a funding rate that stops at the end of last month,
  and `hours_since_funding` would climb to hundreds of hours - a value no
  training row ever had. So by default this stops at the end of the last
  complete month, where both series end together. `include_current_month`
  opts in for bars only, and says so.
- Older kline files have no header row and some newer ones carry microsecond
  timestamps. Both are detected rather than assumed.

Every file is checked against its published checksum before it is parsed. A
truncated download that happened to parse would put a hole in the history that
nothing downstream can tell from a real gap.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from xml.etree import ElementTree

import httpx
import pandas as pd

from cryptopred.ingest.binance import KLINE_COLUMNS
from cryptopred.ingest.storage import ParquetStore

logger = logging.getLogger(__name__)

ARCHIVE_BASE = "https://data.binance.vision"
LISTING_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"

_RAW_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
_FLOAT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]
# Anything above this is microseconds: 1e14 ms is the year 5138.
_MICROSECOND_THRESHOLD = 10**14

_PERIOD = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


class ChecksumMismatch(RuntimeError):
    """The downloaded bytes are not the file Binance published."""


def list_archives(client: httpx.Client, prefix: str) -> list[str]:
    """Every .zip key under `prefix`, oldest first, following S3 pagination."""
    keys: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        response = client.get(LISTING_BASE, params=params)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        page = [el.text or "" for el in root.iter(f"{_S3}Key")]
        keys.extend(k for k in page if k.endswith(".zip"))
        truncated = (root.findtext(f"{_S3}IsTruncated") or "false").lower() == "true"
        if not truncated or not page:
            break
        marker = page[-1]
    return sorted(keys, key=_period_of)


def _period_of(key: str) -> str:
    match = _PERIOD.search(key)
    return match.group(1) if match else ""


def fetch_archive(client: httpx.Client, key: str) -> bytes:
    """Download one zip and verify it against its published SHA-256."""
    response = client.get(f"{ARCHIVE_BASE}/{key}")
    response.raise_for_status()
    payload = response.content

    check = client.get(f"{ARCHIVE_BASE}/{key}.CHECKSUM")
    check.raise_for_status()
    expected = check.text.split()[0].strip().lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ChecksumMismatch(f"{key}: sha256 {actual} != published {expected}")
    return payload


def _read_csv(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        raw = archive.read(name)
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)


def _to_utc(values: pd.Series) -> pd.Series:
    values = values.astype("int64")
    unit = "us" if int(values.max()) >= _MICROSECOND_THRESHOLD else "ms"
    return pd.to_datetime(values, unit=unit, utc=True)


def parse_kline_archive(payload: bytes) -> pd.DataFrame:
    """One monthly or daily kline zip, in the same shape as parse_klines."""
    df = _read_csv(payload)
    df.columns = _RAW_KLINE_COLUMNS[: len(df.columns)]
    df["open_time"] = _to_utc(df["open_time"])
    df["close_time"] = _to_utc(df["close_time"])
    df[_FLOAT_COLUMNS] = df[_FLOAT_COLUMNS].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    df = df.set_index("open_time").sort_index()
    return df[~df.index.duplicated(keep="last")][KLINE_COLUMNS]


def parse_funding_archive(payload: bytes) -> pd.DataFrame:
    """One monthly fundingRate zip, in the same shape as parse_funding.

    The archive stamps each rate a few milliseconds after the hour
    (calc_time). Bars are matched to funding by their close at hh:59:59.999, so
    those milliseconds never move a rate onto a different bar.
    """
    df = _read_csv(payload)
    df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: len(df.columns)]
    out = pd.DataFrame(
        {"funding_rate": df["last_funding_rate"].astype("float64").to_numpy()},
        index=pd.DatetimeIndex(_to_utc(df["calc_time"]), name="funding_time"),
    )
    return out.sort_index()


def _last_complete_month_end(now: pd.Timestamp) -> pd.Timestamp:
    return now.tz_convert("UTC").normalize().replace(day=1)


def ingest_klines(
    client: httpx.Client,
    store: ParquetStore,
    symbol: str,
    interval: str,
    now: pd.Timestamp | None = None,
    include_current_month: bool = False,
) -> int:
    """Fetch every archived month this store does not already cover. Returns
    bars written. Resumes: months before the newest stored bar are skipped."""
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = _last_complete_month_end(now)
    last = store.last_open_time("klines", symbol, interval)
    resume_month = last.strftime("%Y-%m") if last is not None else ""

    monthly = list_archives(client, f"data/futures/um/monthly/klines/{symbol}/{interval}/")
    keys = [k for k in monthly if _period_of(k) >= resume_month]
    if include_current_month:
        # Daily files for every month that has no monthly archive yet - not only
        # the current one. Early in a month Binance has published the new
        # month's daily files but not last month's archive; taking only this
        # month's dailies then stored the 1st beside a hole where the rest of
        # last month belonged, and resuming from the newest bar never went back.
        daily = list_archives(client, f"data/futures/um/daily/klines/{symbol}/{interval}/")
        archived = {_period_of(k) for k in monthly}
        keys += [
            k
            for k in daily
            if _period_of(k)[:7] not in archived and _period_of(k)[:7] >= resume_month
        ]

    written = 0
    for key in keys:
        bars = parse_kline_archive(fetch_archive(client, key))
        if not include_current_month:
            bars = bars[bars.index < cutoff]
        if last is not None:
            bars = bars[bars.index > last]
        if bars.empty:
            continue
        store.append("klines", symbol, interval, bars)
        written += len(bars)
    return written


def ingest_funding(
    client: httpx.Client, store: ParquetStore, symbol: str
) -> int:
    """Every archived month of funding this store does not already cover."""
    last = store.last_open_time("funding", symbol, "8h")
    resume_month = last.strftime("%Y-%m") if last is not None else ""
    keys = [
        k
        for k in list_archives(client, f"data/futures/um/monthly/fundingRate/{symbol}/")
        if _period_of(k) >= resume_month
    ]
    written = 0
    for key in keys:
        rates = parse_funding_archive(fetch_archive(client, key))
        if last is not None:
            rates = rates[rates.index > last]
        if rates.empty:
            continue
        store.append("funding", symbol, "8h", rates)
        written += len(rates)
    return written
