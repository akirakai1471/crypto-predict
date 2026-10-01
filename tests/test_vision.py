"""The archive path must produce exactly what the API path does, or refuse."""

import hashlib
import io
import zipfile

import httpx
import pandas as pd
import pytest

from cryptopred.ingest.binance import KLINE_COLUMNS, parse_klines
from cryptopred.ingest.storage import ParquetStore
from cryptopred.ingest.vision import (
    ChecksumMismatch,
    fetch_archive,
    ingest_funding,
    ingest_klines,
    list_archives,
    parse_funding_archive,
    parse_kline_archive,
)

HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _rows(start: str, n: int, scale: int = 1) -> list[list]:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    for i in range(n):
        open_ms = t0 + i * 3_600_000
        rows.append(
            [open_ms * scale, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0,
             (open_ms + 3_599_999) * scale, 1000.0, 5, 4.0, 400.0, 0]
        )
    return rows


def _zip(name: str, rows: list[list], header: str | None = HEADER) -> bytes:
    lines = ([header] if header else []) + [",".join(str(v) for v in r) for r in rows]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "\n".join(lines) + "\n")
    return buffer.getvalue()


def _listing(keys: list[str]) -> str:
    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{body}</ListBucketResult>"
    )


def _archive_transport(files: dict[str, bytes], corrupt: set[str] = frozenset()):
    """Serve a fake data.binance.vision: listing, zips, and their checksums."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("s3-"):
            prefix = request.url.params["prefix"]
            keys = [k for k in files if k.startswith(prefix)]
            keys += [k + ".CHECKSUM" for k in keys]
            return httpx.Response(200, text=_listing(keys))
        key = request.url.path.lstrip("/")
        if key.endswith(".CHECKSUM"):
            payload = files[key.removesuffix(".CHECKSUM")]
            digest = hashlib.sha256(payload).hexdigest()
            if key.removesuffix(".CHECKSUM") in corrupt:
                digest = "0" * 64
            return httpx.Response(200, text=f"{digest}  {key.rsplit('/', 1)[-1][:-9]}\n")
        if key in files:
            return httpx.Response(200, content=files[key])
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_an_archive_month_parses_to_the_same_frame_as_the_api_does():
    """Same bars through both doors must be the same data, column for column."""
    rows = _rows("2025-08-01", 5)
    from_archive = parse_kline_archive(_zip("x.csv", rows))
    # The API sends times and trade counts as integers and prices as strings.
    as_api = [[r[0], *map(str, r[1:6]), r[6], str(r[7]), r[8], *map(str, r[9:])] for r in rows]
    from_api = parse_klines(as_api)
    pd.testing.assert_frame_equal(from_archive, from_api)
    assert list(from_archive.columns) == KLINE_COLUMNS


def test_old_files_without_a_header_parse_the_same():
    rows = _rows("2020-01-01", 3)
    pd.testing.assert_frame_equal(
        parse_kline_archive(_zip("x.csv", rows, header=None)),
        parse_kline_archive(_zip("x.csv", rows)),
    )


def test_microsecond_timestamps_are_detected_not_misread_as_year_50000():
    rows_us = _rows("2025-01-01", 3, scale=1000)
    parsed = parse_kline_archive(_zip("x.csv", rows_us))
    assert parsed.index[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_funding_archive_matches_the_api_shape():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "f.csv",
            "calc_time,funding_interval_hours,last_funding_rate\n"
            "1754006400001,8,-0.00001408\n1754035200003,8,0.00001549\n",
        )
    df = parse_funding_archive(buffer.getvalue())
    assert list(df.columns) == ["funding_rate"]
    assert df.index.name == "funding_time"
    assert df["funding_rate"].iloc[0] == pytest.approx(-0.00001408)


def test_a_file_that_does_not_match_its_checksum_is_refused():
    """A truncated download that happened to parse would be a silent hole."""
    key = "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2025-08.zip"
    files = {key: _zip("x.csv", _rows("2025-08-01", 3))}
    with (
        httpx.Client(transport=_archive_transport(files, corrupt={key})) as client,
        pytest.raises(ChecksumMismatch),
    ):
        fetch_archive(client, key)


def test_listing_returns_zips_oldest_first_and_skips_checksum_files():
    keys = [
        "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2020-02.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip",
    ]
    files = {k: b"" for k in keys}
    with httpx.Client(transport=_archive_transport(files)) as client:
        listed = list_archives(client, "data/futures/um/monthly/klines/BTCUSDT/1h/")
    assert listed == sorted(keys)


def _two_months():
    base = "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-"
    return {
        base + "2026-07.zip": _zip("a.csv", _rows("2026-07-01", 24 * 31)),
        base + "2026-08.zip": _zip("b.csv", _rows("2026-08-01", 24 * 31)),
        "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-09-01.zip": _zip(
            "c.csv", _rows("2026-09-01", 24)
        ),
    }


def test_ingest_stops_at_the_last_complete_month_by_default(tmp_path):
    """Funding has no daily archive, so bars past last month would carry a
    stale funding rate. By default both series end together."""
    store = ParquetStore(tmp_path)
    with httpx.Client(transport=_archive_transport(_two_months())) as client:
        n = ingest_klines(client, store, "BTCUSDT", "1h", now=pd.Timestamp("2026-09-15", tz="UTC"))
    stored = store.read("klines", "BTCUSDT", "1h")
    assert n == 24 * 62
    assert stored.index.max() < pd.Timestamp("2026-09-01", tz="UTC")


def test_the_current_month_comes_from_daily_files_only_when_asked(tmp_path):
    store = ParquetStore(tmp_path)
    with httpx.Client(transport=_archive_transport(_two_months())) as client:
        ingest_klines(
            client, store, "BTCUSDT", "1h",
            now=pd.Timestamp("2026-09-15", tz="UTC"), include_current_month=True,
        )
    assert store.read("klines", "BTCUSDT", "1h").index.max() == pd.Timestamp(
        "2026-09-01 23:00", tz="UTC"
    )


def test_a_second_run_downloads_nothing_new_and_duplicates_nothing(tmp_path):
    store = ParquetStore(tmp_path)
    now = pd.Timestamp("2026-09-15", tz="UTC")
    with httpx.Client(transport=_archive_transport(_two_months())) as client:
        ingest_klines(client, store, "BTCUSDT", "1h", now=now)
        again = ingest_klines(client, store, "BTCUSDT", "1h", now=now)
    stored = store.read("klines", "BTCUSDT", "1h")
    assert again == 0
    assert stored.index.is_unique and len(stored) == 24 * 62


def test_funding_ingest_resumes_after_the_last_stored_rate(tmp_path):
    def month(start: str) -> bytes:
        t = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "f.csv",
                "calc_time,funding_interval_hours,last_funding_rate\n"
                + "".join(f"{t + i * 28_800_000},8,0.0001\n" for i in range(90)),
            )
        return buffer.getvalue()

    base = "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-"
    files = {base + "2026-07.zip": month("2026-07-01"), base + "2026-08.zip": month("2026-08-01")}
    store = ParquetStore(tmp_path)
    with httpx.Client(transport=_archive_transport(files)) as client:
        assert ingest_funding(client, store, "BTCUSDT") == 180
        assert ingest_funding(client, store, "BTCUSDT") == 0


def test_a_month_not_yet_archived_is_filled_from_its_daily_files(tmp_path):
    """Run mid-September, then on 2 October before Binance has published the
    September archive: only October's dailies were taken, the rest of
    September became a hole, and resuming from the newest bar never went back."""
    daily = "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-"
    files = {
        "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-08.zip": _zip(
            "a.csv", _rows("2026-08-01", 24 * 31)
        ),
    }
    for day in pd.date_range("2026-09-01", "2026-10-01", freq="D"):
        files[daily + day.strftime("%Y-%m-%d") + ".zip"] = _zip(
            "d.csv", _rows(day.strftime("%Y-%m-%d"), 24)
        )
    store = ParquetStore(tmp_path)
    with httpx.Client(transport=_archive_transport(files)) as client:
        ingest_klines(
            client, store, "BTCUSDT", "1h",
            now=pd.Timestamp("2026-10-02", tz="UTC"), include_current_month=True,
        )
    stored = store.read("klines", "BTCUSDT", "1h")
    expected = pd.date_range("2026-08-01", "2026-10-01 23:00", freq="1h", tz="UTC")
    assert len(stored) == len(expected) and (stored.index == expected).all()
