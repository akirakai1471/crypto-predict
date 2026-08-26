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

            def flush(batch: pd.DataFrame, _symbol=symbol, _interval=interval) -> None:
                """Persist each batch as it arrives so an interrupted run resumes
                from where it stopped instead of starting over."""
                store.append("klines", _symbol, _interval, batch)
                typer.echo(
                    f"  +{len(batch)} bars up to {batch.index.max()}", err=False
                )

            written = backfill_klines(
                client, symbol, interval, start=start, end=now, now=now, sink=flush
            )
            if not written:
                typer.echo(f"  no new bars for {symbol} {interval}")
                continue

            total += written
            typer.echo(f"  wrote {written} bars")

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
        start = (
            last + pd.Timedelta(hours=1)
            if last is not None
            else pd.Timestamp(cfg.data.start, tz="UTC")
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
