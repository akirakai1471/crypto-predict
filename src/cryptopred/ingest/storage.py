"""Parquet-backed storage, partitioned by kind/symbol/interval/year.

Year partitioning keeps individual files small enough to rewrite cheaply when
appending, while still allowing a full multi-year read in one call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptopred.atomic import replace_atomically


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    # Through a temporary file: the current year's file is rewritten every
    # hour, and a kill mid-write used to leave it truncated for good.
    replace_atomically(path, lambda tmp: frame.to_parquet(tmp, engine="pyarrow", index=True))


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
            _write_parquet(chunk, directory / f"{year}.parquet")

    def append(self, kind: str, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Merge `df` into stored data, rewriting only the affected year files."""
        if df.empty:
            return
        directory = self._dir(kind, symbol, interval)
        directory.mkdir(parents=True, exist_ok=True)
        for year, chunk in df.groupby(df.index.year):
            path = directory / f"{year}.parquet"
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            _write_parquet(merge_frames(existing, chunk), path)

    def tail_summary(
        self, kind: str, symbol: str, interval: str
    ) -> tuple[int, pd.DataFrame]:
        """(total rows, newest year's rows) without reading every year.

        Row counts come from each file's parquet footer. The dashboard's health
        check asks this every minute, and reading a multi-year 1m store in full
        to report one timestamp cost seconds per symbol.
        """
        import pyarrow.parquet as pq

        directory = self._dir(kind, symbol, interval)
        files = sorted(directory.glob("*.parquet")) if directory.exists() else []
        if not files:
            return 0, pd.DataFrame()
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        return total, pd.read_parquet(files[-1])

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
