"""SQLite store for live predictions and paper trades.

Every prediction the system makes is written here before its outcome is known,
and scored later once enough bars have closed. That ordering is the whole point:
a prediction log written after the fact can be edited by hindsight, deliberately
or not. This one cannot.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    interval        TEXT    NOT NULL,
    bar_close_time  TEXT    NOT NULL,
    prob_down       REAL    NOT NULL,
    prob_flat       REAL    NOT NULL,
    prob_up         REAL    NOT NULL,
    signal          INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    close_price     REAL    NOT NULL,
    model_version   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    actual_return   REAL,
    actual_label    INTEGER,
    is_correct      INTEGER,
    scored_at       TEXT,
    UNIQUE (symbol, interval, bar_close_time, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions (symbol, interval, bar_close_time);

CREATE TABLE IF NOT EXISTS paper_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    interval        TEXT    NOT NULL,
    direction       INTEGER NOT NULL,
    entry_time      TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_time       TEXT,
    exit_price      REAL,
    size_usd        REAL    NOT NULL,
    gross_return    REAL,
    cost            REAL,
    net_return      REAL,
    pnl_usd         REAL,
    status          TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    UNIQUE (symbol, interval, entry_time)
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON paper_trades (status);
"""


class PredictionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        # A fresh connection per operation: SQLite connections are not safe to
        # share across the scheduler thread and the request handlers.
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_prediction(
        self,
        symbol: str,
        interval: str,
        bar_close_time: pd.Timestamp,
        proba: tuple[float, float, float],
        signal: int,
        close_price: float,
        model_version: str,
    ) -> bool:
        """Insert a prediction. Returns False if this bar was already recorded."""
        prob_down, prob_flat, prob_up = proba
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO predictions
                    (symbol, interval, bar_close_time, prob_down, prob_flat, prob_up,
                     signal, confidence, close_price, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    interval,
                    bar_close_time.isoformat(),
                    float(prob_down),
                    float(prob_flat),
                    float(prob_up),
                    int(signal),
                    float(max(proba)),
                    float(close_price),
                    model_version,
                    pd.Timestamp.now(tz="UTC").isoformat(),
                ),
            )
            return cursor.rowcount > 0

    def unscored(self, symbol: str, interval: str) -> pd.DataFrame:
        """Predictions whose outcome has not been filled in yet."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions WHERE symbol = ? AND interval = ? "
                "AND actual_return IS NULL ORDER BY bar_close_time",
                conn,
                params=(symbol, interval),
            )

    def score_prediction(
        self, prediction_id: int, actual_return: float, actual_label: int, is_correct: bool
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE predictions SET actual_return = ?, actual_label = ?, "
                "is_correct = ?, scored_at = ? WHERE id = ?",
                (
                    float(actual_return),
                    int(actual_label),
                    int(is_correct),
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    int(prediction_id),
                ),
            )

    def history(
        self, symbol: str, interval: str, limit: int = 500, scored_only: bool = False
    ) -> pd.DataFrame:
        clause = " AND actual_return IS NOT NULL" if scored_only else ""
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM predictions WHERE symbol = ? AND interval = ?{clause} "
                "ORDER BY bar_close_time DESC LIMIT ?",
                conn,
                params=(symbol, interval, limit),
            )

    def latest_prediction(self, symbol: str, interval: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE symbol = ? AND interval = ? "
                "ORDER BY bar_close_time DESC LIMIT 1",
                (symbol, interval),
            ).fetchone()
            return dict(row) if row else None

    def accuracy_summary(self, symbol: str, interval: str) -> dict[str, Any]:
        """Live accuracy, computed only from predictions that have been scored.

        Deliberately returns `n` alongside every rate: an accuracy figure without
        its sample size invites the reader to trust six observations.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n,
                       SUM(is_correct) AS correct,
                       SUM(CASE WHEN signal != 0 THEN 1 ELSE 0 END) AS n_signals,
                       SUM(CASE WHEN signal != 0 AND is_correct = 1 THEN 1 ELSE 0 END)
                           AS correct_signals
                FROM predictions
                WHERE symbol = ? AND interval = ? AND actual_return IS NOT NULL
                """,
                (symbol, interval),
            ).fetchone()

        n = row["n"] or 0
        n_signals = row["n_signals"] or 0
        return {
            "n_scored": n,
            "accuracy": (row["correct"] / n) if n else None,
            "n_signals": n_signals,
            "signal_accuracy": (row["correct_signals"] / n_signals) if n_signals else None,
        }

    # --- paper trading ---------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        interval: str,
        direction: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        size_usd: float,
        model_version: str,
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO paper_trades
                    (symbol, interval, direction, entry_time, entry_price, size_usd,
                     status, model_version)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    symbol,
                    interval,
                    int(direction),
                    entry_time.isoformat(),
                    float(entry_price),
                    float(size_usd),
                    model_version,
                ),
            )
            return cursor.rowcount > 0

    def close_trade(
        self,
        trade_id: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        gross_return: float,
        cost: float,
        net_return: float,
        pnl_usd: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE paper_trades SET exit_time = ?, exit_price = ?, gross_return = ?, "
                "cost = ?, net_return = ?, pnl_usd = ?, status = 'closed' WHERE id = ?",
                (
                    exit_time.isoformat(),
                    float(exit_price),
                    float(gross_return),
                    float(cost),
                    float(net_return),
                    float(pnl_usd),
                    int(trade_id),
                ),
            )

    def open_trades(self, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM paper_trades WHERE status = 'open'"
        params: tuple = ()
        if symbol:
            query += " AND symbol = ?"
            params = (symbol,)
        with self._connect() as conn:
            return pd.read_sql_query(query + " ORDER BY entry_time", conn, params=params)

    def closed_trades(self, symbol: str | None = None, limit: int = 500) -> pd.DataFrame:
        query = "SELECT * FROM paper_trades WHERE status = 'closed'"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=tuple(params))
