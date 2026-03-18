"""
Async SQLite database layer using aiosqlite.
Manages trade history and daily performance statistics.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = "data/trades.db"


async def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create database tables if they do not already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                direction       TEXT    NOT NULL,
                entry_price     REAL    NOT NULL,
                exit_price      REAL,
                sl_price        REAL,
                tp1_price       REAL,
                tp2_price       REAL,
                tp3_price       REAL,
                lot_size        REAL    NOT NULL DEFAULT 0.01,
                open_time       TEXT    NOT NULL,
                close_time      TEXT,
                status          TEXT    NOT NULL DEFAULT 'OPEN',
                pnl             REAL    DEFAULT 0.0,
                pnl_pips        REAL    DEFAULT 0.0,
                close_reason    TEXT,
                signal_strength INTEGER DEFAULT 1,
                confirmations   INTEGER DEFAULT 0,
                metadata        TEXT    DEFAULT '{}'
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_stats (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT    NOT NULL UNIQUE,
                total_trades  INTEGER DEFAULT 0,
                wins          INTEGER DEFAULT 0,
                losses        INTEGER DEFAULT 0,
                win_rate      REAL    DEFAULT 0.0,
                total_pnl     REAL    DEFAULT 0.0,
                max_drawdown  REAL    DEFAULT 0.0
            )
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades (status)
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_open_time
            ON trades (open_time)
            """
        )
        await db.commit()
    logger.info("Database initialized at %s", db_path)


class TradeHistory:
    """Async interface to the trades and daily_stats tables."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute(
        self,
        sql: str,
        params: tuple = (),
        fetch: str = "none",
    ) -> Any:
        """
        Execute a SQL statement and optionally fetch results.

        Args:
            sql:    SQL string.
            params: Bind parameters.
            fetch:  'one', 'all', or 'none'.

        Returns:
            Row, list of rows, or lastrowid depending on `fetch`.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            if fetch == "one":
                return await cursor.fetchone()
            if fetch == "all":
                return await cursor.fetchall()
            await db.commit()
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Trade CRUD
    # ------------------------------------------------------------------

    async def save_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        tp3_price: float,
        lot_size: float,
        signal_strength: int = 1,
        confirmations: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a new trade record. Returns the new trade ID."""
        meta_json = json.dumps(metadata or {})
        trade_id = await self._execute(
            """
            INSERT INTO trades (
                symbol, direction, entry_price, sl_price,
                tp1_price, tp2_price, tp3_price, lot_size,
                open_time, status, signal_strength, confirmations, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
            """,
            (
                symbol,
                direction,
                entry_price,
                sl_price,
                tp1_price,
                tp2_price,
                tp3_price,
                lot_size,
                datetime.utcnow().isoformat(),
                signal_strength,
                confirmations,
                meta_json,
            ),
        )
        logger.info(
            "Trade saved: id=%s %s %s @ %.2f", trade_id, direction, symbol, entry_price
        )
        return trade_id

    async def update_trade(
        self,
        trade_id: int,
        **kwargs: Any,
    ) -> None:
        """Update arbitrary columns on a trade row."""
        if not kwargs:
            return
        allowed = {
            "sl_price", "tp1_price", "tp2_price", "tp3_price",
            "exit_price", "close_time", "status", "pnl", "pnl_pips",
            "close_reason", "lot_size", "metadata",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        if not filtered:
            logger.warning("update_trade: no valid columns supplied")
            return
        set_clause = ", ".join(f"{col} = ?" for col in filtered)
        values = list(filtered.values()) + [trade_id]
        await self._execute(
            f"UPDATE trades SET {set_clause} WHERE id = ?",
            tuple(values),
        )

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        close_reason: str,
    ) -> None:
        """Mark a trade as CLOSED and record its result."""
        await self._execute(
            """
            UPDATE trades
            SET exit_price = ?, close_time = ?, status = 'CLOSED',
                pnl = ?, pnl_pips = ?, close_reason = ?
            WHERE id = ?
            """,
            (
                exit_price,
                datetime.utcnow().isoformat(),
                pnl,
                pnl_pips,
                close_reason,
                trade_id,
            ),
        )
        logger.info(
            "Trade closed: id=%s reason=%s pnl=%.2f", trade_id, close_reason, pnl
        )

    async def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Return a single trade row as a dict, or None."""
        row = await self._execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,), fetch="one"
        )
        return dict(row) if row else None

    async def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return all currently OPEN trades."""
        rows = await self._execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY open_time",
            fetch="all",
        )
        return [dict(r) for r in rows] if rows else []

    async def get_closed_trades(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return paginated closed trades, newest first."""
        rows = await self._execute(
            """
            SELECT * FROM trades
            WHERE status = 'CLOSED'
            ORDER BY close_time DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
            fetch="all",
        )
        return [dict(r) for r in rows] if rows else []

    # ------------------------------------------------------------------
    # Daily stats
    # ------------------------------------------------------------------

    async def upsert_daily_stats(
        self,
        stat_date: Optional[date] = None,
    ) -> None:
        """
        Compute and upsert today's (or given date's) statistics from the
        closed trades table.
        """
        target = (stat_date or date.today()).isoformat()
        rows = await self._execute(
            """
            SELECT pnl FROM trades
            WHERE status = 'CLOSED'
              AND DATE(close_time) = ?
            """,
            (target,),
            fetch="all",
        )
        pnls = [r["pnl"] for r in rows] if rows else []
        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        total_pnl = sum(pnls)

        # Simple drawdown: max cumulative loss from peak during the day
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        await self._execute(
            """
            INSERT INTO daily_stats (date, total_trades, wins, losses, win_rate, total_pnl, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_trades = excluded.total_trades,
                wins         = excluded.wins,
                losses       = excluded.losses,
                win_rate     = excluded.win_rate,
                total_pnl    = excluded.total_pnl,
                max_drawdown = excluded.max_drawdown
            """,
            (target, total, wins, losses, win_rate, total_pnl, max_dd),
        )

    async def get_daily_stats(
        self,
        stat_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return stats row for a specific date (defaults to today)."""
        target = (stat_date or date.today()).isoformat()
        row = await self._execute(
            "SELECT * FROM daily_stats WHERE date = ?", (target,), fetch="one"
        )
        return dict(row) if row else None

    async def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Return a time-ordered list of {time, cumulative_pnl} points for
        all closed trades. Useful for plotting an equity curve.
        """
        rows = await self._execute(
            """
            SELECT close_time, pnl
            FROM trades
            WHERE status = 'CLOSED'
            ORDER BY close_time
            """,
            fetch="all",
        )
        if not rows:
            return []
        cumulative = 0.0
        curve: List[Dict[str, Any]] = []
        for row in rows:
            cumulative += row["pnl"]
            curve.append(
                {
                    "time": row["close_time"],
                    "cumulative_pnl": round(cumulative, 2),
                }
            )
        return curve

    async def get_all_stats_history(self) -> List[Dict[str, Any]]:
        """Return all daily_stats rows sorted by date."""
        rows = await self._execute(
            "SELECT * FROM daily_stats ORDER BY date", fetch="all"
        )
        return [dict(r) for r in rows] if rows else []

    async def count_open_trades(self) -> int:
        """Return count of currently open trades."""
        row = await self._execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE status = 'OPEN'",
            fetch="one",
        )
        return row["cnt"] if row else 0

    async def get_today_pnl(self) -> float:
        """Return sum of PnL for trades closed today."""
        today = date.today().isoformat()
        row = await self._execute(
            """
            SELECT COALESCE(SUM(pnl), 0.0) AS total
            FROM trades
            WHERE status = 'CLOSED' AND DATE(close_time) = ?
            """,
            (today,),
            fetch="one",
        )
        return float(row["total"]) if row else 0.0
