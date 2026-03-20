"""SQLite trade history database for the supervisor."""

import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            market_id TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            close_price REAL NOT NULL,
            quantity REAL NOT NULL,
            size_usd REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            close_reason TEXT NOT NULL,
            market_type TEXT DEFAULT '',
            entry_time TEXT NOT NULL,
            close_time TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            total_trades INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            total_pnl REAL NOT NULL,
            win_rate REAL NOT NULL,
            avg_win REAL DEFAULT 0,
            avg_loss REAL DEFAULT 0,
            profit_factor REAL DEFAULT 0,
            bankroll REAL DEFAULT 0,
            moneyline_trades INTEGER DEFAULT 0,
            moneyline_win_rate REAL DEFAULT 0,
            spread_trades INTEGER DEFAULT 0,
            spread_win_rate REAL DEFAULT 0,
            totals_trades INTEGER DEFAULT 0,
            totals_win_rate REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS parameter_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            parameter TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time);
        CREATE INDEX IF NOT EXISTS idx_trades_slug ON trades(slug);
    """)
    conn.commit()
    conn.close()


def insert_trade(
    slug: str, market_id: str, side: str, entry_price: float,
    close_price: float, quantity: float, size_usd: float,
    realized_pnl: float, close_reason: str, market_type: str,
    entry_time: datetime, close_time: datetime,
):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO trades
           (slug, market_id, side, entry_price, close_price, quantity, size_usd,
            realized_pnl, close_reason, market_type, entry_time, close_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (slug, market_id, side, entry_price, close_price, quantity, size_usd,
         realized_pnl, close_reason, market_type, entry_time.isoformat(),
         close_time.isoformat()),
    )
    conn.commit()
    conn.close()


def get_trades_since(since: datetime) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE close_time >= ? ORDER BY close_time",
        (since.isoformat(),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_trades(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY close_time DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_daily_summary(summary: Dict[str, Any]):
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_summaries
           (date, total_trades, wins, losses, total_pnl, win_rate,
            avg_win, avg_loss, profit_factor, bankroll,
            moneyline_trades, moneyline_win_rate,
            spread_trades, spread_win_rate,
            totals_trades, totals_win_rate)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (summary["date"], summary["total_trades"], summary["wins"],
         summary["losses"], summary["total_pnl"], summary["win_rate"],
         summary.get("avg_win", 0), summary.get("avg_loss", 0),
         summary.get("profit_factor", 0), summary.get("bankroll", 0),
         summary.get("moneyline_trades", 0), summary.get("moneyline_win_rate", 0),
         summary.get("spread_trades", 0), summary.get("spread_win_rate", 0),
         summary.get("totals_trades", 0), summary.get("totals_win_rate", 0)),
    )
    conn.commit()
    conn.close()


def get_daily_summaries(days: int = 7) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_summaries ORDER BY date DESC LIMIT ?", (days,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_parameter_change(parameter: str, old_value: str, new_value: str, reason: str):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO parameter_changes (timestamp, parameter, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), parameter, old_value, new_value, reason),
    )
    conn.commit()
    conn.close()


def get_recent_parameter_changes(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM parameter_changes ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
