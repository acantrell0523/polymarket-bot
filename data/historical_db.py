"""
Historical market & snapshot tables for the backtest ingestion pipeline.

Two tables are added to the existing data/trades.db:
  historical_markets   — one row per Polymarket market (game or outright)
  historical_snapshots — price time-series for each market

All public functions accept an optional db_path so unit tests can use
a temp file without touching the production database.
"""
import os
import sqlite3
from typing import Any, Dict, List, Optional

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "trades.db"
)


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database."""
    path = db_path or _DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_tables(db_path: Optional[str] = None) -> None:
    """Create historical_markets and historical_snapshots if they don't exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_markets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            slug            TEXT    NOT NULL,
            market_id       TEXT    DEFAULT '',
            condition_id    TEXT    DEFAULT '',
            league          TEXT    DEFAULT 'NBA',
            sport           TEXT    DEFAULT 'basketball',
            question        TEXT    DEFAULT '',
            home_team       TEXT    DEFAULT '',
            away_team       TEXT    DEFAULT '',
            home_abbr       TEXT    DEFAULT '',
            away_abbr       TEXT    DEFAULT '',
            game_start_time TEXT    DEFAULT '',
            espn_game_id    TEXT    DEFAULT '',
            home_score      TEXT    DEFAULT '',
            away_score      TEXT    DEFAULT '',
            settled_outcome TEXT    DEFAULT '',
            market_type     TEXT    DEFAULT '',
            token_id_0      TEXT    DEFAULT '',
            token_id_1      TEXT    DEFAULT '',
            ingest_time     TEXT    DEFAULT (datetime('now')),
            UNIQUE(slug)
        );

        CREATE TABLE IF NOT EXISTS historical_snapshots (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            slug                 TEXT    NOT NULL,
            timestamp            INTEGER NOT NULL,
            polymarket_price     REAL    DEFAULT 0,
            espn_consensus_prob  REAL    DEFAULT 0,
            num_books            INTEGER DEFAULT 0,
            source               TEXT    DEFAULT 'clob_prices_history',
            trade_size_usd       REAL    DEFAULT 0,
            UNIQUE(slug, timestamp, source)
        );

        CREATE INDEX IF NOT EXISTS idx_hm_slug
            ON historical_markets(slug);

        CREATE INDEX IF NOT EXISTS idx_hs_slug
            ON historical_snapshots(slug);

        CREATE INDEX IF NOT EXISTS idx_hs_timestamp
            ON historical_snapshots(timestamp);

        CREATE INDEX IF NOT EXISTS idx_hs_slug_ts
            ON historical_snapshots(slug, timestamp);
    """)
    conn.commit()
    conn.close()


def upsert_historical_market(
    conn: sqlite3.Connection, m: Dict[str, Any]
) -> None:
    """INSERT OR IGNORE one market row (idempotent by slug)."""
    conn.execute(
        """
        INSERT OR IGNORE INTO historical_markets
            (slug, market_id, condition_id, league, sport, question,
             home_team, away_team, home_abbr, away_abbr,
             game_start_time, espn_game_id, home_score, away_score,
             settled_outcome, market_type, token_id_0, token_id_1)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            m.get("slug", ""),
            m.get("market_id", ""),
            m.get("condition_id", ""),
            m.get("league", "NBA"),
            m.get("sport", "basketball"),
            m.get("question", ""),
            m.get("home_team", ""),
            m.get("away_team", ""),
            m.get("home_abbr", ""),
            m.get("away_abbr", ""),
            m.get("game_start_time", ""),
            m.get("espn_game_id", ""),
            m.get("home_score", ""),
            m.get("away_score", ""),
            m.get("settled_outcome", ""),
            m.get("market_type", ""),
            m.get("token_id_0", ""),
            m.get("token_id_1", ""),
        ),
    )


def upsert_snapshots(
    conn: sqlite3.Connection,
    slug: str,
    history: List[Dict[str, Any]],
    source: str = "clob_prices_history",
) -> int:
    """Batch INSERT OR IGNORE snapshot rows. Returns count actually inserted.

    Timestamps are bucketed to UTC day boundaries (floor to 86400 s) so that
    the UNIQUE(slug, timestamp, source) constraint catches duplicates even when
    the CLOB API returns slightly different intra-day timestamps across runs.
    """
    inserted = 0
    for point in history:
        raw_ts = int(point["t"])
        bucketed_ts = (raw_ts // 86400) * 86400          # floor to day boundary
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO historical_snapshots
                (slug, timestamp, polymarket_price, source)
            VALUES (?,?,?,?)
            """,
            (slug, bucketed_ts, float(point["p"]), source),
        )
        inserted += cur.rowcount                          # 1 if inserted, 0 if ignored
    return inserted


# ── Read helpers (used by inspect_historical.py and tests) ────────────────────

def get_all_markets(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM historical_markets ORDER BY slug"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_snapshot_counts(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return (slug, count) sorted by count DESC."""
    conn = get_conn(db_path)
    rows = conn.execute(
        """
        SELECT slug, COUNT(*) as cnt
        FROM historical_snapshots
        GROUP BY slug
        ORDER BY cnt DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_settled_outcome_distribution(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return (settled_outcome, count) pairs."""
    conn = get_conn(db_path)
    rows = conn.execute(
        """
        SELECT settled_outcome, COUNT(*) as cnt
        FROM historical_markets
        GROUP BY settled_outcome
        ORDER BY cnt DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sample_snapshots(
    slug: str, limit: int = 5, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute(
        """
        SELECT * FROM historical_snapshots
        WHERE slug = ?
        ORDER BY timestamp
        LIMIT ?
        """,
        (slug, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_games_by_date(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Group game markets by date extracted from game_start_time."""
    conn = get_conn(db_path)
    rows = conn.execute(
        """
        SELECT
            substr(game_start_time, 1, 10) as game_date,
            COUNT(*) as market_count
        FROM historical_markets
        WHERE market_type = 'moneyline_game'
          AND game_start_time != ''
        GROUP BY game_date
        ORDER BY game_date
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
