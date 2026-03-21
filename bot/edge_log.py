"""Edge analysis and logging system.

Provides:
1. Edge log table — records full signal snapshot for every trade
2. Edge validation report — daily analysis by edge bucket, league, time, signal
3. Edge pattern classification — categorizes trades into exploitable patterns
4. Daily edge scan / morning briefing — top opportunities at 8 AM ET
"""

import json
import sqlite3
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from bot import trade_db

DB_PATH = trade_db.DB_PATH


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_edge_tables():
    """Create edge_log and line_movement tables."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS edge_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            market_type TEXT DEFAULT '',
            league TEXT DEFAULT '',
            polymarket_price REAL NOT NULL,
            consensus_price REAL DEFAULT 0,
            books_used TEXT DEFAULT '',
            num_books INTEGER DEFAULT 0,
            edge_at_entry REAL NOT NULL,
            signal_snapshot TEXT DEFAULT '{}',
            edge_pattern TEXT DEFAULT '',
            final_outcome TEXT DEFAULT '',
            actual_pnl REAL DEFAULT 0,
            time_held_seconds REAL DEFAULT 0,
            price_at_close REAL DEFAULT 0,
            close_reason TEXT DEFAULT '',
            is_live_game INTEGER DEFAULT 0,
            entry_time TEXT DEFAULT '',
            close_time TEXT DEFAULT '',
            closing_line_value REAL DEFAULT 0,
            closing_line_price REAL DEFAULT 0,
            resolution_flag TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS line_movement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            consensus_prob REAL NOT NULL,
            polymarket_price REAL NOT NULL,
            num_books INTEGER DEFAULT 0,
            sharp_consensus REAL DEFAULT 0,
            overall_consensus REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_edge_log_slug ON edge_log(slug);
        CREATE INDEX IF NOT EXISTS idx_edge_log_timestamp ON edge_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_edge_log_league ON edge_log(league);
        CREATE INDEX IF NOT EXISTS idx_line_movement_slug ON line_movement(slug);
        CREATE INDEX IF NOT EXISTS idx_line_movement_timestamp ON line_movement(timestamp);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Edge log CRUD
# ---------------------------------------------------------------------------

def insert_edge_log(
    slug: str,
    polymarket_price: float,
    consensus_price: float,
    books_used: str,
    num_books: int,
    edge_at_entry: float,
    signal_snapshot: Dict,
    edge_pattern: str,
    is_live_game: bool,
    league: str = "",
    market_type: str = "",
    resolution_flag: str = "",
) -> int:
    """Record edge data when a trade is opened. Returns the edge_log ID."""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO edge_log
           (slug, timestamp, market_type, league, polymarket_price, consensus_price,
            books_used, num_books, edge_at_entry, signal_snapshot, edge_pattern,
            is_live_game, entry_time, resolution_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (slug, datetime.now(timezone.utc).isoformat(), market_type, league,
         polymarket_price, consensus_price, books_used, num_books,
         edge_at_entry, json.dumps(signal_snapshot, default=str),
         edge_pattern, int(is_live_game),
         datetime.now(timezone.utc).isoformat(), resolution_flag),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_edge_log_outcome(
    slug: str,
    entry_time_iso: str,
    actual_pnl: float,
    time_held_seconds: float,
    price_at_close: float,
    close_reason: str,
    final_outcome: str,
):
    """Update edge log entry with trade outcome when position closes."""
    conn = _get_conn()
    conn.execute(
        """UPDATE edge_log
           SET final_outcome = ?, actual_pnl = ?, time_held_seconds = ?,
               price_at_close = ?, close_reason = ?,
               close_time = ?
           WHERE slug = ? AND entry_time = ?""",
        (final_outcome, actual_pnl, time_held_seconds, price_at_close,
         close_reason, datetime.now(timezone.utc).isoformat(),
         slug, entry_time_iso),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Closing Line Value (CLV) tracking
# ---------------------------------------------------------------------------

def record_closing_line(slug: str, closing_consensus: float, polymarket_closing: float):
    """Record the closing line (consensus at tip-off/kickoff).

    CLV = entry_price vs closing_line. Positive CLV = we got value.
    Called when a game starts (live scan detects game start).
    """
    conn = _get_conn()
    # Get the most recent edge_log entry for this slug that doesn't have a closing line yet
    row = conn.execute(
        """SELECT id, polymarket_price FROM edge_log
           WHERE slug = ? AND closing_line_price = 0
           ORDER BY timestamp DESC LIMIT 1""",
        (slug,),
    ).fetchone()
    if row:
        entry_price = row["polymarket_price"]
        clv = closing_consensus - entry_price  # positive = we bought cheaper than close
        conn.execute(
            """UPDATE edge_log SET closing_line_value = ?, closing_line_price = ?
               WHERE id = ?""",
            (clv, closing_consensus, row["id"]),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Resolution edge check
# ---------------------------------------------------------------------------

RESOLUTION_FLAG_KEYWORDS = {"over/under", "spread", "total", "handicap", "margin"}


def check_resolution_flag(slug: str, question: str) -> str:
    """Flag markets with potentially ambiguous resolution criteria.

    Returns a flag string if the market needs extra scrutiny, empty string otherwise.
    """
    q_lower = question.lower()
    slug_lower = slug.lower()

    flags = []
    if any(kw in q_lower or kw in slug_lower for kw in RESOLUTION_FLAG_KEYWORDS):
        flags.append("ambiguous_resolution_criteria")

    # Spread markets: slug starts with asc- or contains "spread"
    if slug_lower.startswith("asc-") or "spread" in slug_lower:
        flags.append("spread_market")

    # Totals markets: slug starts with tsc- or contains "pt5"
    if slug_lower.startswith("tsc-") or "pt5" in slug_lower:
        flags.append("totals_market")

    return ",".join(flags)


def log_resolution_dispute(slug: str, reason: str):
    """Log a disputed resolution for later review."""
    conn = _get_conn()
    conn.execute(
        """UPDATE edge_log SET resolution_flag = ?
           WHERE slug = ? AND resolution_flag = ''
           ORDER BY timestamp DESC LIMIT 1""",
        (reason, slug),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Correlated market detection
# ---------------------------------------------------------------------------

def extract_game_id(slug: str) -> str:
    """Extract the game identifier from a slug for correlation detection.

    Examples:
        aec-nba-atl-hou-2026-03-20 → nba-atl-hou-2026-03-20
        asc-nba-atl-hou-pos-5pt5-2026-03-20 → nba-atl-hou-2026-03-20
        tsc-nba-atl-hou-pt5-220-2026-03-20 → nba-atl-hou-2026-03-20

    Different market types (moneyline, spread, totals) on the same game
    share the same game_id.
    """
    parts = slug.split("-")
    if len(parts) < 5:
        return slug

    # Sport is parts[1], teams are parts[2] and parts[3]
    sport = parts[1]
    team1 = parts[2]
    team2 = parts[3]

    # Find the date portion (YYYY-MM-DD) in the remaining parts
    date_parts = []
    for i, p in enumerate(parts):
        if len(p) == 4 and p.isdigit() and i >= 4:
            # Found year, grab year-month-day
            if i + 2 < len(parts):
                date_parts = parts[i:i+3]
            break

    if date_parts:
        return f"{sport}-{team1}-{team2}-{'-'.join(date_parts)}"
    return f"{sport}-{team1}-{team2}"


def get_open_game_ids(open_positions) -> dict:
    """Get a mapping of game_id → list of position slugs for open positions.

    Used to check if we already have a position on a game before opening another.
    """
    game_map = {}
    for pos in open_positions:
        slug = getattr(pos, 'slug', '') or getattr(pos, 'market_id', '')
        game_id = extract_game_id(slug)
        game_map.setdefault(game_id, []).append(slug)
    return game_map


# ---------------------------------------------------------------------------
# Line movement tracking
# ---------------------------------------------------------------------------

def record_line_movement(
    slug: str,
    consensus_prob: float,
    polymarket_price: float,
    num_books: int = 0,
    sharp_consensus: float = 0.0,
    overall_consensus: float = 0.0,
):
    """Record a line movement data point."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO line_movement
           (slug, timestamp, consensus_prob, polymarket_price, num_books,
            sharp_consensus, overall_consensus)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (slug, datetime.now(timezone.utc).isoformat(), consensus_prob,
         polymarket_price, num_books, sharp_consensus, overall_consensus),
    )
    conn.commit()
    conn.close()


def get_line_movement(slug: str, hours: int = 6) -> List[Dict]:
    """Get line movement history for a slug over the last N hours."""
    conn = _get_conn()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        """SELECT * FROM line_movement
           WHERE slug = ? AND timestamp >= ?
           ORDER BY timestamp""",
        (slug, since),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_line_drift(slug: str, hours: int = 1) -> Optional[float]:
    """Compute how much the consensus line has moved in the last N hours.

    Returns the drift (positive = line moved up, negative = line moved down).
    None if insufficient data.
    """
    points = get_line_movement(slug, hours=hours)
    if len(points) < 2:
        return None
    first = points[0]["consensus_prob"]
    last = points[-1]["consensus_prob"]
    return last - first


# ---------------------------------------------------------------------------
# Edge pattern classification
# ---------------------------------------------------------------------------

EDGE_PATTERNS = {
    "late_line_movement": "Books moved but Polymarket hasn't caught up",
    "mean_reversion": "Polymarket overreacting to recent result",
    "liquidity_gap": "Low liquidity market with stale price",
    "weak_consensus": "Sharp disagreement between books",
    "sharp_divergence": "Sharp books disagree with public books",
}


def classify_edge_pattern(signal_snapshot: Dict) -> str:
    """Classify a trade into an edge pattern based on signal data.

    Signal snapshot should contain signal values and metadata from the
    trade's signals at entry time.
    """
    signals = signal_snapshot.get("signals", {})

    # Check for late line movement: consensus moved recently but poly hasn't
    line_drift_1h = signal_snapshot.get("line_drift_1h")
    if line_drift_1h is not None and abs(line_drift_1h) >= 0.05:
        return "late_line_movement"

    # Check for sharp divergence: sharp books differ from overall
    sharp_consensus = signal_snapshot.get("sharp_consensus", 0)
    overall_consensus = signal_snapshot.get("overall_consensus", 0)
    if sharp_consensus > 0 and overall_consensus > 0:
        sharp_diff = abs(sharp_consensus - overall_consensus)
        if sharp_diff >= 0.03:
            return "sharp_divergence"

    # Check for mean reversion: price far from 0.5 with edge pointing back
    polymarket_price = signal_snapshot.get("polymarket_price", 0.5)
    edge = signal_snapshot.get("edge", 0)
    if abs(polymarket_price - 0.5) > 0.25:
        # Price is extreme
        if (polymarket_price > 0.5 and edge < 0) or (polymarket_price < 0.5 and edge > 0):
            return "mean_reversion"

    # Check for weak consensus: high spread across books
    odds_meta = signals.get("odds_value", {}).get("metadata", {})
    book_spread = odds_meta.get("spread", 0)
    if book_spread >= 0.08:
        return "weak_consensus"

    # Check for liquidity gap: low liquidity + stale price
    ob_meta = signals.get("order_book_imbalance", {}).get("metadata", {})
    total_depth = ob_meta.get("bid_depth", 0) + ob_meta.get("ask_depth", 0)
    if total_depth < 200:
        return "liquidity_gap"

    return "unknown"


# ---------------------------------------------------------------------------
# Edge validation report
# ---------------------------------------------------------------------------

def get_edge_log_entries(since: Optional[datetime] = None) -> List[Dict]:
    """Get edge log entries, optionally filtered by time."""
    conn = _get_conn()
    if since:
        rows = conn.execute(
            "SELECT * FROM edge_log WHERE timestamp >= ? ORDER BY timestamp",
            (since.isoformat(),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM edge_log ORDER BY timestamp").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _bucket_edge(edge: float) -> str:
    """Classify edge into a bucket."""
    abs_edge = abs(edge) * 100
    if abs_edge < 3:
        return "<3%"
    elif abs_edge < 5:
        return "3-5%"
    elif abs_edge < 7:
        return "5-7%"
    elif abs_edge < 10:
        return "7-10%"
    else:
        return "10%+"


def _extract_league(slug: str) -> str:
    """Extract league from slug."""
    parts = slug.split("-")
    if len(parts) >= 2:
        return parts[1].upper()
    return "UNKNOWN"


def _is_pregame(entry: Dict) -> bool:
    """Determine if a trade was pre-game based on is_live_game field."""
    return not entry.get("is_live_game", False)


def generate_edge_validation_report(days: int = 1) -> Dict[str, Any]:
    """Generate a comprehensive edge validation report.

    Returns a dict with sections:
        by_edge_bucket: {bucket: {count, wins, win_rate, avg_pnl, total_pnl}}
        by_league: {league: {count, wins, win_rate, avg_pnl, total_pnl}}
        by_time: {pregame/live: {count, wins, win_rate, avg_pnl, total_pnl}}
        by_signal: {signal_name: {avg_when_win, avg_when_loss}}
        by_pattern: {pattern: {count, wins, win_rate, avg_pnl, total_pnl}}
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    entries = get_edge_log_entries(since)

    # Filter to entries with outcomes
    completed = [e for e in entries if e.get("final_outcome")]

    report = {
        "total_entries": len(entries),
        "completed_entries": len(completed),
        "by_edge_bucket": {},
        "by_league": {},
        "by_time": {},
        "by_signal": {},
        "by_pattern": {},
    }

    if not completed:
        return report

    # Helper to compute bucket stats
    def _bucket_stats(items: List[Dict]) -> Dict:
        if not items:
            return {"count": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
        wins = [i for i in items if i.get("actual_pnl", 0) > 0]
        total_pnl = sum(i.get("actual_pnl", 0) for i in items)
        return {
            "count": len(items),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(items), 3) if items else 0,
            "avg_pnl": round(total_pnl / len(items), 2),
            "total_pnl": round(total_pnl, 2),
        }

    # By edge bucket
    buckets: Dict[str, List[Dict]] = {}
    for e in completed:
        bucket = _bucket_edge(e.get("edge_at_entry", 0))
        buckets.setdefault(bucket, []).append(e)
    for bucket_name in ["3-5%", "5-7%", "7-10%", "10%+"]:
        report["by_edge_bucket"][bucket_name] = _bucket_stats(buckets.get(bucket_name, []))

    # By league
    leagues: Dict[str, List[Dict]] = {}
    for e in completed:
        league = e.get("league") or _extract_league(e.get("slug", ""))
        leagues.setdefault(league, []).append(e)
    for league, items in sorted(leagues.items(), key=lambda x: -len(x[1])):
        report["by_league"][league] = _bucket_stats(items)

    # By time (pregame vs live)
    pregame = [e for e in completed if _is_pregame(e)]
    live = [e for e in completed if not _is_pregame(e)]
    report["by_time"]["pregame"] = _bucket_stats(pregame)
    report["by_time"]["live"] = _bucket_stats(live)

    # By signal strength (which signal was strongest in wins vs losses)
    signal_wins: Dict[str, List[float]] = {}
    signal_losses: Dict[str, List[float]] = {}
    for e in completed:
        try:
            snapshot = json.loads(e.get("signal_snapshot", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        signals = snapshot.get("signals", {})
        is_win = e.get("actual_pnl", 0) > 0
        for sig_name, sig_data in signals.items():
            val = sig_data.get("value", 0.5) if isinstance(sig_data, dict) else 0.5
            if is_win:
                signal_wins.setdefault(sig_name, []).append(val)
            else:
                signal_losses.setdefault(sig_name, []).append(val)

    all_sig_names = set(list(signal_wins.keys()) + list(signal_losses.keys()))
    for sig_name in all_sig_names:
        w_vals = signal_wins.get(sig_name, [])
        l_vals = signal_losses.get(sig_name, [])
        report["by_signal"][sig_name] = {
            "avg_when_win": round(sum(w_vals) / len(w_vals), 4) if w_vals else 0,
            "avg_when_loss": round(sum(l_vals) / len(l_vals), 4) if l_vals else 0,
            "win_count": len(w_vals),
            "loss_count": len(l_vals),
        }

    # By pattern
    patterns: Dict[str, List[Dict]] = {}
    for e in completed:
        pattern = e.get("edge_pattern", "unknown")
        patterns.setdefault(pattern, []).append(e)
    for pattern, items in sorted(patterns.items(), key=lambda x: -len(x[1])):
        report["by_pattern"][pattern] = _bucket_stats(items)

    return report


def format_edge_report_slack(report: Dict) -> str:
    """Format the edge validation report for Slack."""
    lines = [":mag: *Edge Validation Report*\n"]

    # By edge bucket
    lines.append("*By Edge Size*")
    for bucket, stats in report.get("by_edge_bucket", {}).items():
        if stats["count"] > 0:
            lines.append(
                f"  `{bucket}`: {stats['count']} trades, "
                f"`{stats['win_rate']*100:.0f}%` WR, "
                f"avg `${stats['avg_pnl']:+.2f}`, "
                f"total `${stats['total_pnl']:+.2f}`"
            )

    # By league
    lines.append("\n*By League*")
    for league, stats in report.get("by_league", {}).items():
        if stats["count"] > 0:
            lines.append(
                f"  `{league}`: {stats['count']} trades, "
                f"`{stats['win_rate']*100:.0f}%` WR, "
                f"`${stats['total_pnl']:+.2f}`"
            )

    # By time
    lines.append("\n*Pre-game vs Live*")
    for period, stats in report.get("by_time", {}).items():
        if stats["count"] > 0:
            lines.append(
                f"  `{period}`: {stats['count']} trades, "
                f"`{stats['win_rate']*100:.0f}%` WR, "
                f"`${stats['total_pnl']:+.2f}`"
            )

    # By signal
    lines.append("\n*Signal Strength (Win vs Loss)*")
    for sig, stats in report.get("by_signal", {}).items():
        if stats["win_count"] + stats["loss_count"] > 0:
            lines.append(
                f"  `{sig}`: win avg `{stats['avg_when_win']:.3f}` "
                f"vs loss avg `{stats['avg_when_loss']:.3f}` "
                f"({stats['win_count']}W / {stats['loss_count']}L)"
            )

    # By pattern
    lines.append("\n*By Edge Pattern*")
    for pattern, stats in report.get("by_pattern", {}).items():
        if stats["count"] > 0:
            label = EDGE_PATTERNS.get(pattern, pattern)
            lines.append(
                f"  `{pattern}`: {stats['count']} trades, "
                f"`{stats['win_rate']*100:.0f}%` WR, "
                f"`${stats['total_pnl']:+.2f}` — _{label}_"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Morning briefing / daily edge scan
# ---------------------------------------------------------------------------

def build_edge_snapshot_for_signal(signal, odds_cache=None, slug: str = "") -> Dict:
    """Build a signal snapshot dict from a TradeSignal's signals list.

    Used both for edge logging and for the morning briefing.
    """
    snapshot = {
        "signals": {},
        "polymarket_price": signal.market_price if hasattr(signal, "market_price") else 0,
        "edge": signal.edge if hasattr(signal, "edge") else 0,
        "sharp_consensus": 0,
        "overall_consensus": 0,
        "line_drift_1h": None,
    }

    for s in (signal.signals if hasattr(signal, "signals") else []):
        snapshot["signals"][s.name] = {
            "value": s.value,
            "confidence": s.confidence,
            "direction": s.direction,
            "metadata": s.metadata or {},
        }

    # Extract consensus data from odds_value signal
    odds_data = snapshot["signals"].get("odds_value", {}).get("metadata", {})
    snapshot["overall_consensus"] = odds_data.get("consensus_prob", 0)

    # Line drift
    if slug:
        drift = compute_line_drift(slug, hours=1)
        snapshot["line_drift_1h"] = drift

    return snapshot


def run_morning_scan(market_data_client, estimator, odds_cache, config) -> List[Dict]:
    """Scan all markets and return top edge opportunities.

    Called at 8 AM ET by the supervisor for the morning briefing.
    Returns a list of dicts sorted by absolute edge.
    """
    from bot.signals.estimator import detect_market_type

    opportunities = []

    try:
        markets = market_data_client.get_active_markets()
    except Exception:
        return []

    for market in markets:
        try:
            snapshot = market_data_client.build_snapshot(market)
            if not snapshot:
                continue

            market_type = detect_market_type(snapshot)
            if market_type != "sports":
                continue

            # Get consensus price
            consensus_data = None
            if odds_cache and odds_cache.enabled:
                consensus_data = odds_cache.get_consensus_odds(snapshot.slug)

            if not consensus_data:
                continue

            # Get probability for this outcome
            prob_result = odds_cache.get_probability_for_slug(snapshot.slug)
            if not prob_result:
                continue

            consensus_prob, num_books = prob_result
            edge = consensus_prob - snapshot.price

            if abs(edge) < 0.02:
                continue

            # Extract league and game time
            parts = snapshot.slug.split("-")
            league = parts[1].upper() if len(parts) >= 2 else "?"
            game_time = market.get("gameStartTime", "")

            opportunities.append({
                "slug": snapshot.slug,
                "question": snapshot.question,
                "polymarket_price": snapshot.price,
                "consensus_prob": consensus_prob,
                "edge": edge,
                "abs_edge": abs(edge),
                "num_books": num_books,
                "league": league,
                "game_time": game_time,
                "home_team": consensus_data.get("home_team", ""),
                "away_team": consensus_data.get("away_team", ""),
            })
        except Exception:
            continue

    # Sort by absolute edge, descending
    opportunities.sort(key=lambda x: x["abs_edge"], reverse=True)
    return opportunities[:20]


def format_morning_briefing(opportunities: List[Dict]) -> str:
    """Format the morning briefing for Slack."""
    if not opportunities:
        return ":sunrise: *Morning Edge Briefing*\nNo significant edges found across NBA/NCAA markets today."

    lines = [
        ":sunrise: *Morning Edge Briefing*",
        f"Top {min(len(opportunities), 10)} edge opportunities for today:\n",
    ]

    for i, opp in enumerate(opportunities[:10], 1):
        edge_pct = opp["edge"] * 100
        direction = ":arrow_up:" if opp["edge"] > 0 else ":arrow_down:"

        game_time_str = ""
        if opp.get("game_time"):
            try:
                gt = datetime.fromisoformat(opp["game_time"].replace("Z", "+00:00"))
                game_time_str = f" | {gt.strftime('%I:%M %p ET')}"
            except (ValueError, TypeError):
                game_time_str = ""

        lines.append(
            f"{i}. {direction} `{opp['league']}` *{opp.get('question', opp['slug'])}*\n"
            f"   Poly: `{opp['polymarket_price']:.3f}` → Books: `{opp['consensus_prob']:.3f}` "
            f"| Edge: `{edge_pct:+.1f}%` | Books: `{opp['num_books']}`{game_time_str}"
        )

    return "\n".join(lines)


# Initialize tables on import
init_edge_tables()
