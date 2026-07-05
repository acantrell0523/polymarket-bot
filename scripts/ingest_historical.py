"""
Historical NBA data ingestion pipeline (Option D — CLOB prices-history).

Data flow
---------
Phase 1 (game-by-game):
  ESPN scoreboard  →  Polymarket slug lookup (Gamma)  →  CLOB prices-history
                   →  ESPN game summary (pickcenter)  →  consensus probability

Phase 2 (season-long outrights):
  Gamma /events for known NBA event slugs  →  CLOB prices-history
  (no consensus source exists for outrights — they stay gated in backtests)

Phase 3 (consensus backfill, also available standalone via --consensus-only):
  For every ingested moneyline_game market, fetch the ESPN game summary's
  `pickcenter` block (free, retained for past games), de-vig each provider's
  moneyline into a win probability, average across providers, figure out
  which team the market's YES token refers to, and write the result into
  historical_snapshots.espn_consensus_prob / num_books. This is the value
  backtest.historical_odds.HistoricalOddsCache.from_db() serves to the
  external validation gate.

  The consensus from pickcenter is effectively the CLOSING line, so by
  default it is only applied to snapshots within --consensus-window-days
  (default 3) before the game start (through one day after) — earlier
  snapshots keep espn_consensus_prob=0 rather than receiving a consensus
  that "knows" late-breaking news (lookahead).

Both price phases filter history to the [--start, --end] date window and
store results in data/trades.db:
  historical_markets   — one row per Polymarket market
  historical_snapshots — (slug, timestamp, price, consensus) time-series

Re-running is fully idempotent: INSERT OR IGNORE for inserts, plain UPDATE
for the consensus backfill. Per-market and per-game failures are logged and
skipped.
"""
import argparse
import datetime
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# Project path so we can import from data/
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.historical_db import (
    get_conn,
    get_consensus_coverage,
    init_tables,
    update_snapshot_consensus,
    upsert_historical_market,
    upsert_snapshots,
)
# Reuse the exact odds→probability conversions the live bot trades with, so
# historical consensus and live consensus are computed identically.
from bot.signals.odds_api import _american_to_prob, _spread_to_moneyline_prob

# ── Constants ─────────────────────────────────────────────────────────────────

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
)
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# Minimum seconds between outbound API calls (polite throttle)
THROTTLE_SECS = 0.2

# Gamma event slugs for season-long NBA outrights to sweep regardless of date
NBA_OUTRIGHT_EVENT_SLUGS = [
    "2026-nba-champion",
    "nba-playoffs-eastern-conference-champion",
    "nba-playoffs-western-conference-champion",
    "nba-western-conference-champion",
    "nba-eastern-conference-champion",
    "nba-champion-2024-2025",
    "nba-eastern-conference-champion",
    "nba-mvp-694",
    "nba-scoring-leader",
    "nba-assists-leader",
    "nba-rebounds-leader",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _throttle(last_call: float) -> float:
    """Sleep if needed to maintain THROTTLE_SECS between calls. Returns now."""
    elapsed = time.time() - last_call
    if elapsed < THROTTLE_SECS:
        time.sleep(THROTTLE_SECS - elapsed)
    return time.time()


def _get(session: requests.Session, url: str,
         params: Optional[Dict] = None, timeout: int = 15) -> Optional[Any]:
    """GET with error handling. Returns parsed JSON or None."""
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("http_get_failed url=%s error=%s", url, e)
        return None


# ── ESPN ──────────────────────────────────────────────────────────────────────

def fetch_espn_games(
    session: requests.Session, date_str: str, last_call: float
) -> Tuple[List[Dict], float]:
    """
    Fetch NBA games for a YYYYMMDD string.
    Returns (list[game_dict], last_call).
    """
    last_call = _throttle(last_call)
    data = _get(session, ESPN_SCOREBOARD, params={"dates": date_str})
    last_call = time.time()

    if not data:
        return [], last_call

    games = []
    for ev in data.get("events", []):
        try:
            comp  = ev.get("competitions", [{}])[0]
            teams = comp.get("competitors", [])
            home  = next((t for t in teams if t.get("homeAway") == "home"), {})
            away  = next((t for t in teams if t.get("homeAway") == "away"), {})

            games.append({
                "espn_id":    ev.get("id", ""),
                "date":       date_str,
                "start_time": ev.get("date", ""),
                "status":     ev.get("status", {}).get("type", {}).get("name", ""),
                "home_name":  home.get("team", {}).get("displayName", ""),
                "home_abbr":  home.get("team", {}).get("abbreviation", "").lower(),
                "home_score": str(home.get("score", "") or ""),
                "away_name":  away.get("team", {}).get("displayName", ""),
                "away_abbr":  away.get("team", {}).get("abbreviation", "").lower(),
                "away_score": str(away.get("score", "") or ""),
            })
        except Exception as exc:
            logging.warning("espn_parse_error event_id=%s error=%s", ev.get("id"), exc)

    return games, last_call


# ── Gamma (market discovery) ──────────────────────────────────────────────────

def _parse_token_ids(market: Dict) -> Tuple[str, str]:
    tok = market.get("clobTokenIds", "")
    if not tok:
        return "", ""
    try:
        tlist = json.loads(tok) if isinstance(tok, str) else tok
        t0 = tlist[0] if len(tlist) > 0 else ""
        t1 = tlist[1] if len(tlist) > 1 else ""
        return str(t0), str(t1)
    except Exception:
        return "", ""


def find_game_market(
    session: requests.Session, game: Dict, last_call: float
) -> Tuple[Optional[Dict], float]:
    """
    Try to find a Polymarket moneyline market for a given ESPN game.
    Constructs the expected aec-nba-{away}-{home}-{date} slug and checks Gamma.
    Returns (market_dict | None, last_call).
    """
    away = game["away_abbr"]
    home = game["home_abbr"]
    date_iso = f"{game['date'][:4]}-{game['date'][4:6]}-{game['date'][6:]}"

    # Normalise a few known ESPN ↔ Polymarket abbreviation differences
    abbr_map = {
        "sa":  "sas",   # San Antonio Spurs
        "gs":  "gsw",   # Golden State Warriors
        "ny":  "nyk",   # New York Knicks
        "no":  "nor",   # New Orleans Pelicans
        "uta": "uta",
    }
    home_pm = abbr_map.get(home, home)
    away_pm = abbr_map.get(away, away)

    candidates = [
        f"aec-nba-{away_pm}-{home_pm}-{date_iso}",
        f"aec-nba-{away}-{home}-{date_iso}",
    ]

    for slug in candidates:
        last_call = _throttle(last_call)
        data = _get(session, f"{GAMMA_BASE}/markets", params={"slug": slug})
        last_call = time.time()
        if data:
            items = data if isinstance(data, list) else data.get("data", [])
            if items:
                return items[0], last_call

    return None, last_call


def fetch_gamma_event_markets(
    session: requests.Session, event_slug: str, last_call: float
) -> Tuple[List[Dict], float]:
    """
    Fetch all markets under a Gamma event slug.
    Returns (list[market_dict], last_call).
    """
    last_call = _throttle(last_call)
    data = _get(session, f"{GAMMA_BASE}/events", params={"slug": event_slug, "limit": 1})
    last_call = time.time()

    if not data:
        return [], last_call

    items = data if isinstance(data, list) else data.get("data", [])
    if not items:
        return [], last_call

    return items[0].get("markets", []), last_call


# ── CLOB prices-history ───────────────────────────────────────────────────────

def fetch_prices_history(
    session: requests.Session,
    token_id: str,
    start_ts: int,
    end_ts: int,
    last_call: float,
) -> Tuple[List[Dict], float]:
    """
    Fetch CLOB prices-history for a token, filtered to [start_ts, end_ts].
    Uses interval=max with fidelity=1440 (daily candles) for broadest coverage.
    Returns (list[{t, p}], last_call).
    """
    last_call = _throttle(last_call)
    data = _get(
        session,
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": "max", "fidelity": 1440},
    )
    last_call = time.time()

    if not data:
        return [], last_call

    history = data.get("history", []) if isinstance(data, dict) else []
    filtered = [p for p in history if start_ts <= p.get("t", 0) <= end_ts]
    return filtered, last_call


# ── Consensus backfill (ESPN pickcenter) ─────────────────────────────────────

def fetch_espn_pickcenter(
    session: requests.Session, espn_game_id: str, last_call: float
) -> Tuple[List[Dict], float]:
    """Fetch the pickcenter (sportsbook odds) block from an ESPN game summary.

    Works for past games — ESPN retains the odds in the summary endpoint.
    Returns (list[provider_odds_dict], last_call).
    """
    last_call = _throttle(last_call)
    data = _get(session, ESPN_SUMMARY, params={"event": espn_game_id})
    last_call = time.time()

    if not data or not isinstance(data, dict):
        return [], last_call
    pickcenter = data.get("pickcenter", [])
    return pickcenter if isinstance(pickcenter, list) else [], last_call


def compute_consensus_from_pickcenter(
    pickcenter: List[Dict],
) -> Optional[Tuple[float, float, int]]:
    """Turn ESPN pickcenter provider odds into (p_home, p_away, num_books).

    Per provider, in preference order:
      1. Moneylines for both teams → implied probabilities, de-vigged by
         normalizing so p_home + p_away = 1 (raw implied probs sum to >1 by
         the bookmaker's margin; the overround carries no information).
      2. Point spread fallback → win probability via the same calibrated
         logistic model the live bot uses. ESPN's `spread` is the HOME
         team's spread (negative = home favored) while the converter expects
         positive = favored, hence the sign flip.

    The consensus is the plain mean across providers. Returns None when no
    provider yields a usable probability.
    """
    home_probs: List[float] = []
    for entry in pickcenter:
        if not isinstance(entry, dict):
            continue
        home_odds = entry.get("homeTeamOdds") or {}
        away_odds = entry.get("awayTeamOdds") or {}

        h_ml = home_odds.get("moneyLine")
        a_ml = away_odds.get("moneyLine")
        if h_ml is not None and a_ml is not None:
            try:
                ph_raw = _american_to_prob(int(h_ml))
                pa_raw = _american_to_prob(int(a_ml))
                total = ph_raw + pa_raw
                if total > 0:
                    home_probs.append(ph_raw / total)
                    continue
            except (ValueError, TypeError):
                pass

        spread = entry.get("spread")
        if spread is not None:
            try:
                home_probs.append(_spread_to_moneyline_prob(-float(spread)))
            except (ValueError, TypeError):
                pass

    if not home_probs:
        return None

    p_home = sum(home_probs) / len(home_probs)
    p_home = max(0.01, min(0.99, p_home))
    return p_home, 1.0 - p_home, len(home_probs)


def infer_yes_team(
    question: str,
    home_name: str,
    away_name: str,
    settled_outcome: str = "",
    last_price: Optional[float] = None,
) -> str:
    """Decide whether the market's YES token refers to the home or away team.

    espn_consensus_prob must price the SAME event as polymarket_price (the
    YES token), so getting this wrong flips the sign of every backtest edge.
    Inference order, strongest evidence first:

    1. Outcome agreement — if we know who actually won (from ESPN scores)
       AND the market's final price was decisive (≥0.75 or ≤0.25), the data
       itself tells us: price→1 means YES was the winner, price→0 means YES
       was the loser. Immune to naming/format assumptions.
    2. Question text — for "Will the Hawks beat the Rockets?" the first team
       nickname mentioned is the YES subject.
    3. Slug convention fallback — game slugs are aec-nba-{away}-{home}-{date}
       and the live OddsCache returns P(first-listed team wins), so YES
       defaults to the away team.

    Returns "home" or "away".
    """
    # 1. Data-driven: settled winner + decisive closing price
    if settled_outcome in ("home", "away") and last_price is not None:
        if last_price >= 0.75:
            return settled_outcome
        if last_price <= 0.25:
            return "away" if settled_outcome == "home" else "home"

    # 2. Question text: first team nickname mentioned is the YES subject
    q = (question or "").lower()

    def _first_mention(team_name: str) -> int:
        nickname = team_name.split()[-1].lower() if team_name else ""
        if not nickname:
            return -1
        return q.find(nickname)

    home_pos = _first_mention(home_name)
    away_pos = _first_mention(away_name)
    if home_pos >= 0 and (away_pos < 0 or home_pos < away_pos):
        return "home"
    if away_pos >= 0:
        return "away"

    # 3. Slug convention: first-listed (away) team is the YES subject
    return "away"


def _parse_game_start_ts(game_start_time: str) -> Optional[int]:
    """ESPN ISO timestamp (e.g. '2025-01-16T00:30Z') → unix ts, or None."""
    if not game_start_time:
        return None
    try:
        iso = game_start_time.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def backfill_consensus_for_market(
    session: requests.Session,
    conn,
    market_row: Dict,
    window_days: int,
    last_call: float,
) -> Tuple[int, float]:
    """Populate espn_consensus_prob for one moneyline_game market's snapshots.

    Returns (snapshot_rows_updated, last_call). 0 rows means either no
    pickcenter odds exist for the game or no snapshots fall in the window.
    """
    slug = market_row.get("slug", "")
    espn_id = market_row.get("espn_game_id", "")
    if not espn_id:
        return 0, last_call

    pickcenter, last_call = fetch_espn_pickcenter(session, espn_id, last_call)
    consensus = compute_consensus_from_pickcenter(pickcenter)
    if consensus is None:
        logging.info("no_pickcenter_odds slug=%s espn_id=%s", slug, espn_id)
        return 0, last_call
    p_home, p_away, num_books = consensus

    # Final price for the YES-team inference (step 1 needs it)
    row = conn.execute(
        "SELECT polymarket_price FROM historical_snapshots "
        "WHERE slug = ? ORDER BY timestamp DESC LIMIT 1",
        (slug,),
    ).fetchone()
    last_price = float(row[0]) if row else None

    yes_team = infer_yes_team(
        question=market_row.get("question", ""),
        home_name=market_row.get("home_team", ""),
        away_name=market_row.get("away_team", ""),
        settled_outcome=market_row.get("settled_outcome", ""),
        last_price=last_price,
    )
    yes_prob = p_home if yes_team == "home" else p_away

    # Closing-line lookahead guard: only stamp snapshots near the game.
    # Snapshot timestamps are floored to UTC day boundaries, so the window
    # start is floored too — otherwise a game at 02:00 UTC would exclude its
    # own game-day snapshot.
    start_ts = end_ts = None
    game_ts = _parse_game_start_ts(market_row.get("game_start_time", ""))
    if game_ts is not None:
        start_ts = ((game_ts - window_days * 86400) // 86400) * 86400
        end_ts = game_ts + 86400

    updated = update_snapshot_consensus(
        conn, slug, yes_prob, num_books, start_ts, end_ts
    )
    conn.commit()

    logging.info(
        "consensus_backfilled slug=%s yes_team=%s prob=%.4f books=%d rows=%d",
        slug, yes_team, yes_prob, num_books, updated,
    )
    return updated, last_call


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def _settled_from_price(last_price: float) -> str:
    if last_price >= 0.90:
        return "YES"
    if last_price <= 0.10:
        return "NO"
    return "open"


def ingest_game_market(
    session: requests.Session,
    game: Dict,
    market: Dict,
    start_ts: int,
    end_ts: int,
    last_call: float,
    conn,
) -> Tuple[int, float]:
    """Insert one game market + its price snapshots. Returns (snap_count, last_call)."""
    slug = market.get("slug", "")
    t0, t1 = _parse_token_ids(market)

    if not t0:
        logging.warning("no_token_id slug=%s", slug)
        return 0, last_call

    history, last_call = fetch_prices_history(session, t0, start_ts, end_ts, last_call)

    # Determine settled outcome from score or price
    try:
        h_score = int(game["home_score"]) if game["home_score"] else None
        a_score = int(game["away_score"]) if game["away_score"] else None
        if h_score is not None and a_score is not None:
            settled = "home" if h_score > a_score else "away"
        elif history:
            settled = _settled_from_price(history[-1]["p"])
        else:
            settled = ""
    except (ValueError, TypeError):
        settled = _settled_from_price(history[-1]["p"]) if history else ""

    m_row = {
        "slug":            slug,
        "market_id":       str(market.get("id", "")),
        "condition_id":    market.get("conditionId", ""),
        "question":        market.get("question", ""),
        "home_team":       game["home_name"],
        "away_team":       game["away_name"],
        "home_abbr":       game["home_abbr"],
        "away_abbr":       game["away_abbr"],
        "game_start_time": game["start_time"],
        "espn_game_id":    game["espn_id"],
        "home_score":      game["home_score"],
        "away_score":      game["away_score"],
        "settled_outcome": settled,
        "market_type":     "moneyline_game",
        "token_id_0":      t0,
        "token_id_1":      t1,
    }
    upsert_historical_market(conn, m_row)
    n = upsert_snapshots(conn, slug, history)
    conn.commit()

    logging.info(
        "game_market_ingested slug=%s snapshots=%d settled=%s", slug, n, settled
    )
    return n, last_call


def ingest_outright_market(
    session: requests.Session,
    market: Dict,
    start_ts: int,
    end_ts: int,
    last_call: float,
    conn,
) -> Tuple[int, float]:
    """Insert one outright market + its price snapshots. Returns (snap_count, last_call)."""
    slug = market.get("slug", "")
    t0, t1 = _parse_token_ids(market)

    if not t0:
        logging.warning("no_token_id slug=%s", slug)
        return 0, last_call

    history, last_call = fetch_prices_history(session, t0, start_ts, end_ts, last_call)

    if not history:
        logging.debug("no_window_data slug=%s", slug)
        return 0, last_call

    last_price = history[-1]["p"]
    settled = _settled_from_price(last_price)

    m_row = {
        "slug":            slug,
        "market_id":       str(market.get("id", "")),
        "condition_id":    market.get("conditionId", ""),
        "question":        market.get("question", ""),
        "market_type":     "nba_outright",
        "settled_outcome": settled,
        "token_id_0":      t0,
        "token_id_1":      t1,
    }
    upsert_historical_market(conn, m_row)
    n = upsert_snapshots(conn, slug, history)
    conn.commit()

    logging.info(
        "outright_ingested slug=%s snapshots=%d last_p=%.4f settled=%s",
        slug, n, last_price, settled,
    )
    return n, last_call


# ── Main ──────────────────────────────────────────────────────────────────────

def run_consensus_backfill(
    session: requests.Session, conn, window_days: int, last_call: float
) -> Tuple[int, int, float]:
    """Phase 3: populate espn_consensus_prob for every ingested game market.

    Sweeps all moneyline_game rows that have an ESPN game id — including ones
    ingested on previous runs — so it doubles as a standalone backfill for
    databases populated before consensus support existed (--consensus-only).
    Returns (markets_updated, snapshot_rows_updated, last_call).
    """
    rows = conn.execute(
        "SELECT slug, espn_game_id, question, home_team, away_team, "
        "       settled_outcome, game_start_time "
        "FROM historical_markets "
        "WHERE market_type = 'moneyline_game' AND espn_game_id != '' "
        "ORDER BY slug"
    ).fetchall()

    markets_updated = 0
    rows_updated = 0
    for r in rows:
        market_row = dict(r)
        try:
            n, last_call = backfill_consensus_for_market(
                session, conn, market_row, window_days, last_call
            )
            if n > 0:
                markets_updated += 1
                rows_updated += n
        except Exception as exc:
            logging.error(
                "consensus_error slug=%s error=%s", market_row.get("slug"), exc
            )
    return markets_updated, rows_updated, last_call


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest historical NBA Polymarket data into data/trades.db"
    )
    parser.add_argument(
        "--start", help="Start date (YYYY-MM-DD, inclusive)"
    )
    parser.add_argument(
        "--end", help="End date (YYYY-MM-DD, inclusive)"
    )
    parser.add_argument(
        "--consensus-only", action="store_true",
        help="Skip price ingestion; only backfill espn_consensus_prob for "
             "game markets already in the database (no --start/--end needed)"
    )
    parser.add_argument(
        "--consensus-window-days", type=int, default=3,
        help="Apply the (closing-line) consensus only to snapshots within "
             "this many days before game start (default 3). The pickcenter "
             "line reflects information up to tip-off, so stamping it onto "
             "much older snapshots would leak future information into the "
             "backtest."
    )
    parser.add_argument(
        "--skip-consensus", action="store_true",
        help="Skip the consensus backfill phase entirely"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Standalone consensus backfill mode ────────────────────────────────────
    if args.consensus_only:
        init_tables()
        session = requests.Session()
        session.headers.update({"User-Agent": "polymarket-bot-historical/1.0"})
        conn = get_conn()
        m_updated, s_updated, _ = run_consensus_backfill(
            session, conn, args.consensus_window_days, 0.0
        )
        conn.close()
        coverage = get_consensus_coverage()
        logging.info("=== CONSENSUS BACKFILL COMPLETE ===")
        logging.info("markets_backfilled:       %d", m_updated)
        logging.info("snapshot_rows_updated:    %d", s_updated)
        logging.info("coverage: %d/%d snapshots have consensus (%d slugs)",
                     coverage["with_consensus"], coverage["total"],
                     coverage["slugs_with_consensus"])
        return

    if not args.start or not args.end:
        parser.error("--start and --end are required (unless --consensus-only)")

    start_dt = datetime.date.fromisoformat(args.start)
    end_dt   = datetime.date.fromisoformat(args.end)
    start_ts = int(
        datetime.datetime(
            start_dt.year, start_dt.month, start_dt.day,
            tzinfo=datetime.timezone.utc
        ).timestamp()
    )
    end_ts = int(
        datetime.datetime(
            end_dt.year, end_dt.month, end_dt.day, 23, 59, 59,
            tzinfo=datetime.timezone.utc
        ).timestamp()
    )

    logging.info(
        "ingestion_start start=%s end=%s start_ts=%d end_ts=%d",
        args.start, args.end, start_ts, end_ts,
    )

    # Initialise tables (idempotent)
    init_tables()
    logging.info("tables_ready")

    session = requests.Session()
    session.headers.update({"User-Agent": "polymarket-bot-historical/1.0"})

    conn = get_conn()
    last_call = 0.0

    total_espn_games      = 0
    total_game_matched    = 0
    total_game_snaps      = 0
    total_outright_mrkts  = 0
    total_outright_snaps  = 0
    errors                = 0

    # ── Phase 1: ESPN game-by-game sweep ─────────────────────────────────────
    logging.info("=== Phase 1: ESPN game-by-game sweep ===")
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            games, last_call = fetch_espn_games(session, date_str, last_call)
            logging.info("date=%s espn_games=%d", date_str, len(games))
            total_espn_games += len(games)

            for game in games:
                try:
                    market, last_call = find_game_market(session, game, last_call)
                    if market is None:
                        logging.info(
                            "no_market %s @ %s on %s",
                            game["away_name"], game["home_name"], date_str,
                        )
                        continue

                    n, last_call = ingest_game_market(
                        session, game, market, start_ts, end_ts, last_call, conn
                    )
                    total_game_matched += 1
                    total_game_snaps   += n

                except Exception as exc:
                    errors += 1
                    logging.error(
                        "game_error espn_id=%s error=%s", game.get("espn_id"), exc
                    )

        except Exception as exc:
            errors += 1
            logging.error("date_error date=%s error=%s", date_str, exc)

        current += datetime.timedelta(days=1)

    # ── Phase 2: NBA outright / futures sweep ─────────────────────────────────
    logging.info("=== Phase 2: NBA outright sweep ===")
    seen_event_slugs = set()
    for event_slug in NBA_OUTRIGHT_EVENT_SLUGS:
        if event_slug in seen_event_slugs:
            continue
        seen_event_slugs.add(event_slug)
        try:
            markets, last_call = fetch_gamma_event_markets(session, event_slug, last_call)
            logging.info("event=%s markets=%d", event_slug, len(markets))

            for market in markets:
                try:
                    n, last_call = ingest_outright_market(
                        session, market, start_ts, end_ts, last_call, conn
                    )
                    if n > 0:
                        total_outright_mrkts += 1
                        total_outright_snaps += n
                except Exception as exc:
                    errors += 1
                    logging.error(
                        "outright_error slug=%s error=%s", market.get("slug"), exc
                    )

        except Exception as exc:
            errors += 1
            logging.error("event_error event=%s error=%s", event_slug, exc)

    # ── Phase 3: consensus backfill for game markets ──────────────────────────
    consensus_markets = 0
    consensus_rows = 0
    if args.skip_consensus:
        logging.info("=== Phase 3: consensus backfill SKIPPED (--skip-consensus) ===")
    else:
        logging.info("=== Phase 3: consensus backfill (ESPN pickcenter) ===")
        consensus_markets, consensus_rows, last_call = run_consensus_backfill(
            session, conn, args.consensus_window_days, last_call
        )

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_markets = total_game_matched + total_outright_mrkts
    total_snaps   = total_game_snaps + total_outright_snaps
    coverage = get_consensus_coverage()

    logging.info("=== INGESTION COMPLETE ===")
    logging.info("date_range:              %s → %s", args.start, args.end)
    logging.info("espn_games_found:        %d", total_espn_games)
    logging.info("game_markets_matched:    %d", total_game_matched)
    logging.info("game_snapshots:          %d", total_game_snaps)
    logging.info("outright_markets:        %d", total_outright_mrkts)
    logging.info("outright_snapshots:      %d", total_outright_snaps)
    logging.info("total_markets_ingested:  %d", total_markets)
    logging.info("total_snapshots_ingested:%d", total_snaps)
    logging.info("consensus_markets:       %d", consensus_markets)
    logging.info("consensus_rows_updated:  %d", consensus_rows)
    logging.info("consensus_coverage:      %d/%d snapshots (%d slugs)",
                 coverage["with_consensus"], coverage["total"],
                 coverage["slugs_with_consensus"])
    logging.info("errors:                  %d", errors)


if __name__ == "__main__":
    main()
