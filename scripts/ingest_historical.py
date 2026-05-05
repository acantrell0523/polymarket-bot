"""
Historical NBA data ingestion pipeline (Option D — CLOB prices-history).

Data flow
---------
Phase 1 (game-by-game):
  ESPN scoreboard  →  Polymarket slug lookup (Gamma)  →  CLOB prices-history

Phase 2 (season-long outrights):
  Gamma /events for known NBA event slugs  →  CLOB prices-history

Both phases filter price history to the [--start, --end] date window and
store results in data/trades.db:
  historical_markets   — one row per Polymarket market
  historical_snapshots — (slug, timestamp, price) time-series

Re-running is fully idempotent: INSERT OR IGNORE, no duplicates.
Per-market and per-game failures are logged and skipped.
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
    init_tables,
    upsert_historical_market,
    upsert_snapshots,
)

# ── Constants ─────────────────────────────────────────────────────────────────

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest historical NBA Polymarket data into data/trades.db"
    )
    parser.add_argument(
        "--start", required=True, help="Start date (YYYY-MM-DD, inclusive)"
    )
    parser.add_argument(
        "--end", required=True, help="End date (YYYY-MM-DD, inclusive)"
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

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_markets = total_game_matched + total_outright_mrkts
    total_snaps   = total_game_snaps + total_outright_snaps

    logging.info("=== INGESTION COMPLETE ===")
    logging.info("date_range:              %s → %s", args.start, args.end)
    logging.info("espn_games_found:        %d", total_espn_games)
    logging.info("game_markets_matched:    %d", total_game_matched)
    logging.info("game_snapshots:          %d", total_game_snaps)
    logging.info("outright_markets:        %d", total_outright_mrkts)
    logging.info("outright_snapshots:      %d", total_outright_snaps)
    logging.info("total_markets_ingested:  %d", total_markets)
    logging.info("total_snapshots_ingested:%d", total_snaps)
    logging.info("errors:                  %d", errors)


if __name__ == "__main__":
    main()
