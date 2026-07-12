"""
Sports data recorder — CURRENT sports, today and forward.

The bot trades whatever is in season NOW (July: MLB, WNBA, MLS; NBA/NHL/NFL
pick up automatically when their seasons start — see bot/leagues.py). This
script records the data those backtests need, and its defaults are
forward-looking: run it with no date arguments and it sweeps the last few
days up to TODAY across every registered league. Run it daily (cron, or the
supervisor's built-in 05:30 ET job) and the historical dataset accumulates
forward from today. Off-season leagues return zero games and cost nothing.

Data flow
---------
Phase 1 (game-by-game, all registered leagues):
  ESPN scoreboard  →  Polymarket slug lookup (Gamma)  →  CLOB prices-history
                   →  ESPN game summary (pickcenter)  →  consensus probability

Phase 2 (season-long NBA outrights — OPT-IN via --include-nba-outrights;
  disabled by default because the NBA season is over):
  Gamma /events for known NBA event slugs  →  CLOB prices-history
  (no consensus source exists for outrights — they stay gated in backtests)

Phase 3 (consensus backfill, also available standalone via --consensus-only):
  For every ingested moneyline_game market, fetch the ESPN game summary's
  `pickcenter` block for that market's league, de-vig each provider's
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

Housekeeping: --prune-leagues nba deletes previously ingested markets and
snapshots for finished seasons you no longer care about.

Both price phases filter history to the resolved date window and store
results in data/trades.db:
  historical_markets   — one row per Polymarket market (league column set)
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
from bot.leagues import LEAGUES, all_league_codes, scoreboard_url, summary_url

# ── Constants ─────────────────────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# Minimum seconds between outbound API calls (polite throttle)
THROTTLE_SECS = 0.2

# Leagues that use 3-outcome (win/draw/win) markets on Polymarket, which use
# the atc- slug family with a team-outcome suffix instead of aec-.
SOCCER_LEAGUES = {"mls", "epl"}

# Default sweep window when no dates are given: the recent past up to today.
DEFAULT_DAYS_BACK = 3

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
    session: requests.Session, league: str, date_str: str, last_call: float
) -> Tuple[List[Dict], float]:
    """
    Fetch a league's games for a YYYYMMDD string from the ESPN scoreboard.
    Off-season leagues return an empty list. Returns (list[game_dict], last_call).
    """
    url = scoreboard_url(league)
    if not url:
        return [], last_call

    last_call = _throttle(last_call)
    data = _get(session, url, params={"dates": date_str})
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
                "league":     league,
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
            logging.warning("espn_parse_error league=%s event_id=%s error=%s",
                            league, ev.get("id"), exc)

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


# Known ESPN ↔ Polymarket abbreviation differences, per league.
# Verified live against Gamma on 2026-07-11 (MLB/WNBA); other leagues'
# mismatches surface as "no_market" log lines — add mappings as discovered.
ABBR_MAP = {
    "nba": {
        "sa":  "sas",   # San Antonio Spurs
        "gs":  "gsw",   # Golden State Warriors
        "ny":  "nyk",   # New York Knicks
        "no":  "nor",   # New Orleans Pelicans
    },
    "mlb": {
        "chw": "cws",   # Chicago White Sox
        "ath": "oak",   # Athletics (Polymarket kept the oak code)
    },
    "wnba": {
        "gs":  "gsv",   # Golden State Valkyries
        "con": "conn",  # Connecticut Sun
        "lv":  "las",   # Las Vegas Aces
        "ny":  "nyl",   # New York Liberty
    },
}


def candidate_slugs(game: Dict) -> List[str]:
    """Possible Polymarket slugs for one ESPN game, most likely first.

    Verified live against Gamma (2026-07-11): game markets use BARE slugs —
    {league}-{away}-{home}-{date}, e.g. mlb-mil-pit-2026-07-10 — with
    outcomes [away_team, home_team] on a single market. The aec-/atc-
    prefixed families are the polymarket.us gateway convention and are kept
    as fallbacks. Each candidate costs one Gamma lookup, so the list is
    kept short.
    """
    league = game["league"]
    away = game["away_abbr"]
    home = game["home_abbr"]
    date_iso = f"{game['date'][:4]}-{game['date'][4:6]}-{game['date'][6:]}"

    amap = ABBR_MAP.get(league, {})
    away_pm = amap.get(away, away)
    home_pm = amap.get(home, home)

    slugs = [f"{league}-{away_pm}-{home_pm}-{date_iso}"]
    if (away_pm, home_pm) != (away, home):
        slugs.append(f"{league}-{away}-{home}-{date_iso}")

    if league in SOCCER_LEAGUES:
        base = f"atc-{league}-{away_pm}-{home_pm}-{date_iso}"
        slugs += [f"{base}-{away_pm}", f"{base}-{home_pm}"]
    else:
        slugs.append(f"aec-{league}-{away_pm}-{home_pm}-{date_iso}")
    return slugs


def find_game_market(
    session: requests.Session, game: Dict, last_call: float
) -> Tuple[Optional[Dict], float]:
    """
    Try to find a Polymarket moneyline market for a given ESPN game.
    Returns (market_dict | None, last_call).

    Gamma's /markets?slug= excludes CLOSED markets by default (verified live
    2026-07-11: a resolved game returns 0 rows without closed=true). The
    recorder mostly sweeps finished games, so each candidate slug is tried
    open-first, then with closed=true.
    """
    for slug in candidate_slugs(game):
        for extra in ({}, {"closed": "true"}):
            last_call = _throttle(last_call)
            data = _get(session, f"{GAMMA_BASE}/markets",
                        params={"slug": slug, **extra})
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
    session: requests.Session, espn_game_id: str, last_call: float,
    league: str = "nba",
) -> Tuple[List[Dict], float]:
    """Fetch the pickcenter (sportsbook odds) block from an ESPN game summary.

    Works for past games — ESPN retains the odds in the summary endpoint.
    Returns (list[provider_odds_dict], last_call).
    """
    url = summary_url(league) or summary_url("nba")
    last_call = _throttle(last_call)
    data = _get(session, url, params={"event": espn_game_id})
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


def multi_book_home_prob(
    aggregator, league: str, home_abbr: str, away_abbr: str
) -> Optional[Tuple[float, int]]:
    """Point-in-time multi-book consensus (FanDuel + Pinnacle) for a game.

    Sportsbooks only quote UPCOMING games, so this naturally applies to the
    pre-game recorder pass and returns None for finished games — which is
    exactly the no-lookahead property we want: consensus recorded before
    tip-off, never reconstructed after the result is known.

    Returns (p_home, num_books) or None.
    """
    if aggregator is None:
        return None
    info = LEAGUES.get(league)
    if not info:
        return None
    amap = ABBR_MAP.get(league, {})
    try:
        game = aggregator.find_game(
            info["odds_api_key"],
            amap.get(home_abbr, home_abbr),
            amap.get(away_abbr, away_abbr),
        )
    except Exception as exc:
        logging.debug("multi_book_lookup_failed league=%s error=%s", league, exc)
        return None
    if not game:
        return None
    return float(game["home_prob"]), int(game.get("num_books", 1))


def existing_num_books(conn, slug: str) -> int:
    """Best num_books already recorded for a slug's snapshots."""
    row = conn.execute(
        "SELECT MAX(num_books) FROM historical_snapshots WHERE slug = ?", (slug,)
    ).fetchone()
    return int(row[0] or 0) if row else 0


def backfill_consensus_for_market(
    session: requests.Session,
    conn,
    market_row: Dict,
    window_days: int,
    last_call: float,
    aggregator=None,
) -> Tuple[int, float]:
    """Populate espn_consensus_prob for one moneyline_game market's snapshots.

    Consensus sources, all de-vigged to P(home):
      * ESPN pickcenter (DraftKings closing line) — works for past games
      * FanDuel + Pinnacle via MultiBookAggregator — pre-game only (books
        drop games once played), recorded point-in-time

    The blend weights each source by its book count; num_books is the total
    across DISTINCT books (DK never double-counts — the aggregator fetches
    only FanDuel and Pinnacle).

    NO-DOWNGRADE GUARD: a later run (after the game finishes) sees only the
    1-book pickcenter line and must not overwrite a richer pre-game 3-book
    consensus already recorded.

    Returns (snapshot_rows_updated, last_call).
    """
    slug = market_row.get("slug", "")
    espn_id = market_row.get("espn_game_id", "")
    if not espn_id:
        return 0, last_call

    # Legacy rows stored league as 'NBA'; new rows store the lowercase code.
    league = (market_row.get("league") or "nba").lower()
    pickcenter, last_call = fetch_espn_pickcenter(session, espn_id, last_call, league)
    consensus = compute_consensus_from_pickcenter(pickcenter)

    # Blend in live sportsbooks (pre-game only; None once the game is played)
    books_result = multi_book_home_prob(
        aggregator, league,
        market_row.get("home_abbr", ""), market_row.get("away_abbr", ""),
    )

    if consensus is None and books_result is None:
        logging.info("no_consensus_source slug=%s espn_id=%s", slug, espn_id)
        return 0, last_call

    parts = []  # (p_home, num_books) per source
    if consensus is not None:
        parts.append((consensus[0], consensus[2]))
    if books_result is not None:
        parts.append(books_result)

    total_books = sum(n for _, n in parts)
    p_home = sum(p * n for p, n in parts) / total_books
    p_home = max(0.01, min(0.99, p_home))
    p_away = 1.0 - p_home
    num_books = total_books

    # No-downgrade guard
    already = existing_num_books(conn, slug)
    if already > num_books:
        logging.info(
            "consensus_kept_richer slug=%s existing_books=%d new_books=%d",
            slug, already, num_books,
        )
        return 0, last_call

    # Which team does the stored price series (token 0) refer to?
    # token0_side is exact — read from the market's outcomes array at ingest.
    # Rows ingested before that column existed fall back to inference.
    yes_team = (market_row.get("token0_side") or "").lower()
    if yes_team not in ("home", "away"):
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

    # Which side does token 0 (the price series we store) refer to?
    # Game markets aren't Yes/No — Gamma returns outcomes as a JSON string,
    # e.g. '["Milwaukee Brewers", "Pittsburgh Pirates"]', ordered like
    # clobTokenIds. Matching outcomes[0] against the ESPN team names gives an
    # exact answer, so the consensus backfill never has to guess.
    token0_side = _token0_side_from_outcomes(
        market.get("outcomes"), game["home_name"], game["away_name"]
    )

    m_row = {
        "slug":            slug,
        "market_id":       str(market.get("id", "")),
        "condition_id":    market.get("conditionId", ""),
        "league":          game.get("league", "nba"),
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
        "token0_side":     token0_side,
    }
    upsert_historical_market(conn, m_row)
    n = upsert_snapshots(conn, slug, history)
    conn.commit()

    logging.info(
        "game_market_ingested slug=%s snapshots=%d settled=%s token0=%s",
        slug, n, settled, token0_side or "?",
    )
    return n, last_call


def _token0_side_from_outcomes(
    outcomes, home_name: str, away_name: str
) -> str:
    """Match outcomes[0] to the home or away team. Returns "home"/"away"/""."""
    import json as _json
    if isinstance(outcomes, str):
        try:
            outcomes = _json.loads(outcomes)
        except ValueError:
            return ""
    if not isinstance(outcomes, list) or not outcomes:
        return ""
    first = str(outcomes[0]).lower()
    home_nick = home_name.split()[-1].lower() if home_name else ""
    away_nick = away_name.split()[-1].lower() if away_name else ""
    if away_nick and away_nick in first:
        return "away"
    if home_nick and home_nick in first:
        return "home"
    return ""


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

    For games still listed by the sportsbooks (i.e. not yet played), a
    MultiBookAggregator blends FanDuel + Pinnacle into the ESPN/DraftKings
    pickcenter line — recorded point-in-time, giving real multi-book
    num_books that unlock the ">7% edge needs 3 books" rule in backtests.
    Returns (markets_updated, snapshot_rows_updated, last_call).
    """
    rows = conn.execute(
        "SELECT slug, espn_game_id, league, question, home_team, away_team, "
        "       home_abbr, away_abbr, settled_outcome, game_start_time, "
        "       token0_side "
        "FROM historical_markets "
        "WHERE market_type = 'moneyline_game' AND espn_game_id != '' "
        "ORDER BY slug"
    ).fetchall()

    # One aggregator for the whole sweep — it caches per-sport fetches
    # (TTL 300s), so 15 MLB games cost 2 upstream calls, not 30.
    aggregator = None
    try:
        from bot.signals.book_scrapers import MultiBookAggregator
        aggregator = MultiBookAggregator(cache_ttl=300)
    except Exception as exc:
        logging.warning("multi_book_aggregator_unavailable error=%s", exc)

    markets_updated = 0
    rows_updated = 0
    for r in rows:
        market_row = dict(r)
        try:
            n, last_call = backfill_consensus_for_market(
                session, conn, market_row, window_days, last_call,
                aggregator=aggregator,
            )
            if n > 0:
                markets_updated += 1
                rows_updated += n
        except Exception as exc:
            logging.error(
                "consensus_error slug=%s error=%s", market_row.get("slug"), exc
            )
    return markets_updated, rows_updated, last_call


def prune_leagues(conn, leagues: List[str]) -> Tuple[int, int]:
    """Delete markets + snapshots for finished seasons you no longer track.

    Matches historical_markets.league case-insensitively and also removes
    NBA outright markets when 'nba' is pruned (they have market_type
    'nba_outright'). Returns (markets_deleted, snapshots_deleted).
    """
    codes = [l.lower() for l in leagues]
    placeholders = ",".join("?" for _ in codes)
    where = f"LOWER(league) IN ({placeholders})"
    if "nba" in codes:
        where += " OR market_type = 'nba_outright'"

    slugs = [r[0] for r in conn.execute(
        f"SELECT slug FROM historical_markets WHERE {where}", codes
    ).fetchall()]

    snaps_deleted = 0
    if slugs:
        slug_ph = ",".join("?" for _ in slugs)
        snaps_deleted = conn.execute(
            f"DELETE FROM historical_snapshots WHERE slug IN ({slug_ph})", slugs
        ).rowcount
    markets_deleted = conn.execute(
        f"DELETE FROM historical_markets WHERE {where}", codes
    ).rowcount
    conn.commit()
    return markets_deleted, snaps_deleted


def run_ingest(
    start_dt: datetime.date,
    end_dt: datetime.date,
    leagues: List[str],
    include_nba_outrights: bool = False,
    consensus_window_days: int = 3,
    skip_consensus: bool = False,
) -> Dict[str, int]:
    """Run the full recorder (phases 1–3) and return a summary dict.

    Callable programmatically (the supervisor's daily recorder job uses this)
    as well as from the CLI.
    """
    start_ts = int(datetime.datetime(
        start_dt.year, start_dt.month, start_dt.day,
        tzinfo=datetime.timezone.utc).timestamp())
    end_ts = int(datetime.datetime(
        end_dt.year, end_dt.month, end_dt.day, 23, 59, 59,
        tzinfo=datetime.timezone.utc).timestamp())

    logging.info(
        "ingestion_start start=%s end=%s leagues=%s",
        start_dt.isoformat(), end_dt.isoformat(), ",".join(leagues),
    )

    init_tables()
    session = requests.Session()
    session.headers.update({"User-Agent": "polymarket-bot-historical/1.0"})
    conn = get_conn()
    last_call = 0.0

    totals = {
        "espn_games": 0, "game_markets": 0, "game_snapshots": 0,
        "outright_markets": 0, "outright_snapshots": 0,
        "consensus_markets": 0, "consensus_rows": 0, "errors": 0,
    }

    # ── Phase 1: ESPN game-by-game sweep, every registered league ────────────
    # Off-season leagues return zero games from ESPN — the sweep automatically
    # tracks whatever sports are actually being played in the window.
    logging.info("=== Phase 1: ESPN game sweep (leagues: %s) ===", ",".join(leagues))
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        for league in leagues:
            try:
                games, last_call = fetch_espn_games(session, league, date_str, last_call)
                if games:
                    logging.info("date=%s league=%s espn_games=%d",
                                 date_str, league, len(games))
                totals["espn_games"] += len(games)

                for game in games:
                    try:
                        market, last_call = find_game_market(session, game, last_call)
                        if market is None:
                            logging.info(
                                "no_market league=%s %s @ %s on %s",
                                league, game["away_name"], game["home_name"], date_str,
                            )
                            continue

                        n, last_call = ingest_game_market(
                            session, game, market, start_ts, end_ts, last_call, conn
                        )
                        totals["game_markets"] += 1
                        totals["game_snapshots"] += n

                    except Exception as exc:
                        totals["errors"] += 1
                        logging.error(
                            "game_error league=%s espn_id=%s error=%s",
                            league, game.get("espn_id"), exc,
                        )

            except Exception as exc:
                totals["errors"] += 1
                logging.error("date_error date=%s league=%s error=%s",
                              date_str, league, exc)

        current += datetime.timedelta(days=1)

    # ── Phase 2: NBA outright / futures sweep (opt-in; season is over) ───────
    if include_nba_outrights:
        logging.info("=== Phase 2: NBA outright sweep ===")
        for event_slug in dict.fromkeys(NBA_OUTRIGHT_EVENT_SLUGS):
            try:
                markets, last_call = fetch_gamma_event_markets(session, event_slug, last_call)
                logging.info("event=%s markets=%d", event_slug, len(markets))
                for market in markets:
                    try:
                        n, last_call = ingest_outright_market(
                            session, market, start_ts, end_ts, last_call, conn
                        )
                        if n > 0:
                            totals["outright_markets"] += 1
                            totals["outright_snapshots"] += n
                    except Exception as exc:
                        totals["errors"] += 1
                        logging.error("outright_error slug=%s error=%s",
                                      market.get("slug"), exc)
            except Exception as exc:
                totals["errors"] += 1
                logging.error("event_error event=%s error=%s", event_slug, exc)
    else:
        logging.info("=== Phase 2: NBA outrights SKIPPED "
                     "(season over; --include-nba-outrights to opt in) ===")

    # ── Phase 3: consensus backfill for game markets ──────────────────────────
    if skip_consensus:
        logging.info("=== Phase 3: consensus backfill SKIPPED (--skip-consensus) ===")
    else:
        logging.info("=== Phase 3: consensus backfill (ESPN pickcenter) ===")
        totals["consensus_markets"], totals["consensus_rows"], last_call = (
            run_consensus_backfill(session, conn, consensus_window_days, last_call)
        )

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    coverage = get_consensus_coverage()
    logging.info("=== INGESTION COMPLETE ===")
    logging.info("date_range:              %s → %s",
                 start_dt.isoformat(), end_dt.isoformat())
    logging.info("leagues:                 %s", ",".join(leagues))
    logging.info("espn_games_found:        %d", totals["espn_games"])
    logging.info("game_markets_matched:    %d", totals["game_markets"])
    logging.info("game_snapshots:          %d", totals["game_snapshots"])
    logging.info("outright_markets:        %d", totals["outright_markets"])
    logging.info("outright_snapshots:      %d", totals["outright_snapshots"])
    logging.info("consensus_markets:       %d", totals["consensus_markets"])
    logging.info("consensus_rows_updated:  %d", totals["consensus_rows"])
    logging.info("consensus_coverage:      %d/%d snapshots (%d slugs)",
                 coverage["with_consensus"], coverage["total"],
                 coverage["slugs_with_consensus"])
    logging.info("errors:                  %d", totals["errors"])
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record current-sports Polymarket data into data/trades.db. "
                    "With no date arguments, sweeps the last %d days up to "
                    "today across all registered leagues — run it daily and "
                    "the dataset builds forward." % DEFAULT_DAYS_BACK
    )
    parser.add_argument(
        "--start", help="Start date (YYYY-MM-DD, inclusive). "
                        "Default: today minus --days-back"
    )
    parser.add_argument(
        "--end", help="End date (YYYY-MM-DD, inclusive). Default: today"
    )
    parser.add_argument(
        "--days-back", type=int, default=DEFAULT_DAYS_BACK,
        help="With no --start, sweep this many days back from today "
             f"(default {DEFAULT_DAYS_BACK})"
    )
    parser.add_argument(
        "--leagues", default=",".join(all_league_codes()),
        help="Comma-separated league codes to sweep "
             f"(default: all registered — {','.join(all_league_codes())}). "
             "Off-season leagues cost one empty call per day."
    )
    parser.add_argument(
        "--include-nba-outrights", action="store_true",
        help="Also sweep season-long NBA outright markets (off by default — "
             "the NBA season is over)"
    )
    parser.add_argument(
        "--prune-leagues", default="",
        help="Comma-separated league codes whose ingested markets/snapshots "
             "should be DELETED before this run (e.g. --prune-leagues nba "
             "to drop last season's data)"
    )
    parser.add_argument(
        "--consensus-only", action="store_true",
        help="Skip price ingestion; only backfill espn_consensus_prob for "
             "game markets already in the database (no dates needed)"
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

    # ── Optional prune of finished-season data ────────────────────────────────
    if args.prune_leagues.strip():
        init_tables()
        conn = get_conn()
        codes = [c.strip() for c in args.prune_leagues.split(",") if c.strip()]
        m_del, s_del = prune_leagues(conn, codes)
        conn.close()
        logging.info("pruned leagues=%s markets=%d snapshots=%d",
                     ",".join(codes), m_del, s_del)

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

    # Forward-looking defaults: [today - days_back, today]
    today = datetime.datetime.now(datetime.timezone.utc).date()
    end_dt = datetime.date.fromisoformat(args.end) if args.end else today
    if args.start:
        start_dt = datetime.date.fromisoformat(args.start)
    else:
        start_dt = end_dt - datetime.timedelta(days=args.days_back)

    leagues = [c.strip().lower() for c in args.leagues.split(",") if c.strip()]
    unknown = [c for c in leagues if c not in LEAGUES]
    if unknown:
        parser.error(f"unknown league code(s): {','.join(unknown)} "
                     f"(registered: {','.join(all_league_codes())})")

    run_ingest(
        start_dt, end_dt, leagues,
        include_nba_outrights=args.include_nba_outrights,
        consensus_window_days=args.consensus_window_days,
        skip_consensus=args.skip_consensus,
    )


if __name__ == "__main__":
    main()
