"""Central league registry — the single source of truth for supported sports.

The bot is strictly focused on sports happening NOW and going forward. Rather
than hardcoding which leagues are "in season" (which would need editing every
few months), every consumer sweeps ALL registered leagues and lets ESPN's
scoreboard self-select: an off-season league simply returns zero games and
costs one cached HTTP call. When the NBA comes back in October, the bot picks
it up automatically with no code change; in July it's MLB/WNBA/MLS that have
games.

Consumers:
  * bot/game_schedule.py      — live-mode wake/sleep + game clocks
  * bot/signals/odds_api.py   — ESPN fallback odds endpoints
  * scripts/ingest_historical.py — daily forward data recorder

The `slug` key is the league code Polymarket uses in game slugs
(aec-{league}-{away}-{home}-{date}) and in per-league minimum-edge rules.
"""

from typing import Dict, List, Optional

# ESPN site API root; a league's scoreboard is {ESPN_SITE_API}/{espn_path}/scoreboard
ESPN_SITE_API = "https://site.api.espn.com/apis/site/v2/sports"

LEAGUES: Dict[str, Dict] = {
    # ── Summer (in season as of July) ────────────────────────────────────────
    "mlb": {
        "espn_path": "baseball/mlb",
        "odds_api_key": "baseball_mlb",
        # Baseball has no game clock — last-5-minutes blocking doesn't apply.
        "clock": None,
    },
    "wnba": {
        "espn_path": "basketball/wnba",
        "odds_api_key": "basketball_wnba",
        # 4 quarters × 10 minutes
        "clock": {"periods": 4, "minutes": 10, "count_up": False},
    },
    "mls": {
        "espn_path": "soccer/usa.1",
        "odds_api_key": "soccer_usa_mls",
        # 2 halves × 45 minutes; soccer clocks count UP
        "clock": {"periods": 2, "minutes": 45, "count_up": True},
    },
    # ── Fall/winter (auto-activate when their seasons start) ─────────────────
    "nba": {
        "espn_path": "basketball/nba",
        "odds_api_key": "basketball_nba",
        "clock": {"periods": 4, "minutes": 12, "count_up": False},
    },
    "cbb": {
        "espn_path": "basketball/mens-college-basketball",
        "odds_api_key": "basketball_ncaab",
        "clock": {"periods": 2, "minutes": 20, "count_up": False},
    },
    "nhl": {
        "espn_path": "hockey/nhl",
        "odds_api_key": "icehockey_nhl",
        "clock": {"periods": 3, "minutes": 20, "count_up": False},
    },
    "nfl": {
        "espn_path": "football/nfl",
        "odds_api_key": "americanfootball_nfl",
        "clock": {"periods": 4, "minutes": 15, "count_up": False},
    },
    "epl": {
        "espn_path": "soccer/eng.1",
        "odds_api_key": "soccer_epl",
        "clock": {"periods": 2, "minutes": 45, "count_up": True},
    },
}


def all_league_codes() -> List[str]:
    return list(LEAGUES.keys())


def scoreboard_url(league: str) -> Optional[str]:
    """ESPN scoreboard URL for a league code, or None if unregistered."""
    info = LEAGUES.get(league)
    if not info:
        return None
    return f"{ESPN_SITE_API}/{info['espn_path']}/scoreboard"


def summary_url(league: str) -> Optional[str]:
    """ESPN game-summary URL (pickcenter odds source) for a league code."""
    info = LEAGUES.get(league)
    if not info:
        return None
    return f"{ESPN_SITE_API}/{info['espn_path']}/summary"


def game_seconds_remaining(league: str, period: int, clock_seconds: float) -> Optional[float]:
    """Estimate seconds remaining in a live game from period + display clock.

    Returns None for clockless sports (baseball) and unknown leagues — the
    caller treats None as "no last-5-minutes block".
    """
    info = LEAGUES.get(league)
    if not info or not info.get("clock"):
        return None
    clock = info["clock"]
    period_seconds = clock["minutes"] * 60

    if clock["count_up"]:
        # Soccer: displayClock counts up within the match (e.g. 67' = 4020s).
        total = clock["periods"] * period_seconds
        return max(0.0, total - clock_seconds)

    # Count-down clocks: remaining = full future periods + current clock
    periods_left = max(0, clock["periods"] - period)
    return periods_left * period_seconds + clock_seconds
