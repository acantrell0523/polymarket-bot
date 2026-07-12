"""Live in-game win probabilities from ESPN — the live-edge signal source.

THE live-edge thesis: during a game, big swings (a home run, a 10-0 run, a
red card) reprice sportsbooks and ESPN's win-probability model within
seconds, while Polymarket order books lag — that lag IS the edge. This
module supplies the model side of that comparison.

Verified live 2026-07-11 against an in-progress MLB game (TOR@SD, top 7th):
the ESPN summary endpoint's `winprobability` array carries per-play
`homeWinPercentage` (0..1) with play IDs — a real, continuously updated
model, free, for every registered league.

Design:
  * One scoreboard sweep per league (TTL ~20s) finds in-progress games and
    their team abbreviations.
  * One summary fetch per live game (TTL ~20s) reads the latest
    winprobability point.
  * `get_live_prob(slug)` maps a Polymarket game slug to the probability of
    the slug's FIRST-LISTED team (= away team = the YES side of aec- game
    markets, the convention verified across this codebase) plus the data's
    age, so the signal can decay confidence with staleness.

Freshness matters more than anything here: a 3-minute-old win probability
during a live game is misinformation. Callers get (prob, age_seconds) and
must decay/refuse stale data.
"""

import time
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bot.leagues import LEAGUES, normalize_abbr, scoreboard_url, summary_url, league_from_slug

# Only sweep leagues whose games ESPN models live. All registered leagues
# qualify today; kept explicit so exotic additions don't silently join.
LIVE_WP_LEAGUES = tuple(LEAGUES.keys())


def slug_game_teams(slug: str) -> Optional[Tuple[str, str, str]]:
    """Parse a game slug into (league, away_abbr, home_abbr) — or None.

    Handles both families:
      aec-mlb-nyy-bos-2026-07-11 → ("mlb", "nyy", "bos")
      mlb-nyy-bos-2026-07-11     → ("mlb", "nyy", "bos")
    First-listed team = away = the YES side (verified convention).
    """
    parts = slug.lower().split("-")
    league = league_from_slug(slug)
    if not league:
        return None
    idx = 0 if parts[0] == league else 1
    if len(parts) < idx + 4:
        return None
    return league, parts[idx + 1], parts[idx + 2]


class LiveWinProbCache:
    """Fetches and caches live ESPN win probabilities, keyed for slug lookup.

    All methods are best-effort and never raise: the live signal degrades to
    confidence 0 rather than blocking a scan cycle.
    """

    def __init__(self, scoreboard_ttl: float = 20.0, summary_ttl: float = 20.0):
        self.scoreboard_ttl = scoreboard_ttl
        self.summary_ttl = summary_ttl
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503])))

        # league -> (fetched_at, {(away, home): espn_game_id})
        self._live_games: Dict[str, Tuple[float, Dict[Tuple[str, str], str]]] = {}
        # espn_game_id -> (fetched_at, home_win_pct)
        self._win_prob: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Fetching (overridable in tests)
    # ------------------------------------------------------------------

    def _fetch_json(self, url: str, params: Optional[dict] = None):
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def _live_games_for_league(self, league: str) -> Dict[Tuple[str, str], str]:
        """(away_abbr, home_abbr) -> espn_game_id for IN-PROGRESS games only."""
        now = time.time()
        cached = self._live_games.get(league)
        if cached and now - cached[0] < self.scoreboard_ttl:
            return cached[1]

        games: Dict[Tuple[str, str], str] = {}
        data = self._fetch_json(scoreboard_url(league))
        for ev in (data or {}).get("events", []):
            if ev.get("status", {}).get("type", {}).get("state") != "in":
                continue
            comp = (ev.get("competitions") or [{}])[0]
            away = home = ""
            for c in comp.get("competitors", []):
                abbr = normalize_abbr(
                    league, c.get("team", {}).get("abbreviation", "").lower())
                if c.get("homeAway") == "home":
                    home = abbr
                else:
                    away = abbr
            if away and home and ev.get("id"):
                games[(away, home)] = str(ev["id"])

        self._live_games[league] = (now, games)
        return games

    def _home_win_pct(self, league: str, espn_game_id: str) -> Optional[float]:
        """Latest homeWinPercentage from the game summary's winprobability."""
        now = time.time()
        cached = self._win_prob.get(espn_game_id)
        if cached and now - cached[0] < self.summary_ttl:
            return cached[1]

        data = self._fetch_json(summary_url(league), params={"event": espn_game_id})
        wp = (data or {}).get("winprobability") or []
        if not wp:
            return None
        try:
            home_pct = float(wp[-1].get("homeWinPercentage"))
        except (TypeError, ValueError):
            return None
        if not (0.0 <= home_pct <= 1.0):
            return None

        self._win_prob[espn_game_id] = (now, home_pct)
        return home_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_live_prob(self, slug: str) -> Optional[Tuple[float, float]]:
        """Live win probability for the slug's YES team (first-listed = away).

        Returns (prob, age_seconds) or None when the game isn't live, isn't
        found, or ESPN has no model output yet. prob = 1 - homeWinPercentage
        because game slugs list the away team first and that team is the
        YES side.
        """
        parsed = slug_game_teams(slug)
        if not parsed:
            return None
        league, away, home = parsed

        games = self._live_games_for_league(league)
        game_id = games.get((away, home)) or games.get((home, away))
        if not game_id:
            return None

        home_pct = self._home_win_pct(league, game_id)
        if home_pct is None:
            return None

        fetched_at = self._win_prob.get(game_id, (time.time(),))[0]
        age = max(0.0, time.time() - fetched_at)
        return 1.0 - home_pct, age
