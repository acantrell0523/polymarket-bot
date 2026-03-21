"""Live odds tracker for in-game edge detection.

During March Madness, sportsbook lines move fast on momentum swings.
Polymarket often lags behind. This module detects that lag.
"""

import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bot.signals.odds_api import _spread_to_moneyline_prob, TEAM_ABBREVS


# Historical seed win rates (NCAA tournament since 1985)
SEED_WIN_RATES = {
    (1, 16): 0.993, (2, 15): 0.943, (3, 14): 0.857, (4, 13): 0.793,
    (5, 12): 0.649, (6, 11): 0.627, (7, 10): 0.607, (8, 9): 0.514,
    (1, 8): 0.800, (1, 9): 0.860, (2, 7): 0.710, (2, 10): 0.770,
    (3, 6): 0.590, (3, 11): 0.750, (4, 5): 0.550, (4, 12): 0.700,
    (4, 13): 0.793, (1, 4): 0.720, (1, 5): 0.810, (2, 3): 0.560,
    (1, 2): 0.530, (1, 3): 0.650,
}

ESPN_NCAA_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


class LiveOddsTracker:
    """Tracks live odds movement during games.

    Compares ESPN's live implied probability (from in-game spread)
    to Polymarket's current price. When the gap exceeds a threshold,
    it signals a trade opportunity.
    """

    def __init__(self, cache_ttl: int = 30):
        self.cache_ttl = cache_ttl
        self._last_fetch: float = 0
        self._scoreboard: Optional[dict] = None
        self._history: Dict[str, List[dict]] = {}  # slug -> list of snapshots

    def _fetch_scoreboard(self) -> Optional[dict]:
        now = time.time()
        if self._scoreboard and (now - self._last_fetch) < self.cache_ttl:
            return self._scoreboard
        try:
            resp = requests.get(ESPN_NCAA_URL, timeout=10)
            if resp.status_code == 200:
                self._scoreboard = resp.json()
                self._last_fetch = now
                return self._scoreboard
        except Exception:
            pass
        return self._scoreboard

    def get_live_games(self) -> List[dict]:
        """Get all currently live NCAA games with scores, odds, and context."""
        scoreboard = self._fetch_scoreboard()
        if not scoreboard:
            return []

        live = []
        for event in scoreboard.get("events", []):
            status = event.get("status", {}).get("type", {})
            if status.get("name") not in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME",
                                           "STATUS_END_PERIOD"):
                continue

            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            odds = comps.get("odds", [])

            if len(competitors) < 2:
                continue

            game = {
                "event_name": event.get("name", ""),
                "status": status.get("detail", ""),
                "status_name": status.get("name", ""),
                "neutral_site": comps.get("neutralSite", False),
                "home": None,
                "away": None,
                "spread": None,
                "over_under": None,
                "home_implied_prob": None,
            }

            for c in competitors:
                team = c.get("team", {})
                seed = c.get("curatedRank", {}).get("current", 0)
                info = {
                    "name": team.get("displayName", ""),
                    "abbr": team.get("abbreviation", "").lower(),
                    "seed": seed,
                    "score": int(c.get("score", "0") or "0"),
                    "record": c.get("records", [{}])[0].get("summary", "") if c.get("records") else "",
                }
                if c.get("homeAway") == "home":
                    game["home"] = info
                else:
                    game["away"] = info

            # Live odds from ESPN
            for o in odds:
                spread = o.get("spread")
                ou = o.get("overUnder")
                details = o.get("details", "")
                if spread is not None:
                    try:
                        spread_val = float(spread)
                        # ESPN: positive spread = away favored
                        home_spread = -spread_val
                        game["spread"] = spread_val
                        game["home_implied_prob"] = _spread_to_moneyline_prob(home_spread)
                    except (ValueError, TypeError):
                        pass
                if ou is not None:
                    try:
                        game["over_under"] = float(ou)
                    except (ValueError, TypeError):
                        pass

            if game["home"] and game["away"]:
                live.append(game)

        return live

    def get_all_games(self) -> List[dict]:
        """Get ALL today's NCAA games (live, scheduled, final) with odds and seeds."""
        scoreboard = self._fetch_scoreboard()
        if not scoreboard:
            return []

        games = []
        for event in scoreboard.get("events", []):
            status = event.get("status", {}).get("type", {})
            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            odds = comps.get("odds", [])

            if len(competitors) < 2:
                continue

            game = {
                "event_name": event.get("name", ""),
                "date": event.get("date", ""),
                "status": status.get("detail", ""),
                "status_name": status.get("name", ""),
                "completed": status.get("completed", False),
                "neutral_site": comps.get("neutralSite", False),
                "venue": comps.get("venue", {}).get("fullName", ""),
                "home": None,
                "away": None,
                "spread": None,
                "over_under": None,
                "home_implied_prob": None,
            }

            for c in competitors:
                team = c.get("team", {})
                seed = c.get("curatedRank", {}).get("current", 0)
                info = {
                    "name": team.get("displayName", ""),
                    "abbr": team.get("abbreviation", "").lower(),
                    "seed": seed,
                    "score": int(c.get("score", "0") or "0"),
                    "record": c.get("records", [{}])[0].get("summary", "") if c.get("records") else "",
                }
                if c.get("homeAway") == "home":
                    game["home"] = info
                else:
                    game["away"] = info

            for o in odds:
                spread = o.get("spread")
                ou = o.get("overUnder")
                if spread is not None:
                    try:
                        spread_val = float(spread)
                        game["spread"] = spread_val
                        game["home_implied_prob"] = _spread_to_moneyline_prob(-spread_val)
                    except (ValueError, TypeError):
                        pass
                if ou is not None:
                    try:
                        game["over_under"] = float(ou)
                    except (ValueError, TypeError):
                        pass

            if game["home"] and game["away"]:
                games.append(game)

        return games

    def detect_live_edges(self, polymarket_prices: Dict[str, float],
                          min_edge: float = 0.05) -> List[dict]:
        """Compare live ESPN odds to Polymarket prices.

        Args:
            polymarket_prices: {slug: price} for current Polymarket markets
            min_edge: minimum edge threshold

        Returns list of edge opportunities with:
            slug, poly_price, espn_prob, edge, game_status, scores
        """
        live_games = self.get_live_games()
        edges = []

        for game in live_games:
            if game["home_implied_prob"] is None:
                continue

            home = game["home"]
            away = game["away"]

            # Try to match against Polymarket slugs
            for slug, poly_price in polymarket_prices.items():
                if "cbb" not in slug:
                    continue
                parts = slug.split("-")
                if len(parts) < 4:
                    continue
                slug_t1 = parts[2]
                slug_t2 = parts[3]

                # Match teams
                h_match = self._abbr_matches(slug_t1, home["abbr"], home["name"])
                a_match = self._abbr_matches(slug_t2, away["abbr"], away["name"])
                h_match2 = self._abbr_matches(slug_t1, away["abbr"], away["name"])
                a_match2 = self._abbr_matches(slug_t2, home["abbr"], home["name"])

                if not ((h_match and a_match) or (h_match2 and a_match2)):
                    continue

                # Determine which team the slug is pricing
                # Slug first team = the team being bet on
                if self._abbr_matches(slug_t1, home["abbr"], home["name"]):
                    espn_prob = game["home_implied_prob"]
                else:
                    espn_prob = 1.0 - game["home_implied_prob"]

                edge = espn_prob - poly_price

                if abs(edge) >= min_edge:
                    edges.append({
                        "slug": slug,
                        "poly_price": poly_price,
                        "espn_prob": espn_prob,
                        "edge": edge,
                        "abs_edge": abs(edge),
                        "side": "buy" if edge > 0 else "sell",
                        "home_team": home["name"],
                        "away_team": away["name"],
                        "home_score": home["score"],
                        "away_score": away["score"],
                        "status": game["status"],
                        "spread": game.get("spread"),
                        "home_seed": home.get("seed", 0),
                        "away_seed": away.get("seed", 0),
                    })

                    # Record history
                    self._history.setdefault(slug, []).append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "espn_prob": espn_prob,
                        "poly_price": poly_price,
                        "edge": edge,
                        "status": game["status"],
                        "score": f"{away['score']}-{home['score']}",
                    })

        edges.sort(key=lambda x: x["abs_edge"], reverse=True)
        return edges

    def _abbr_matches(self, slug_abbr: str, espn_abbr: str, full_name: str) -> bool:
        """Match a Polymarket slug abbreviation to an ESPN team."""
        if slug_abbr == espn_abbr:
            return True
        known = TEAM_ABBREVS.get(slug_abbr, "")
        if known and known in full_name.lower():
            return True
        return False

    def get_seed_edge(self, higher_seed: int, lower_seed: int) -> Optional[float]:
        """Get historical win rate for a seed matchup.

        Returns the probability that the higher (better) seed wins.
        """
        key = (min(higher_seed, lower_seed), max(higher_seed, lower_seed))
        return SEED_WIN_RATES.get(key)

    def get_history(self, slug: str) -> List[dict]:
        """Get live edge tracking history for a slug."""
        return self._history.get(slug, [])
