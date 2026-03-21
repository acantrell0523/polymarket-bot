"""Real-time sports data from ESPN for injury/lineup and game context signals."""

import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple


# ESPN API endpoints by sport
ESPN_ENDPOINTS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "basketball_ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
}

# Map from the-odds-api sport keys to ESPN keys
ODDS_TO_ESPN = {
    "basketball_nba": "basketball_nba",
    "basketball_ncaab": "basketball_ncaab",
}


class ESPNCache:
    """Caches ESPN scoreboard data per sport with TTL (default 5 minutes)."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, dict]] = {}  # sport_key -> (timestamp, data)

    def _fetch(self, sport_key: str) -> Optional[dict]:
        url = ESPN_ENDPOINTS.get(sport_key)
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def get_scoreboard(self, sport_key: str) -> Optional[dict]:
        """Get cached scoreboard, refreshing if stale."""
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < self.cache_ttl:
                return data
        data = self._fetch(sport_key)
        if data:
            self._cache[sport_key] = (now, data)
        return data

    def find_game(self, sport_key: str, home_abbr: str, away_abbr: str) -> Optional[dict]:
        """Find a specific game by team abbreviations."""
        scoreboard = self.get_scoreboard(sport_key)
        if not scoreboard:
            return None

        events = scoreboard.get("events", [])
        for event in events:
            competitors = []
            for comp in event.get("competitions", [{}])[0].get("competitors", []):
                team = comp.get("team", {})
                competitors.append({
                    "abbr": team.get("abbreviation", "").lower(),
                    "name": team.get("displayName", "").lower(),
                    "short": team.get("shortDisplayName", "").lower(),
                    "home_away": comp.get("homeAway", ""),
                    "record": comp.get("records", [{}])[0].get("summary", "") if comp.get("records") else "",
                    "score": comp.get("score", "0"),
                })

            abbrs = [c["abbr"] for c in competitors]
            names = [c["name"] for c in competitors]

            home_match = home_abbr in abbrs or any(home_abbr in n for n in names)
            away_match = away_abbr in abbrs or any(away_abbr in n for n in names)

            if home_match and away_match:
                competition = event.get("competitions", [{}])[0]
                status = event.get("status", {})
                return {
                    "event": event,
                    "competitors": competitors,
                    "status_type": status.get("type", {}).get("name", ""),
                    "status_detail": status.get("type", {}).get("detail", ""),
                    "is_live": status.get("type", {}).get("name") == "STATUS_IN_PROGRESS",
                    "is_final": status.get("type", {}).get("completed", False),
                    "venue": competition.get("venue", {}).get("fullName", ""),
                    "neutral_site": competition.get("neutralSite", False),
                    "date": event.get("date", ""),
                }

        return None

    def get_injuries(self, sport_key: str, team_abbr: str) -> List[dict]:
        """Extract injury info for a team from the scoreboard data."""
        scoreboard = self.get_scoreboard(sport_key)
        if not scoreboard:
            return []

        # ESPN scoreboard doesn't always include full injury reports,
        # but competition notes sometimes contain key injuries
        injuries = []
        for event in scoreboard.get("events", []):
            for comp in event.get("competitions", [{}])[0].get("competitors", []):
                team = comp.get("team", {})
                if team.get("abbreviation", "").lower() == team_abbr.lower():
                    # Check for injury headlines in competition notes
                    notes = event.get("competitions", [{}])[0].get("notes", [])
                    for note in notes:
                        if "injury" in note.get("headline", "").lower():
                            injuries.append({
                                "headline": note.get("headline", ""),
                                "type": "note",
                            })
        return injuries


class GameContextAnalyzer:
    """Analyzes game context for confidence modifiers."""

    def __init__(self, espn_cache: ESPNCache):
        self.espn_cache = espn_cache
        self._schedule_cache: Dict[str, Tuple[float, List[str]]] = {}  # team -> (ts, dates)

    def analyze(self, sport_key: str, home_abbr: str, away_abbr: str) -> dict:
        """Analyze game context and return modifier data.

        Returns dict with:
            home_advantage: float (confidence boost for home team)
            fatigue_home: bool (back-to-back for home)
            fatigue_away: bool (back-to-back for away)
            is_conference_tourney: bool
            home_record: str
            away_record: str
            context_modifier: float (overall confidence adjustment, -0.1 to +0.1)
        """
        result = {
            "home_advantage": 0.0,
            "fatigue_home": False,
            "fatigue_away": False,
            "is_conference_tourney": False,
            "home_record": "",
            "away_record": "",
            "context_modifier": 0.0,
            "is_live": False,
            "neutral_site": False,
        }

        game = self.espn_cache.find_game(sport_key, home_abbr, away_abbr)
        if not game:
            return result

        result["is_live"] = game.get("is_live", False)
        result["neutral_site"] = game.get("neutral_site", False)

        # Home advantage
        if not game.get("neutral_site", False):
            if sport_key == "basketball_nba":
                result["home_advantage"] = 0.03  # ~60% home win rate
            elif sport_key == "basketball_ncaab":
                result["home_advantage"] = 0.05  # ~65% home win rate
        else:
            result["home_advantage"] = 0.0

        # Records
        for comp in game.get("competitors", []):
            if comp["home_away"] == "home":
                result["home_record"] = comp.get("record", "")
            else:
                result["away_record"] = comp.get("record", "")

        # Conference tournament detection
        event = game.get("event", {})
        season_type = event.get("season", {}).get("type", 0)
        event_name = event.get("name", "").lower()
        if season_type == 3 or "tournament" in event_name or "conference" in event_name:
            result["is_conference_tourney"] = True

        # Back-to-back detection (NBA only — NCAA rarely plays B2B)
        if sport_key == "basketball_nba":
            result["fatigue_home"] = self._check_back_to_back(sport_key, home_abbr)
            result["fatigue_away"] = self._check_back_to_back(sport_key, away_abbr)

        # Compute overall context modifier
        modifier = 0.0
        # Home team boost
        modifier += result["home_advantage"]
        # Fatigue penalty for home
        if result["fatigue_home"]:
            modifier -= 0.02
        # Fatigue penalty for away (helps home)
        if result["fatigue_away"]:
            modifier += 0.01
        # Conference tournament — more variance, less predictable
        if result["is_conference_tourney"]:
            modifier *= 0.8  # reduce confidence

        result["context_modifier"] = round(modifier, 4)
        return result

    def _check_back_to_back(self, sport_key: str, team_abbr: str) -> bool:
        """Check if a team played yesterday (back-to-back game)."""
        scoreboard = self.espn_cache.get_scoreboard(sport_key)
        if not scoreboard:
            return False

        # Look for the team's game date and check if they played yesterday
        # ESPN scoreboard shows today's games; we check if the team is in
        # yesterday's completed games by looking at event dates
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        for event in scoreboard.get("events", []):
            event_date_str = event.get("date", "")
            if not event_date_str:
                continue
            try:
                event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00")).date()
            except (ValueError, TypeError):
                continue

            if event_date == yesterday:
                for comp in event.get("competitions", [{}])[0].get("competitors", []):
                    if comp.get("team", {}).get("abbreviation", "").lower() == team_abbr.lower():
                        return True
        return False
