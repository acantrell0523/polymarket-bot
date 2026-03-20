"""External odds client using the-odds-api.com for consensus line comparison."""

import os
import time
import requests
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone


# Sport key mapping: Polymarket slug prefix → the-odds-api sport key
SPORT_MAP = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
    "cbb": "basketball_ncaab",
    "epl": "soccer_epl",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "mls": "soccer_usa_mls",
    "laliga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "seriea": "soccer_italy_serie_a",
    "ligue1": "soccer_france_ligue_one",
}

# Common team abbreviation → full name fragments for matching
TEAM_ABBREVS = {
    # NBA
    "atl": "atlanta", "bos": "boston", "bkn": "brooklyn", "cha": "charlotte",
    "chi": "chicago", "cle": "cleveland", "dal": "dallas", "den": "denver",
    "det": "detroit", "gs": "golden state", "hou": "houston", "ind": "indiana",
    "lac": "la clippers", "lal": "los angeles lakers", "mem": "memphis",
    "mia": "miami", "mil": "milwaukee", "min": "minnesota", "no": "new orleans",
    "ny": "new york knicks", "okc": "oklahoma", "orl": "orlando", "phi": "philadelphia",
    "phx": "phoenix", "por": "portland", "sac": "sacramento", "sa": "san antonio",
    "tor": "toronto", "uta": "utah", "was": "washington",
    # NHL
    "ana": "anaheim", "arz": "arizona", "buf": "buffalo", "cgy": "calgary",
    "car": "carolina", "col": "colorado", "cbj": "columbus", "edm": "edmonton",
    "fla": "florida", "la": "los angeles kings", "nj": "new jersey",
    "nyi": "new york islanders", "nyr": "new york rangers", "ott": "ottawa",
    "pit": "pittsburgh", "stl": "st. louis", "sj": "san jose", "sea": "seattle",
    "tb": "tampa bay", "van": "vancouver", "vgk": "vegas", "wpg": "winnipeg",
    # NCAA basketball (partial — common teams)
    "uk": "kentucky", "ala": "alabama", "kan": "kansas", "txtech": "texas tech",
    "vir": "virginia", "ucf": "ucf", "ucla": "ucla", "miaoh": "miami oh",
    "tenn": "tennessee", "fl": "florida", "arz": "arizona", "liub": "long island",
    "hofst": "hofstra", "wrght": "wright", "sanclr": "santa clara",
    "cabap": "cal baptist", "niowa": "northern iowa", "stjohn": "st. john",
    "missr": "missouri", "iowast": "iowa state", "tenst": "tennessee state",
    "akron": "akron", "vill": "villanova", "clmsn": "clemson", "iowa": "iowa",
    "pvam": "prairie view",
    # EPL
    "bha": "brighton", "liv": "liverpool", "bor": "bournemouth", "mnu": "manchester united",
    "mci": "manchester city", "ars": "arsenal", "che": "chelsea", "tot": "tottenham",
    "avl": "aston villa", "new": "newcastle", "whu": "west ham", "cry": "crystal palace",
    "ful": "fulham", "wol": "wolverhampton", "eve": "everton", "lei": "leicester",
    "not": "nottingham", "ips": "ipswich", "sou": "southampton",
}

# Major bookmakers to average
TARGET_BOOKS = {"fanduel", "draftkings", "betmgm", "pointsbetus", "bovada", "williamhill_us"}


class OddsCache:
    """Caches odds per sport with a TTL to stay within API rate limits."""

    def __init__(self, api_key: str, cache_ttl: int = 300):
        self.api_key = api_key
        self.cache_ttl = cache_ttl  # seconds
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}  # sport_key → (timestamp, events)
        self.enabled = bool(api_key)

    def _fetch_sport(self, sport_key: str) -> List[dict]:
        """Fetch odds for a sport from the-odds-api."""
        if not self.enabled:
            return []
        try:
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_events(self, sport_key: str) -> List[dict]:
        """Get cached events for a sport, refreshing if stale."""
        now = time.time()
        if sport_key in self._cache:
            cached_time, events = self._cache[sport_key]
            if now - cached_time < self.cache_ttl:
                return events

        events = self._fetch_sport(sport_key)
        self._cache[sport_key] = (now, events)
        return events

    def get_consensus_odds(self, slug: str) -> Optional[Dict]:
        """Look up consensus odds for a Polymarket market slug.

        Returns dict with:
            home_team, away_team, home_prob, away_prob, draw_prob (if applicable),
            num_books, spread (max - min probability across books)
        Or None if no match found.
        """
        sport_key, home_abbr, away_abbr = self._parse_slug(slug)
        if not sport_key:
            return None

        events = self.get_events(sport_key)
        if not events:
            return None

        # Find matching event
        for event in events:
            home = event.get("home_team", "").lower()
            away = event.get("away_team", "").lower()

            home_match = self._team_matches(home_abbr, home) or self._team_matches(home_abbr, away)
            away_match = self._team_matches(away_abbr, away) or self._team_matches(away_abbr, home)

            if not (home_match and away_match):
                continue

            # Extract odds from target bookmakers
            bookmaker_odds = []
            for bookmaker in event.get("bookmakers", []):
                bk_key = bookmaker.get("key", "")
                if bk_key not in TARGET_BOOKS:
                    continue
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {}
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "").lower()
                        price = outcome.get("price", 0)
                        if price > 0:
                            prob = 1.0 / price
                            outcomes[name] = prob
                    if outcomes:
                        bookmaker_odds.append(outcomes)

            if len(bookmaker_odds) < 2:
                continue

            # Average across books
            all_teams = set()
            for bo in bookmaker_odds:
                all_teams.update(bo.keys())

            avg_probs = {}
            prob_ranges = {}
            for team in all_teams:
                probs = [bo[team] for bo in bookmaker_odds if team in bo]
                if probs:
                    avg_probs[team] = sum(probs) / len(probs)
                    prob_ranges[team] = max(probs) - min(probs)

            # Normalize so probabilities sum to ~1
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}

            # Map back to home/away/draw
            result = {
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "num_books": len(bookmaker_odds),
                "probs": avg_probs,
                "spread": max(prob_ranges.values()) if prob_ranges else 0,
            }
            return result

        return None

    def _parse_slug(self, slug: str) -> Tuple[Optional[str], str, str]:
        """Extract sport key and team abbreviations from a Polymarket slug.

        Examples:
            aec-nba-atl-hou-2026-03-20 → (basketball_nba, atl, hou)
            atc-epl-bha-liv-2026-03-21-bha → (soccer_epl, bha, liv)
        """
        parts = slug.split("-")
        if len(parts) < 4:
            return None, "", ""

        # Sport is typically parts[1]
        sport_abbr = parts[1]
        sport_key = SPORT_MAP.get(sport_abbr)
        if not sport_key:
            return None, "", ""

        # Teams are parts[2] and parts[3]
        home_abbr = parts[2]
        away_abbr = parts[3]

        return sport_key, home_abbr, away_abbr

    @staticmethod
    def _team_matches(abbr: str, full_name: str) -> bool:
        """Check if a team abbreviation matches a full team name."""
        full_lower = full_name.lower()
        # Direct abbreviation lookup
        known = TEAM_ABBREVS.get(abbr, "")
        if known and known in full_lower:
            return True
        # Fallback: abbreviation is substring of full name
        if len(abbr) >= 3 and abbr in full_lower:
            return True
        return False

    def get_probability_for_slug(self, slug: str) -> Optional[Tuple[float, int]]:
        """Get the consensus probability for the outcome a Polymarket slug represents.

        For moneyline slugs like aec-nba-atl-hou-2026-03-20, returns P(home team wins).
        For draw slugs like atc-epl-bha-liv-2026-03-21-draw, returns P(draw).

        Returns (probability, num_books) or None if no match.
        """
        consensus = self.get_consensus_odds(slug)
        if not consensus:
            return None

        parts = slug.split("-")
        sport_abbr = parts[1] if len(parts) > 1 else ""
        probs = consensus["probs"]
        num_books = consensus["num_books"]

        # Check if this is a draw market
        if slug.endswith("-draw"):
            draw_prob = probs.get("draw", probs.get("Draw", None))
            if draw_prob is not None:
                return (draw_prob, num_books)
            return None

        # For moneyline: find the probability for the team in the slug
        # The team we're betting on is typically the home team (parts[2])
        # But we need to check the last part for specific outcome indicators
        home_abbr = parts[2] if len(parts) > 2 else ""
        away_abbr = parts[3] if len(parts) > 3 else ""

        # Check if slug ends with a team abbr (e.g., atc-epl-bha-liv-2026-03-21-bha)
        outcome_abbr = parts[-1] if len(parts) > 5 else home_abbr

        # Match against consensus teams
        for team_name, prob in probs.items():
            if team_name.lower() == "draw":
                continue
            if self._team_matches(outcome_abbr, team_name):
                return (prob, num_books)

        # Fallback: return home team probability
        home_team = consensus["home_team"].lower()
        for team_name, prob in probs.items():
            if self._team_matches(home_abbr, team_name):
                return (prob, num_books)

        return None
