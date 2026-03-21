"""Free sportsbook odds clients: FanDuel, Pinnacle, ESPN/DraftKings.

Each client fetches current moneyline odds, normalizes to implied probability,
maps team names, and caches with a 5-minute TTL.
"""

import json
import math
import time
import requests
from typing import Dict, List, Optional, Tuple

from bot.signals.odds_api import TEAM_ABBREVS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def american_to_prob(odds: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 0.5


def _normalize_team(name: str) -> str:
    """Lowercase, strip common suffixes."""
    return name.lower().strip()


def _match_abbr(full_name: str) -> str:
    """Reverse-lookup: find the abbreviation for a full team name."""
    name = full_name.lower()
    for abbr, fragment in TEAM_ABBREVS.items():
        if fragment and fragment in name:
            return abbr
    return ""


# ---------------------------------------------------------------------------
# FanDuel client
# ---------------------------------------------------------------------------

FANDUEL_BASE = "https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page"
FANDUEL_SPORTS = {
    "basketball_nba": "nba",
    "basketball_ncaab": "ncaab",
    "icehockey_nhl": "nhl",
}


class FanDuelClient:
    """Fetches odds from FanDuel's public API."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}
        self.name = "fanduel"

    def _fetch(self, sport_id: str) -> Optional[dict]:
        try:
            resp = requests.get(
                FANDUEL_BASE,
                params={
                    "page": "CUSTOM",
                    "customPageId": sport_id,
                    "_ak": "FhMFpcPWXMeyZxOx",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            return json.loads(resp.text, strict=False)
        except Exception:
            return None

    def get_odds(self, sport_key: str) -> List[dict]:
        """Get moneyline odds for a sport. Returns list of event dicts.

        Each dict: {home_team, away_team, home_prob, away_prob, book}
        """
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < self.cache_ttl:
                return data

        fd_sport = FANDUEL_SPORTS.get(sport_key)
        if not fd_sport:
            return []

        raw = self._fetch(fd_sport)
        if not raw:
            return []

        results = self._parse(raw)
        self._cache[sport_key] = (now, results)
        return results

    def _parse(self, data: dict) -> List[dict]:
        """Parse FanDuel response into normalized odds."""
        attachments = data.get("attachments", {})
        events = attachments.get("events", {})
        markets = attachments.get("markets", {})

        # Build event lookup
        event_map = {}
        for eid, ev in events.items():
            name = ev.get("name", "")
            if " @ " in name or " v " in name:
                parts = name.replace(" v ", " @ ").split(" @ ")
                if len(parts) == 2:
                    event_map[str(ev.get("eventId", eid))] = {
                        "away_team": parts[0].strip(),
                        "home_team": parts[1].strip(),
                    }

        # Find moneyline markets
        results = []
        for mid, mkt in markets.items():
            if mkt.get("marketName", "").lower() not in ("moneyline", "money line"):
                continue
            event_id = str(mkt.get("eventId", ""))
            ev_info = event_map.get(event_id)
            if not ev_info:
                continue

            runners = mkt.get("runners", [])
            if len(runners) < 2:
                continue

            team_probs = {}
            for runner in runners:
                name = runner.get("runnerName", "")
                odds_data = runner.get("winRunnerOdds", {})
                am_odds = odds_data.get("americanDisplayOdds", {}).get("americanOdds")
                if am_odds is not None:
                    prob = american_to_prob(int(am_odds))
                    team_probs[name] = prob

            if len(team_probs) < 2:
                continue

            home = ev_info["home_team"]
            away = ev_info["away_team"]
            home_prob = team_probs.get(home, 0)
            away_prob = team_probs.get(away, 0)

            # Normalize
            total = home_prob + away_prob
            if total > 0:
                home_prob /= total
                away_prob /= total

            results.append({
                "home_team": home,
                "away_team": away,
                "home_prob": home_prob,
                "away_prob": away_prob,
                "book": self.name,
            })

        return results


# ---------------------------------------------------------------------------
# Pinnacle client
# ---------------------------------------------------------------------------

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
PINNACLE_LEAGUES = {
    "basketball_nba": 487,
    "basketball_ncaab": 493,
    "icehockey_nhl": 1456,
}


class PinnacleClient:
    """Fetches odds from Pinnacle's public guest API. Pinnacle is a sharp book."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}
        self.name = "pinnacle"

    def _get(self, url: str) -> Optional[dict]:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def get_odds(self, sport_key: str) -> List[dict]:
        """Get moneyline odds. Returns list of event dicts."""
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < self.cache_ttl:
                return data

        league_id = PINNACLE_LEAGUES.get(sport_key)
        if not league_id:
            return []

        # Fetch matchups
        matchups_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/matchups")
        if not matchups_raw:
            return []

        # Fetch markets
        markets_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight")
        if not markets_raw:
            return []

        results = self._parse(matchups_raw, markets_raw)
        self._cache[sport_key] = (now, results)
        return results

    def _parse(self, matchups: list, markets: list) -> List[dict]:
        """Parse Pinnacle matchups + markets into normalized odds."""
        # Build matchup lookup: id → {home, away}
        matchup_map = {}
        for m in matchups:
            mid = m.get("id")
            participants = m.get("participants", [])
            if len(participants) < 2:
                continue
            home = ""
            away = ""
            for p in participants:
                if p.get("alignment") == "home":
                    home = p.get("name", "")
                elif p.get("alignment") == "away":
                    away = p.get("name", "")
            if home and away:
                matchup_map[mid] = {"home_team": home, "away_team": away}

        # Find moneyline markets (type=moneyline, period=0, isAlternate=false)
        ml_odds = {}  # matchup_id → {home_price, away_price}
        for mkt in markets:
            if mkt.get("type") != "moneyline":
                continue
            if mkt.get("period") != 0:
                continue
            if mkt.get("isAlternate", False):
                continue

            mid = mkt.get("matchupId")
            prices = mkt.get("prices", [])
            for p in prices:
                designation = p.get("designation", "")
                price = p.get("price", 0)
                if not price:
                    continue
                if mid not in ml_odds:
                    ml_odds[mid] = {}
                ml_odds[mid][designation] = price

        results = []
        for mid, teams in matchup_map.items():
            odds = ml_odds.get(mid)
            if not odds:
                continue
            home_price = odds.get("home", 0)
            away_price = odds.get("away", 0)
            if not home_price or not away_price:
                continue

            home_prob = american_to_prob(int(home_price))
            away_prob = american_to_prob(int(away_price))

            total = home_prob + away_prob
            if total > 0:
                home_prob /= total
                away_prob /= total

            results.append({
                "home_team": teams["home_team"],
                "away_team": teams["away_team"],
                "home_prob": home_prob,
                "away_prob": away_prob,
                "book": self.name,
            })

        return results


# ---------------------------------------------------------------------------
# Multi-book aggregator
# ---------------------------------------------------------------------------

class MultiBookAggregator:
    """Aggregates odds across FanDuel, Pinnacle, and ESPN/DraftKings."""

    # Pinnacle is a sharp book — weight its line higher
    SHARP_BOOKS = {"pinnacle"}
    SHARP_WEIGHT = 1.5  # sharp books count 50% more in consensus

    def __init__(self, cache_ttl: int = 300):
        self.fanduel = FanDuelClient(cache_ttl=cache_ttl)
        self.pinnacle = PinnacleClient(cache_ttl=cache_ttl)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, dict]] = {}

    def get_all_odds(self, sport_key: str) -> Dict[str, List[dict]]:
        """Fetch odds from all books for a sport.

        Returns: {game_key: [odds_from_each_book]}
        where game_key is a normalized "away @ home" string.
        """
        now = time.time()
        cache_key = f"all_{sport_key}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return data

        all_events: List[dict] = []

        # FanDuel
        try:
            fd = self.fanduel.get_odds(sport_key)
            all_events.extend(fd)
        except Exception:
            pass

        # Pinnacle
        try:
            pin = self.pinnacle.get_odds(sport_key)
            all_events.extend(pin)
        except Exception:
            pass

        # Group by game
        games: Dict[str, List[dict]] = {}
        for ev in all_events:
            key = self._game_key(ev["home_team"], ev["away_team"])
            if key:
                games.setdefault(key, []).append(ev)

        self._cache[cache_key] = (now, games)
        return games

    def get_consensus(self, sport_key: str) -> List[dict]:
        """Get consensus odds across all books.

        Returns list of: {home_team, away_team, home_prob, away_prob,
                          num_books, books, sharp_home_prob, sharp_away_prob, spread}
        """
        games = self.get_all_odds(sport_key)
        results = []

        for key, events in games.items():
            if not events:
                continue

            home_team = events[0]["home_team"]
            away_team = events[0]["away_team"]
            books = []

            # Weighted average: sharp books count more
            weighted_home = 0.0
            weighted_away = 0.0
            weight_sum = 0.0
            sharp_home = 0.0
            sharp_away = 0.0
            sharp_count = 0
            home_probs = []
            away_probs = []

            for ev in events:
                book = ev.get("book", "unknown")
                hp = ev["home_prob"]
                ap = ev["away_prob"]
                books.append(book)
                home_probs.append(hp)
                away_probs.append(ap)

                w = self.SHARP_WEIGHT if book in self.SHARP_BOOKS else 1.0
                weighted_home += hp * w
                weighted_away += ap * w
                weight_sum += w

                if book in self.SHARP_BOOKS:
                    sharp_home += hp
                    sharp_away += ap
                    sharp_count += 1

            if weight_sum == 0:
                continue

            cons_home = weighted_home / weight_sum
            cons_away = weighted_away / weight_sum

            # Normalize
            total = cons_home + cons_away
            if total > 0:
                cons_home /= total
                cons_away /= total

            # Sharp consensus
            if sharp_count > 0:
                s_home = sharp_home / sharp_count
                s_away = sharp_away / sharp_count
                s_total = s_home + s_away
                if s_total > 0:
                    s_home /= s_total
                    s_away /= s_total
            else:
                s_home = cons_home
                s_away = cons_away

            # Spread: max - min probability across books
            spread = max(home_probs) - min(home_probs) if len(home_probs) > 1 else 0

            results.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_prob": cons_home,
                "away_prob": cons_away,
                "num_books": len(events),
                "books": books,
                "sharp_home_prob": s_home,
                "sharp_away_prob": s_away,
                "spread": spread,
                "book_details": events,
            })

        return results

    def _game_key(self, home: str, away: str) -> str:
        """Normalize game into a matchable key using abbreviations."""
        h_abbr = _match_abbr(home)
        a_abbr = _match_abbr(away)
        if h_abbr and a_abbr:
            return f"{a_abbr}@{h_abbr}"
        # Fallback: use lowercase fragments
        h = home.lower().split()[0] if home else ""
        a = away.lower().split()[0] if away else ""
        return f"{a}@{h}" if h and a else ""

    def find_game(self, sport_key: str, home_abbr: str, away_abbr: str) -> Optional[dict]:
        """Find consensus for a specific game by team abbreviations."""
        consensus = self.get_consensus(sport_key)
        for game in consensus:
            h = _match_abbr(game["home_team"])
            a = _match_abbr(game["away_team"])
            # Cross-check both orderings
            if (h == home_abbr and a == away_abbr) or (h == away_abbr and a == home_abbr):
                return game
        return None
