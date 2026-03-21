"""External odds client using the-odds-api.com + ESPN fallback for consensus line comparison."""

import math
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

# ESPN API endpoints for fallback odds
ESPN_ODDS_ENDPOINTS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "basketball_ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
}


def _spread_to_moneyline_prob(spread: float) -> float:
    """Convert a point spread to an implied moneyline probability.

    Uses the empirical relationship between spread and win probability
    in NBA/NCAA basketball. A spread of 0 = 50%. Each point of spread
    is worth roughly 3-4% win probability.

    For NBA: P(win) ≈ 0.50 + spread * 0.033
    For NCAA: similar but slightly stronger home effect.
    """
    # Standard logistic model calibrated to NBA/NCAA data
    # spread > 0 means the team is favored
    k = 0.14  # steepness factor (calibrated to ~3.3% per point)
    prob = 1.0 / (1.0 + math.exp(-k * spread))
    return max(0.01, min(0.99, prob))


def _american_to_prob(moneyline: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if moneyline > 0:
        return 100.0 / (moneyline + 100)
    elif moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    return 0.5


def _fetch_espn_odds(sport_key: str) -> List[dict]:
    """Fetch odds from ESPN scoreboard API and convert to the-odds-api format.

    ESPN is free, no API key needed, and includes DraftKings lines.
    """
    url = ESPN_ODDS_ENDPOINTS.get(sport_key)
    if not url:
        return []

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    events = []
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        odds_list = comps.get("odds", [])

        if len(competitors) < 2:
            continue

        # Identify home and away teams
        home_team = ""
        away_team = ""
        for c in competitors:
            team = c.get("team", {})
            name = team.get("displayName", "")
            if c.get("homeAway") == "home":
                home_team = name
            else:
                away_team = name

        if not home_team or not away_team:
            continue

        # Build bookmaker odds from ESPN odds data
        bookmakers = []
        for odds in odds_list:
            provider = odds.get("provider", {}).get("name", "")
            spread = odds.get("spread")
            details = odds.get("details", "")

            home_ml = odds.get("homeTeamOdds", {}).get("moneyLine")
            away_ml = odds.get("awayTeamOdds", {}).get("moneyLine")

            # Build outcomes
            outcomes = []

            if home_ml is not None and away_ml is not None:
                # Use moneyline directly
                home_prob = _american_to_prob(int(home_ml))
                away_prob = _american_to_prob(int(away_ml))
                outcomes.append({"name": home_team, "price": 1.0 / home_prob if home_prob > 0 else 100})
                outcomes.append({"name": away_team, "price": 1.0 / away_prob if away_prob > 0 else 100})
            elif spread is not None:
                # Convert spread to implied probability
                try:
                    spread_val = float(spread)
                except (ValueError, TypeError):
                    continue

                # ESPN spread convention:
                #   positive = away team favored (home is underdog)
                #   negative = home team favored (away is underdog)
                # _spread_to_moneyline_prob expects:
                #   positive = team is favored → high probability
                # So for home team: negate ESPN's spread
                home_spread = -spread_val

                home_prob = _spread_to_moneyline_prob(home_spread)
                away_prob = 1.0 - home_prob
                outcomes.append({"name": home_team, "price": 1.0 / home_prob if home_prob > 0 else 100})
                outcomes.append({"name": away_team, "price": 1.0 / away_prob if away_prob > 0 else 100})
            else:
                continue

            if outcomes:
                bookmakers.append({
                    "key": provider.lower().replace(" ", ""),
                    "title": provider,
                    "markets": [{"key": "h2h", "outcomes": outcomes}],
                })

        if bookmakers:
            events.append({
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("date", ""),
                "bookmakers": bookmakers,
            })

    return events

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
    # NCAA basketball — March Madness / tournament teams
    "uk": "kentucky", "ala": "alabama", "kan": "kansas", "txtech": "texas tech",
    "vir": "virginia", "ucf": "ucf", "ucla": "ucla", "miaoh": "miami oh",
    "tenn": "tennessee", "fl": "florida", "arz": "arizona", "liub": "long island",
    "hofst": "hofstra", "wrght": "wright", "sanclr": "santa clara",
    "cabap": "cal baptist", "niowa": "northern iowa", "stjohn": "st. john",
    "missr": "missouri", "iowast": "iowa state", "tenst": "tennessee state",
    "akron": "akron", "vill": "villanova", "clmsn": "clemson", "iowa": "iowa",
    "pvam": "prairie view",
    # Major NCAA tournament teams (2026 bracket)
    "duke": "duke", "gonz": "gonzaga", "hou": "houston", "mich": "michigan",
    "mst": "michigan state", "lou": "louisville", "tex": "texas",
    "tcu": "tcu", "ark": "arkansas", "nebr": "nebraska", "vand": "vanderbilt",
    "conn": "connecticut", "uconn": "connecticut", "unc": "north carolina",
    "aub": "auburn", "purd": "purdue", "illini": "illinois", "ill": "illinois",
    "marq": "marquette", "baylor": "baylor", "wisc": "wisconsin",
    "creigh": "creighton", "oreg": "oregon", "okst": "oklahoma state",
    "slu": "saint louis", "vcu": "vcu", "day": "dayton", "hpnt": "high point",
    "wichi": "wichita", "librty": "liberty", "nevada": "nevada",
    "ncw": "wilmington", "txam": "texas a&m", "syr": "syracuse",
    "usc": "usc", "lsu": "lsu", "unlv": "unlv", "tulsa": "tulsa",
    # EPL
    "bha": "brighton", "liv": "liverpool", "bor": "bournemouth", "mnu": "manchester united",
    "mci": "manchester city", "ars": "arsenal", "che": "chelsea", "tot": "tottenham",
    "avl": "aston villa", "new": "newcastle", "whu": "west ham", "cry": "crystal palace",
    "ful": "fulham", "wol": "wolverhampton", "eve": "everton", "lei": "leicester",
    "not": "nottingham", "ips": "ipswich", "sou": "southampton",
}

# Major bookmakers to average
TARGET_BOOKS = {"fanduel", "draftkings", "betmgm", "pointsbetus", "bovada", "williamhill_us"}

# Sharp books — their lines are considered more accurate
SHARP_BOOKS = {"pinnacle", "circa"}


class OddsCache:
    """Caches odds per sport with a TTL to stay within API rate limits.

    Sources (in priority order):
    1. the-odds-api.com (paid, multi-book) — if API key available and not exhausted
    2. Multi-book aggregator (free): FanDuel + Pinnacle + ESPN/DraftKings
    """

    def __init__(self, api_key: str = "", cache_ttl: int = 300):
        self.api_key = api_key
        self.cache_ttl = cache_ttl  # seconds
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}  # sport_key → (timestamp, events)
        # Enable even without API key — free sources work without one
        self.enabled = True
        self._odds_api_available = bool(api_key)
        # Multi-book aggregator (FanDuel + Pinnacle)
        self._multi_book = None  # lazy init to avoid circular imports
        self._multi_book_cache: Dict[str, Tuple[float, Dict]] = {}

    def _get_multi_book(self):
        if self._multi_book is None:
            from bot.signals.book_scrapers import MultiBookAggregator
            self._multi_book = MultiBookAggregator(cache_ttl=self.cache_ttl)
        return self._multi_book

    def _fetch_sport(self, sport_key: str) -> List[dict]:
        """Fetch odds for a sport. Tries the-odds-api first, falls back to ESPN."""
        # Try the-odds-api if we have a key
        if self._odds_api_available:
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
                    events = resp.json()
                    if events:
                        return events
                # If 401/429 (quota exhausted), disable for this session
                if resp.status_code in (401, 429):
                    self._odds_api_available = False
            except Exception:
                pass

        # Fallback: ESPN (free, no key needed)
        espn_events = _fetch_espn_odds(sport_key)
        if espn_events:
            return espn_events

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

    def _get_multi_book_consensus(self, sport_key: str, home_abbr: str, away_abbr: str) -> Optional[Dict]:
        """Get consensus from multi-book aggregator (FanDuel + Pinnacle + ESPN)."""
        try:
            mb = self._get_multi_book()
            game = mb.find_game(sport_key, home_abbr, away_abbr)
            if not game:
                return None

            # Also add ESPN data if available
            espn_events = _fetch_espn_odds(sport_key)
            espn_match = None
            for ev in espn_events:
                h = ev.get("home_team", "").lower()
                a = ev.get("away_team", "").lower()
                hm = self._team_matches(home_abbr, h) or self._team_matches(home_abbr, a)
                am = self._team_matches(away_abbr, a) or self._team_matches(away_abbr, h)
                if hm and am:
                    # Extract ESPN probability
                    for bk in ev.get("bookmakers", []):
                        for mkt in bk.get("markets", []):
                            if mkt.get("key") != "h2h":
                                continue
                            for outcome in mkt.get("outcomes", []):
                                price = outcome.get("price", 0)
                                if price > 0:
                                    prob = 1.0 / price
                                    name = outcome.get("name", "")
                                    if self._team_matches(home_abbr, name.lower()):
                                        espn_match = {"home_prob": prob, "away_prob": 1.0 - prob}
                                    elif self._team_matches(away_abbr, name.lower()):
                                        espn_match = {"home_prob": 1.0 - prob, "away_prob": prob}
                    break

            # Merge ESPN into book_details if not already there
            # Only add ESPN if we have fewer than 2 books (ESPN spread-to-prob
            # is less accurate than direct moneyline from FanDuel/Pinnacle)
            books = list(game.get("books", []))
            book_details = list(game.get("book_details", []))
            if espn_match and "espn_dk" not in books and len(book_details) < 2:
                books.append("espn_dk")
                book_details.append({
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_prob": espn_match["home_prob"],
                    "away_prob": espn_match["away_prob"],
                    "book": "espn_dk",
                })

            # Recompute consensus with all sources
            total_w = 0
            w_home = 0
            w_away = 0
            sharp_home = 0
            sharp_away = 0
            sharp_n = 0
            all_home = []

            for bd in book_details:
                hp = bd["home_prob"]
                ap = bd["away_prob"]
                is_sharp = bd["book"] in ("pinnacle",)
                w = 1.5 if is_sharp else 1.0
                w_home += hp * w
                w_away += ap * w
                total_w += w
                all_home.append(hp)
                if is_sharp:
                    sharp_home += hp
                    sharp_away += ap
                    sharp_n += 1

            if total_w == 0:
                return None

            cons_home = w_home / total_w
            cons_away = w_away / total_w
            t = cons_home + cons_away
            if t > 0:
                cons_home /= t
                cons_away /= t

            if sharp_n > 0:
                sh = sharp_home / sharp_n
                sa = sharp_away / sharp_n
                st = sh + sa
                if st > 0:
                    sh /= st
                    sa /= st
            else:
                sh = cons_home
                sa = cons_away

            spread = (max(all_home) - min(all_home)) if len(all_home) > 1 else 0

            return {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "num_books": len(book_details),
                "probs": {
                    game["home_team"].lower(): cons_home,
                    game["away_team"].lower(): cons_away,
                },
                "sharp_probs": {
                    game["home_team"].lower(): sh,
                    game["away_team"].lower(): sa,
                },
                "spread": spread,
                "books_used": ",".join(books),
            }
        except Exception:
            return None

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

        # Try multi-book aggregator first (FanDuel + Pinnacle + ESPN)
        multi = self._get_multi_book_consensus(sport_key, home_abbr, away_abbr)
        if multi and multi.get("num_books", 0) >= 2:
            return multi

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

            # Extract odds from target bookmakers AND sharp books
            bookmaker_odds = []
            sharp_odds = []
            book_names = []
            for bookmaker in event.get("bookmakers", []):
                bk_key = bookmaker.get("key", "")
                is_target = bk_key in TARGET_BOOKS
                is_sharp = bk_key in SHARP_BOOKS
                if not is_target and not is_sharp:
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
                        if is_target:
                            bookmaker_odds.append(outcomes)
                            book_names.append(bk_key)
                        if is_sharp:
                            sharp_odds.append(outcomes)
                            if not is_target:
                                book_names.append(bk_key)

            if not bookmaker_odds:
                continue

            # Average across all target books (overall consensus)
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

            # Sharp book consensus (Pinnacle, Circa)
            sharp_probs = {}
            if sharp_odds:
                for team in all_teams:
                    s_probs = [bo[team] for bo in sharp_odds if team in bo]
                    if s_probs:
                        sharp_probs[team] = sum(s_probs) / len(s_probs)
                s_total = sum(sharp_probs.values())
                if s_total > 0:
                    sharp_probs = {k: v / s_total for k, v in sharp_probs.items()}

            # Map back to home/away/draw
            result = {
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "num_books": len(bookmaker_odds) + len(sharp_odds),
                "probs": avg_probs,
                "sharp_probs": sharp_probs,
                "spread": max(prob_ranges.values()) if prob_ranges else 0,
                "books_used": ",".join(book_names),
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
        # Direct abbreviation lookup (preferred — unambiguous)
        known = TEAM_ABBREVS.get(abbr, "")
        if known and known in full_lower:
            return True
        # Fallback: abbreviation must match the START of a word in the name
        # (avoids "tor" matching "sena-tor-s")
        if len(abbr) >= 3:
            words = full_lower.split()
            for word in words:
                if word.startswith(abbr):
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
