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


# League-scoped team fragments, keyed by the ABBR_MAP-normalized ESPN
# abbreviation the recorder uses (== Polymarket's where verified live).
# Nicknames are unique WITHIN a league, which sidesteps the flat-dict
# collisions: "Las Vegas Aces" contains NHL's "vegas" fragment (vgk),
# "Chicago Cubs" contains NBA's "chicago" (chi), "Golden State Valkyries"
# contains "golden state" (gs) — city-first matching returned the wrong
# league's abbreviation for all of them.
LEAGUE_TEAM_FRAGMENTS = {
    "icehockey_nhl": {
        # Polymarket US codes (was/nas/veg/mon differ from ESPN's wsh/nsh/vgk/mtl).
        "ana": "ducks", "bos": "bruins", "buf": "sabres", "car": "hurricanes",
        "cbj": "blue jackets", "cgy": "flames", "chi": "blackhawks", "col": "avalanche",
        "dal": "stars", "det": "red wings", "edm": "oilers", "fla": "panthers",
        "la": "kings", "min": "wild", "mon": "canadiens", "nas": "predators",
        "nj": "devils", "nyi": "islanders", "nyr": "rangers", "ott": "senators",
        "phi": "flyers", "pit": "penguins", "sea": "kraken", "sj": "sharks",
        "stl": "blues", "tb": "lightning", "tor": "maple leafs", "uta": "mammoth",
        "van": "canucks", "veg": "golden knights", "was": "capitals", "wpg": "jets",
    },
    "americanfootball_nfl": {
        # Polymarket slug abbreviations (ESPN's "wsh" is "was" here — see
        # bot.leagues.ABBR_MAP). Nicknames are unique within the NFL.
        "ari": "cardinals", "atl": "falcons", "bal": "ravens", "buf": "bills",
        "car": "panthers", "chi": "bears", "cin": "bengals", "cle": "browns",
        "dal": "cowboys", "den": "broncos", "det": "lions", "gb": "packers",
        "hou": "texans", "ind": "colts", "jax": "jaguars", "kc": "chiefs",
        "lv": "raiders", "lac": "chargers", "lar": "rams", "mia": "dolphins",
        "min": "vikings", "ne": "patriots", "no": "saints", "nyg": "giants",
        "nyj": "jets", "phi": "eagles", "pit": "steelers", "sf": "49ers",
        "sea": "seahawks", "tb": "buccaneers", "ten": "titans", "was": "commanders",
    },
    "baseball_mlb": {
        "ari": "diamondbacks", "atl": "braves", "bal": "orioles",
        "bos": "red sox", "chc": "cubs", "cws": "white sox",
        "cin": "reds", "cle": "guardians", "col": "rockies",
        "det": "tigers", "hou": "astros", "kc": "royals",
        "laa": "angels", "lad": "dodgers", "mia": "marlins",
        "mil": "brewers", "min": "twins", "nym": "mets",
        "nyy": "yankees", "oak": "athletics", "phi": "phillies",
        "pit": "pirates", "sd": "padres", "sea": "mariners",
        "sf": "giants", "stl": "cardinals", "tb": "rays",
        "tex": "rangers", "tor": "blue jays", "wsh": "nationals",
    },
    "basketball_wnba": {
        "atl": "dream", "chi": "sky", "conn": "sun", "dal": "wings",
        "gsv": "valkyries", "ind": "fever", "la": "sparks",
        "las": "aces", "min": "lynx", "nyl": "liberty",
        "phx": "mercury", "por": "fire", "sea": "storm",
        "tor": "tempo", "wsh": "mystics",
    },
    "soccer_usa_mls": {
        # Pinnacle/ESPN use distinctive club names; extend as matches surface
        "atl": "atlanta united", "atx": "austin", "clt": "charlotte fc",
        "chi": "chicago fire", "cin": "fc cincinnati", "col": "colorado rapids",
        "clb": "columbus crew", "dal": "fc dallas", "dc": "d.c. united",
        "hou": "dynamo", "skc": "sporting kansas city", "la": "la galaxy",
        "lafc": "lafc", "mia": "inter miami", "min": "minnesota united",
        "mtl": "montr", "nsh": "nashville", "ne": "new england",
        "nyc": "new york city", "rbny": "red bulls", "orl": "orlando city",
        "phi": "philadelphia union", "por": "timbers", "rsl": "real salt lake",
        "sd": "san diego fc", "sj": "earthquakes", "sea": "sounders",
        "stl": "st. louis city", "tor": "toronto fc", "van": "whitecaps",
    },
}


# College football fragments = ESPN `location` ("Oregon State", "Miami",
# "Miami (OH)") from configs/cfb_teams.json. Matching below prefers an exact
# name, then the LONGEST fragment, so "ohio" never claims "ohio state".
from bot.leagues import CFB_TEAMS as _CFB_TEAMS
LEAGUE_TEAM_FRAGMENTS["americanfootball_ncaaf"] = {
    code: v["location"].lower() for code, v in _CFB_TEAMS.items() if v.get("location")
}


def fighter_code(full_name: str) -> str:
    """Polymarket's UFC fighter code: first 3 of first name + first 3 of last.

    Verified against tonight's card (2026-07-11): "Max Holloway" -> maxhol,
    "Conor McGregor" -> conmcg (slug aec-ufc-maxhol-conmcg-2026-07-11).
    """
    parts = [p for p in full_name.lower().replace(".", "").split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0][:6]
    return (parts[0][:3] + parts[-1][:3])


def _match_abbr(full_name: str, sport_key: str = "") -> str:
    """Reverse-lookup: find the abbreviation for a full team name.

    League-scoped nickname fragments take precedence when the caller knows
    the sport; MMA derives the fighter code from the name (no fixed roster);
    the flat cross-league TEAM_ABBREVS remains the fallback for the original
    NBA/NHL/NCAA paths.
    """
    name = full_name.lower()
    if sport_key == "mma_mixed_martial_arts":
        return fighter_code(full_name)
    league_fragments = LEAGUE_TEAM_FRAGMENTS.get(sport_key)
    if league_fragments:
        clean = name.replace("é", "e").strip()
        for abbr, fragment in league_fragments.items():
            if fragment and fragment == clean:
                return abbr
        best = ""
        best_len = 0
        for abbr, fragment in league_fragments.items():
            if fragment and fragment in clean and len(fragment) > best_len:
                best, best_len = abbr, len(fragment)
        if best:
            return best
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
    # Summer sports (verified live 2026-07-11: mlb=27 ML markets, wnba=5).
    # No MLS custom page exists — MLS coverage comes from Pinnacle + ESPN.
    "baseball_mlb": "mlb",
    "basketball_wnba": "wnba",
    # NFL custom page verified live 2026-09-20: 15 moneylines, 15 spreads,
    # 15 totals, inPlay flag set on in-progress games.
    "americanfootball_nfl": "nfl",
    # College football page verified 2026-09-20: 58 spreads, inPlay flags.
    "americanfootball_ncaaf": "ncaaf",
}


class FanDuelClient:
    """Fetches odds from FanDuel's public API."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}
        self.name = "fanduel"

    def _fetch(self, sport_id: str, max_age: float = 300.0) -> Optional[dict]:
        from bot.http_cache import get_json
        return get_json(FANDUEL_BASE,
                        params={"page": "CUSTOM", "customPageId": sport_id,
                                "_ak": "FhMFpcPWXMeyZxOx"},
                        headers={"User-Agent": "Mozilla/5.0"}, max_age=max_age)

    def get_odds(self, sport_key: str, max_age: Optional[float] = None) -> List[dict]:
        """Get moneyline odds for a sport. Returns list of event dicts.

        Each dict: {home_team, away_team, home_prob, away_prob, book, live}.
        max_age (seconds) overrides the cache TTL — in-play callers pass ~20.
        """
        ttl = self.cache_ttl if max_age is None else max_age
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < ttl:
                return data

        fd_sport = FANDUEL_SPORTS.get(sport_key)
        if not fd_sport:
            return []

        raw = self._fetch(fd_sport, ttl)
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
            if mkt.get("marketStatus") not in (None, "OPEN"):
                continue  # suspended (common in-play after big plays): no quote
            event_id = str(mkt.get("eventId", ""))
            ev_info = event_map.get(event_id)
            if not ev_info:
                continue

            runners = mkt.get("runners", [])
            if len(runners) < 2:
                continue

            team_probs = {}
            for runner in runners:
                if runner.get("runnerStatus") not in (None, "ACTIVE"):
                    continue
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
                "live": bool(mkt.get("inPlay")),
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
    # Discovered live 2026-07-11 via /0.1/sports/{id}/leagues
    "baseball_mlb": 246,
    "basketball_wnba": 578,
    "soccer_usa_mls": 2663,
    "mma_mixed_martial_arts": 1624,   # UFC (Pinnacle serves live fight lines)
    # Football: sport 15 → NFL 889, NCAA 880 (discovered live 2026-09-20)
    "americanfootball_nfl": 889,
    "americanfootball_ncaaf": 880,
}


class PinnacleClient:
    """Fetches odds from Pinnacle's public guest API. Pinnacle is a sharp book."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}
        self.name = "pinnacle"

    def _get(self, url: str, max_age: float = 300.0) -> Optional[dict]:
        from bot.http_cache import get_json
        return get_json(url, headers={"User-Agent": "Mozilla/5.0"}, max_age=max_age)

    def get_odds(self, sport_key: str, max_age: Optional[float] = None) -> List[dict]:
        """Get moneyline odds. Returns list of event dicts (live=True for
        in-play child matchups). max_age overrides the cache TTL."""
        ttl = self.cache_ttl if max_age is None else max_age
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < ttl:
                return data

        league_id = PINNACLE_LEAGUES.get(sport_key)
        if not league_id:
            return []

        # Fetch matchups
        matchups_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/matchups", ttl)
        if not matchups_raw:
            return []

        # Fetch markets
        markets_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight", ttl)
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
                matchup_map[mid] = {"home_team": home, "away_team": away,
                                    "live": bool(m.get("isLive"))}

        # Find moneyline markets (type=moneyline, period=0, isAlternate=false)
        ml_odds = {}  # matchup_id → {home_price, away_price}
        for mkt in markets:
            if mkt.get("type") != "moneyline":
                continue
            if mkt.get("period") != 0:
                continue
            if mkt.get("isAlternate", False):
                continue
            if mkt.get("status") not in (None, "open"):
                continue  # suspended in-play

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
                "live": teams.get("live", False),
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
        self.actionnetwork = ActionNetworkClient(cache_ttl=cache_ttl)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, dict]] = {}

    def get_all_odds(self, sport_key: str, max_age: Optional[float] = None) -> Dict[str, List[dict]]:
        """Fetch odds from all books for a sport.

        Returns: {game_key: [odds_from_each_book]}
        where game_key is a normalized "away @ home" string. max_age (s)
        overrides the cache TTL for every source (in-play callers pass ~20).
        """
        ttl = self.cache_ttl if max_age is None else max_age
        now = time.time()
        cache_key = f"all_{sport_key}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < ttl:
                return data

        all_events: List[dict] = []

        # FanDuel
        try:
            fd = self.fanduel.get_odds(sport_key, max_age=max_age)
            all_events.extend(fd)
        except Exception:
            pass

        # Pinnacle
        try:
            pin = self.pinnacle.get_odds(sport_key, max_age=max_age)
            all_events.extend(pin)
        except Exception:
            pass

        # Action Network: DraftKings / BetMGM / Caesars / bet365 / BetRivers
        # (pregame rows before kickoff, in-play rows during the game)
        try:
            all_events.extend(self.actionnetwork.get_odds(sport_key, max_age=max_age))
        except Exception:
            pass

        # Group by game (league-aware abbr matching), ONE entry per book —
        # FanDuel sometimes lists duplicate moneyline markets for a game,
        # which inflated num_books (observed live: a 3-book game reporting 5)
        # and over-weighted that book in the consensus.
        games: Dict[str, List[dict]] = {}
        for ev in all_events:
            key = self._game_key(ev["home_team"], ev["away_team"], sport_key)
            if not key:
                continue
            entries = games.setdefault(key, [])
            book = ev.get("book", "unknown")
            existing = next((e for e in entries if e.get("book", "unknown") == book), None)
            if existing is None:
                entries.append(ev)
            elif ev.get("live") and not existing.get("live"):
                # One entry per book, but an IN-PLAY line supersedes the
                # pregame line: a live quote only exists while the game is
                # on, and the pregame number is stale the moment it starts.
                # This is the live moneyline source for leagues ESPN has no
                # win-probability model for (NHL).
                entries[entries.index(existing)] = ev

        self._cache[cache_key] = (now, games)
        return games

    def live_consensus(self, sport_key: str, token0_abbr: str, other_abbr: str,
                       max_age: float = 20.0) -> Optional[dict]:
        """In-play consensus that `token0_abbr` wins, from LIVE quotes only.

        Polymarket's aec- token 0 is the away team (slug parts[2]); books key
        games away@home, so a reversed match (neutral site) reads home_prob.
        Only entries flagged live count (Pinnacle live child matchups,
        FanDuel inPlay markets, Action Network in-play rows under 150s old);
        a pregame line during a game is stale and never used. Sources are
        refetched when older than max_age. Pinnacle counts 1.5x.
        """
        games = self.get_all_odds(sport_key, max_age=max_age)
        direct, flipped = f"{token0_abbr}@{other_abbr}", f"{other_abbr}@{token0_abbr}"
        flip = False
        entries = games.get(direct)
        if not entries:
            entries, flip = games.get(flipped), True
        if not entries:
            return None
        live = [e for e in entries if e.get("live")]
        if not live:
            return None
        weighted, weight_sum, sharp, probs = 0.0, 0.0, [], []
        for e in live:
            p = e["home_prob"] if flip else e["away_prob"]
            w = self.SHARP_WEIGHT if e.get("book") in self.SHARP_BOOKS else 1.0
            weighted += w * p
            weight_sum += w
            probs.append(p)
            if e.get("book") in self.SHARP_BOOKS:
                sharp.append(p)
        return {
            "prob": weighted / weight_sum,
            "num_books": len(live),
            "books": [e.get("book", "unknown") for e in live],
            "sharp_prob": (sum(sharp) / len(sharp)) if sharp else None,
            "spread": (max(probs) - min(probs)) if len(probs) > 1 else 0.0,
        }

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

    def _game_key(self, home: str, away: str, sport_key: str = "") -> str:
        """Normalize game into a matchable key using abbreviations."""
        h_abbr = _match_abbr(home, sport_key)
        a_abbr = _match_abbr(away, sport_key)
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
            h = _match_abbr(game["home_team"], sport_key)
            a = _match_abbr(game["away_team"], sport_key)
            # Cross-check both orderings
            if (h == home_abbr and a == away_abbr) or (h == away_abbr and a == home_abbr):
                return game
        return None


# ---------------------------------------------------------------------------
# Action Network (multi-book: DraftKings, BetMGM, Caesars, bet365, BetRivers).
# Unofficial JSON the Action Network app itself loads; no auth. Each game
# carries per-book odds rows by `type`: "game" (pregame full-game line, frozen
# at kickoff) and "live" (the in-play line, refreshed through the game —
# verified on IND@KC 2026-09-20: live rows stamped minutes before the end).
# Scheduled games use the "game" rows; in-progress games use "live" rows no
# older than AN_LIVE_MAX_AGE seconds, tagged live=True so the aggregator and
# LinesCache treat them as in-play quotes.
# ---------------------------------------------------------------------------

AN_BASE = "https://api.actionnetwork.com/web/v1/scoreboard"
AN_UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
# FanDuel (69) is excluded: the direct FanDuel client already supplies it (live-aware).
AN_BOOKS = {68: "an_draftkings", 75: "an_betmgm", 123: "an_caesars", 79: "an_bet365", 71: "an_betrivers"}
AN_LIVE_MAX_AGE = 150   # seconds; a live row older than this is a stale line, not a quote
AN_LIVE_TTL = 25        # refetch cadence while any game is in progress
AN_LEAGUES = {
    "americanfootball_nfl": ("nfl", ""),
    "americanfootball_ncaaf": ("ncaaf", "&division=FBS"),
    "icehockey_nhl": ("nhl", ""),
}


class ActionNetworkClient:
    """Pregame moneylines, spreads and totals per book from Action Network."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self.name = "actionnetwork"
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}

    def _fetch(self, sport_key: str, max_age: float = 300.0) -> List[dict]:
        from bot.http_cache import get_json
        league = AN_LEAGUES.get(sport_key)
        if not league:
            return []
        import datetime as _dt
        path, extra = league
        games: List[dict] = []
        # nfl/ncaaf return the whole week for any date in it; nhl is per day.
        days = [0, 1] if path == "nhl" else [0]
        seen = set()
        for d in days:
            date = (_dt.datetime.utcnow() - _dt.timedelta(hours=5) + _dt.timedelta(days=d)).strftime("%Y%m%d")
            url = (f"{AN_BASE}/{path}?bookIds={','.join(str(b) for b in AN_BOOKS)}"
                   f"&date={date}&periods=event{extra}")
            try:
                payload = get_json(url, headers={"User-Agent": AN_UA}, max_age=max_age)
                if not isinstance(payload, dict):
                    continue
                for g in payload.get("games", []):
                    if g.get("id") in seen:
                        continue
                    seen.add(g.get("id"))
                    games.append(g)
            except Exception:
                continue
        return games

    @staticmethod
    def _row_age(o: dict, now: float) -> Optional[float]:
        """Seconds since the row's `inserted` stamp (ISO 8601, UTC)."""
        import datetime as _dt
        try:
            ts = _dt.datetime.fromisoformat(str(o.get("inserted", "")).replace("Z", "+00:00"))
            return now - ts.timestamp()
        except (TypeError, ValueError):
            return None

    def parse_games(self, raw_games: List[dict], now: Optional[float] = None) -> List[dict]:
        """Normalize raw games: scheduled → `game` rows; in progress → fresh `live` rows."""
        now = now or time.time()
        out = []
        for g in raw_games:
            status = g.get("status")
            if status == "scheduled":
                wanted, live = "game", False
            elif status == "inprogress":
                wanted, live = "live", True
            else:
                continue
            teams = {t.get("id"): t for t in g.get("teams", [])}
            home = teams.get(g.get("home_team_id"), {}).get("full_name", "")
            away = teams.get(g.get("away_team_id"), {}).get("full_name", "")
            if not home or not away:
                continue
            rows = []
            for o in g.get("odds", []):
                if o.get("type") != wanted or o.get("book_id") not in AN_BOOKS:
                    continue
                if live:
                    age = self._row_age(o, now)
                    if age is None or age > AN_LIVE_MAX_AGE:
                        continue  # stale in-play line: no quote is safer than an old one
                rows.append(o)
            if rows:
                out.append({"home_team": home, "away_team": away, "start": g.get("start_time", ""),
                            "live": live, "rows": rows})
        return out

    def games(self, sport_key: str, max_age: Optional[float] = None) -> List[dict]:
        """Normalized games with per-book odds rows (cached; short TTL while live)."""
        now = time.time()
        cached = self._cache.get(sport_key)
        if max_age is not None:
            ttl = max_age
        elif cached and any(g.get("live") for g in cached[1]):
            ttl = AN_LIVE_TTL
        else:
            ttl = self.cache_ttl
        if cached and now - cached[0] < ttl:
            return cached[1]
        out = self.parse_games(self._fetch(sport_key, ttl), now)
        self._cache[sport_key] = (now, out)
        return out

    @staticmethod
    def _suspended(o: dict, *fields) -> bool:
        """Action Network flags an unavailable line with line_status 2."""
        status = o.get("line_status") or {}
        return any(status.get(f) == 2 for f in fields)

    def get_odds(self, sport_key: str, max_age: Optional[float] = None) -> List[dict]:
        """Moneyline events, one per book per game (same shape as FanDuel/Pinnacle)."""
        results = []
        for g in self.games(sport_key, max_age=max_age):
            for o in g["rows"]:
                mh, ma = o.get("ml_home"), o.get("ml_away")
                if not mh or not ma or self._suspended(o, "ml_home", "ml_away"):
                    continue
                hp, ap = american_to_prob(int(mh)), american_to_prob(int(ma))
                tot = hp + ap
                if tot <= 0:
                    continue
                results.append({"home_team": g["home_team"], "away_team": g["away_team"],
                                "home_prob": hp / tot, "away_prob": ap / tot,
                                "book": AN_BOOKS[o["book_id"]], "live": g["live"]})
        return results

    def line_quotes(self, sport_key: str, max_age: Optional[float] = None):
        """(game key, kind, quote) tuples for LinesCache: away spread + total per book."""
        for g in self.games(sport_key, max_age=max_age):
            key_a, key_h = _match_abbr(g["away_team"], sport_key), _match_abbr(g["home_team"], sport_key)
            if not key_a or not key_h:
                continue
            key = f"{key_a}@{key_h}"
            for o in g["rows"]:
                book = AN_BOOKS[o["book_id"]]
                sa, pa, ph = o.get("spread_away"), o.get("spread_away_line"), o.get("spread_home_line")
                if sa is not None and pa and ph and not self._suspended(o, "spread_away", "spread_home"):
                    p_away = american_to_prob(int(pa)); p_home = american_to_prob(int(ph))
                    yield key, "spread", {"book": book, "kind": "spread", "points": float(sa),
                                          "p": p_away / (p_away + p_home), "main": True, "live": g["live"]}
                t, po, pu = o.get("total"), o.get("over"), o.get("under")
                if t is not None and po and pu and not self._suspended(o, "over", "under"):
                    p_o = american_to_prob(int(po)); p_u = american_to_prob(int(pu))
                    yield key, "total", {"book": book, "kind": "total", "points": float(t),
                                         "p": p_o / (p_o + p_u), "main": True, "live": g["live"]}
