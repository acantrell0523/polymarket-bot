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
    # Person sports: Polymarket encodes people as first3(first)+first3(last)
    # ("Max Holloway" -> maxhol, "Timo Legout" -> timleg; short last names
    # keep their full length: "Haoran Hu" -> haohu). Verified live for both
    # UFC (2026-07-11) and ITF tennis (2026-07-14).
    if sport_key == "mma_mixed_martial_arts" or sport_key in TENNIS_CIRCUIT_PREFIXES:
        return fighter_code(full_name)
    league_fragments = LEAGUE_TEAM_FRAGMENTS.get(sport_key)
    if league_fragments:
        for abbr, fragment in league_fragments.items():
            if fragment and fragment in name:
                return abbr
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

    def get_lines(self, sport_key: str) -> List[dict]:
        """Spread/total lines (best-effort; see _fanduel_lines_from_payload)."""
        now = time.time()
        ck = f"lines_{sport_key}"
        if ck in self._cache:
            ts, data = self._cache[ck]
            if now - ts < self.cache_ttl:
                return data
        fd_sport = FANDUEL_SPORTS.get(sport_key)
        if not fd_sport:
            return []
        raw = self._fetch(fd_sport)
        rows = _fanduel_lines_from_payload(raw) if raw else []
        self._cache[ck] = (now, rows)
        return rows

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


def _fanduel_lines_from_payload(data: dict) -> List[dict]:
    """Best-effort FanDuel spread/total extraction.

    UNVERIFIED against live game payloads (built during the All-Star break
    when only futures were listed) — shapes are guarded so mismatches
    return nothing instead of wrong lines. Verify on Friday's MLB slate.
    """
    rows: List[dict] = []
    try:
        attachments = data.get("attachments", {})
        events = attachments.get("events", {})
        markets = attachments.get("markets", {})
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
        for mid, mkt in markets.items():
            mname = mkt.get("marketName", "").lower()
            teams = event_map.get(str(mkt.get("eventId", "")))
            if not teams:
                continue
            runners = mkt.get("runners", [])
            if len(runners) != 2:
                continue

            def _prob(r):
                odds = r.get("winRunnerOdds", {}).get(
                    "americanDisplayOdds", {}).get("americanOdds")
                return american_to_prob(int(odds)) if odds is not None else None

            if "run line" in mname or "runline" in mname:
                away_r = next((r for r in runners
                               if teams["away_team"].lower() in r.get("runnerName", "").lower()), None)
                home_r = next((r for r in runners
                               if teams["home_team"].lower() in r.get("runnerName", "").lower()), None)
                if not away_r or not home_r:
                    continue
                pa, ph = _prob(away_r), _prob(home_r)
                hcp = away_r.get("handicap")
                if pa is None or ph is None or hcp is None or pa + ph <= 0:
                    continue
                rows.append({"book": "fanduel", **teams, "kind": "spread",
                             "away_points": float(hcp),
                             "prob_away": pa / (pa + ph)})
            elif mname.startswith("total") or "total runs" in mname:
                over_r = next((r for r in runners
                               if r.get("runnerName", "").lower().startswith("over")), None)
                under_r = next((r for r in runners
                                if r.get("runnerName", "").lower().startswith("under")), None)
                if not over_r or not under_r:
                    continue
                po, pu = _prob(over_r), _prob(under_r)
                hcp = over_r.get("handicap")
                if po is None or pu is None or hcp is None or po + pu <= 0:
                    continue
                rows.append({"book": "fanduel", **teams, "kind": "total",
                             "points": float(hcp),
                             "prob_over": po / (po + pu)})
    except Exception:
        return []
    return rows


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
}

# Tennis: Pinnacle has no stable league id — every tournament is its own
# league (e.g. "ITF Men Slobozia - R1", id 272345), rotating weekly. The
# client discovers current tournament ids from the sport-33 league list
# (cached 1h) and aggregates matchups across tournaments whose name matches
# the circuit prefix. Doubles tournaments are skipped (Polymarket lists
# singles).
TENNIS_SPORT_ID = 33
TENNIS_CIRCUIT_PREFIXES = {
    "tennis_itf_men": ("ITF Men",),
    "tennis_itf_women": ("ITF Women",),
}
TENNIS_MAX_TOURNAMENTS = 20   # bound the per-refresh API cost


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

    def _tennis_league_ids(self, sport_key: str) -> List[int]:
        """Current tournament league ids for a tennis circuit (cached 1h)."""
        now = time.time()
        cached = self._cache.get("_tennis_leagues")
        if cached and now - cached[0] < 3600:
            leagues = cached[1]
        else:
            leagues = self._get(
                f"{PINNACLE_BASE}/sports/{TENNIS_SPORT_ID}/leagues?all=false") or []
            self._cache["_tennis_leagues"] = (now, leagues)

        prefixes = TENNIS_CIRCUIT_PREFIXES.get(sport_key, ())
        ids = [
            lg["id"] for lg in leagues
            if any(lg.get("name", "").startswith(p) for p in prefixes)
            and "Doubles" not in lg.get("name", "")
            and lg.get("matchupCount", 0) > 0
        ]
        return ids[:TENNIS_MAX_TOURNAMENTS]

    def _fetch_league_raw(self, league_id: int):
        matchups_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/matchups")
        if not matchups_raw:
            return None, None
        markets_raw = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight")
        if not markets_raw:
            return None, None
        return matchups_raw, markets_raw

    def _fetch_league(self, league_id: int) -> List[dict]:
        matchups_raw, markets_raw = self._fetch_league_raw(league_id)
        if not matchups_raw:
            return []
        return self._parse(matchups_raw, markets_raw)

    def _build_matchup_map(self, matchups: list) -> Dict:
        matchup_map = {}
        for m in matchups:
            participants = m.get("participants", [])
            home = away = ""
            for pt in participants:
                if pt.get("alignment") == "home":
                    home = pt.get("name", "")
                elif pt.get("alignment") == "away":
                    away = pt.get("name", "")
            if home and away:
                matchup_map[m.get("id")] = {"home_team": home, "away_team": away}
        return matchup_map

    def _parse_lines(self, matchups: list, markets: list) -> List[dict]:
        """Extract spread/total lines (INCLUDING alternates — Polymarket
        lists a ladder of lines per game, so alternates are needed to match).

        Row shapes (probabilities de-vigged so each pair sums to 1):
          {"book","home_team","away_team","kind":"spread",
           "away_points": +1.5, "prob_away": P(away covers away_points)}
          {"book","home_team","away_team","kind":"total",
           "points": 8.0, "prob_over": P(total > points)}
        """
        matchup_map = self._build_matchup_map(matchups)
        rows: List[dict] = []
        for mkt in markets:
            if mkt.get("period") != 0 or mkt.get("status") not in (None, "open"):
                continue
            teams = matchup_map.get(mkt.get("matchupId"))
            if not teams:
                continue
            prices = {p.get("designation"): p for p in mkt.get("prices", [])}
            if mkt.get("type") == "spread":
                h, a = prices.get("home"), prices.get("away")
                if not h or not a:
                    continue
                ph = american_to_prob(int(h.get("price", 0)))
                pa = american_to_prob(int(a.get("price", 0)))
                if ph + pa <= 0:
                    continue
                rows.append({
                    "book": self.name, **teams, "kind": "spread",
                    "away_points": float(a.get("points", 0)),
                    "prob_away": pa / (ph + pa),
                })
            elif mkt.get("type") == "total":
                o, u = prices.get("over"), prices.get("under")
                if not o or not u:
                    continue
                po = american_to_prob(int(o.get("price", 0)))
                pu = american_to_prob(int(u.get("price", 0)))
                if po + pu <= 0:
                    continue
                rows.append({
                    "book": self.name, **teams, "kind": "total",
                    "points": float(o.get("points", 0)),
                    "prob_over": po / (po + pu),
                })
        return rows

    def get_lines(self, sport_key: str) -> List[dict]:
        """Spread/total lines for a sport (same league resolution as odds)."""
        now = time.time()
        ck = f"lines_{sport_key}"
        if ck in self._cache:
            ts, data = self._cache[ck]
            if now - ts < self.cache_ttl:
                return data

        if sport_key in TENNIS_CIRCUIT_PREFIXES:
            league_ids = self._tennis_league_ids(sport_key)
        else:
            lid = PINNACLE_LEAGUES.get(sport_key)
            league_ids = [lid] if lid else []

        rows: List[dict] = []
        for lid in league_ids:
            try:
                matchups_raw, markets_raw = self._fetch_league_raw(lid)
                if matchups_raw:
                    rows.extend(self._parse_lines(matchups_raw, markets_raw))
            except Exception:
                continue
        self._cache[ck] = (now, rows)
        return rows

    def get_odds(self, sport_key: str) -> List[dict]:
        """Get moneyline odds. Returns list of event dicts."""
        now = time.time()
        if sport_key in self._cache:
            ts, data = self._cache[sport_key]
            if now - ts < self.cache_ttl:
                return data

        if sport_key in TENNIS_CIRCUIT_PREFIXES:
            results: List[dict] = []
            for lid in self._tennis_league_ids(sport_key):
                try:
                    results.extend(self._fetch_league(lid))
                except Exception:
                    continue
            self._cache[sport_key] = (now, results)
            return results

        league_id = PINNACLE_LEAGUES.get(sport_key)
        if not league_id:
            return []

        results = self._fetch_league(league_id)
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
            if any(e.get("book", "unknown") == book for e in entries):
                continue  # dedup: keep the first entry per book
            entries.append(ev)

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

    def get_lines(self, sport_key: str) -> List[dict]:
        """Spread/total line rows from all books (cached)."""
        now = time.time()
        ck = f"lines_{sport_key}"
        if ck in self._cache:
            ts, data = self._cache[ck]
            if now - ts < self.cache_ttl:
                return data
        rows: List[dict] = []
        for client in (self.fanduel, self.pinnacle):
            try:
                rows.extend(client.get_lines(sport_key))
            except Exception:
                pass
        self._cache[ck] = (now, rows)
        return rows

    def find_line(self, sport_key: str, away_abbr: str, home_abbr: str,
                  kind: str, line: float) -> Optional[Tuple[float, int]]:
        """Book probability for a derivative market's YES side.

        Polymarket semantics (verified 2026-07-14):
          spread: YES = FIRST-listed (away) team covers `line` (signed) —
                  matched against the away side's points on each book row.
          total:  YES = OVER `line`.

        Returns (mean de-vigged prob across DISTINCT books quoting exactly
        this line, num_books) or None.
        """
        probs: Dict[str, float] = {}   # book -> prob (dedup per book)
        for row in self.get_lines(sport_key):
            if row.get("kind") != kind:
                continue
            h = _match_abbr(row.get("home_team", ""), sport_key)
            a = _match_abbr(row.get("away_team", ""), sport_key)
            if not ((h == home_abbr and a == away_abbr)
                    or (h == away_abbr and a == home_abbr)):
                continue
            flipped = (h == away_abbr and a == home_abbr)
            book = row.get("book", "unknown")
            if kind == "spread":
                pts = row.get("away_points")
                prob = row.get("prob_away")
                if flipped:
                    # row's away is OUR home: our away's points/prob mirror
                    pts = -pts if pts is not None else None
                    prob = 1 - prob if prob is not None else None
                if pts is None or prob is None or abs(pts - line) > 0.01:
                    continue
                probs.setdefault(book, prob)
            else:  # total — orientation-independent
                if row.get("points") is None or abs(row["points"] - line) > 0.01:
                    continue
                if row.get("prob_over") is None:
                    continue
                probs.setdefault(book, row["prob_over"])
        if not probs:
            return None
        return sum(probs.values()) / len(probs), len(probs)

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
