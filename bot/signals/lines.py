"""Spread and total pricing for the asc-/tsc- game-market families.

The moneyline engine (odds_value / live_win_prob) prices P(team wins). It
has nothing to say about "Will the Colts cover -3.5" or "Will the total be
more than 44.5", so those markets were either mispriced (moneyline consensus
compared against a spread price) or excluded. This module prices them.

Slug families on polymarket.us (verified live 2026-09-20):

    asc-nfl-ind-kc-2026-09-20-neg-8pt5   token 0 = AWAY team (parts[2]) covers -8.5
    asc-nfl-ind-kc-2026-09-20-pos-2pt5   token 0 = AWAY team covers +2.5
    tsc-nfl-phi-ten-2026-09-20-total-44pt5   token 0 = OVER 44.5

The `outcomes` labels on these markets are NOT reliably ordered; the
`marketSides[0].team` / `.description` fields confirm token 0 is always the
away team's side of the spread, and always Over for totals. Period variants
(-1h-, -1q-, team totals -tt-) are separate markets and are not supported.

Model
-----
Final margin M = away − home is treated as Normal(mu, sigma) and the game
total T as Normal(mu_t, sigma_t), the standard closing-line model. Each
book quote (a line with two prices) is de-vigged and inverted into an
estimate of mu; quotes are averaged (Pinnacle 1.5x, alternate lines 0.5x)
so a book's whole ladder of alternates sharpens the estimate instead of
just the main number. When a book quotes the market's exact line, its
de-vigged price is blended 50/50 with the model (the book already priced
that number, including key-number effects the normal model ignores).

Live games use ONLY live quotes (Pinnacle live child matchups, FanDuel
inPlay markets) and shrink sigma by sqrt(fraction of game remaining); a
pregame line is stale the moment the game starts.

Sources are all free: Pinnacle guest API (main + alternate lines, live),
FanDuel custom page (main line, inPlay flag), ESPN scoreboard (pregame
main line, DraftKings, no prices — assumed −110 each side).
"""

import math
import time
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import requests

from bot.leagues import LEAGUES, scoreboard_url, game_seconds_remaining
from bot.signals.book_scrapers import (
    american_to_prob, _match_abbr,
    PINNACLE_BASE, PINNACLE_LEAGUES, FANDUEL_BASE, FANDUEL_SPORTS,
)
from utils.models import MarketSnapshot, Signal

# Pregame standard deviation (points) of (final margin, final total).
# NFL: margin ~13.5, total ~10 — the long-run closing-line residuals.
# Only leagues listed here get asc-/tsc- markets admitted into the scan.
SIGMA: Dict[str, Tuple[float, float]] = {
    "nfl": (13.5, 10.0),
    "cfb": (16.0, 12.5),   # wider outcomes than the NFL
    "nhl": (2.4, 2.0),     # low-scoring; the exact-line book blend does most of the work
    "nba": (12.0, 18.0),
    "cbb": (11.0, 14.0),
    "wnba": (11.0, 14.0),
}
SUPPORTED_LEAGUES = tuple(SIGMA)

LINE_PREFIXES = ("asc-", "tsc-")

_N = NormalDist()


# ---------------------------------------------------------------------------
# Slug parsing
# ---------------------------------------------------------------------------

def is_line_market(slug: str) -> bool:
    return str(slug).startswith(LINE_PREFIXES)


def parse_line_slug(slug: str) -> Optional[Dict]:
    """asc-/tsc- full-game slug → dict, or None for unsupported variants.

    Returns {kind: "spread"|"total", league, away, home, date, line} where
    `line` is the away team's spread (negative = away favored) or the total.
    """
    parts = str(slug).lower().split("-")
    if len(parts) != 9:
        return None
    fam, league, away, home = parts[0], parts[1], parts[2], parts[3]
    kind, num = parts[7], parts[8]
    try:
        value = float(num.replace("pt", "."))
    except ValueError:
        return None
    base = {"league": league, "away": away, "home": home,
            "date": "-".join(parts[4:7])}
    if fam == "asc" and kind in ("neg", "pos"):
        return {**base, "kind": "spread", "line": -value if kind == "neg" else value}
    if fam == "tsc" and kind == "total":
        return {**base, "kind": "total", "line": value}
    return None


# ---------------------------------------------------------------------------
# Normal-model math
# ---------------------------------------------------------------------------

def prob_away_covers(mu: float, sigma: float, line: float) -> float:
    """P(M + line > 0) with M = away − home ~ N(mu, sigma)."""
    return _N.cdf((mu + line) / sigma)


def prob_over(mu: float, sigma: float, line: float) -> float:
    return 1.0 - _N.cdf((line - mu) / sigma)


def mu_from_spread_quote(away_points: float, p_away: float, sigma: float) -> float:
    """Invert a de-vigged spread quote into an estimate of E[away − home]."""
    p = min(max(p_away, 0.02), 0.98)
    return sigma * _N.inv_cdf(p) - away_points


def mu_from_total_quote(line: float, p_over: float, sigma: float) -> float:
    p = min(max(p_over, 0.02), 0.98)
    return line + sigma * _N.inv_cdf(p)


def devig(p_a: float, p_b: float) -> float:
    total = p_a + p_b
    return p_a / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Quote sources
# ---------------------------------------------------------------------------
# A quote: {"book", "kind": "spread"|"total", "points": away spread or total
#           line, "p": de-vigged P(away covers) or P(over), "main": bool,
#           "live": bool}

class LinesCache:
    """Per-league spread/total quotes from Pinnacle, FanDuel and ESPN."""

    SHARP_WEIGHT = 1.5
    ALT_WEIGHT = 0.5

    def __init__(self, cache_ttl: int = 120, game_schedule=None):
        self.cache_ttl = cache_ttl
        self.game_schedule = game_schedule
        # league → (fetched_at, {"away@home": {"spread": [..], "total": [..]}})
        self._games: Dict[str, Tuple[float, Dict[str, Dict[str, List[dict]]]]] = {}

    # -- fetch ---------------------------------------------------------------

    LIVE_TTL = 20.0   # seconds between refreshes while any quote is in-play

    def _get(self, url: str, params: Optional[dict] = None,
             browser_ua: bool = True) -> Optional[object]:
        # Pinnacle/FanDuel want a browser UA; ESPN's CDN returns 403 to the
        # bare "Mozilla/5.0" string and 200 to the default client UA.
        # Shared across profiles and between LinesCache and the moneyline
        # book clients (same URLs), at this refresh's max age.
        from bot.http_cache import get_json
        headers = {"User-Agent": "Mozilla/5.0"} if browser_ua else {}
        return get_json(url, params=params, headers=headers,
                        max_age=getattr(self, "_fetch_age", self.cache_ttl))

    def _get_uncached(self, url: str, params: Optional[dict] = None,
                      browser_ua: bool = True) -> Optional[object]:
        headers = {"User-Agent": "Mozilla/5.0"} if browser_ua else {}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def _league_games(self, league: str) -> Dict[str, Dict[str, List[dict]]]:
        now = time.time()
        cached = self._games.get(league)
        ttl = self.cache_ttl
        if cached and any(q.get("live") for g in cached[1].values()
                          for kind in ("spread", "total") for q in g[kind]):
            ttl = self.LIVE_TTL   # in-play lines move every play
        if cached and now - cached[0] < ttl:
            return cached[1]
        self._fetch_age = ttl
        games: Dict[str, Dict[str, List[dict]]] = {}
        sport_key = LEAGUES.get(league, {}).get("odds_api_key", "")
        for fetch in (self._pinnacle, self._fanduel, self._espn, self._actionnetwork):
            try:
                for key, kind, quote in fetch(league, sport_key):
                    games.setdefault(key, {"spread": [], "total": []})[kind].append(quote)
            except Exception:
                continue
        self._games[league] = (now, games)
        return games

    @staticmethod
    def _key(away_name: str, home_name: str, sport_key: str) -> str:
        a = _match_abbr(away_name, sport_key)
        h = _match_abbr(home_name, sport_key)
        return f"{a}@{h}" if a and h else ""

    def _pinnacle(self, league: str, sport_key: str):
        league_id = PINNACLE_LEAGUES.get(sport_key)
        if not league_id:
            return
        matchups = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/matchups") or []
        markets = self._get(f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight") or []
        info: Dict[int, Tuple[str, bool]] = {}   # matchup id → (game key, live)
        for m in matchups:
            if m.get("type") != "matchup":
                continue
            parts = m.get("participants") or []
            if len(parts) < 2 and m.get("parent"):
                parts = m["parent"].get("participants") or []
            home = next((p.get("name", "") for p in parts if p.get("alignment") == "home"), "")
            away = next((p.get("name", "") for p in parts if p.get("alignment") == "away"), "")
            key = self._key(away, home, sport_key)
            if key:
                info[m.get("id")] = (key, bool(m.get("isLive")))
        for mk in markets:
            if mk.get("period") != 0 or mk.get("type") not in ("spread", "total"):
                continue
            meta = info.get(mk.get("matchupId"))
            if not meta:
                continue
            key, live = meta
            prices = {p.get("designation"): p for p in mk.get("prices", [])}
            main = not mk.get("isAlternate", False)
            if mk["type"] == "spread":
                a, h = prices.get("away"), prices.get("home")
                if not a or not h or a.get("points") is None:
                    continue
                p_away = devig(american_to_prob(int(a["price"])), american_to_prob(int(h["price"])))
                yield key, "spread", {"book": "pinnacle", "kind": "spread",
                                      "points": float(a["points"]), "p": p_away,
                                      "main": main, "live": live}
            else:
                o, u = prices.get("over"), prices.get("under")
                if not o or not u or o.get("points") is None:
                    continue
                p_over = devig(american_to_prob(int(o["price"])), american_to_prob(int(u["price"])))
                yield key, "total", {"book": "pinnacle", "kind": "total",
                                     "points": float(o["points"]), "p": p_over,
                                     "main": main, "live": live}

    def _fanduel(self, league: str, sport_key: str):
        page = FANDUEL_SPORTS.get(sport_key)
        if not page:
            return
        raw = self._get(FANDUEL_BASE, {"page": "CUSTOM", "customPageId": page,
                                       "_ak": "FhMFpcPWXMeyZxOx"})
        if not raw:
            return
        att = raw.get("attachments", {})
        events = {}
        for eid, ev in att.get("events", {}).items():
            name = ev.get("name", "")
            if " @ " in name:
                away, home = name.split(" @ ", 1)
                events[str(ev.get("eventId", eid))] = (away.strip(), home.strip())
        for mk in att.get("markets", {}).values():
            mtype = mk.get("marketType", "")
            if mtype not in ("MATCH_HANDICAP_(2-WAY)", "TOTAL_POINTS_(OVER/UNDER)"):
                continue
            teams = events.get(str(mk.get("eventId", "")))
            if not teams:
                continue
            key = self._key(teams[0], teams[1], sport_key)
            if not key:
                continue
            live = bool(mk.get("inPlay"))
            runners = {}
            for r in mk.get("runners", []):
                odds = r.get("winRunnerOdds", {}).get("americanDisplayOdds", {}).get("americanOdds")
                if odds is None or r.get("handicap") is None:
                    continue
                runners[r.get("runnerName", "")] = (float(r["handicap"]), american_to_prob(int(odds)))
            if len(runners) < 2:
                continue
            if mtype == "MATCH_HANDICAP_(2-WAY)":
                away_abbr = key.split("@")[0]
                away_r = next((v for n, v in runners.items() if _match_abbr(n, sport_key) == away_abbr), None)
                home_r = next((v for n, v in runners.items() if _match_abbr(n, sport_key) != away_abbr), None)
                if not away_r or not home_r:
                    continue
                yield key, "spread", {"book": "fanduel", "kind": "spread",
                                      "points": away_r[0], "p": devig(away_r[1], home_r[1]),
                                      "main": True, "live": live}
            else:
                over, under = runners.get("Over"), runners.get("Under")
                if not over or not under:
                    continue
                yield key, "total", {"book": "fanduel", "kind": "total",
                                     "points": over[0], "p": devig(over[1], under[1]),
                                     "main": True, "live": live}

    def _espn(self, league: str, sport_key: str):
        """Pregame DraftKings line from the scoreboard (no prices: assume −110)."""
        url = scoreboard_url(league)
        if not url:
            return
        data = self._get(url, browser_ua=False) or {}
        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            odds = (comp.get("odds") or [{}])[0]
            if not odds or odds.get("details") is None:
                continue  # live/final games carry no line
            teams = {c.get("homeAway"): c.get("team", {}).get("displayName", "")
                     for c in comp.get("competitors", [])}
            key = self._key(teams.get("away", ""), teams.get("home", ""), sport_key)
            if not key:
                continue
            spread = odds.get("spread")          # home spread, negative = home favored
            if spread is not None:
                yield key, "spread", {"book": "espn_dk", "kind": "spread",
                                      "points": -float(spread), "p": 0.5,
                                      "main": True, "live": False}
            ou = odds.get("overUnder")
            if ou is not None:
                yield key, "total", {"book": "espn_dk", "kind": "total",
                                     "points": float(ou), "p": 0.5,
                                     "main": True, "live": False}

    def _actionnetwork(self, league: str, sport_key: str):
        """Pregame spreads/totals from DraftKings, BetMGM, Caesars, bet365, BetRivers."""
        from bot.signals.book_scrapers import ActionNetworkClient
        if not hasattr(self, "_an"):
            self._an = ActionNetworkClient(cache_ttl=self.cache_ttl)
        yield from self._an.line_quotes(sport_key, max_age=getattr(self, "_fetch_age", None))

    # -- pricing -------------------------------------------------------------

    def _fraction_remaining(self, league: str, away: str, home: str) -> Optional[float]:
        if self.game_schedule is None:
            return None
        try:
            remaining = self.game_schedule.get_game_time_remaining(league, home, away)
        except Exception:
            return None
        if remaining is None:
            return None
        clock = LEAGUES.get(league, {}).get("clock")
        if not clock:
            return None
        full = clock["periods"] * clock["minutes"] * 60
        return max(0.0, min(1.0, remaining / full))

    def price_market(self, parsed: Dict, is_live: bool) -> Optional[Dict]:
        """Model probability for token 0 of a parsed asc-/tsc- market."""
        league = parsed["league"]
        if league not in SIGMA:
            return None
        games = self._league_games(league)
        game = games.get(f"{parsed['away']}@{parsed['home']}")
        if not game:
            return None
        quotes = [q for q in game[parsed["kind"]] if bool(q["live"]) == bool(is_live)]
        if not quotes:
            return None

        sigma_full = SIGMA[league][0 if parsed["kind"] == "spread" else 1]
        clock_known = True
        if is_live:
            frac = self._fraction_remaining(league, parsed["away"], parsed["home"])
            if frac is None:
                frac, clock_known = 0.5, False
            sigma = sigma_full * math.sqrt(max(frac, 0.03))
        else:
            sigma = sigma_full

        w_sum = mu_sum = 0.0
        for q in quotes:
            w = (self.SHARP_WEIGHT if q["book"] == "pinnacle" else 1.0) * (1.0 if q["main"] else self.ALT_WEIGHT)
            if parsed["kind"] == "spread":
                mu_q = mu_from_spread_quote(q["points"], q["p"], sigma)
            else:
                mu_q = mu_from_total_quote(q["points"], q["p"], sigma)
            mu_sum += w * mu_q
            w_sum += w
        mu = mu_sum / w_sum
        line = parsed["line"]
        if parsed["kind"] == "spread":
            p_model = prob_away_covers(mu, sigma, line)
        else:
            p_model = prob_over(mu, sigma, line)

        exact = [q["p"] for q in quotes if abs(q["points"] - line) < 0.01]
        prob = 0.5 * p_model + 0.5 * (sum(exact) / len(exact)) if exact else p_model
        books = sorted({q["book"] for q in quotes})
        return {
            "prob": min(max(prob, 0.01), 0.99),
            "num_books": len(books),
            "books": books,
            "mu": mu,
            "sigma": sigma,
            "exact_line_quotes": len(exact),
            "live": is_live,
            "clock_known": clock_known,
        }


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def spread_total_signal(snapshot: MarketSnapshot, config, lines_cache: Optional[LinesCache]) -> Signal:
    """Primary external signal for asc-/tsc- markets. Confidence 0 blocks."""
    def blocked(reason: str, **extra) -> Signal:
        return Signal(name="spread_total", value=0.5, confidence=0.0,
                      direction="neutral", metadata={"reason": reason, **extra})

    parsed = parse_line_slug(snapshot.slug)
    if not parsed:
        return blocked("unsupported_line_market")
    if lines_cache is None:
        return blocked("no_lines_cache")
    res = lines_cache.price_market(parsed, bool(getattr(snapshot, "is_live", False)))
    if res is None:
        return blocked("no_lines_for_game")
    num_books = res["num_books"]
    if num_books < 2:
        return blocked(f"only_{num_books}_books", num_books=num_books)

    prob = res["prob"]
    edge = prob - snapshot.price
    if abs(edge) > 0.07 and num_books < 3:
        return blocked(f"edge_{abs(edge)*100:.0f}pct_needs_3_books_have_{num_books}",
                       edge=edge, num_books=num_books, consensus_prob=prob)

    confidence = min(num_books / 3, 1.0) * min(abs(edge) * 5, 1.0)
    confidence = max(0.1, min(confidence, 1.0))
    if res["live"] and not res["clock_known"]:
        confidence *= 0.5
    direction = "bullish" if edge > 0.02 else "bearish" if edge < -0.02 else "neutral"
    return Signal(
        name="spread_total",
        value=float(prob),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "consensus_prob": float(prob),
            "polymarket_price": float(snapshot.price),
            "edge": float(edge),
            "num_books": num_books,
            "books_used": ",".join(res["books"]),
            "kind": parsed["kind"],
            "line": parsed["line"],
            "mu": round(res["mu"], 2),
            "sigma": round(res["sigma"], 2),
            "exact_line_quotes": res["exact_line_quotes"],
            "is_live": res["live"],
        },
    )
