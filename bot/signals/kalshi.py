"""Kalshi cross-venue signal.

Kalshi lists the same games as Polymarket US (NFL, college football, NHL
moneylines; NFL/college spreads; NFL totals). Where the two venues disagree
on the same outcome there is either information on one side or a stale
price on the other — a second prediction-market print is the closest thing
to an independent, tradeable reference. This is an AUXILIARY signal: the
sportsbook consensus / ESPN live model stays primary and the direction
guard means Kalshi can temper an edge, never manufacture or flip one.

Public read-only API, no key: https://api.elections.kalshi.com/trade-api/v2

Ticker shapes (verified 2026-09-20):
    KXNFLGAME-26SEP21NYGLAR-NYG        away+home concatenated, outcome suffix
    KXNCAAFGAME-26OCT04SJSUHAW-SJSU    college abbreviations ≈ ESPN's
    KXNHLGAME-26SEP24EDMVAN-VAN
    KXNFLSPREAD-26SEP27LARDEN-LAR8     "LA Rams wins by over 7.5" (floor_strike)
    KXNFLTOTAL-26SEP20INDKC-67         "over 66.5 points" (floor_strike)
Prices arrive as *_dollars strings (yes_bid_dollars, yes_ask_dollars).
"""

import json
import re
import time
import urllib.request
from datetime import datetime
from typing import Dict, Optional, Tuple

from bot.leagues import ABBR_MAP, CFB_TEAMS, normalize_abbr
from bot.signals.lines import parse_line_slug
from utils.models import MarketSnapshot, Signal

API = "https://api.elections.kalshi.com/trade-api/v2/markets"
SERIES = {
    "nfl": {"ml": "KXNFLGAME", "spread": "KXNFLSPREAD", "total": "KXNFLTOTAL"},
    "cfb": {"ml": "KXNCAAFGAME", "spread": "KXNCAAFSPREAD", "total": "KXNCAAFTOTAL"},
    "nhl": {"ml": "KXNHLGAME"},
}
MAX_WIDTH = 0.08   # bid/ask wider than this = no usable print
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_TICK = re.compile(r"^(KX[A-Z]+)-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)-([A-Z]*?)(\d*)$")  # totals: outcome is just the strike index

# Kalshi's own codes that differ from both ESPN's and Polymarket's.
KALSHI_ALIASES = {
    "nfl": {"was": "was", "wsh": "was", "lar": "lar", "la": "lar", "lac": "lac", "jac": "jax", "gnb": "gb",
            "kan": "kc", "sfo": "sf", "tam": "tb", "nor": "no", "nwe": "ne", "lvr": "lv"},
    "nhl": {"vgk": "veg", "wsh": "was", "nsh": "nas", "mtl": "mon", "lak": "la", "sjs": "sj", "njd": "nj",
            "tbl": "tb", "utah": "uta"},
}


_CFB_DISPLAY = {v["display_abbr"]: code for code, v in CFB_TEAMS.items() if v.get("display_abbr")}


def _pm_code(league: str, code: str) -> str:
    """Kalshi abbreviation → Polymarket slug code for the league."""
    c = code.lower()
    alias = KALSHI_ALIASES.get(league, {})
    if c in alias:
        return alias[c]
    if league == "cfb":
        if c in CFB_TEAMS:
            return c
        if c in _CFB_DISPLAY:            # Kalshi tickers use Polymarket's display codes
            return _CFB_DISPLAY[c]
        return ABBR_MAP.get("cfb", {}).get(c, c)
    return normalize_abbr(league, c)


def _parse_ticker(ticker: str) -> Optional[Dict]:
    m = _TICK.match(ticker)
    if not m:
        return None
    series, yy, mon, dd, teams, outcome, strike_idx = m.groups()
    try:
        date = datetime(2000 + int(yy), MONTHS[mon], int(dd)).date().isoformat()
    except (KeyError, ValueError):
        return None
    if teams.startswith(outcome) and len(teams) > len(outcome):
        away, home = outcome, teams[len(outcome):]
    elif teams.endswith(outcome) and len(teams) > len(outcome):
        away, home = teams[:-len(outcome)], outcome
    else:
        # totals: outcome is a strike index, teams must be split by the
        # event's known abbreviations — handled by the caller via the event map
        away, home = None, None
    return {"series": series, "date": date, "teams": teams, "outcome": outcome,
            "away": away, "home": home}


def _mid(m: dict) -> Optional[Tuple[float, float]]:
    try:
        bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (bid + ask) / 2, ask - bid


class KalshiCache:
    """Open Kalshi game markets indexed by (league, date, away_pm, home_pm)."""

    def __init__(self, cache_ttl: int = 60):
        self.cache_ttl = cache_ttl
        self._index: Dict[str, Tuple[float, Dict]] = {}   # league → (fetched_at, index)

    def _fetch_series(self, series: str):
        cursor = None
        while True:
            url = f"{API}?limit=200&status=open&series_ticker={series}" + (f"&cursor={cursor}" if cursor else "")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                            timeout=15) as r:
                    d = json.load(r)
            except Exception:
                return
            for m in d.get("markets", []):
                yield m
            cursor = d.get("cursor")
            if not cursor:
                return

    def _build(self, league: str) -> Dict:
        idx: Dict[Tuple, Dict] = {}
        events: Dict[str, Tuple[str, str]] = {}   # "26SEP21NYGLAR" → (away, home) learned from ML tickers
        series = SERIES.get(league, {})
        ml = list(self._fetch_series(series["ml"])) if "ml" in series else []
        for m in ml:
            t = _parse_ticker(m["ticker"])
            if not t or not t["away"]:
                continue
            events[t["teams"]] = (t["away"], t["home"])
            key = (league, t["date"], _pm_code(league, t["away"]), _pm_code(league, t["home"]))
            q = _mid(m)
            if q:
                idx.setdefault(key, {"ml": {}, "spread": {}, "total": {}})["ml"][_pm_code(league, t["outcome"])] = q
        for kind in ("spread", "total"):
            if kind not in series:
                continue
            for m in self._fetch_series(series[kind]):
                t = _parse_ticker(m["ticker"])
                if not t:
                    continue
                ev = events.get(t["teams"])
                if not ev:
                    continue
                key = (league, t["date"], _pm_code(league, ev[0]), _pm_code(league, ev[1]))
                strike = m.get("floor_strike")
                if strike is None:
                    continue
                q = _mid(m)
                if not q:
                    continue
                bucket = idx.setdefault(key, {"ml": {}, "spread": {}, "total": {}})
                if kind == "spread":
                    bucket["spread"][(_pm_code(league, t["outcome"]), float(strike))] = q
                else:
                    bucket["total"][float(strike)] = q
        return idx

    def index(self, league: str) -> Dict:
        now = time.time()
        cached = self._index.get(league)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]
        idx = self._build(league) if league in SERIES else {}
        self._index[league] = (now, idx)
        return idx

    def quote_for_slug(self, slug: str) -> Optional[Dict]:
        """Kalshi mid for token 0 of a Polymarket slug, or None."""
        parts = slug.split("-")
        line = parse_line_slug(slug)
        if line:
            league, away, home, date, kind = line["league"], line["away"], line["home"], line["date"], line["kind"]
        elif len(parts) == 7 and parts[0] == "aec":
            league, away, home, date, kind = parts[1], parts[2], parts[3], "-".join(parts[4:7]), "ml"
        else:
            return None
        game = self.index(league).get((league, date, away, home))
        if not game:
            return None
        if kind == "ml":
            q = game["ml"].get(away)
            return {"prob": q[0], "width": q[1], "kind": kind} if q else None
        if kind == "total":
            q = game["total"].get(line["line"])
            return {"prob": q[0], "width": q[1], "kind": kind} if q else None
        # spread: token 0 = away covers `line`.
        L = line["line"]
        if L < 0:                       # away favored: "away wins by over |L|"
            q = game["spread"].get((away, -L))
            return {"prob": q[0], "width": q[1], "kind": kind} if q else None
        q = game["spread"].get((home, L))   # away +L covers unless home wins by over L
        return {"prob": 1.0 - q[0], "width": q[1], "kind": kind} if q else None


def kalshi_cross_signal(snapshot: MarketSnapshot, config, cache: Optional[KalshiCache]) -> Signal:
    def off(reason):
        return Signal(name="kalshi_cross", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": reason})
    if cache is None:
        return off("no_kalshi_cache")
    try:
        q = cache.quote_for_slug(snapshot.slug)
    except Exception as e:  # never let a venue outage break a scan
        return off(f"kalshi_error_{type(e).__name__}")
    if not q:
        return off("no_kalshi_market")
    if q["width"] > MAX_WIDTH:
        return off("kalshi_book_too_wide")
    prob = min(max(q["prob"], 0.01), 0.99)
    edge = prob - snapshot.price
    confidence = max(0.0, 0.6 * (1.0 - q["width"] / MAX_WIDTH))
    direction = "bullish" if edge > 0.02 else "bearish" if edge < -0.02 else "neutral"
    return Signal(name="kalshi_cross", value=float(prob), confidence=float(confidence), direction=direction,
                  metadata={"kalshi_prob": float(prob), "polymarket_price": float(snapshot.price),
                            "edge": float(edge), "width": float(q["width"]), "kind": q["kind"]})
