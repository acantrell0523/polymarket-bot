"""Kalshi cross-exchange price source (public API, no auth).

A regulated exchange's order book on the SAME instrument is the strongest
external validation available — stronger than sportsbook consensus, because
there is no vig to strip and no line to translate.

Verified live 2026-07-18:
  * GET /trade-api/v2/markets?series_ticker=KXHIGHNY returns real quotes
    (yes_bid_dollars / yes_ask_dollars as decimal strings).
  * Polymarket tc-temp-nychigh-2026-07-18-lt79f traded 0.82 while Kalshi
    KXHIGHNY-26JUL18-T79 ("78° or below") quoted 0.81/0.83 — identical
    market, two venues, in sync. Divergence between them IS the signal.

First vertical: daily high-temperature markets. Matching is MECHANICAL
(city map + date token + strike bounds) — no fuzzy text matching, which is
what made the PredictIt source untrustworthy.

Bucket semantics (verified against live subtitles):
  Polymarket                      Kalshi
  lt79f        (high <= 78)   ==  cap_strike=79   ("78° or below")
  gte87f       (high >= 87)   ==  floor_strike=86 ("87° or above")
  gte84lt86f   (84 <= h <= 85)==  floor=84, cap=85 ("84° to 85°")
Only exact-equivalent buckets are matched; anything else returns None
(Kalshi interiors are 2-degree, Polymarket's are often 1-degree — a
mismatched bucket is not a price, it's a trap).
"""

import re
import time
import requests
from typing import Dict, List, Optional, Tuple

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Polymarket temp-slug city token -> Kalshi series. Only VERIFIED pairs;
# sfohigh stays unmapped until the Kalshi series is confirmed.
TEMP_CITY_SERIES = {
    "nychigh": "KXHIGHNY",
    "miahigh": "KXHIGHMIA",
    "laxhigh": "KXHIGHLAX",
    "mdwhigh": "KXHIGHCHI",   # Kalshi Chicago series settles on Midway (MDW)
}

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_temp_slug(slug: str) -> Optional[Dict]:
    """Parse tc-temp-{city}-{Y}-{M}-{D}-{bucket} into components.

    Buckets: ltXf | gteXf | gteXltYf  (temperatures in whole °F).
    Returns {"city", "date_token" (Kalshi 26JUL18 style), "kind", bounds}.
    """
    parts = slug.lower().split("-")
    if len(parts) != 7 or parts[0] != "tc" or parts[1] != "temp":
        return None
    city, y, mo, d, bucket = parts[2], parts[3], parts[4], parts[5], parts[6]
    if not (y.isdigit() and mo.isdigit() and d.isdigit()):
        return None
    date_token = f"{int(y) % 100:02d}{_MONTHS[int(mo) - 1]}{int(d):02d}"

    m = re.fullmatch(r"lt(\d+)f", bucket)
    if m:
        return {"city": city, "date_token": date_token,
                "kind": "below", "bound": int(m.group(1))}
    m = re.fullmatch(r"gte(\d+)f", bucket)
    if m:
        return {"city": city, "date_token": date_token,
                "kind": "above", "bound": int(m.group(1))}
    m = re.fullmatch(r"gte(\d+)lt(\d+)f", bucket)
    if m:
        return {"city": city, "date_token": date_token, "kind": "range",
                "lo": int(m.group(1)), "hi": int(m.group(2))}
    return None


class KalshiCache:
    """Cached Kalshi market fetcher + mechanical matchers."""

    def __init__(self, cache_ttl: int = 120):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}

    def _series_markets(self, series_ticker: str) -> List[dict]:
        now = time.time()
        if series_ticker in self._cache:
            ts, data = self._cache[series_ticker]
            if now - ts < self.cache_ttl:
                return data
        try:
            resp = requests.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": 100, "status": "open",
                        "series_ticker": series_ticker},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            markets = (resp.json().get("markets", [])
                       if resp.status_code == 200 else [])
        except Exception:
            markets = []
        self._cache[series_ticker] = (now, markets)
        return markets

    @staticmethod
    def _mid_and_spread(m: dict) -> Optional[Tuple[float, float]]:
        """Mid of yes bid/ask (dollar-decimal strings). None if one-sided."""
        try:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return (bid + ask) / 2.0, ask - bid

    def match_temp_market(self, slug: str) -> Optional[Dict]:
        """Find the EXACT Kalshi counterpart of a Polymarket temp bucket.

        Returns {"prob", "spread", "ticker"} or None (no partial matches —
        a Kalshi 2-degree bucket is NOT a price for a 1-degree PM bucket).
        """
        parsed = parse_temp_slug(slug)
        if not parsed:
            return None
        series = TEMP_CITY_SERIES.get(parsed["city"])
        if not series:
            return None

        for m in self._series_markets(series):
            if parsed["date_token"] not in m.get("ticker", ""):
                continue
            floor = m.get("floor_strike")
            cap = m.get("cap_strike")
            matched = False
            if parsed["kind"] == "below":
                # PM lt79f == Kalshi cap_strike 79, no floor
                matched = (floor is None and cap == parsed["bound"])
            elif parsed["kind"] == "above":
                # PM gte87f == Kalshi floor_strike 86, no cap
                matched = (cap is None and floor == parsed["bound"] - 1)
            else:  # range gte{lo}lt{hi}: closed [lo, hi-1] == floor lo, cap hi-1
                matched = (floor == parsed["lo"] and cap == parsed["hi"] - 1)
            if not matched:
                continue
            ms = self._mid_and_spread(m)
            if ms is None:
                return None
            mid, spread = ms
            return {"prob": mid, "spread": spread, "ticker": m.get("ticker")}
        return None

    def match_slug(self, slug: str) -> Optional[Dict]:
        """Route a Polymarket slug to the right matcher (temp only, v1)."""
        if slug.startswith("tc-temp-"):
            return self.match_temp_market(slug)
        return None
