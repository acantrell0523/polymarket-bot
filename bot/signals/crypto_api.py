"""Crypto probability estimation using CoinGecko price data and simple math."""

import re
import time
import math
import requests
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone


COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Map common crypto names/tickers to CoinGecko IDs
CRYPTO_IDS = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "solana": "solana",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "xrp": "ripple", "ripple": "ripple",
    "ada": "cardano", "cardano": "cardano",
    "avax": "avalanche-2", "avalanche": "avalanche-2",
    "dot": "polkadot", "polkadot": "polkadot",
    "matic": "matic-network", "polygon": "matic-network",
    "link": "chainlink", "chainlink": "chainlink",
    "bnb": "binancecoin",
}

# Annualized volatility estimates (updated periodically)
# These are rough defaults; the cache fetches real volatility when possible
DEFAULT_VOLATILITY = {
    "bitcoin": 0.60,
    "ethereum": 0.75,
    "solana": 1.00,
    "dogecoin": 1.20,
    "ripple": 0.90,
}

# Annualized-vol sanity band. Crypto majors realize ~0.4-1.0; the 30d
# realized estimate can spike absurdly high (a single 30% day) or collapse
# in a quiet month. The barrier/terminal models are VERY sensitive to vol
# (σ√T is the whole distribution width), so clamp it.
VOL_FLOOR = 0.30
VOL_CAP = 2.50


def _terminal_above_prob(S: float, K: float, T: float, vol: float) -> float:
    """Risk-neutral P(S_T >= K) under GBM with r=0.

    ln S_T = ln S_0 + (r - vol^2/2)T + vol*sqrt(T)*Z, r=0.
    P(S_T >= K) = Phi( (ln(S/K) + (r - vol^2/2)T) / (vol*sqrt(T)) ).
    (The old model used +vol^2/2 — WRONG SIGN — which overstated "above".)
    """
    nu = -0.5 * vol * vol
    d = (math.log(S / K) + nu * T) / (vol * math.sqrt(T))
    return _norm_cdf(d)


def _barrier_hit_prob(S: float, K: float, T: float, vol: float, direction: str) -> float:
    """P(price TOUCHES the barrier K at any time before T) — first passage.

    Most Polymarket crypto markets are "will X hit $Y by date" TOUCH options,
    not "be above $Y AT date" terminal options. For an out-of-the-money target
    the touch probability is up to ~2x the terminal probability (reflection
    principle), so terminal-pricing a touch market underprices YES massively.

    Reflection principle for arithmetic BM X_t = nu*t + vol*B_t (X_0=0),
    barrier level b = ln(K/S) in log space, nu = -vol^2/2 (r=0):
      up-barrier (K>S, b>0):   P(max X >= b) = Phi((-b+nu T)/s) + e^(2 nu b/vol^2) Phi((-b-nu T)/s)
      down-barrier (K<S, b<0): P(min X <= b) = Phi(( b-nu T)/s) + e^(2 nu b/vol^2) Phi(( b+nu T)/s)
    with s = vol*sqrt(T). Driftless sanity: both reduce to 2*Phi(-|b|/s) = 2x terminal.
    """
    if direction == "above" and S >= K:
        return 1.0   # already touched
    if direction == "below" and S <= K:
        return 1.0
    nu = -0.5 * vol * vol
    s = vol * math.sqrt(T)
    b = math.log(K / S)
    expo = max(-50.0, min(50.0, 2.0 * nu * b / (vol * vol)))  # guard overflow
    if direction == "above":
        p = _norm_cdf((-b + nu * T) / s) + math.exp(expo) * _norm_cdf((-b - nu * T) / s)
    else:
        p = _norm_cdf((b - nu * T) / s) + math.exp(expo) * _norm_cdf((b + nu * T) / s)
    return max(0.0, min(1.0, p))


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    from datetime import date
    return (date(year, month + 1, 1) - __import__("datetime").timedelta(days=1)).day


class CryptoCache:
    """Caches crypto price and volatility data from CoinGecko."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._price_cache: Dict[str, Tuple[float, float, float]] = {}  # id → (timestamp, price, vol)

    def _fetch_price_and_vol(self, coin_id: str) -> Optional[Tuple[float, float]]:
        """Fetch current price and compute 30-day volatility."""
        try:
            # Current price
            resp = requests.get(
                f"{COINGECKO_BASE}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            price = resp.json().get(coin_id, {}).get("usd")
            if not price:
                return None

            # 30-day historical for volatility
            vol = DEFAULT_VOLATILITY.get(coin_id, 0.80)
            try:
                hist_resp = requests.get(
                    f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": "30", "interval": "daily"},
                    timeout=10,
                )
                if hist_resp.status_code == 200:
                    prices = [p[1] for p in hist_resp.json().get("prices", [])]
                    if len(prices) > 5:
                        returns = []
                        for i in range(1, len(prices)):
                            if prices[i - 1] > 0:
                                returns.append(math.log(prices[i] / prices[i - 1]))
                        if returns:
                            daily_vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
                            vol = daily_vol * math.sqrt(365)  # annualize
            except Exception:
                pass

            self._price_cache[coin_id] = (time.time(), float(price), vol)
            return (float(price), vol)

        except Exception:
            return None

    def get_price_and_vol(self, coin_id: str) -> Optional[Tuple[float, float]]:
        """Get cached price and volatility, refreshing if stale."""
        if coin_id in self._price_cache:
            ts, price, vol = self._price_cache[coin_id]
            if time.time() - ts < self.cache_ttl:
                return (price, vol)
        return self._fetch_price_and_vol(coin_id)

    def estimate_probability(
        self,
        question: str,
        polymarket_price: float,
        slug: str = "",
    ) -> Optional[Tuple[float, Dict]]:
        """Estimate the YES probability for a crypto price-target market.

        Handles both option styles, which the OLD model conflated:
          * BARRIER / touch ("Will X hit/reach $Y by date"): P(touch <= T),
            priced with the reflection principle. This is the DOMINANT
            Polymarket style (cpc-btc-150k-12-31-2026: "When will Bitcoin
            hit $150k?").
          * TERMINAL ("Will X be above $Y ON date"): P(S_T >= K).

        Parsing is SLUG-FIRST: the live "hit" markets put the deadline (and
        often the target) in the SLUG, not the question ("When will Bitcoin
        hit $150k?" has no date). The old question-only parser defaulted to
        30 days — catastrophically wrong for an 18-month barrier. Falls back
        to question parsing when the slug isn't a recognizable cpc- slug.

        Returns (probability, metadata) or None if unparseable / no data.
        """
        parsed = self._parse_crypto_slug(slug) if slug else None
        source = "slug"
        if not parsed:
            parsed = self._parse_crypto_question(question)
            source = "question"
        if not parsed:
            return None

        coin_id, target_price, direction, days_remaining, is_barrier = parsed

        data = self.get_price_and_vol(coin_id)
        if not data:
            return None
        current_price, annual_vol = data
        vol = max(VOL_FLOOR, min(VOL_CAP, annual_vol))

        if current_price <= 0 or target_price <= 0:
            return None

        # Already resolved by deadline passing.
        if days_remaining <= 0:
            if direction == "above":
                prob = 1.0 if current_price >= target_price else 0.0
            else:
                prob = 1.0 if current_price <= target_price else 0.0
            return (prob, {"coin": coin_id, "current_price": current_price,
                           "target_price": target_price, "direction": direction,
                           "days_remaining": 0, "model": "expired", "parse_source": source})

        T = days_remaining / 365.0

        if is_barrier:
            prob = _barrier_hit_prob(current_price, target_price, T, vol, direction)
            model = f"barrier_{direction}"
        else:
            above = _terminal_above_prob(current_price, target_price, T, vol)
            prob = above if direction == "above" else 1.0 - above
            model = f"terminal_{direction}"

        prob = max(0.01, min(0.99, prob))
        metadata = {
            "coin": coin_id,
            "current_price": current_price,
            "target_price": target_price,
            "direction": direction,
            "is_barrier": is_barrier,
            "days_remaining": days_remaining,
            "annual_vol": round(vol, 3),
            "model": model,
            "model_prob": round(prob, 4),
            "parse_source": source,
        }
        return (prob, metadata)

    def _parse_crypto_slug(self, slug: str) -> Optional[Tuple[str, float, str, float, bool]]:
        """Parse a Polymarket crypto-price slug into components.

        Verified live 2026-07-15:
          cpc-btc-150k-12-31-2026                 -> btc, 150000, above, barrier
          cpc-btc-hitprice-high-yr-12-31-2026-200k-> btc, 200000, above, barrier
        Date is MM-DD-YYYY somewhere in the slug; target is a k/m-suffixed
        token (150k) or a bare >=1000 integer outside the date; coin is any
        token in CRYPTO_IDS. "hit/hitprice/high/low/reach/touch" => barrier
        (the cpc- family is touch markets). "close/settle/on" => terminal.

        Returns (coin_id, target, direction, days_remaining, is_barrier) or None.
        """
        if not slug:
            return None
        parts = slug.lower().split("-")
        tokens = set(parts)

        coin_id = None
        for tok in parts:
            if tok in CRYPTO_IDS:
                coin_id = CRYPTO_IDS[tok]
                break
        if coin_id is None:
            return None

        # Date MM-DD-YYYY: three consecutive numeric tokens ending in a 4-digit year.
        now = datetime.now(timezone.utc)
        days_remaining = None
        date_idx = set()
        for i in range(len(parts) - 2):
            a, b, c = parts[i], parts[i + 1], parts[i + 2]
            if a.isdigit() and b.isdigit() and c.isdigit() and len(c) == 4:
                try:
                    deadline = datetime(int(c), int(a), int(b), tzinfo=timezone.utc)
                except ValueError:
                    continue
                days_remaining = (deadline - now).days
                date_idx = {i, i + 1, i + 2}
                break
        if days_remaining is None:
            return None  # no reliable deadline in slug -> let question parser try

        # Target: k/m suffixed token, else a bare >=1000 integer not in the date.
        target = None
        for i, tok in enumerate(parts):
            m = re.fullmatch(r'(\d+(?:pt\d+)?)(k|m)?', tok)
            if not m:
                continue
            base = float(m.group(1).replace("pt", "."))
            suffix = m.group(2)
            if suffix == "k":
                target = base * 1_000
            elif suffix == "m":
                target = base * 1_000_000
            elif i not in date_idx and base >= 1000:
                target = base
            if target is not None:
                break
        if target is None:
            return None

        direction = "above"
        if any(w in tokens for w in ("low", "below", "under", "drop", "dip")):
            direction = "below"
        # Terminal only if the slug explicitly says settle/close/on-date.
        is_barrier = not any(w in tokens for w in ("close", "settle", "closeprice"))

        return (coin_id, target, direction, float(days_remaining), is_barrier)

    def _parse_crypto_question(self, question: str) -> Optional[Tuple[str, float, str, float, bool]]:
        """Parse a crypto market question (fallback when the slug isn't a cpc- slug).

        Returns (coin_id, target_price, direction, days_remaining, is_barrier)
        or None.
        """
        q = question.lower()

        coin_id = None
        for keyword, cid in CRYPTO_IDS.items():
            if keyword in q:
                coin_id = cid
                break
        if not coin_id:
            return None

        # Target price. Handle "$100k"/"$1.5m" (the old $-regex captured "100"
        # from "$100k" and dropped the k -> off by 1000x) as well as bare "100k".
        target_price = None
        dk = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*([km])?', question, re.IGNORECASE)
        if dk:
            base = float(dk.group(1).replace(",", ""))
            sfx = (dk.group(2) or "").lower()
            target_price = base * (1000 if sfx == "k" else 1_000_000 if sfx == "m" else 1)
        else:
            bare = re.search(r'(\d+(?:\.\d+)?)\s*([km])\b', q)
            if bare:
                base = float(bare.group(1))
                target_price = base * (1000 if bare.group(2) == "k" else 1_000_000)
            else:
                return None

        direction = "above"
        if any(w in q for w in ["below", "under", "drop", "fall", "less than", "dip"]):
            direction = "below"

        # Barrier vs terminal from phrasing. "hit/reach/touch/when will" => touch.
        # "on <date>/close/settle/end at" => terminal. Default barrier (the
        # dominant Polymarket style).
        is_barrier = True
        if re.search(r'\b(hit|reach|touch|hits|reaches|cross|when will)\b', q):
            is_barrier = True
        elif re.search(r'\b(close|settle|end (the )?(year|month|day)|on \w+ \d)\b', q):
            is_barrier = False

        now = datetime.now(timezone.utc)
        days_remaining = 30  # default (question with no date is low-confidence)

        month_match = re.search(
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            q,
        )
        if month_match:
            month_names = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
            }
            month = month_names[month_match.group(1)]
            year = int(month_match.group(2))
            try:
                deadline = datetime(year, month, _last_day_of_month(year, month),
                                    tzinfo=timezone.utc)
                days_remaining = max(1, (deadline - now).days)
            except ValueError:
                pass
        else:
            iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
            if iso_match:
                try:
                    from dateutil import parser as dateutil_parser
                    deadline = dateutil_parser.isoparse(iso_match.group(1)).replace(tzinfo=timezone.utc)
                    days_remaining = max(1, (deadline - now).days)
                except Exception:
                    pass

        return (coin_id, target_price, direction, float(days_remaining), is_barrier)

def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))
