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
    ) -> Optional[Tuple[float, Dict]]:
        """Estimate the probability of a crypto price target being hit.

        Parses questions like:
          "Will Bitcoin hit $100,000 by June 2026?"
          "Will ETH be above $5000 on March 31?"

        Uses a simple log-normal model:
          P(S > K) = Φ((ln(S/K) + (r - σ²/2)T) / (σ√T))

        Returns (probability, metadata) or None if we can't parse the question.
        """
        parsed = self._parse_crypto_question(question)
        if not parsed:
            return None

        coin_id, target_price, direction, days_remaining = parsed

        data = self.get_price_and_vol(coin_id)
        if not data:
            return None

        current_price, annual_vol = data

        if days_remaining <= 0:
            # Already past deadline
            if direction == "above":
                prob = 1.0 if current_price >= target_price else 0.0
            else:
                prob = 1.0 if current_price <= target_price else 0.0
            return (prob, {
                "coin": coin_id, "current": current_price, "target": target_price,
                "direction": direction, "days": 0, "model": "expired",
            })

        # Time in years
        T = days_remaining / 365.0
        vol = annual_vol
        r = 0.0  # risk-free rate ≈ 0 for crypto

        # Log-normal probability
        if current_price <= 0 or target_price <= 0:
            return None

        d = (math.log(current_price / target_price) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))

        # Φ(d) = probability price ends above target
        prob_above = _norm_cdf(d)

        if direction == "above":
            prob = prob_above
        else:
            prob = 1.0 - prob_above

        prob = max(0.01, min(0.99, prob))

        metadata = {
            "coin": coin_id,
            "current_price": current_price,
            "target_price": target_price,
            "direction": direction,
            "days_remaining": days_remaining,
            "annual_vol": round(annual_vol, 3),
            "model_prob": round(prob, 4),
        }

        return (prob, metadata)

    def _parse_crypto_question(self, question: str) -> Optional[Tuple[str, float, str, float]]:
        """Parse a crypto market question into components.

        Returns (coin_id, target_price, direction, days_remaining) or None.
        """
        q = question.lower()

        # Find the crypto asset
        coin_id = None
        for keyword, cid in CRYPTO_IDS.items():
            if keyword in q:
                coin_id = cid
                break
        if not coin_id:
            return None

        # Find target price — look for $ amounts
        price_match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', question)
        if not price_match:
            # Try bare numbers with k suffix
            price_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', q)
            if price_match:
                target_price = float(price_match.group(1).replace(",", "")) * 1000
            else:
                return None
        else:
            target_price = float(price_match.group(1).replace(",", ""))

        # Determine direction
        direction = "above"  # default
        if any(word in q for word in ["below", "under", "drop", "fall", "less than"]):
            direction = "below"

        # Find deadline — look for dates
        now = datetime.now(timezone.utc)
        days_remaining = 30  # default

        # Try to find a date like "March 2026", "June 30", "2026-06-30"
        month_match = re.search(
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            q
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
                deadline = datetime(year, month, 28, tzinfo=timezone.utc)  # end of month approx
                days_remaining = max(1, (deadline - now).days)
            except ValueError:
                pass
        else:
            # Try ISO date
            iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
            if iso_match:
                try:
                    from dateutil import parser as dateutil_parser
                    deadline = dateutil_parser.isoparse(iso_match.group(1)).replace(tzinfo=timezone.utc)
                    days_remaining = max(1, (deadline - now).days)
                except Exception:
                    pass

        return (coin_id, target_price, direction, days_remaining)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))
