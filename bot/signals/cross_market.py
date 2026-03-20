"""Cross-market arbitrage signal using PredictIt public API."""

import time
import requests
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher


PREDICTIT_API_URL = "https://www.predictit.org/api/marketdata/all"


class PredictItCache:
    """Caches PredictIt market data with a TTL."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._markets: List[dict] = []
        self._last_fetch: float = 0
        self._contracts_index: Dict[str, dict] = {}  # lowered name → contract data

    def _fetch(self):
        """Fetch all PredictIt markets."""
        try:
            resp = requests.get(PREDICTIT_API_URL, timeout=15)
            if resp.status_code != 200:
                return
            data = resp.json()
            self._markets = data.get("markets", [])
            self._build_index()
            self._last_fetch = time.time()
        except Exception:
            pass

    def _build_index(self):
        """Build a lookup index of contract names for fuzzy matching."""
        self._contracts_index = {}
        for market in self._markets:
            for contract in market.get("contracts", []):
                name = contract.get("name", "").lower().strip()
                if name:
                    self._contracts_index[name] = {
                        "market_name": market.get("name", ""),
                        "contract_name": contract.get("name", ""),
                        "lastTradePrice": contract.get("lastTradePrice"),
                        "bestBuyYesCost": contract.get("bestBuyYesCost"),
                        "bestBuyNoCost": contract.get("bestBuyNoCost"),
                        "bestSellYesCost": contract.get("bestSellYesCost"),
                        "bestSellNoCost": contract.get("bestSellNoCost"),
                    }

    def refresh_if_stale(self):
        if time.time() - self._last_fetch > self.cache_ttl:
            self._fetch()

    def find_match(self, question: str, keywords: Optional[List[str]] = None) -> Optional[dict]:
        """Find a PredictIt contract matching a Polymarket question.

        Uses fuzzy string matching on the question text.
        Returns contract data dict or None.
        """
        self.refresh_if_stale()
        if not self._contracts_index:
            return None

        q_lower = question.lower().strip()

        # Try exact substring match first
        for name, data in self._contracts_index.items():
            if name in q_lower or q_lower in name:
                return data

        # Try keyword-based matching if provided
        if keywords:
            kw_lower = [k.lower() for k in keywords]
            best_match = None
            best_score = 0
            for name, data in self._contracts_index.items():
                matches = sum(1 for kw in kw_lower if kw in name)
                if matches > best_score and matches >= 2:
                    best_score = matches
                    best_match = data
            if best_match:
                return best_match

        # Fuzzy match as last resort
        best_match = None
        best_ratio = 0
        for name, data in self._contracts_index.items():
            ratio = SequenceMatcher(None, q_lower[:80], name[:80]).ratio()
            if ratio > best_ratio and ratio > 0.55:
                best_ratio = ratio
                best_match = data

        return best_match

    def get_probability(self, question: str, keywords: Optional[List[str]] = None) -> Optional[Tuple[float, str]]:
        """Get PredictIt probability for a matching market.

        Returns (probability, matched_contract_name) or None.
        """
        match = self.find_match(question, keywords)
        if not match:
            return None

        # Use lastTradePrice as probability (PredictIt prices are 0-1 probabilities)
        price = match.get("lastTradePrice")
        if price is None:
            price = match.get("bestBuyYesCost")
        if price is None:
            return None

        return (float(price), match["contract_name"])
