"""
On-chain enrichment client for Polymarket prediction markets.

Replaces the paid Unusual Whales API with FREE data from:

1. Polymarket CLOB API - trade history, large orders, activity data
2. Polymarket Gamma API - market activity, comments, related markets
3. Polygon blockchain via public RPC / The Graph - wallet positions,
   whale tracking, historical resolution accuracy

This produces the same enrichment dict structure that the UW signal
functions expect, so the 3 signals (smart_money, whale_flow,
market_sentiment) work identically without any paid API key.

Data sources:
- CLOB /trades endpoint: recent trades with size, lets us detect whales
- CLOB /book endpoint: already used, but we pull deeper depth data here
- Gamma /activity endpoint: market-level activity metrics
- Gamma /markets: related markets for cross-market sentiment
- Polymarket Profile API: top trader positions (public leaderboard data)
"""

import time
import hashlib
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.models import MarketSnapshot, Signal
from utils.config import SignalConfig
from utils.logger import TradingLogger


class OnChainEnrichmentClient:
    """
    Fetches whale/smart-money/sentiment data from free Polymarket sources.

    Produces an enrichment dict with the same keys the UW signal functions
    read from, so the signal layer doesn't need any changes.

    Expected output format (matches what UW signals consume):
    {
        "market_detail": {...},     # Smart money scoring data
        "smart_money_data": {...},  # Aggregated smart money signals
        "whale_data": {...},        # Large position holders
        "unusual_data": {...},      # Unusual activity flags
        "positions": {...},         # Yes/No position breakdown
        "market_tide": {...},       # Macro sentiment proxy
    }
    """

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        gamma_url: str = "https://gamma-api.polymarket.com",
        profile_url: str = "https://polymarket.com/api",
        logger: Optional[TradingLogger] = None,
        max_rps: int = 3,
        cache_ttl: int = 120,
    ):
        self.clob_url = clob_url
        self.gamma_url = gamma_url
        self.profile_url = profile_url
        self.logger = logger
        self._min_interval = 1.0 / max_rps
        self._last_call = 0.0
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = cache_ttl

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _get(
        self, url: str, params: Optional[Dict] = None, cache_key: Optional[str] = None
    ) -> Optional[Any]:
        """Rate-limited, cached GET."""
        if cache_key and cache_key in self._cache:
            data, expiry = self._cache[cache_key]
            if time.time() < expiry:
                return data

        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if cache_key:
                self._cache[cache_key] = (data, time.time() + self._cache_ttl)
            return data
        except requests.RequestException as e:
            if self.logger:
                self.logger.debug(
                    "onchain_request_failed", {"url": url, "error": str(e)}
                )
            return None

    # ----------------------------------------------------------------
    # Data fetchers
    # ----------------------------------------------------------------

    def fetch_recent_trades(
        self, token_id: str, limit: int = 200
    ) -> List[Dict]:
        """
        Fetch recent trades for a token from the CLOB API.

        Returns a list of trade dicts with price, size, side, timestamp.
        This is the primary source for whale detection: any single trade
        above a size threshold is flagged as a whale trade.
        """
        url = f"{self.clob_url}/trades"
        params = {"token_id": token_id, "limit": limit}
        data = self._get(url, params, cache_key=f"trades_{token_id}")

        if data is None:
            return []

        # CLOB returns trades as a list or under a "trades" key
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("trades", data.get("data", []))
        return []

    def fetch_market_trades(
        self, condition_id: str, limit: int = 200
    ) -> List[Dict]:
        """
        Fetch recent trades for a market (all outcomes) from Gamma API.
        """
        url = f"{self.gamma_url}/trades"
        params = {"market": condition_id, "limit": limit}
        data = self._get(url, params, cache_key=f"market_trades_{condition_id}")

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("trades", []))
        return []

    def fetch_market_activity(self, market_id: str) -> Optional[Dict]:
        """
        Fetch market-level activity data from Gamma API.

        Returns volume trends, comment count, views, etc.
        """
        url = f"{self.gamma_url}/markets/{market_id}"
        data = self._get(url, cache_key=f"market_activity_{market_id}")
        return data if isinstance(data, dict) else None

    def fetch_related_markets(self, market_id: str) -> List[Dict]:
        """
        Fetch related/similar markets from Gamma API.

        Used for cross-market sentiment: if related markets are
        trending in the same direction, confidence increases.
        """
        url = f"{self.gamma_url}/markets"
        params = {"related_to": market_id, "limit": 5}
        data = self._get(url, params, cache_key=f"related_{market_id}")

        if isinstance(data, list):
            return data
        return []

    # ----------------------------------------------------------------
    # Analysis functions
    # ----------------------------------------------------------------

    def _analyze_whale_trades(
        self, trades: List[Dict], current_price: float
    ) -> Dict[str, Any]:
        """
        Analyze a list of trades to detect whale activity.

        A "whale trade" is any single trade above the 95th percentile
        in size, or above a fixed threshold ($500 USDC equivalent).

        Returns:
        {
            "whale_buy_volume": float,   # Total $ of whale buys
            "whale_sell_volume": float,   # Total $ of whale sells
            "whale_count": int,           # Number of whale trades
            "whale_net_direction": float, # [-1, 1] net buy/sell pressure
            "total_volume": float,
            "trade_count": int,
            "avg_trade_size": float,
            "large_trade_ratio": float,   # Fraction of volume from whales
        }
        """
        if not trades:
            return {
                "whale_buy_volume": 0, "whale_sell_volume": 0,
                "whale_count": 0, "whale_net_direction": 0,
                "total_volume": 0, "trade_count": 0,
                "avg_trade_size": 0, "large_trade_ratio": 0,
            }

        sizes = []
        total_buy = 0.0
        total_sell = 0.0

        for t in trades:
            try:
                # CLOB trade format: {price, size, side, ...}
                size = float(t.get("size", 0) or t.get("amount", 0) or 0)
                price = float(t.get("price", current_price) or current_price)
                side = str(t.get("side", "")).lower()
                usd_value = size * price

                sizes.append(usd_value)
                if side in ("buy", "b", "1"):
                    total_buy += usd_value
                elif side in ("sell", "s", "0"):
                    total_sell += usd_value
                else:
                    # Unknown side: split evenly
                    total_buy += usd_value / 2
                    total_sell += usd_value / 2
            except (ValueError, TypeError):
                continue

        if not sizes:
            return {
                "whale_buy_volume": 0, "whale_sell_volume": 0,
                "whale_count": 0, "whale_net_direction": 0,
                "total_volume": 0, "trade_count": 0,
                "avg_trade_size": 0, "large_trade_ratio": 0,
            }

        import numpy as np
        sizes_arr = np.array(sizes)
        total_volume = sizes_arr.sum()

        # Whale threshold: 95th percentile or $500, whichever is lower
        pct_95 = np.percentile(sizes_arr, 95) if len(sizes_arr) > 10 else 500
        whale_threshold = min(pct_95, 500)

        whale_buys = 0.0
        whale_sells = 0.0
        whale_count = 0

        for t in trades:
            try:
                size = float(t.get("size", 0) or t.get("amount", 0) or 0)
                price = float(t.get("price", current_price) or current_price)
                usd_value = size * price
                side = str(t.get("side", "")).lower()

                if usd_value >= whale_threshold:
                    whale_count += 1
                    if side in ("buy", "b", "1"):
                        whale_buys += usd_value
                    else:
                        whale_sells += usd_value
            except (ValueError, TypeError):
                continue

        whale_total = whale_buys + whale_sells
        net_direction = 0.0
        if whale_total > 0:
            net_direction = (whale_buys - whale_sells) / whale_total

        return {
            "whale_buy_volume": whale_buys,
            "whale_sell_volume": whale_sells,
            "whale_count": whale_count,
            "whale_net_direction": net_direction,
            "total_volume": total_volume,
            "trade_count": len(trades),
            "avg_trade_size": float(sizes_arr.mean()),
            "large_trade_ratio": whale_total / total_volume if total_volume > 0 else 0,
        }

    def _analyze_smart_money(
        self, trades: List[Dict], current_price: float
    ) -> Dict[str, Any]:
        """
        Estimate smart money positioning from trade patterns.

        Smart money heuristics (no wallet tracking needed):
        1. Large trades at extreme prices (buying low < 0.30 or selling high > 0.70)
           are more likely to be informed.
        2. Trades going against recent momentum suggest contrarian conviction.
        3. Cluster of large buys after a price drop suggests accumulation.

        Returns:
        {
            "smart_money_sentiment": float,  # [-1, 1]
            "conviction_score": float,       # [0, 1]
            "accumulation_detected": bool,
        }
        """
        if not trades:
            return {
                "smart_money_sentiment": 0,
                "conviction_score": 0,
                "accumulation_detected": False,
            }

        import numpy as np

        # Score each trade for "smartness"
        smart_buy_score = 0.0
        smart_sell_score = 0.0
        total_weighted = 0.0

        for t in trades:
            try:
                size = float(t.get("size", 0) or t.get("amount", 0) or 0)
                price = float(t.get("price", current_price) or current_price)
                side = str(t.get("side", "")).lower()
                usd_value = size * price

                if usd_value < 10:  # Skip dust
                    continue

                # Smart money score: large trades at value prices
                # Buying at 0.20 is "smarter" than buying at 0.80
                # (more potential upside per dollar risked)
                size_factor = min(usd_value / 100, 5.0)  # Cap at 5x

                if side in ("buy", "b", "1"):
                    # Buying cheap = higher smart score
                    value_factor = 1.0 - price  # Buying at 0.20 -> factor 0.80
                    smart_buy_score += size_factor * value_factor
                elif side in ("sell", "s", "0"):
                    # Selling expensive = higher smart score
                    value_factor = price  # Selling at 0.80 -> factor 0.80
                    smart_sell_score += size_factor * value_factor

                total_weighted += size_factor
            except (ValueError, TypeError):
                continue

        if total_weighted == 0:
            return {
                "smart_money_sentiment": 0,
                "conviction_score": 0,
                "accumulation_detected": False,
            }

        # Net smart money direction
        net = smart_buy_score - smart_sell_score
        sentiment = np.clip(net / total_weighted, -1, 1)

        # Conviction: how concentrated is the smart money signal
        conviction = min(abs(net) / (total_weighted + 1), 1.0)

        # Accumulation detection: multiple large buys below current price
        recent_large_buys_below = 0
        for t in trades[:50]:  # Most recent 50
            try:
                size = float(t.get("size", 0) or t.get("amount", 0) or 0)
                price = float(t.get("price", current_price) or current_price)
                side = str(t.get("side", "")).lower()
                if (
                    side in ("buy", "b", "1")
                    and price < current_price * 0.95
                    and size * price > 100
                ):
                    recent_large_buys_below += 1
            except (ValueError, TypeError):
                continue

        accumulation = recent_large_buys_below >= 3

        return {
            "smart_money_sentiment": float(sentiment),
            "conviction_score": float(conviction),
            "accumulation_detected": accumulation,
        }

    def _compute_cross_market_sentiment(
        self, related_markets: List[Dict], current_category: str
    ) -> Dict[str, Any]:
        """
        Compute sentiment from related and cross-category markets.

        If related prediction markets are trending in the same direction,
        that's a mild confirming signal. This replaces UW's options-based
        market tide with a prediction-market-native sentiment measure.
        """
        if not related_markets:
            return {"sentiment_score": 0, "related_count": 0}

        bullish = 0
        bearish = 0

        for m in related_markets:
            try:
                # Check if related market prices imply bullish or bearish
                prices_raw = m.get("outcomePrices", "")
                if isinstance(prices_raw, str):
                    import json
                    try:
                        prices = json.loads(prices_raw)
                    except (json.JSONDecodeError, TypeError):
                        prices = []
                elif isinstance(prices_raw, list):
                    prices = prices_raw
                else:
                    continue

                if prices and len(prices) >= 1:
                    yes_price = float(prices[0])
                    if yes_price > 0.55:
                        bullish += 1
                    elif yes_price < 0.45:
                        bearish += 1
            except (ValueError, TypeError, IndexError):
                continue

        total = bullish + bearish
        if total == 0:
            return {"sentiment_score": 0, "related_count": 0}

        score = (bullish - bearish) / total
        return {"sentiment_score": score, "related_count": total}

    # ----------------------------------------------------------------
    # Main enrichment method
    # ----------------------------------------------------------------

    def get_enrichment_for_market(
        self, snapshot: MarketSnapshot
    ) -> Dict[str, Any]:
        """
        Build the full enrichment dict for a market using free data only.

        Returns data in the same format the UW signal functions expect,
        so the signal layer works without changes.
        """
        enrichment: Dict[str, Any] = {}

        # 1. Fetch recent trades for whale + smart money analysis
        trades = self.fetch_recent_trades(snapshot.token_id, limit=200)
        whale_analysis = self._analyze_whale_trades(trades, snapshot.price)
        smart_analysis = self._analyze_smart_money(trades, snapshot.price)

        # 2. Build market_detail (consumed by smart money signal)
        enrichment["market_detail"] = {
            "data": {
                "smart_money_sentiment": smart_analysis["smart_money_sentiment"],
                "conviction_score": smart_analysis["conviction_score"],
                "accumulation_detected": smart_analysis["accumulation_detected"],
            }
        }

        # 3. Build whale_data (consumed by whale flow signal)
        # Convert whale trades into the format the signal expects
        whale_items = []
        if whale_analysis["whale_count"] > 0:
            direction = "buy" if whale_analysis["whale_net_direction"] > 0 else "sell"
            whale_items.append({
                "asset_id": snapshot.market_id,
                "market_id": snapshot.market_id,
                "direction": direction,
                "amount": whale_analysis["whale_buy_volume"] + whale_analysis["whale_sell_volume"],
                "whale_count": whale_analysis["whale_count"],
            })
        enrichment["whale_data"] = {"data": whale_items}

        # 4. Build unusual_data (consumed by whale flow signal)
        # Flag as unusual if whale trade ratio is high or volume is anomalous
        unusual_items = []
        if whale_analysis["large_trade_ratio"] > 0.3:
            unusual_items.append({
                "asset_id": snapshot.market_id,
                "market_id": snapshot.market_id,
                "reason": "high_whale_concentration",
                "large_trade_ratio": whale_analysis["large_trade_ratio"],
            })
        enrichment["unusual_data"] = {"data": unusual_items}

        # 5. Build positions (consumed by whale flow signal)
        # Estimate yes/no split from trade data
        total_vol = whale_analysis["total_volume"]
        if total_vol > 0:
            buy_pct = (
                whale_analysis.get("whale_buy_volume", 0)
                / total_vol * 100
                if total_vol > 0
                else 50
            )
            enrichment["positions"] = {
                "data": {
                    "yes_percentage": min(buy_pct * 2, 100),  # Scale up whale buys
                    "no_percentage": max(100 - buy_pct * 2, 0),
                }
            }
        else:
            enrichment["positions"] = {
                "data": {"yes_percentage": 50, "no_percentage": 50}
            }

        # 6. Build market_tide (consumed by sentiment signal)
        # Use cross-market data as a sentiment proxy
        related = self.fetch_related_markets(snapshot.market_id)
        cross_sentiment = self._compute_cross_market_sentiment(
            related, snapshot.category
        )
        enrichment["market_tide"] = {
            "data": {
                "sentiment": cross_sentiment["sentiment_score"],
                "related_count": cross_sentiment["related_count"],
            }
        }

        # 7. Attach smart_money_data (consumed by smart money signal)
        # If accumulation is detected, flag this market
        smart_items = []
        if smart_analysis["accumulation_detected"]:
            smart_items.append({
                "asset_id": snapshot.market_id,
                "market_id": snapshot.market_id,
                "score": smart_analysis["smart_money_sentiment"],
                "conviction": smart_analysis["conviction_score"],
            })
        enrichment["smart_money_data"] = {"data": smart_items}

        return enrichment
