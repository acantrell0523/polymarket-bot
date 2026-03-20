"""Polymarket US API + CLOB API client."""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateutil_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.models import MarketSnapshot, OrderBook, OrderBookLevel
from utils.config import APIConfig, FilterConfig
from utils.logger import TradingLogger


class MarketDataClient:
    """Fetches market data from Polymarket US API (primary) and CLOB API (fallback for order books)."""

    def __init__(self, config: APIConfig, logger: Optional[TradingLogger] = None, filters: Optional[FilterConfig] = None):
        self.config = config
        self.logger = logger
        self.filters = filters or FilterConfig()
        self._min_interval = 1.0 / config.max_requests_per_second
        self._last_call = 0.0

        self.session = requests.Session()
        retries = Retry(
            total=config.max_retries,
            backoff_factor=config.retry_backoff_base,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)

        # Polymarket US API base URL (gateway is the public endpoint)
        self.us_api_url = "https://gateway.polymarket.us"

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if self.logger:
                err_str = str(e)
                if "prices-history" in url and "400" in err_str:
                    self.logger.debug("api_request_failed", {"url": url, "error": err_str})
                else:
                    self.logger.error("api_request_failed", {"url": url, "error": err_str})
            return None

    def get_active_markets(self, limit: int = 500) -> List[Dict]:
        """Fetch active markets from Polymarket US API, filtered by time window.

        Sports markets (with gameStartTime):
          - Upcoming: included if game starts within 24 hours
          - Live: included if game started up to 4 hours ago (currently in progress)
          - Stale: excluded if game started more than 4 hours ago (probably over)
        Non-sports markets: included if endDate is within 14 days.
        Both types (upcoming only): excluded if resolving in under min_hours_to_expiry.
        """
        url = f"{self.us_api_url}/v1/markets"
        params = {"active": "true", "closed": "false", "limit": limit}
        data = self._get(url, params)

        if isinstance(data, dict):
            markets = data.get("markets", [])
        elif isinstance(data, list):
            markets = data
        else:
            return []

        now = datetime.now(timezone.utc)
        min_expiry = now + timedelta(hours=self.filters.min_hours_to_expiry)
        sports_cutoff = now + timedelta(hours=24)
        max_live_age = timedelta(hours=4)
        nonsports_cutoff = now + timedelta(days=14)

        filtered = []
        live_count = 0
        for m in markets:
            game_start_str = m.get("gameStartTime")
            end_date_str = m.get("endDate")

            if game_start_str:
                game_start = self._parse_datetime(game_start_str)
                if game_start is None:
                    continue

                if game_start <= now:
                    # Game has started — live or finished
                    if (now - game_start) > max_live_age:
                        continue  # game probably over, skip
                    # Live game — include it
                    live_count += 1
                else:
                    # Upcoming game
                    if game_start < min_expiry or game_start > sports_cutoff:
                        continue
            else:
                # Non-sports market: filter by endDate
                ref_time = self._parse_datetime(end_date_str)
                if ref_time is None:
                    continue
                if ref_time < min_expiry or ref_time > nonsports_cutoff:
                    continue

            filtered.append(m)

        if self.logger:
            self.logger.info("markets_filtered", {
                "total": len(markets),
                "after_time_filter": len(filtered),
                "live_games": live_count,
                "sports_window_hours": 24,
                "nonsports_window_days": 14,
            })

        return filtered

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse an ISO datetime string, returning a timezone-aware datetime or None."""
        if not dt_str:
            return None
        try:
            dt = dateutil_parser.isoparse(dt_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    def get_live_price(self, slug: str) -> Optional[float]:
        """Fetch current price for a market via the BBO endpoint."""
        url = f"{self.us_api_url}/v1/markets/{slug}/bbo"
        data = self._get(url)
        if not data or not isinstance(data, dict):
            return None
        market_data = data.get("marketData", data)
        # Try lastTradePx first, then currentPx
        for field in ("lastTradePx", "currentPx"):
            px = market_data.get(field)
            if px and isinstance(px, dict):
                try:
                    val = float(px.get("value", 0))
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass
        return None

    def get_us_order_book(self, slug: str) -> OrderBook:
        """Fetch order book from Polymarket US API."""
        url = f"{self.us_api_url}/v1/markets/{slug}/book"
        data = self._get(url)

        if not data or not isinstance(data, dict):
            return OrderBook()

        market_data = data.get("marketData", data)

        bids = []
        for b in market_data.get("bids", []):
            px = b.get("px", {})
            price = float(px.get("value", 0)) if isinstance(px, dict) else float(px or 0)
            qty = float(b.get("qty", 0))
            bids.append(OrderBookLevel(price=price, size=qty))

        asks = []
        for a in market_data.get("offers", []):
            px = a.get("px", {})
            price = float(px.get("value", 0)) if isinstance(px, dict) else float(px or 0)
            qty = float(a.get("qty", 0))
            asks.append(OrderBookLevel(price=price, size=qty))

        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(bids=bids, asks=asks)

    def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch order book from CLOB API (fallback)."""
        url = f"{self.config.clob_url}/book"
        data = self._get(url, {"token_id": token_id})

        if not data or not isinstance(data, dict):
            return OrderBook()

        bids = [
            OrderBookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
            for a in data.get("asks", [])
        ]

        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(bids=bids, asks=asks)

    def get_price_history(self, token_id: str, limit: int = 100) -> List[float]:
        """Fetch price history from CLOB API, trying multiple interval formats."""
        url = f"{self.config.clob_url}/prices-history"

        # Try these intervals in order
        for interval in ("1w", "1d", "6h"):
            data = self._get(url, {"token_id": token_id, "interval": interval, "limit": limit})
            prices = self._parse_price_history(data)
            if prices:
                return prices

        # Fallback: try startTs/endTs (last 7 days)
        now = int(time.time())
        seven_days_ago = now - 7 * 24 * 3600
        data = self._get(url, {"token_id": token_id, "startTs": seven_days_ago, "endTs": now})
        prices = self._parse_price_history(data)
        if prices:
            return prices

        return []

    def _parse_price_history(self, data: Any) -> List[float]:
        """Parse price history response into a list of floats."""
        if not data:
            return []

        if isinstance(data, dict):
            history = data.get("history", [])
        elif isinstance(data, list):
            history = data
        else:
            return []

        prices = []
        for point in history:
            if isinstance(point, dict):
                p = point.get("p", point.get("price", 0))
                prices.append(float(p))
            elif isinstance(point, (int, float)):
                prices.append(float(point))

        return prices

    def build_snapshot(self, market: Dict) -> Optional[MarketSnapshot]:
        """Build a MarketSnapshot from a Polymarket US market dict."""
        try:
            market_id = str(market.get("id", ""))
            slug = market.get("slug", "")
            question = market.get("question", "")

            if not slug:
                return None

            # Extract price from outcomePrices, bestBid/bestAsk, or marketSides
            price = self._extract_price(market)

            volume = float(market.get("volume", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)

            # Extract token_id from marketSides identifier if available
            market_sides = market.get("marketSides", [])
            token_id = ""
            if market_sides:
                token_id = market_sides[0].get("identifier", "")

            # Fetch order book from US API first, fall back to CLOB
            order_book = self.get_us_order_book(slug)
            if order_book.bid_depth == 0 and order_book.ask_depth == 0 and token_id:
                order_book = self.get_order_book(token_id)

            # Fetch price history from CLOB if we have a token_id
            price_history = []
            if token_id:
                price_history = self.get_price_history(token_id)

            # Skip markets that have no price history AND no order book data
            has_order_book = order_book.bid_depth > 0 or order_book.ask_depth > 0
            if not price_history and not has_order_book:
                return None

            if not price_history:
                price_history = [price]

            # Determine if game is live and compute hours_to_expiry
            now = datetime.now(timezone.utc)
            is_live = False
            hours_to_expiry = 72.0

            game_start = self._parse_datetime(market.get("gameStartTime"))
            end_date = self._parse_datetime(market.get("endDate"))

            if game_start:
                if game_start <= now:
                    is_live = True
                    # Estimate time remaining (assume ~3h game from start)
                    hours_to_expiry = max(0, 3.0 - (now - game_start).total_seconds() / 3600)
                else:
                    hours_to_expiry = (game_start - now).total_seconds() / 3600
            elif end_date:
                hours_to_expiry = max(0, (end_date - now).total_seconds() / 3600)

            return MarketSnapshot(
                market_id=market_id,
                token_id=token_id or slug,
                question=question,
                price=price,
                volume_24h=volume,
                liquidity=liquidity,
                order_book=order_book,
                price_history=price_history,
                timestamp=datetime.now(timezone.utc),
                category=market.get("category", ""),
                slug=slug,
                hours_to_expiry=hours_to_expiry,
                is_live=is_live,
            )
        except Exception as e:
            if self.logger:
                self.logger.error("build_snapshot_failed", {"slug": market.get("slug", ""), "error": str(e)})
            return None

    def _extract_price(self, market: Dict) -> float:
        """Extract the best available price from a US API market object.

        Uses the long side (Yes) price from marketSides first, then
        bestBid/bestAsk midpoint, then falls back to 0.5.
        """
        # Best source: marketSides long side price
        market_sides = market.get("marketSides", [])
        for side in market_sides:
            if side.get("long") is True:
                side_price = side.get("price")
                if side_price is not None:
                    try:
                        return float(side_price)
                    except (ValueError, TypeError):
                        pass

        # Fallback: first side with a price
        for side in market_sides:
            side_price = side.get("price")
            if side_price is not None:
                try:
                    return float(side_price)
                except (ValueError, TypeError):
                    pass

        # Try bestBid/bestAsk midpoint
        best_bid = market.get("bestBid")
        best_ask = market.get("bestAsk")
        if best_bid is not None and best_ask is not None:
            try:
                bid = float(best_bid)
                ask = float(best_ask)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
            except (ValueError, TypeError):
                pass

        return 0.5
