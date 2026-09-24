"""Polymarket US API + CLOB API client."""

import os
import time
import os
import json
import fcntl
import tempfile
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateutil_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.models import MarketSnapshot, OrderBook, OrderBookLevel
from utils.config import APIConfig, FilterConfig
from utils.logger import TradingLogger


# Game-winner slug families on polymarket.us: aec- = 2-way moneyline,
# atc- = 3-way (soccer). Spreads (asc-), totals (tsc-), props (astatc-…)
# and futures (tec-) are excluded from the scan universe.
from bot.leagues import LEAGUES
from bot.signals.lines import LINE_PREFIXES, SUPPORTED_LEAGUES, parse_line_slug

MONEYLINE_PREFIXES = ("aec-", "atc-")
# Spread/total markets (asc-/tsc-) are admitted for leagues in
# bot.signals.lines.SUPPORTED_LEAGUES when token 0 is priced in this band.
LINE_MIN_PRICE = 0.30
LINE_MAX_PRICE = 0.70
MAX_LINES_PER_GAME = 3   # per game per kind (spread / total)


class MarketDataClient:
    """Fetches market data from Polymarket US API (primary) and CLOB API (fallback for order books)."""

    def __init__(self, config: APIConfig, logger: Optional[TradingLogger] = None, filters: Optional[FilterConfig] = None):
        self.config = config
        self.logger = logger
        self.filters = filters or FilterConfig()
        self._min_interval = 1.0 / config.max_requests_per_second
        self._last_call = 0.0
        # The gateway rate-limits /book and /bbo far harder than /markets:
        # measured 2026-09-20 at ~5 requests per 10-second window (429 with
        # Retry-After up to 10s beyond that), while 80 list pages at 4 rps
        # never 429'd. Book calls get their own, slower limiter.
        self._book_min_interval = 2.0
        self._last_book_call = 0.0
        # Shared cache for side-by-side strategy processes. When
        # POLYBOT_SHARED_DIR is set, the market list and every order book
        # fetched are written there and reused by any other bot process for
        # a short TTL, and /book calls are serialized across processes with
        # a file lock so the gateway sees ONE scanner regardless of how many
        # strategies are running.
        self.shared_dir = os.environ.get("POLYBOT_SHARED_DIR") or None
        if self.shared_dir:
            os.makedirs(os.path.join(self.shared_dir, "books"), exist_ok=True)
        self.shared_list_ttl = 90.0
        self.shared_book_ttl = 8.0
        self.book_feed = None   # attached by the trading loop when the websocket feed is on
        # Every slug in the latest raw active-market list, and whether that
        # list is complete (not truncated by a cap). A held market missing
        # from a complete list has closed, which triggers a settlement check.
        self.active_slugs: set = set()
        self.active_slugs_complete = False

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

    def _book_rate_limit(self):
        if self.shared_dir:
            # Cross-process limiter: hold the lock while waiting so the
            # combined /book cadence of all processes stays at one call per
            # _book_min_interval.
            lock_path = os.path.join(self.shared_dir, "book_rate.lock")
            with open(lock_path, "a+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    last = float(fh.read().strip() or 0)
                except ValueError:
                    last = 0.0
                wait = self._book_min_interval - (time.time() - last)
                if wait > 0:
                    time.sleep(wait)
                fh.seek(0); fh.truncate(); fh.write(str(time.time())); fh.flush()
                fcntl.flock(fh, fcntl.LOCK_UN)
            return
        elapsed = time.time() - self._last_book_call
        if elapsed < self._book_min_interval:
            time.sleep(self._book_min_interval - elapsed)
        self._last_book_call = time.time()

    # -- shared cache helpers (no-ops without POLYBOT_SHARED_DIR) ------------

    def _shared_read(self, name: str, ttl: float) -> Optional[Any]:
        if not self.shared_dir:
            return None
        path = os.path.join(self.shared_dir, name)
        try:
            if time.time() - os.path.getmtime(path) > ttl:
                return None
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _shared_write(self, name: str, data: Any) -> None:
        if not self.shared_dir:
            return
        path = os.path.join(self.shared_dir, name)
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)  # atomic: readers never see a partial file
        except OSError:
            pass

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        if "/book" in url or "/bbo" in url or "/settlement" in url:
            self._book_rate_limit()
        else:
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

    @staticmethod
    def _token0_price(market: Dict) -> Optional[float]:
        """outcomePrices[0] as float; the gateway serializes it as a JSON string."""
        raw = market.get("outcomePrices")
        try:
            if isinstance(raw, str):
                import json as _json
                raw = _json.loads(raw)
            return float(raw[0])
        except Exception:
            return None

    def get_active_markets(self, page_size: int = 500, max_markets: int = 40000) -> List[Dict]:
        """Fetch ALL active markets from Polymarket US API, filtered by time window.

        PAGINATED: the gateway lists thousands of active markets (measured
        live 2026-07-11: 6,000+, with WNBA game markets first appearing at
        offset ~2,500 and MLB at ~3,500). A single limit=500 request never
        sees a single tradeable game, so we page with limit+offset until a
        short page or the max_markets safety cap.

        Failure semantics: if the FIRST page fails we return [] (degraded
        mode handles it); if a later page fails we keep what we have and log,
        so a mid-scan blip degrades coverage instead of blanking the scan.

        Sports markets (with gameStartTime):
          - Upcoming: included if game starts within 24 hours
          - Live: included if game started up to 4 hours ago (currently in progress)
          - Stale: excluded if game started more than 4 hours ago (probably over)
        Non-sports markets: included if endDate is within 14 days.
        Both types (upcoming only): excluded if resolving in under min_hours_to_expiry.
        """
        url = f"{self.us_api_url}/v1/markets"
        markets: List[Dict] = []
        offset = 0
        pages = 0

        cached = self._shared_read("markets_list.json", self.shared_list_ttl)
        if cached:
            markets = cached
            offset = max_markets  # skip pagination; counts logged below
            if self.logger:
                self.logger.info("markets_list_from_shared_cache", {"raw_markets": len(markets)})

        while offset < max_markets:
            params = {"active": "true", "closed": "false",
                      "limit": page_size, "offset": offset}
            data = self._get(url, params)

            if isinstance(data, dict):
                page = data.get("markets", [])
            elif isinstance(data, list):
                page = data
            else:
                if offset == 0:
                    return []  # total failure — nothing to scan
                if self.logger:
                    self.logger.warning("market_pagination_partial_failure", {
                        "offset": offset, "markets_so_far": len(markets),
                    })
                break

            markets.extend(page)
            pages += 1
            if len(page) < page_size:
                break  # last page
            offset += page_size

        if not cached and markets:
            self._shared_write("markets_list.json", markets)
        self.active_slugs = {m.get("slug") for m in markets if m.get("slug")}
        self.active_slugs_complete = bool(markets) and len(markets) < max_markets
        if offset >= max_markets and not cached:
            if self.logger:
                self.logger.warning("market_pagination_cap_reached", {
                    "max_markets": max_markets,
                    "message": "active market list larger than scan cap; raise max_markets",
                })
        if self.logger:
            self.logger.info("markets_paginated", {"pages": pages, "raw_markets": len(markets)})

        now = datetime.now(timezone.utc)
        min_expiry = now + timedelta(hours=self.filters.min_hours_to_expiry)
        # Windows are config-driven (filters.sports_window_hours /
        # filters.nonsports_window_days) so deployments can widen or narrow
        # the scan universe without code changes.
        sports_window_hours = getattr(self.filters, "sports_window_hours", 24.0)
        nonsports_window_days = getattr(self.filters, "nonsports_window_days", 14.0)
        sports_cutoff = now + timedelta(hours=sports_window_hours)
        max_live_age = timedelta(hours=4)
        nonsports_cutoff = now + timedelta(days=nonsports_window_days)

        filtered = []
        live_count = 0
        non_moneyline = 0
        unregistered = 0
        line_out_of_band = 0
        line_candidates = []  # (parsed, token0 price, market)
        for m in markets:
            game_start_str = m.get("gameStartTime")
            end_date_str = m.get("endDate")

            if game_start_str:
                # Moneyline-only universe. The gateway lists ~300 markets per
                # NFL game (spreads, totals, player props, 1H/1Q variants);
                # the odds_value signal prices moneylines only, and building
                # 3,700 live order books at 5 req/s stalls the fast scan for
                # 12+ minutes. Keep the aec- (2-way) and atc- (3-way soccer)
                # game-winner families only.
                slug = str(m.get("slug", ""))
                if slug.startswith(LINE_PREFIXES):
                    # Full-game spreads/totals for leagues the lines model
                    # supports, and only lines priced inside the band the
                    # trade filter would accept anyway (the ladder of ±20
                    # alternates at 2c/98c is untradeable and would cost an
                    # order-book call each).
                    parsed = parse_line_slug(slug)
                    if not parsed or parsed["league"] not in SUPPORTED_LEAGUES:
                        non_moneyline += 1
                        continue
                    p0 = self._token0_price(m)
                    if p0 is None or not (LINE_MIN_PRICE <= p0 <= LINE_MAX_PRICE):
                        line_out_of_band += 1
                        continue
                    # Same time window as moneylines: live (started < 4h ago)
                    # or starting within sports_window_hours.
                    gs = self._parse_datetime(game_start_str)
                    if gs is None:
                        continue
                    if gs <= now:
                        if (now - gs) > max_live_age:
                            continue
                    elif gs < min_expiry or gs > sports_cutoff:
                        continue
                    line_candidates.append((parsed, p0, m))
                    continue  # admitted below, capped per game
                elif not slug.startswith(MONEYLINE_PREFIXES):
                    non_moneyline += 1
                    continue
                parts = slug.split("-")
                # Only leagues in the registry have an odds source; the
                # gateway also lists La Liga, Bundesliga, Norway, Iceland,
                # Dota 2, ... which would burn order-book calls for nothing.
                if len(parts) < 2 or parts[1] not in LEAGUES:
                    unregistered += 1
                    continue
                # Full-game slug only: aec-{lg}-{away}-{home}-{y}-{m}-{d} is
                # 7 parts (atc- adds the outcome: 8). Period variants such as
                # aec-nfl-car-atl-2026-09-20-1h are separate markets.
                if len(parts) != (7 if slug.startswith("aec-") else 8):
                    non_moneyline += 1
                    continue
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

        # Spread/total ladder: keep only the MAX_LINES_PER_GAME lines nearest
        # 50c per game and kind. A game lists ~40 spreads and ~30 totals;
        # the ones near even money are where a mispricing is tradeable, and
        # every admitted market costs an order-book call per scan.
        by_game: Dict[tuple, list] = {}
        for parsed, p0, m in line_candidates:
            key = (parsed["league"], parsed["away"], parsed["home"], parsed["date"], parsed["kind"])
            by_game.setdefault(key, []).append((abs(p0 - 0.5), m))
        line_kept = 0
        for key, rows in by_game.items():
            rows.sort(key=lambda r: r[0])
            for _, m in rows[:MAX_LINES_PER_GAME]:
                filtered.append(m)
                line_kept += 1
                if self._parse_datetime(m.get("gameStartTime")) <= now:
                    live_count += 1

        if self.logger:
            self.logger.info("markets_filtered", {
                "total": len(markets),
                "after_time_filter": len(filtered),
                "live_games": live_count,
                "dropped_non_moneyline": non_moneyline,
                "dropped_unregistered_league": unregistered,
                "dropped_line_out_of_band": line_out_of_band,
                "line_markets_kept": line_kept,
                "sports_window_hours": sports_window_hours,
                "nonsports_window_days": nonsports_window_days,
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

    def get_market_resolutions(self, slugs: List[str]) -> Dict[str, float]:
        """Settlement values for any of `slugs` whose market has RESOLVED.

        One /v1/markets call per 50 slugs (the list endpoint accepts repeated
        `slug` params and returns closed markets too). The value is what token
        0 (the long side: away team / away spread / Over) paid: 1.0 or 0.0.
        Source of truth is /v1/markets/{slug}/settlement; outcomePrices[0] of
        the resolved market is the fallback (outcomePrices follow marketSides,
        long side first, and read "1"/"0" once resolved — verified 2026-09-23).
        Unresolved or unknown markets are simply absent from the result.
        """
        out: Dict[str, float] = {}
        wanted = [s for s in dict.fromkeys(slugs) if s]
        for i in range(0, len(wanted), 50):
            chunk = wanted[i:i + 50]
            data = self._get(f"{self.us_api_url}/v1/markets", {"slug": chunk, "limit": len(chunk)})
            markets = data.get("markets", []) if isinstance(data, dict) else (data or [])
            for m in markets:
                slug = m.get("slug")
                if slug not in chunk or m.get("status") != "MARKET_STATUS_RESOLVED":
                    continue
                value = self._settlement_value(slug, m)
                if value is not None:
                    out[slug] = value
        return out

    def _settlement_value(self, slug: str, market: Dict) -> Optional[float]:
        data = self._get(f"{self.us_api_url}/v1/markets/{slug}/settlement")
        if isinstance(data, dict) and data.get("settlement") is not None:
            try:
                value = float(data["settlement"])
                if 0.0 <= value <= 1.0:
                    return value
            except (TypeError, ValueError):
                pass
        raw = market.get("outcomePrices")
        try:
            if isinstance(raw, str):
                import json as _json
                raw = _json.loads(raw)
            value = float(raw[0])
        except Exception:
            return None
        return value if value in (0.0, 1.0) else None

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
        cache_name = os.path.join("books", f"{slug}.json")
        # 1. streamed book (websocket feed, zero REST calls) — same payload
        #    shape as REST, so the parser below handles both
        data = None
        feed = getattr(self, "book_feed", None)
        if feed is not None:
            payload = feed.get_book(slug)
            if payload is not None:
                data = {"marketData": payload}
                self.books_from_feed = getattr(self, "books_from_feed", 0) + 1
        # 2. shared cache (a sibling process or the feed leader wrote it)
        if data is None:
            data = self._shared_read(cache_name, self.shared_book_ttl)
        # 3. REST, rate-limited
        if data is None:
            data = self._get(url)
            if data:
                self._shared_write(cache_name, data)

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

    def build_snapshot(self, market: Dict, fetch_book: bool = True) -> Optional[MarketSnapshot]:
        """Build a MarketSnapshot from a Polymarket US market dict.

        fetch_book=False builds a LIGHT snapshot from the list price alone
        (empty order book, zero HTTP calls). The trading loop pre-screens
        every market this way and fetches real books only for the few that
        show edge, because the gateway allows ~0.5 book calls per second.
        """
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

            # Fetch order book from US API (or an empty book for a light snapshot)
            order_book = self.get_us_order_book(slug) if fetch_book else OrderBook()

            # No price history needed — signals use order book and external odds
            price_history = [price]

            # Skip markets with no order book data
            has_order_book = order_book.bid_depth > 0 or order_book.ask_depth > 0
            if fetch_book and not has_order_book:
                return None  # a real book that came back empty is untradeable

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
