"""Streaming order books from the Polymarket US websocket (SUBSCRIPTION_TYPE_MARKET_DATA).

Why: the gateway's REST /book endpoint allows ~5 calls per 10 seconds, so the
fast live loop could only refresh a handful of books per pass (25-45s per
cycle). The websocket pushes every book change for every subscribed market,
so a live pass costs zero REST calls and the loop can run at its 3s cadence.

The feed authenticates with the API key pair (POLYMARKET_KEY_ID /
POLYMARKET_SECRET_KEY); a 401 means the key is dead or missing, in which
case the feed disables itself and the client silently keeps using REST.

Design:
  * One background thread runs an asyncio loop: connect → subscribe to the
    current slug universe → store each `marketData` payload (same shape as
    the REST /book response, so MarketDataClient parses both identically)
    → reconnect with backoff on close/error.
  * `set_slugs()` is called by the trading loop after each full scan; a
    changed universe triggers unsubscribe + resubscribe.
  * When POLYBOT_SHARED_DIR is set the leader process also mirrors each
    update into the shared book cache, so sibling strategy processes get
    streamed books through the cache they already read.
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

MAX_AGE = 15.0        # seconds; older than this the book is stale, fall back to REST
MIRROR_MIN_GAP = 1.0  # seconds between shared-cache writes per slug


class BookFeed:
    def __init__(self, key_id: str, secret_key: str, logger=None, shared_dir: Optional[str] = None):
        self.key_id, self.secret_key = key_id, secret_key
        self.logger = logger
        self.shared_dir = shared_dir
        self.books: Dict[str, Tuple[float, dict]] = {}
        self._slugs: List[str] = []
        self._slugs_version = 0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.enabled = bool(key_id and secret_key)
        self.connected = False
        self.disabled_reason = "" if self.enabled else "no_api_key"
        self._last_mirror: Dict[str, float] = {}
        self.messages = 0

    # -- public ---------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="book-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def set_slugs(self, slugs: List[str]) -> None:
        new = sorted(set(s for s in slugs if s))
        with self._lock:
            if new != self._slugs:
                self._slugs = new
                self._slugs_version += 1

    def get_book(self, slug: str, max_age: float = MAX_AGE) -> Optional[dict]:
        """The latest streamed book payload for a slug, or None if absent/stale."""
        entry = self.books.get(slug)
        if not entry or time.time() - entry[0] > max_age:
            return None
        return entry[1]

    def status(self) -> dict:
        return {"enabled": self.enabled, "connected": self.connected, "subscribed": len(self._slugs),
                "books": len(self.books), "messages": self.messages, "disabled_reason": self.disabled_reason}

    # -- message handling (sync, testable) -------------------------------------

    def handle_market_data(self, message: dict) -> Optional[str]:
        payload = message.get("marketData") or {}
        slug = payload.get("marketSlug")
        if not slug:
            return None
        self.books[slug] = (time.time(), payload)
        self.messages += 1
        self._mirror(slug, payload)
        return slug

    def _mirror(self, slug: str, payload: dict) -> None:
        if not self.shared_dir:
            return
        now = time.time()
        if now - self._last_mirror.get(slug, 0.0) < MIRROR_MIN_GAP:
            return
        self._last_mirror[slug] = now
        path = os.path.join(self.shared_dir, "books", f"{slug}.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ws-")
            with os.fdopen(fd, "w") as fh:
                json.dump({"marketData": payload}, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    # -- connection loop -------------------------------------------------------

    def _log(self, level: str, event: str, data: dict) -> None:
        if self.logger:
            getattr(self.logger, level, self.logger.info)(event, data)

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        from polymarket_us.websocket import MarketsWebSocket
        backoff = 2.0
        while not self._stop.is_set():
            ws = MarketsWebSocket(key_id=self.key_id, secret_key=self.secret_key)
            closed = asyncio.Event()
            ws.on("market_data", self.handle_market_data)
            ws.on("close", lambda *a: closed.set())
            ws.on("error", lambda e, *a: self._log("warning", "book_feed_error", {"error": str(e)[:200]}))
            try:
                await ws.connect()
            except Exception as e:
                msg = str(e)
                if "401" in msg or "403" in msg:
                    self.enabled = False
                    self.disabled_reason = "auth_rejected"
                    self._log("warning", "book_feed_disabled", {"reason": "websocket auth rejected (dead or missing API key); using REST books"})
                    return
                self._log("warning", "book_feed_connect_failed", {"error": msg[:200], "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2.0
            self.connected = True
            self._log("info", "book_feed_connected", {})
            subscribed_version = -1
            request_id = None
            try:
                while not self._stop.is_set() and not closed.is_set():
                    with self._lock:
                        version, slugs = self._slugs_version, list(self._slugs)
                    if version != subscribed_version and slugs:
                        if request_id:
                            try:
                                await ws.unsubscribe(request_id)
                            except Exception:
                                pass
                        request_id = f"books-{int(time.time())}-{version}"
                        await ws.subscribe_market_data(request_id, slugs)
                        subscribed_version = version
                        self._log("info", "book_feed_subscribed", {"markets": len(slugs)})
                    await asyncio.sleep(1.0)
            except Exception as e:
                self._log("warning", "book_feed_loop_error", {"error": str(e)[:200]})
            finally:
                self.connected = False
                try:
                    await ws.close()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._log("warning", "book_feed_reconnecting", {"in": backoff})
                await asyncio.sleep(backoff)
