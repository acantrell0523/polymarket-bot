#!/usr/bin/env python3
"""One-shot websocket check: connect with the .env API key, subscribe to a
few live markets, print the first books. Run after generating a new key at
polymarket.us/developer:  python scripts/ws_smoke.py aec-nfl-nyg-lar-2026-09-21
"""
import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from bot.book_feed import BookFeed

slugs = sys.argv[1:] or ["aec-nfl-nyg-lar-2026-09-21"]
feed = BookFeed(os.environ.get("POLYMARKET_KEY_ID", ""), os.environ.get("POLYMARKET_SECRET_KEY", ""))
feed.set_slugs(slugs)
feed.start()
for i in range(20):
    time.sleep(1)
    st = feed.status()
    print(f"{i+1:2}s  connected={st['connected']} books={st['books']} messages={st['messages']} {st['disabled_reason'] or ''}")
    if st["books"]:
        for s in slugs:
            b = feed.get_book(s)
            if b:
                bids, offers = b.get("bids", []), b.get("offers", [])
                print(f"   {s}: best bid {bids[0] if bids else None} best offer {offers[0] if offers else None}")
        break
    if not feed.enabled:
        break
feed.stop()
