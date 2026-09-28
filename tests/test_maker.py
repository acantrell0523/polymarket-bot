"""2026-09-26: the maker strategy (bot/maker.py) and its paper fill model."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot.maker import Maker, quote_prices, tape
from bot.market_data import MarketDataClient
from bot.strategies.fees import booked_fee_usd
from utils.config import TradingConfig
from utils.models import Position

ML = "aec-nfl-kc-mia-2026-09-27"
SPREAD = "asc-nfl-kc-mia-2026-09-27-neg-3pt5"


def _payload(bids, asks, shares=1000.0, last_px=0.50, last_qty=5.0, last_time="t0"):
    return {"marketSlug": "x", "state": "MARKET_STATE_OPEN",
            "bids": [{"px": {"value": f"{p:.4f}"}, "qty": f"{q:.4f}"} for p, q in bids],
            "offers": [{"px": {"value": f"{p:.4f}"}, "qty": f"{q:.4f}"} for p, q in asks],
            "stats": {"sharesTraded": f"{shares:.4f}", "lastTradePx": {"value": f"{last_px:.4f}"},
                      "lastTradeQty": f"{last_qty:.4f}", "lastTradeSetTime": last_time}}


def _market(slug, hours=20.0):
    start = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"slug": slug, "id": "1", "question": slug, "gameStartTime": start}


def _bot(payloads, positions=(), **cfg):
    from bot.trading_loop import TradingBot
    from bot.strategies.risk import RiskManager
    bot = TradingBot.__new__(TradingBot)
    cfg = {"maker_wide_seconds": 0.0, "maker_live_quotes": True, **cfg}
    bot.config = SimpleNamespace(trading=TradingConfig(strategy="maker", hold_to_settlement=True, **cfg))
    bot.logger = Mock()
    bot.risk = RiskManager(bot.config.trading)
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: list(positions), get_total_exposure=lambda: 0.0,
                                    bankroll=1000.0, open_position=Mock(), close_position=Mock(return_value=1.5),
                                    close_partial=Mock(return_value=0.5))
    md = Mock()
    md.book_feed = None
    md._leader_streams.return_value = True
    md._shared_read.side_effect = lambda name, ttl: (
        {"marketData": payloads[name.split("/")[-1][:-5]]} if name.split("/")[-1][:-5] in payloads else None)
    md._parse_us_book = MarketDataClient._parse_us_book
    md._parse_datetime = MarketDataClient._parse_datetime.__get__(md)
    md.build_snapshot.return_value = SimpleNamespace(slug=ML, is_live=False, category="sports")
    bot.market_data = md
    bot.lines_cache = Mock()
    bot.lines_cache.price_market.return_value = {"prob": 0.55, "num_books": 4}
    bot.odds_cache = Mock()
    bot.odds_cache.get_probability_for_slug.return_value = (0.50, 4)
    bot.entries_paused_until = None
    bot._log_decision = Mock()
    return bot


def _position(side="buy", qty=40.0, price=0.47):
    return Position(market_id="1", token_id=ML, side=side, entry_price=price, size_usd=price * qty,
                    quantity=qty, estimated_prob=0.5, entry_time=datetime.now(timezone.utc),
                    current_price=price, slug=ML, peak_price=price)


# ── quote placement ─────────────────────────────────────────────────────────

def test_quotes_sit_inside_the_consensus_and_inside_the_book():
    assert quote_prices(0.50, 0.40, 0.60, 0.03, 0.02) == (0.47, 0.53)
    assert quote_prices(0.50, 0.46, 0.60, 0.03, 0.02) == (0.47, 0.53)     # still the best bid by 1c
    assert quote_prices(0.50, 0.48, 0.60, 0.03, 0.02) == (None, 0.53)     # would leave under 2c of edge
    assert quote_prices(0.50, 0.40, 0.52, 0.03, 0.02) == (0.47, None)
    assert quote_prices(0.50, 0.49, 0.51, 0.03, 0.02) == (None, None)     # nothing to improve
    assert quote_prices(0.05, 0.01, 0.20, 0.03, 0.02) == (0.02, 0.08)
    assert quote_prices(0.03, 0.005, 0.20, 0.03, 0.02) == (None, 0.06)    # no bids under 2c


def test_tape_reads_the_streamed_stats():
    assert tape(_payload([(0.4, 10)], [(0.6, 10)], 1234.5, 0.45, 7.0, "T")) == (1234.5, 0.45, 7.0, "T")
    assert tape({"stats": {}}) == (0.0, None, 0.0, "")


def test_refresh_quotes_wide_priced_books_only(monkeypatch):
    rows = []
    monkeypatch.setattr("bot.trade_db.insert_maker_log", lambda *a: rows.append(a))
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)]), SPREAD: _payload([(0.45, 100)], [(0.70, 100)]),
                "aec-nfl-buf-nyj-2026-09-27": _payload([(0.30, 10)], [(0.70, 10)]),
                "aec-nfl-sf-ari-2026-09-27": _payload([(0.40, 10)], [(0.60, 10)])}
    bot = _bot(payloads)
    bot.odds_cache.get_probability_for_slug.side_effect = lambda s: (0.50, 2) if "buf" in s else (0.50, 4)
    m = Maker(bot)
    m.refresh([_market(ML), _market(SPREAD), _market("aec-nfl-buf-nyj-2026-09-27"),
               _market("aec-nfl-sf-ari-2026-09-27", hours=0.1), {"slug": "tsc-nfl-kc-mia-2026-09-27-total-44pt5"}])
    assert set(m.quotes) == {ML, SPREAD}                      # 2 books: out; 6 min to kickoff: out; no start: out
    q = m.quotes[ML]
    assert (q.bid, q.ask) == (0.47, 0.53) and q.bid_contracts == 53 and q.ask_contracts == 53
    assert m.quotes[SPREAD].fair == 0.55 and m.quotes[SPREAD].source == "lines"
    assert [r[9] for r in rows] == ["quote", "quote"]
    m.refresh([_market(ML)])                                   # unchanged quote stays, the other is pulled
    assert set(m.quotes) == {ML} and rows[-1][9] == "pull" and rows[-1][1] == SPREAD


def test_refresh_pulls_everything_when_entries_are_blocked():
    bot = _bot({ML: _payload([(0.40, 500)], [(0.60, 500)])})
    m = Maker(bot)
    m.refresh([_market(ML)])
    assert ML in m.quotes
    bot.entries_paused_until = datetime.now(timezone.utc)
    m.refresh([_market(ML)])
    assert m.quotes == {}


# ── fills ───────────────────────────────────────────────────────────────────

def test_a_print_at_or_below_the_bid_fills_it_at_our_price():
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)], shares=1000.0)}
    bot = _bot(payloads)
    m = Maker(bot)
    m.refresh([_market(ML)])
    payloads[ML] = _payload([(0.40, 460)], [(0.60, 500)], shares=1040.0, last_px=0.40, last_qty=40.0, last_time="t1")
    m.poll()
    sig, trade = bot.portfolio.open_position.call_args.args
    assert trade.side == "buy" and trade.price == 0.47 and trade.quantity == 40
    assert trade.fees == pytest.approx(booked_fee_usd(40, 0.47, -0.0125)) and trade.fees < 0
    assert sig.exec_price == 0.47 and sig.estimated_prob == 0.50
    assert bot.risk.daily_trade_count == 1
    assert m.quotes[ML].bid is None and m.quotes[ML].ask == 0.53        # ask still working
    m.poll()                                                            # same tape: nothing new
    assert bot.portfolio.open_position.call_count == 1


def test_a_print_between_the_quotes_fills_nothing():
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads)
    m = Maker(bot)
    m.refresh([_market(ML)])
    payloads[ML] = _payload([(0.40, 500)], [(0.60, 500)], shares=1010.0, last_px=0.50, last_qty=10.0, last_time="t1")
    m.poll()
    bot.portfolio.open_position.assert_not_called()


def test_a_bid_posted_through_our_ask_fills_the_ask():
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads)
    m = Maker(bot)
    m.refresh([_market(ML)])
    payloads[ML] = _payload([(0.54, 20), (0.40, 500)], [(0.60, 500)])        # no trade, a crossing bid
    m.poll()
    sig, trade = bot.portfolio.open_position.call_args.args
    assert trade.side == "sell" and trade.price == 0.53 and trade.quantity == 20
    assert trade.size_usd == pytest.approx(20 * 0.47)


def test_the_opposite_quote_flips_inventory_and_books_the_spread():
    pos = _position("buy", 40.0, 0.47)
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads, positions=[pos])
    m = Maker(bot)
    m.refresh([_market(ML)])
    assert m.quotes[ML].bid is None and m.quotes[ML].ask == 0.53          # long: only the ask works
    payloads[ML] = _payload([(0.40, 500)], [(0.60, 500)], shares=1060.0, last_px=0.56, last_qty=60.0, last_time="t1")
    m.poll()
    args, kw = bot.portfolio.close_position.call_args
    assert args[0] is pos and args[1] == 0.53 and args[2] == "maker_flip" and kw["exit_fees"] < 0
    bot.portfolio.open_position.assert_not_called()
    assert bot.risk.daily_pnl == pytest.approx(1.5) and ML not in m.quotes


def test_kickoff_pulls_the_quote():
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads)
    m = Maker(bot)
    m.refresh([_market(ML)])
    m.quotes[ML].game_start = datetime.now(timezone.utc) - timedelta(seconds=1)
    m.poll()
    assert m.quotes == {}


def test_no_fill_when_the_leader_is_not_streaming_the_market():
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads)
    m = Maker(bot)
    m.refresh([_market(ML)])
    bot.market_data._leader_streams.return_value = False
    payloads[ML] = _payload([(0.40, 500)], [(0.60, 500)], shares=1040.0, last_px=0.40, last_qty=40.0, last_time="t1")
    m.poll()
    bot.portfolio.open_position.assert_not_called()


# ── live alternate lines ────────────────────────────────────────────────────

def test_live_lines_are_quoted_off_the_live_model_and_pulled_late(monkeypatch):
    live_slug = "asc-cfb-tx-tenn-2026-09-26-pos-14pt5"
    payloads = {live_slug: _payload([(0.30, 40)], [(0.60, 40)]), ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads)
    bot.lines_cache.price_market.side_effect = lambda parsed, live: {"prob": 0.45, "num_books": 3, "clock_known": True} if live else None
    bot.game_state = Mock()
    bot.game_state.state_for.return_value = {"state": "in", "league": "cfb", "period": 3, "clock_seconds": 600}
    m = Maker(bot)
    m.refresh([_market(live_slug, hours=-1.0), _market(ML), _market("aec-cfb-tx-tenn-2026-09-26", hours=-1.0)])
    assert set(m.quotes) == {live_slug, ML}                     # live moneyline: never quoted
    q = m.quotes[live_slug]
    assert q.is_live and q.source == "lines_live" and (q.bid, q.ask) == (0.42, 0.48)
    m.refresh([_market(live_slug, hours=-1.0)], live_only=True)   # a live-only pass keeps the pregame quote
    assert set(m.quotes) == {live_slug, ML}
    bot.game_state.state_for.return_value = {"state": "in", "league": "cfb", "period": 4, "clock_seconds": 200}
    m.refresh([_market(live_slug, hours=-1.0)], live_only=True)   # 3:20 left: pulled
    assert set(m.quotes) == {ML}


def test_a_book_must_stay_wide_before_it_is_quoted(monkeypatch):
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)])}
    bot = _bot(payloads, maker_wide_seconds=300.0)
    m = Maker(bot)
    m.refresh([_market(ML)])
    assert m.quotes == {}                                    # first sighting of a wide book: wait
    m._wide_since[ML] -= 400
    m.refresh([_market(ML)])
    assert ML in m.quotes                                    # wide for 5 minutes: quote
    payloads[ML] = _payload([(0.49, 500)], [(0.51, 500)])    # the maker is back: pulled, clock reset
    m.refresh([_market(ML)])
    assert m.quotes == {} and ML not in m._wide_since


def test_live_quoting_is_off_by_default():
    live_slug = "asc-cfb-tx-tenn-2026-09-26-pos-14pt5"
    bot = _bot({live_slug: _payload([(0.30, 40)], [(0.60, 40)])}, maker_live_quotes=False)
    bot.lines_cache.price_market.return_value = {"prob": 0.45, "num_books": 3, "clock_known": True}
    m = Maker(bot)
    m.refresh([_market(live_slug, hours=-1.0)])
    assert m.quotes == {}


def test_leader_writes_a_price_tape(tmp_path, monkeypatch):
    from bot.book_feed import BookFeed, TAPE_MIN_GAP
    import json as _json
    feed = BookFeed("k", "s", logger=None, shared_dir=str(tmp_path))
    payload = _payload([(0.40, 100)], [(0.60, 50)], shares=10.0, last_px=0.5, last_qty=2.0, last_time="t0")
    payload["marketSlug"] = ML
    feed.handle_market_data({"marketData": payload})
    feed.handle_market_data({"marketData": payload})                     # unchanged: no second line
    files = list((tmp_path / "tape").glob("*.jsonl"))
    rows = [_json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["s"] == ML and rows[0]["b"] == 0.40 and rows[0]["a"] == 0.60
    assert rows[0]["lp"] == "0.5000" and rows[0]["st"] == "10.0000"
    feed._last_tape[ML] = (feed._last_tape[ML][0] - TAPE_MIN_GAP - 1, feed._last_tape[ML][1])
    payload["bids"][0]["px"]["value"] = "0.4100"
    feed.handle_market_data({"marketData": payload})
    assert len(files[0].read_text().splitlines()) == 2


def test_one_auth_rejection_does_not_disable_the_feed(monkeypatch):
    import asyncio, types, sys
    from bot import book_feed as bf
    attempts = []

    class FakeWS:
        def __init__(self, **kw): pass
        def on(self, *a): pass
        async def connect(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("HTTP 401 Unauthorized")
            raise RuntimeError("connection refused")           # any other error: normal backoff
        async def close(self): pass
    fake_mod = types.SimpleNamespace(MarketsWebSocket=FakeWS)
    monkeypatch.setitem(sys.modules, "polymarket_us.websocket", fake_mod)
    monkeypatch.setattr(bf, "AUTH_RETRY_SECONDS", 0.0)
    sleeps = []
    async def fake_sleep(s): sleeps.append(s); (_ for _ in ()).throw(SystemExit) if len(sleeps) >= 2 else None
    monkeypatch.setattr(bf.asyncio, "sleep", fake_sleep)
    feed = bf.BookFeed("k", "s", logger=None)
    try:
        asyncio.run(feed._main())
    except SystemExit:
        pass
    assert feed.enabled and feed.disabled_reason == "" and len(attempts) == 2


def test_maker_holds_one_position_per_game():
    pos = _position("buy", 40.0, 0.47)                      # long the moneyline of KC-MIA
    other = "asc-nfl-kc-mia-2026-09-27-neg-3pt5"
    payloads = {ML: _payload([(0.40, 500)], [(0.60, 500)]), other: _payload([(0.40, 100)], [(0.60, 100)])}
    bot = _bot(payloads, positions=[pos])
    m = Maker(bot)
    m.refresh([_market(ML), _market(other)])
    assert set(m.quotes) == {ML}                            # the spread of the same game is not quoted
    assert m.quotes[ML].bid is None and m.quotes[ML].ask == 0.53
