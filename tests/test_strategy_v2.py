"""2026-09-23 strategy review: live book primary, no re-entry after stops,
round-trip edge, league allowlist, event universe, fresh prices, trailing
toggle, shared caches."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot import trade_db
from utils.config import TradingConfig, SignalConfig, load_config
from utils.models import MarketSnapshot, OrderBook, OrderBookLevel, Position, TradeSignal


def snap(slug="aec-nfl-nyg-lar-2026-09-21", price=0.30, live=True, bid=0.29, ask=0.31):
    ob = OrderBook([OrderBookLevel(bid, 5000)], [OrderBookLevel(ask, 5000)])
    return MarketSnapshot(market_id="m", token_id="t", question="q", price=price, volume_24h=1e5,
                          liquidity=1e5, order_book=ob, price_history=[price],
                          timestamp=datetime.now(timezone.utc), slug=slug, is_live=live)


# ── round-trip edge ─────────────────────────────────────────────────────────

def _signal(net_edge, exec_price=0.51, spread=0.02, edge=0.10):
    s = TradeSignal("m", "t", "buy", 0.6, 0.5, edge, 20, slug="aec-nfl-aaa-bbb-2026-09-27")
    s.net_edge, s.exec_price, s.spread = net_edge, exec_price, spread
    return s


def test_round_trip_gate_requires_exit_fee_and_half_spread():
    from bot.strategies.trade_filter import validate_trade
    from bot.strategies.fees import fee_per_contract
    kw = dict(snapshot=snap(price=0.5, live=False, bid=0.49, ask=0.51), num_books=5, open_game_ids=set(),
              game_id="g", daily_trades=0, max_daily_trades=15, min_liquidity_usd=0, min_net_edge=0.02)
    exit_cost = fee_per_contract(0.51) + 0.01
    assert validate_trade(signal=_signal(0.03), **kw) is None                       # entry-only gate
    reason = validate_trade(signal=_signal(0.03), round_trip=True, **kw)
    assert reason and reason.startswith("net_edge_") and "round_trip" in reason
    assert validate_trade(signal=_signal(0.02 + exit_cost + 0.001), round_trip=True, **kw) is None


# ── trailing stop toggle ────────────────────────────────────────────────────

def _position(side="buy", entry=0.40, qty=100, peak=0.60):
    return Position(market_id="m", token_id="t", side=side, entry_price=entry,
                    size_usd=entry * qty, quantity=qty, estimated_prob=0.9,
                    entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
                    current_price=entry, slug="aec-nfl-aaa-bbb-2026-09-27", peak_price=peak)


def test_trailing_stop_can_be_disabled():
    from bot.strategies.risk import RiskManager
    on = RiskManager(TradingConfig(aggressive_exit_pct=5.0))
    off = RiskManager(TradingConfig(aggressive_exit_pct=5.0, trailing_stop_enabled=False))
    # peak +50% of entry, now given back to +20%: trailing stop territory
    assert on.check_position(_position(), 0.48, 0.9, allow_quote_resolution=False) == "trailing_stop"
    assert off.check_position(_position(), 0.48, 0.9, allow_quote_resolution=False) != "trailing_stop"


# ── live sportsbook consensus as the in-game primary ────────────────────────

def _entries(live_books, pregame_books=(), away_prob=0.62):
    rows = [{"home_team": "Los Angeles Rams", "away_team": "New York Giants",
             "home_prob": 1 - away_prob, "away_prob": away_prob, "book": b, "live": True}
            for b in live_books]
    rows += [{"home_team": "Los Angeles Rams", "away_team": "New York Giants",
              "home_prob": 0.73, "away_prob": 0.27, "book": b, "live": False} for b in pregame_books]
    return rows


def _aggregator(rows):
    from bot.signals.book_scrapers import MultiBookAggregator
    agg = MultiBookAggregator(cache_ttl=999)
    agg.fanduel.get_odds = lambda sk, **kw: [r for r in rows if r["book"] == "fanduel"]
    agg.pinnacle.get_odds = lambda sk, **kw: [r for r in rows if r["book"] == "pinnacle"]
    agg.actionnetwork.get_odds = lambda sk, **kw: [r for r in rows if r["book"].startswith("an_")]
    return agg


def test_live_consensus_uses_only_in_play_quotes():
    agg = _aggregator(_entries(["pinnacle", "an_draftkings"], pregame_books=["fanduel"]))
    res = agg.live_consensus("americanfootball_nfl", "nyg", "lar", max_age=20)
    assert res["num_books"] == 2 and set(res["books"]) == {"pinnacle", "an_draftkings"}
    assert res["prob"] == pytest.approx(0.62)
    assert agg.live_consensus("americanfootball_nfl", "lar", "nyg")["prob"] == pytest.approx(0.38)  # flipped
    assert _aggregator(_entries([], pregame_books=["fanduel"])).live_consensus(
        "americanfootball_nfl", "nyg", "lar") is None


def _odds_cache(rows):
    from bot.signals.odds_api import OddsCache
    oc = OddsCache(api_key="", cache_ttl=999)
    oc._multi_book = _aggregator(rows)
    return oc


def test_live_odds_value_needs_two_live_books_and_prices_off_them():
    from bot.signals.signals import odds_value_signal
    cfg = SignalConfig(live_primary="books")
    one = odds_value_signal(snap(price=0.55), cfg, _odds_cache(_entries(["pinnacle"], ["fanduel", "an_betmgm"])))
    assert one.confidence == 0 and one.metadata["reason"] == "only_1_live_books"
    two = odds_value_signal(snap(price=0.58), cfg, _odds_cache(_entries(["pinnacle", "fanduel"])))
    assert two.confidence > 0 and two.metadata["live"] and two.value == pytest.approx(0.62)
    assert two.metadata["edge"] == pytest.approx(0.04)


def test_books_mode_keeps_odds_value_primary_and_caps_espn_weight():
    from bot.signals.estimator import ProbabilityEstimator
    from utils.models import Signal
    live_sig = Signal("live_win_prob", 0.9, 0.9, "bullish", {"edge": 0.6})
    odds = Signal("odds_value", 0.62, 0.5, "bullish", {"edge": 0.04})
    books = ProbabilityEstimator(SignalConfig(live_primary="books"))
    espn = ProbabilityEstimator(SignalConfig(live_primary="espn"))
    assert books._get_primary_signal([live_sig, odds], "sports").name == "odds_value"
    assert espn._get_primary_signal([live_sig, odds], "sports").name == "live_win_prob"
    assert books._effective_weights("sports")["live_win_prob"] == pytest.approx(0.15)
    assert espn._effective_weights("sports")["live_win_prob"] == pytest.approx(0.55)


# ── shared HTTP cache ───────────────────────────────────────────────────────

def test_http_cache_shares_responses_across_processes(tmp_path, monkeypatch):
    from bot import http_cache
    monkeypatch.setenv("POLYBOT_SHARED_DIR", str(tmp_path))
    calls = []

    class Resp:
        status_code = 200
        text = '{"n": 1}'
    monkeypatch.setattr(http_cache.requests, "get", lambda *a, **k: calls.append(1) or Resp())
    http_cache._memo.clear()
    assert http_cache.get_json("https://x/y", {"a": 1}, max_age=60) == {"n": 1}
    http_cache._memo.clear()                     # a different process: no memo, shared file hit
    assert http_cache.get_json("https://x/y", {"a": 1}, max_age=60) == {"n": 1}
    assert len(calls) == 1
    http_cache._memo.clear()
    old = time.time() - 120
    for f in (tmp_path / "http").iterdir():
        os.utime(f, (old, old))
    http_cache.get_json("https://x/y", {"a": 1}, max_age=20)   # live caller: too old → refetch
    assert len(calls) == 2


# ── event universe + league allowlist ───────────────────────────────────────

def _market(slug, start, price="0.45", **extra):
    m = {"id": slug, "slug": slug, "question": slug, "gameStartTime": start, "active": True,
         "closed": False, "outcomePrices": json.dumps([price, str(1 - float(price))]),
         "marketSides": [{"identifier": slug, "long": True, "price": price, "team": {"abbreviation": "x"}}],
         "bestBidQuote": {"value": "0.44"}, "bestAskQuote": {"value": "0.46"}}
    m.update(extra)
    return m


def test_event_universe_keeps_game_markets_for_allowed_leagues(tmp_path, monkeypatch):
    from bot.market_data import MarketDataClient
    monkeypatch.delenv("POLYBOT_SHARED_DIR", raising=False)
    cfg = load_config()
    cfg.filters.leagues = "nfl,cfb"
    md = MarketDataClient(cfg.api, None, cfg.filters)
    soon = (datetime.now(timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = {
        "nfl": [{"startTime": soon, "markets": [
            _market("aec-nfl-nyg-lar-2026-09-27", soon),
            _market("asc-nfl-nyg-lar-2026-09-27-pos-6pt5", soon),
            _market("asc-nfl-nyg-lar-2026-09-27-1h-pos-3pt5", soon),        # first half: dropped
            _market("astatc-nfl-nyg-lar-2026-09-27-tds-2pt5", soon),         # prop: dropped
            _market("aec-nfl-nyg-lar-2026-09-27-x", soon)]}],                # malformed: dropped
        "cfb": [{"startTime": soon, "markets": [_market("aec-cfb-army-templ-2026-09-27", soon)]}],
    }
    tags = []

    def fake_get(url, params=None):
        assert url.endswith("/v1/events")
        tags.append(params["tagSlug"])
        return {"events": events.get(params["tagSlug"], [])}
    md._get = fake_get
    out = md.get_active_markets()
    assert sorted(tags) == ["cfb", "nfl"]
    assert {m["slug"] for m in out} == {"aec-nfl-nyg-lar-2026-09-27", "asc-nfl-nyg-lar-2026-09-27-pos-6pt5",
                                        "aec-cfb-army-templ-2026-09-27"}
    assert md.active_slugs_complete and "aec-cfb-army-templ-2026-09-27" in md.active_slugs


def test_league_allowlist_drops_other_sports_and_non_sports_in_fallback():
    from bot.market_data import MarketDataClient
    cfg = load_config()
    md = MarketDataClient(cfg.api, None, cfg.filters)
    soon = (datetime.now(timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    markets = [_market("aec-nfl-nyg-lar-2026-09-27", soon), _market("aec-mlb-nym-atl-2026-09-27", soon),
               {"slug": "will-x-happen", "endDate": soon, "question": "politics"}]
    kept = md._filter_markets(markets, {"nfl", "cfb", "nhl"})
    assert [m["slug"] for m in kept] == ["aec-nfl-nyg-lar-2026-09-27"]


def test_snapshot_prices_off_the_book_mid_then_list_quotes():
    from bot.market_data import MarketDataClient
    cfg = load_config()
    md = MarketDataClient(cfg.api, None, cfg.filters)
    soon = (datetime.now(timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = _market("aec-nfl-nyg-lar-2026-09-27", soon, price="0.30")
    md.get_cached_book = lambda slug: OrderBook([OrderBookLevel(0.52, 10)], [OrderBookLevel(0.54, 10)])
    assert md.build_snapshot(m, fetch_book=False).price == pytest.approx(0.53)
    md.get_cached_book = lambda slug: None
    assert md.build_snapshot(m, fetch_book=False).price == pytest.approx(0.45)   # list best bid/ask mid


# ── no re-entry after a stop loss; shared feed subscription ─────────────────

def test_stop_loss_blocks_reentry_and_survives_restart():
    from bot.trading_loop import TradingBot
    bot = TradingBot.__new__(TradingBot)
    bot.config = SimpleNamespace(trading=TradingConfig(reentry_after_stop=False))
    bot.logger = Mock()
    bot._block_reentry("aec-nfl-nyg-lar-2026-09-21", "stop_loss")
    assert "aec-nfl-nyg-lar-2026-09-21" in bot._reentry_blocked
    assert trade_db.load_reentry_blocks() == {"aec-nfl-nyg-lar-2026-09-21"}
    allow = TradingBot.__new__(TradingBot)
    allow.config = SimpleNamespace(trading=TradingConfig(reentry_after_stop=True))
    allow.logger = Mock()
    allow._block_reentry("aec-nfl-kc-den-2026-09-21", "stop_loss")
    assert not getattr(allow, "_reentry_blocked", set())


def test_blocked_market_is_skipped_by_prescreen_and_pipeline():
    from bot.trading_loop import TradingBot
    bot = TradingBot.__new__(TradingBot)
    bot.running = True
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [])
    bot._reentry_blocked = {"aec-nfl-nyg-lar-2026-09-21"}
    bot.market_data = Mock()
    bot.logger = Mock()
    bot._detect_edge = Mock(return_value=("sig", "snap"))
    keep = bot._prescreen([{"slug": "aec-nfl-nyg-lar-2026-09-21"}, {"slug": "aec-nfl-kc-den-2026-09-21"}], "t")
    assert [m["slug"] for m in keep] == ["aec-nfl-kc-den-2026-09-21"]


def test_feed_leader_streams_every_profiles_held_markets(tmp_path, monkeypatch):
    from bot.trading_loop import TradingBot
    monkeypatch.setenv("POLYBOT_SHARED_DIR", str(tmp_path))
    held = tmp_path / "held"
    held.mkdir()
    (held / "ride.json").write_text(json.dumps({"t": time.time(), "slugs": ["asc-nfl-a-b-2026-09-27-pos-3pt5"]}))
    (held / "old.json").write_text(json.dumps({"t": time.time() - 3600, "slugs": ["stale-slug"]}))
    bot = TradingBot.__new__(TradingBot)
    bot._cached_markets = [{"slug": "aec-nfl-a-b-2026-09-27"}]
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [SimpleNamespace(slug="aec-nhl-c-d-2026-09-27")])
    bot.book_feed = Mock()
    bot._refresh_feed_subscription()
    assert set(bot.book_feed.set_slugs.call_args[0][0]) == {
        "aec-nfl-a-b-2026-09-27", "aec-nhl-c-d-2026-09-27", "asc-nfl-a-b-2026-09-27-pos-3pt5"}
    monkeypatch.setenv("POLYBOT_PROFILE", "no_trail")
    bot._publish_held()
    assert json.loads((held / "no_trail.json").read_text())["slugs"] == ["aec-nhl-c-d-2026-09-27"]
