"""Spread/total pricing (bot/signals/lines.py)."""
import time
from datetime import datetime, timezone

import pytest

from bot.signals import lines as L
from utils.models import MarketSnapshot, OrderBook, Signal


def _snap(slug, price, live=False):
    ob = OrderBook()
    return MarketSnapshot(
        market_id="m1", token_id="t1", question="q", price=price, volume_24h=1000.0,
        liquidity=1000.0, order_book=ob, price_history=[price],
        timestamp=datetime.now(timezone.utc), slug=slug, is_live=live,
    )


class TestParse:
    def test_spread_neg_is_away_favored(self):
        p = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-neg-8pt5")
        assert p == {"kind": "spread", "league": "nfl", "away": "ind", "home": "kc",
                     "date": "2026-09-20", "line": -8.5}

    def test_spread_pos(self):
        assert L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-2pt5")["line"] == 2.5

    def test_total(self):
        p = L.parse_line_slug("tsc-nfl-phi-ten-2026-09-20-total-44pt5")
        assert p["kind"] == "total" and p["line"] == 44.5

    @pytest.mark.parametrize("slug", [
        "asc-nfl-car-atl-2026-09-20-1h-neg-10pt5",   # first half
        "tsc-nfl-ind-kc-2026-09-20-1q-5pt5",         # quarter total
        "tsc-nfl-phi-ten-2026-09-20-tt-phi-10pt5",   # team total
        "aec-nfl-ind-kc-2026-09-20",                 # moneyline
        "asc-nfl-ind-kc-2026-09-20-neg-x",
    ])
    def test_unsupported(self, slug):
        assert L.parse_line_slug(slug) is None

    def test_is_line_market(self):
        assert L.is_line_market("asc-nfl-a-b-2026-09-20-pos-1pt5")
        assert not L.is_line_market("aec-nfl-a-b-2026-09-20")


class TestMath:
    def test_pickem_is_coin_flip(self):
        assert L.prob_away_covers(mu=0.0, sigma=13.5, line=0.0) == pytest.approx(0.5)

    def test_favorite_covers_small_line_more_than_half(self):
        # away expected to win by 7; covering -3.5 should be > 50%
        assert L.prob_away_covers(mu=7.0, sigma=13.5, line=-3.5) > 0.55

    def test_over_symmetry(self):
        assert L.prob_over(mu=44.5, sigma=10, line=44.5) == pytest.approx(0.5)
        assert L.prob_over(mu=44.5, sigma=10, line=40.5) > 0.6

    def test_mu_inversion_roundtrip(self):
        mu = L.mu_from_spread_quote(away_points=2.5, p_away=0.5, sigma=13.5)
        assert mu == pytest.approx(-2.5)
        mu_t = L.mu_from_total_quote(line=45.5, p_over=0.5, sigma=10)
        assert mu_t == pytest.approx(45.5)

    def test_devig(self):
        assert L.devig(0.55, 0.55) == pytest.approx(0.5)


def _cache_with(quotes_spread=(), quotes_total=(), schedule=None):
    lc = L.LinesCache(cache_ttl=999, game_schedule=schedule)
    lc._games["nfl"] = (time.time(), {"ind@kc": {"spread": list(quotes_spread),
                                                 "total": list(quotes_total)}})
    return lc


def _q(book, points, p, main=True, live=False, kind="spread"):
    return {"book": book, "kind": kind, "points": points, "p": p, "main": main, "live": live}


class TestPricing:
    def test_prices_token0_as_away_side(self):
        # Books: Colts +6 (away underdog). Token 0 of neg-8pt5 = Colts -8.5 → small.
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5)])
        p = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-neg-8pt5")
        r = lc.price_market(p, is_live=False)
        assert r["num_books"] == 2 and r["mu"] == pytest.approx(-6.0)
        assert r["prob"] < 0.20
        # Colts +8.5 (a 6-point dog getting 8.5) → better than a coin flip
        p2 = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-8pt5")
        assert 0.55 < lc.price_market(p2, is_live=False)["prob"] < 0.65

    def test_exact_line_quote_blends_in(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5),
                          _q("pinnacle", 2.5, 0.30, main=False)])
        p = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-2pt5")
        r = lc.price_market(p, is_live=False)
        assert r["exact_line_quotes"] == 1
        model_only = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5)]) \
            .price_market(p, is_live=False)["prob"]
        assert r["prob"] < model_only  # the 0.30 quote pulls it down

    def test_live_uses_only_live_quotes(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5)])
        p = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-2pt5")
        assert lc.price_market(p, is_live=True) is None  # pregame lines are stale
        lc2 = _cache_with([_q("pinnacle", -3.0, 0.5, live=True), _q("fanduel", -3.0, 0.5, live=True)])
        r = lc2.price_market(p, is_live=True)
        assert r is not None and r["live"] and not r["clock_known"]
        assert r["sigma"] < 13.5  # shrunk (unknown clock → half game)

    def test_live_sigma_scales_with_clock(self):
        class Sched:
            def get_game_time_remaining(self, league, home, away):
                return 900.0  # 15 minutes left of 60
        lc = _cache_with([_q("pinnacle", -3.0, 0.5, live=True), _q("fanduel", -3.0, 0.5, live=True)],
                         schedule=Sched())
        p = L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-2pt5")
        r = lc.price_market(p, is_live=True)
        assert r["clock_known"] and r["sigma"] == pytest.approx(13.5 * 0.5)

    def test_total(self):
        lc = _cache_with(quotes_total=[_q("pinnacle", 45.5, 0.5, kind="total"),
                                       _q("espn_dk", 45.5, 0.5, kind="total")])
        p = L.parse_line_slug("tsc-nfl-ind-kc-2026-09-20-total-41pt5")
        r = lc.price_market(p, is_live=False)
        assert r["prob"] > 0.6  # over 41.5 when the books say 45.5

    def test_unknown_game_or_league(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5)])
        assert lc.price_market(L.parse_line_slug("asc-nfl-car-atl-2026-09-20-pos-2pt5"), False) is None
        assert lc.price_market({**L.parse_line_slug("asc-nfl-ind-kc-2026-09-20-pos-2pt5"), "league": "xyz"}, False) is None


class TestSignal:
    def test_blocks_without_two_books(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5)])
        s = L.spread_total_signal(_snap("asc-nfl-ind-kc-2026-09-20-pos-2pt5", 0.40), None, lc)
        assert s.confidence == 0 and s.metadata["reason"] == "only_1_books"

    def test_blocks_big_edge_with_two_books(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5)])
        s = L.spread_total_signal(_snap("asc-nfl-ind-kc-2026-09-20-pos-2pt5", 0.20), None, lc)
        assert s.confidence == 0 and "needs_3_books" in s.metadata["reason"]

    def test_emits_edge_and_books(self):
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5), _q("espn_dk", 6.0, 0.5)])
        s = L.spread_total_signal(_snap("asc-nfl-ind-kc-2026-09-20-pos-2pt5", 0.30), None, lc)
        assert s.name == "spread_total" and s.confidence > 0
        assert s.metadata["num_books"] == 3
        assert s.metadata["edge"] == pytest.approx(s.value - 0.30)
        assert s.direction == "bullish"

    def test_unsupported_slug(self):
        s = L.spread_total_signal(_snap("tsc-nfl-ind-kc-2026-09-20-1q-5pt5", 0.5), None, _cache_with())
        assert s.confidence == 0 and s.metadata["reason"] == "unsupported_line_market"


class TestEstimatorRouting:
    def test_line_market_uses_spread_total_as_primary(self):
        from bot.signals.estimator import ProbabilityEstimator
        from utils.config import SignalConfig
        lc = _cache_with([_q("pinnacle", 6.0, 0.5), _q("fanduel", 6.0, 0.5), _q("espn_dk", 6.0, 0.5)])
        est = ProbabilityEstimator(SignalConfig(), odds_cache=None, lines_cache=lc)
        snap = _snap("asc-nfl-ind-kc-2026-09-20-pos-2pt5", 0.30)
        sigs = est.compute_signals(snap, "sports")
        names = {s.name for s in sigs}
        assert "spread_total" in names and "odds_value" not in names
        primary = est._get_primary_signal(sigs, "sports")
        assert primary.name == "spread_total"
        ts = est.detect_edge(snap, min_edge=0.03, max_edge=0.4)
        assert ts is not None and ts.side == "buy"


class TestLightSnapshot:
    def test_light_snapshot_makes_no_book_call(self):
        from bot.market_data import MarketDataClient
        from utils.config import load_config
        cfg = load_config()
        md = MarketDataClient(cfg.api, None, cfg.filters)
        md.get_us_order_book = lambda slug: (_ for _ in ()).throw(AssertionError("book fetched"))
        m = {"id": "1", "slug": "asc-nfl-ind-kc-2026-09-20-pos-2pt5", "question": "q",
             "marketSides": [{"identifier": "x", "long": True, "price": "0.41"}],
             "gameStartTime": "2099-01-01T00:00:00Z"}
        snap = md.build_snapshot(m, fetch_book=False)
        assert snap is not None and snap.price == pytest.approx(0.41)
        assert snap.order_book.bid_depth == 0 and not snap.is_live

    def test_book_route_uses_slow_limiter(self):
        from bot.market_data import MarketDataClient
        from utils.config import load_config
        cfg = load_config()
        md = MarketDataClient(cfg.api, None, cfg.filters)
        assert md._book_min_interval >= 2.0


class TestLiveBookPreference:
    def test_live_quote_replaces_pregame_from_same_book(self):
        from bot.signals.book_scrapers import MultiBookAggregator
        agg = MultiBookAggregator(cache_ttl=999)
        pre = {"home_team": "Calgary Flames", "away_team": "Seattle Kraken", "home_prob": 0.6, "away_prob": 0.4, "book": "pinnacle", "live": False}
        live = {"home_team": "Calgary Flames", "away_team": "Seattle Kraken", "home_prob": 0.8, "away_prob": 0.2, "book": "pinnacle", "live": True}
        agg.fanduel.get_odds = lambda k: []
        agg.actionnetwork.get_odds = lambda k: []   # hermetic: no live Action Network call
        agg.pinnacle.get_odds = lambda k: [pre, live]
        games = agg.get_all_odds("icehockey_nhl")
        assert list(games) == ["sea@cgy"]            # Polymarket codes, league-scoped
        assert games["sea@cgy"][0]["home_prob"] == 0.8  # the live line won

    def test_nhl_codes_normalize(self):
        from bot.leagues import normalize_abbr
        assert normalize_abbr("nhl", "vgk") == "veg" and normalize_abbr("nhl", "wsh") == "was"


class TestKalshi:
    def test_ticker_parsing_and_lookup(self):
        from bot.signals import kalshi as K
        t = K._parse_ticker("KXNFLGAME-26SEP21NYGLAR-NYG")
        assert t["date"] == "2026-09-21" and (t["away"], t["home"]) == ("NYG", "LAR")
        t = K._parse_ticker("KXNFLSPREAD-26SEP27LARDEN-LAR8")
        assert (t["away"], t["home"], t["outcome"]) == ("LAR", "DEN", "LAR")
        c = K.KalshiCache(cache_ttl=999)
        c._index["nfl"] = (9e12, {("nfl", "2026-09-21", "nyg", "lar"): {
            "ml": {"nyg": (0.265, 0.01), "lar": (0.735, 0.01)},
            "spread": {("lar", 6.5): (0.52, 0.02)},
            "total": {47.5: (0.5, 0.02)}}})
        assert c.quote_for_slug("aec-nfl-nyg-lar-2026-09-21")["prob"] == 0.265
        assert c.quote_for_slug("asc-nfl-nyg-lar-2026-09-21-pos-6pt5")["prob"] == pytest.approx(0.48)
        assert c.quote_for_slug("tsc-nfl-nyg-lar-2026-09-21-total-47pt5")["prob"] == 0.5
        assert c.quote_for_slug("aec-nfl-kc-den-2026-09-21") is None

    def test_signal_is_aux_and_width_gated(self):
        from bot.signals import kalshi as K
        c = K.KalshiCache(cache_ttl=999)
        c._index["nfl"] = (9e12, {("nfl", "2026-09-21", "nyg", "lar"): {"ml": {"nyg": (0.30, 0.01)}, "spread": {}, "total": {}}})
        s = K.kalshi_cross_signal(_snap("aec-nfl-nyg-lar-2026-09-21", 0.25), None, c)
        assert s.name == "kalshi_cross" and 0 < s.confidence <= 0.6 and s.metadata["edge"] == pytest.approx(0.05)
        c._index["nfl"] = (9e12, {("nfl", "2026-09-21", "nyg", "lar"): {"ml": {"nyg": (0.30, 0.20)}, "spread": {}, "total": {}}})
        assert K.kalshi_cross_signal(_snap("aec-nfl-nyg-lar-2026-09-21", 0.25), None, c).confidence == 0


class TestActionNetworkLive:
    def _game(self, status, rows):
        return {"status": status, "home_team_id": 1, "away_team_id": 2,
                "teams": [{"id": 1, "full_name": "Kansas City Chiefs"}, {"id": 2, "full_name": "Indianapolis Colts"}],
                "start_time": "2026-09-21T00:20:00Z", "odds": rows}

    def test_live_rows_used_in_progress_with_freshness(self):
        from bot.signals.book_scrapers import ActionNetworkClient
        import datetime as dt
        now = dt.datetime(2026, 9, 21, 3, 0, tzinfo=dt.timezone.utc).timestamp()
        fresh = dt.datetime.fromtimestamp(now - 60, dt.timezone.utc).isoformat()
        stale = dt.datetime.fromtimestamp(now - 900, dt.timezone.utc).isoformat()
        rows = [{"type": "game", "book_id": 68, "ml_home": -280, "ml_away": 230, "inserted": stale},
                {"type": "live", "book_id": 68, "ml_home": -1160, "ml_away": 720, "spread_away": 2.5,
                 "spread_away_line": -110, "spread_home_line": -110, "total": 57.5, "over": -110, "under": -110, "inserted": fresh},
                {"type": "live", "book_id": 75, "ml_home": -900, "ml_away": 600, "inserted": stale}]
        an = ActionNetworkClient()
        games = an.parse_games([self._game("inprogress", rows)], now)
        assert len(games) == 1 and games[0]["live"] is True
        assert [r["book_id"] for r in games[0]["rows"]] == [68]   # stale BetMGM live row dropped, pregame row ignored
        import time as _t
        an._cache["americanfootball_nfl"] = (_t.time(), games)   # fresh cache, no network
        ev = an.get_odds("americanfootball_nfl")
        assert ev[0]["live"] is True and ev[0]["home_prob"] > 0.85  # -1160/+720 de-vigged = 0.883
        quotes = list(an.line_quotes("americanfootball_nfl"))
        assert {q[2]["live"] for q in quotes} == {True} and {q[1] for q in quotes} == {"spread", "total"}

    def test_scheduled_uses_pregame_rows_only(self):
        from bot.signals.book_scrapers import ActionNetworkClient
        rows = [{"type": "game", "book_id": 68, "ml_home": -280, "ml_away": 230, "inserted": "2026-09-20T20:00:00+00:00"},
                {"type": "live", "book_id": 68, "ml_home": -1160, "ml_away": 720, "inserted": "2026-09-20T20:00:00+00:00"}]
        games = ActionNetworkClient().parse_games([self._game("scheduled", rows)])
        assert games[0]["live"] is False and games[0]["rows"][0]["ml_home"] == -280


class TestBookFeed:
    def test_feed_book_replaces_rest_and_expires(self):
        from bot.book_feed import BookFeed
        from bot.market_data import MarketDataClient
        from utils.config import load_config
        cfg = load_config(); md = MarketDataClient(cfg.api, None, cfg.filters)
        feed = BookFeed("k", "s"); md.book_feed = feed
        md._get = lambda url, params=None: (_ for _ in ()).throw(AssertionError("REST called"))
        feed.handle_market_data({"marketData": {"marketSlug": "aec-nfl-a-b-2026-09-21",
                                                 "bids": [{"px": {"value": "0.45"}, "qty": "120"}],
                                                 "offers": [{"px": {"value": "0.47"}, "qty": "80"}]}})
        ob = md.get_us_order_book("aec-nfl-a-b-2026-09-21")
        assert ob.bids[0].price == 0.45 and ob.asks[0].price == 0.47 and md.books_from_feed == 1
        feed.books["aec-nfl-a-b-2026-09-21"] = (0.0, feed.books["aec-nfl-a-b-2026-09-21"][1])  # ancient
        assert feed.get_book("aec-nfl-a-b-2026-09-21") is None   # stale → caller falls back to REST

    def test_feed_disabled_without_key(self):
        from bot.book_feed import BookFeed
        f = BookFeed("", ""); f.start()
        assert not f.enabled and f.status()["disabled_reason"] == "no_api_key" and f._thread is None

    def test_set_slugs_bumps_version_only_on_change(self):
        from bot.book_feed import BookFeed
        f = BookFeed("k", "s"); v0 = f._slugs_version
        f.set_slugs(["b", "a", "a"]); assert f._slugs == ["a", "b"] and f._slugs_version == v0 + 1
        f.set_slugs(["a", "b"]); assert f._slugs_version == v0 + 1
