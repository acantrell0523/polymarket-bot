"""2026-09-26: the certainty strategy (bot/certainty.py) and its loop hook."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot import certainty
from utils.config import TradingConfig
from utils.models import OrderBook, OrderBookLevel


def _cfg(**kw):
    return TradingConfig(strategy="certainty", hold_to_settlement=True, **kw)


def _gs(state="in", away=24, home=7, period=4, clock="4:59", league="cfb", completed=False):
    return {"state": state, "completed": completed, "detail": "", "period": period,
            "clock_seconds": certainty._clock_seconds(clock), "league": league,
            "away": "tx", "home": "tenn", "away_score": away, "home_score": home,
            "scores": {"tx": away, "tenn": home}}


ML = "aec-cfb-tx-tenn-2026-09-26"


# ── the decision ────────────────────────────────────────────────────────────

def test_final_buys_the_winner_on_moneylines_spreads_and_totals():
    gs = _gs(state="post", away=31, home=20, completed=True)
    d = certainty.decide(ML, gs, None, _cfg())
    assert d["side"] == "buy" and d["limit"] == 0.99 and d["why"] == "final"
    d = certainty.decide("aec-cfb-tx-tenn-2026-09-26", _gs(state="post", away=20, home=31), None, _cfg())
    assert d["side"] == "sell" and d["limit"] == pytest.approx(0.01)        # home won: short token 0
    assert certainty.decide("asc-cfb-tx-tenn-2026-09-26-neg-10pt5", gs, None, _cfg())["side"] == "buy"   # 31-20 covers -10.5
    assert certainty.decide("asc-cfb-tx-tenn-2026-09-26-neg-11pt5", gs, None, _cfg())["side"] == "sell"  # not -11.5
    assert certainty.decide("tsc-cfb-tx-tenn-2026-09-26-total-50pt5", gs, None, _cfg())["side"] == "buy"  # 51 > 50.5
    assert certainty.decide("tsc-cfb-tx-tenn-2026-09-26-total-51pt5", gs, None, _cfg())["side"] == "sell"
    assert certainty.decide(ML, _gs(state="post", away=20, home=20), None, _cfg()) is None      # tie
    assert certainty.decide("tsc-cfb-tx-tenn-2026-09-26-total-51pt0", gs, None, _cfg()) is None  # push


def test_late_lead_needs_time_margin_and_espn_to_agree():
    cfg = _cfg()
    live = (0.985, 5.0)                                     # ESPN: away 98.5%
    d = certainty.decide(ML, _gs(away=24, home=7, clock="4:59"), live, cfg)
    assert d and d["side"] == "buy" and d["limit"] == 0.96 and d["why"] == "late_lead"
    assert d["seconds_left"] == pytest.approx(299) and d["margin"] == 17
    assert certainty.decide(ML, _gs(away=24, home=7, clock="8:30"), live, cfg) is None      # too early
    assert certainty.decide(ML, _gs(away=24, home=17, clock="4:59"), live, cfg) is None     # one score, 5 min
    assert certainty.decide(ML, _gs(away=24, home=15, clock="3:59"), live, cfg) is not None  # 9 up inside 4:00
    assert certainty.decide(ML, _gs(away=24, home=7, clock="4:59"), (0.95, 5.0), cfg) is None   # ESPN < 97%
    assert certainty.decide(ML, _gs(away=24, home=7, clock="4:59"), (0.985, 90.0), cfg) is None  # stale ESPN
    assert certainty.decide(ML, _gs(away=24, home=7, clock="4:59"), None, cfg) is None         # no ESPN
    assert certainty.decide(ML, _gs(away=24, home=7, period=5, clock="0:00"), live, cfg) is None  # overtime
    home = certainty.decide(ML, _gs(away=7, home=24, clock="4:59"), (0.02, 5.0), cfg)
    assert home["side"] == "sell" and home["limit"] == pytest.approx(0.04) and home["leader"] == "home"
    assert certainty.decide("asc-cfb-tx-tenn-2026-09-26-neg-10pt5", _gs(clock="1:00"), live, cfg) is None  # lines: finals only
    assert certainty.decide(ML, _gs(clock="4:59"), live, _cfg(certainty_live_entries=False)) is None


def test_scoreboard_parsing_keys_games_by_polymarket_codes(monkeypatch):
    payload = {"events": [{"status": {"period": 4, "displayClock": "2:10", "type": {"state": "in", "completed": False, "shortDetail": "2:10 - 4th"}},
                           "competitions": [{"competitors": [
                               {"homeAway": "home", "score": "20", "team": {"abbreviation": "TENN"}},
                               {"homeAway": "away", "score": "31", "team": {"abbreviation": "TEX"}}]}]}]}
    monkeypatch.setattr(certainty, "get_json", lambda url, **kw: payload)
    gs = certainty.GameStateCache().state_for(ML)
    assert gs and gs["away_score"] == 31 and gs["home_score"] == 20 and gs["clock_seconds"] == 130
    assert certainty.seconds_left(gs) == 130 and not certainty.in_overtime(gs)
    assert certainty.GameStateCache().state_for("aec-cfb-nd-pur-2026-09-26") is None


def test_leader_quote_reads_the_right_side_of_the_book():
    book = OrderBook([OrderBookLevel(0.03, 400)], [OrderBookLevel(0.05, 250)])
    assert certainty.leader_quote(book, "away") == (0.05, 250)
    assert certainty.leader_quote(book, "home") == (pytest.approx(0.97), 400)


# ── the loop hook ───────────────────────────────────────────────────────────

def _scan_bot(book, gs, wp, **cfg):
    from bot.trading_loop import TradingBot
    from bot.strategies.risk import RiskManager
    bot = TradingBot.__new__(TradingBot)
    bot.config = SimpleNamespace(trading=_cfg(**cfg))
    bot.running = True
    bot.logger = Mock()
    bot.risk = RiskManager(bot.config.trading)
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [], get_total_exposure=lambda: 0.0,
                                    bankroll=1000.0, open_position=Mock())
    bot.market_data = Mock()
    bot.market_data.get_us_order_book.return_value = book
    bot.market_data.build_snapshot.return_value = SimpleNamespace(slug=ML, is_live=True, category="sports")
    bot.executor = Mock()
    bot.executor.execute_trade.return_value = SimpleNamespace(price=0.95, size_usd=95.0, quantity=100, fees=0.29)
    bot.live_cache = Mock(); bot.live_cache.get_live_prob.return_value = wp
    bot.game_state = Mock(); bot.game_state.state_for.return_value = gs
    bot._settled_slugs = set()
    bot._log_decision = Mock()
    bot.entries_paused_until = None
    return bot


def test_scan_buys_a_late_leader_at_the_ask_and_logs_the_quote(monkeypatch):
    rows = []
    monkeypatch.setattr("bot.trade_db.insert_certainty", lambda *a: rows.append(a))
    book = OrderBook([OrderBookLevel(0.93, 300)], [OrderBookLevel(0.95, 200)])
    bot = _scan_bot(book, _gs(away=24, home=7, clock="4:59"), (0.985, 3.0))
    bot._certainty_scan([{"slug": ML, "id": "1", "question": "Texas vs Tennessee"}])
    sig = bot.executor.execute_trade.call_args.args[0]
    assert sig.side == "buy" and sig.exec_price == 0.96 and sig.position_size_usd == 100.0
    assert sig.estimated_prob == pytest.approx(0.985)
    bot.portfolio.open_position.assert_called_once()
    assert bot.risk.daily_trade_count == 1
    assert len(rows) == 1 and rows[0][1] == ML and rows[0][8] == 0.95 and rows[0][10] == "late_lead" and rows[0][11]
    # the same market a second later: held now, so nothing new
    bot.portfolio.get_open_positions = lambda: [SimpleNamespace(slug=ML)]
    bot._certainty_scan([{"slug": ML, "id": "1"}])
    assert bot.executor.execute_trade.call_count == 1


def test_scan_skips_an_ask_above_the_limit_but_still_measures_it(monkeypatch):
    rows = []
    monkeypatch.setattr("bot.trade_db.insert_certainty", lambda *a: rows.append(a))
    book = OrderBook([OrderBookLevel(0.96, 300)], [OrderBookLevel(0.98, 200)])
    bot = _scan_bot(book, _gs(away=24, home=7, clock="4:59"), (0.985, 3.0))
    bot._certainty_scan([{"slug": ML, "id": "1"}])
    bot.executor.execute_trade.assert_not_called()
    assert len(rows) == 1 and rows[0][8] == 0.98 and not rows[0][11]


def test_scan_respects_market_kinds_exposure_and_the_daily_cap():
    book = OrderBook([OrderBookLevel(0.97, 300)], [OrderBookLevel(0.985, 200)])
    gs = _gs(state="post", away=31, home=20, completed=True)
    bot = _scan_bot(book, gs, None, market_kinds="ml")
    bot._certainty_scan([{"slug": "tsc-cfb-tx-tenn-2026-09-26-total-50pt5", "id": "2"}])
    bot.executor.execute_trade.assert_not_called()                     # totals excluded
    bot._certainty_scan([{"slug": ML, "id": "1"}])
    assert bot.executor.execute_trade.call_args.args[0].exec_price == 0.99
    capped = _scan_bot(book, gs, None, max_daily_trades=1)
    capped.risk.record_trade_opened()
    capped._certainty_scan([{"slug": ML, "id": "1"}])
    capped.executor.execute_trade.assert_not_called()
    assert [c.args[0] for c in capped.logger.warning.call_args_list] == ["entries_capped"]
    tight = _scan_bot(book, gs, None, max_portfolio_exposure_usd=50.0)
    tight._certainty_scan([{"slug": ML, "id": "1"}])
    assert tight.executor.execute_trade.call_args.args[0].position_size_usd == 50.0


def test_value_profiles_ignore_the_certainty_path():
    from bot.trading_loop import TradingBot
    bot = TradingBot.__new__(TradingBot)
    bot.config = SimpleNamespace(trading=TradingConfig())
    assert bot._strategy() == "value"
    bot.config = SimpleNamespace(trading=_cfg())
    bot.running = True
    bot.logger = Mock()
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [])
    assert bot._prescreen([{"slug": ML}], "full") == []
