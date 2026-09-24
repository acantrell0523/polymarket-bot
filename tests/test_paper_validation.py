"""Regression coverage for executable paper fills and restart-safe accounting."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot import trade_db
from bot.execution import ExecutionEngine
from bot.paper import executable_level
from bot.portfolio import Portfolio
from bot.strategies.risk import RiskManager
from bot.supervisor import Supervisor
from bot.trading_loop import TradingBot
from utils.config import TradingConfig
from utils.models import OrderBook, OrderBookLevel, TradeSignal


def book(bid=.48, ask=.52, depth=1000):
    return OrderBook([OrderBookLevel(bid, depth)], [OrderBookLevel(ask, depth)])


def signal(side='buy', price=.52, budget=52):
    return TradeSignal('m', 't', side, .7, .5, .2, budget,
                       timestamp=datetime.now(timezone.utc) - timedelta(minutes=20),
                       slug='aec-nfl-aaa-bbb-2026-09-22', exec_price=price)


def opened(side='buy', price=.52, budget=52, bid=.48, ask=.52):
    sig = signal(side, price, budget)
    trade = ExecutionEngine(TradingConfig()).execute_trade(sig, order_book=book(bid, ask))
    portfolio = Portfolio(initial_bankroll=1000, restore_state=True)
    return portfolio, portfolio.open_position(sig, trade), trade


def test_entry_pays_ask_and_respects_fee_inclusive_budget():
    trade = ExecutionEngine(TradingConfig()).execute_trade(signal(), order_book=book())
    assert trade.price == .52  # never the .50 midpoint
    assert trade.quantity == 97
    assert trade.size_usd + trade.fees <= 52


def test_short_budget_uses_collateral_not_sale_proceeds():
    p, pos, trade = opened('sell', .80, 20, .80, .82)
    assert trade.quantity == 95  # $0.20 collateral + $0.0096 fee per contract
    assert trade.size_usd == pytest.approx(19)
    assert p.bankroll == pytest.approx(980.09)
    # A 6c adverse move loses 30% of collateral; the old denominator said 7.5%.
    assert RiskManager(TradingConfig()).check_position(pos, .86, .7) == 'stop_loss'


@pytest.mark.parametrize('bad_book', [None, OrderBook(),
    OrderBook(bids=[OrderBookLevel(.5, 100)]), book(.6, .4),
    book(float('nan'), .6), book(depth=-1)])
def test_no_phantom_fill_without_valid_two_sided_book(bad_book):
    assert ExecutionEngine(TradingConfig()).execute_trade(signal(), order_book=bad_book) is None


def test_partial_entry_never_exceeds_visible_depth_or_its_limit():
    engine = ExecutionEngine(TradingConfig())
    trade = engine.execute_trade(signal(), order_book=book(depth=3.5))
    assert trade.quantity == 3
    assert engine.execute_trade(signal(price=.51), order_book=book()) is None


def test_extreme_paper_quote_can_trigger_stop_but_not_fake_resolution():
    p, pos, _ = opened()
    risk = RiskManager(TradingConfig())
    assert risk.check_position(pos, .0005, .7, allow_quote_resolution=False) == 'stop_loss'


def test_take_profit_requires_five_dollars_after_both_fees():
    p, pos, trade = opened()
    risk = RiskManager(TradingConfig())
    # $5.82 gross at 58c, but less than $5 after fees.
    assert risk.check_position(pos, .58, .59, allow_quote_resolution=False) is None


@pytest.mark.parametrize('side,entry,exit_px,bid,ask', [
    ('buy', .52, .60, .48, .52), ('sell', .80, .65, .80, .82)])
def test_cash_net_pnl_and_equity_survive_open_and_closed_restarts(side,entry,exit_px,bid,ask):
    from bot.strategies.fees import booked_fee_usd
    p, pos, trade = opened(side, entry, 52, bid, ask)
    restarted = Portfolio(initial_bankroll=9999, restore_state=True)
    assert restarted.initial_bankroll == 1000
    assert restarted.bankroll == pytest.approx(p.bankroll)
    pos = restarted.get_open_positions()[0]
    assert pos.entry_fees == trade.fees
    assert pos.entry_time == trade.timestamp
    gross = (exit_px-entry if side=='buy' else entry-exit_px) * trade.quantity
    net = gross - trade.fees - booked_fee_usd(trade.quantity, exit_px)
    pos.current_price = exit_px
    assert restarted.get_equity() == pytest.approx(1000 + net)
    pnl = restarted.close_position(pos, exit_px, 'take_profit')
    assert pnl == pytest.approx(net)
    final = Portfolio(initial_bankroll=9999, restore_state=True)
    assert final.bankroll == pytest.approx(1000 + net)
    assert final.get_equity() == pytest.approx(final.bankroll)
    assert final.get_open_positions() == []
    row = trade_db.get_recent_trades()[0]
    assert row['accounting_version'] == 2
    assert row['realized_pnl'] == pytest.approx(net)
    assert row['entry_fees'] == trade.fees
    assert restarted.close_position(pos, exit_px, 'take_profit') == 0
    assert len(trade_db.get_recent_trades()) == 1


def test_close_is_atomic_if_trade_insert_fails(monkeypatch):
    p, pos, _ = opened()
    cash = p.bankroll
    def fail(**kw):
        raise RuntimeError('database write failed')
    monkeypatch.setattr(trade_db, 'insert_trade', fail)
    with pytest.raises(RuntimeError):
        p.close_position(pos, .6, 'take_profit')
    assert pos.status == 'open'
    assert p.bankroll == pytest.approx(cash)
    restarted = Portfolio(initial_bankroll=1000, restore_state=True)
    assert restarted.bankroll == pytest.approx(cash)
    assert len(restarted.get_open_positions()) == 1


def test_legacy_database_cannot_silently_reset_bankroll():
    p, pos, _ = opened()
    conn = trade_db._get_conn()
    conn.execute('DELETE FROM paper_portfolio'); conn.commit(); conn.close()
    with pytest.raises(ValueError, match='Legacy paper history'):
        Portfolio(initial_bankroll=1000, restore_state=True)


def test_paper_supervisor_never_reads_real_account(tmp_path, monkeypatch):
    import json, time
    import bot.supervisor as mod
    monkeypatch.setattr(mod, 'KILL_SWITCH_PATH', str(tmp_path/'kill_switch'))
    sup = Supervisor.__new__(Supervisor)
    sup.config = SimpleNamespace(trading=TradingConfig())
    sup._get_client = Mock(side_effect=AssertionError('must never call live API'))
    sup._starting_value = 1000
    sup._activate_kill_switch = Mock()
    heartbeat = {'timestamp': time.time(), 'mode': 'PAPER', 'accounting_version': 2,
                 'cash': 950, 'equity': 999, 'positions': []}
    path = tmp_path/'heartbeat.json'; path.write_text(json.dumps(heartbeat))
    assert sup._get_account_value() == 999
    assert sup._get_exchange_balance() == 950
    sup.check_kill_switch()
    sup._activate_kill_switch.assert_not_called()
    heartbeat['equity'] = 400
    path.write_text(json.dumps(heartbeat))
    sup.check_kill_switch()
    sup._activate_kill_switch.assert_called_once()
    heartbeat['timestamp'] -= 600
    path.write_text(json.dumps(heartbeat))
    assert sup._get_account_value() is None


def bot_with_position(exit_book):
    p, pos, _ = opened()
    bot = TradingBot.__new__(TradingBot)
    bot.config = SimpleNamespace(trading=TradingConfig())
    bot.portfolio = p
    bot.risk = RiskManager(bot.config.trading)
    bot.market_data = Mock()
    bot.market_data.get_us_order_book.return_value = exit_book
    bot.market_data.get_live_price.side_effect = AssertionError('midpoint exit forbidden')
    bot.executor = ExecutionEngine(bot.config.trading)
    bot.logger = Mock()
    bot.alerter = None
    bot._settled_slugs = set()
    bot._slug_cooldowns = {}
    bot._cooldown_seconds = 600
    bot._get_exchange_pnl = Mock(return_value=None)
    bot._enforce_daily_loss_pause = Mock()
    return bot, pos


def test_actual_loop_closes_at_bid_and_books_fees():
    bot, pos = bot_with_position(book(.35, .39))
    bot.check_positions()
    assert pos.status == 'closed'
    assert pos.close_price == .35
    assert pos.realized_pnl < (.35-.52)*pos.quantity


def test_actual_loop_cannot_close_without_a_book():
    bot, pos = bot_with_position(OrderBook())
    bot.check_positions()
    assert pos.status == 'open'
    assert trade_db.get_recent_trades() == []


def test_thin_book_closes_the_visible_slice_and_keeps_the_rest():
    bot, pos = bot_with_position(book(.35, .39, depth=1))
    qty = pos.quantity
    bot.check_positions()
    assert pos.status == 'open' and pos.quantity == pytest.approx(qty - 1)
    rows = trade_db.get_recent_trades()
    assert len(rows) == 1 and rows[0]['quantity'] == pytest.approx(1)
    assert rows[0]['close_reason'] == 'stop_loss_partial'


# ── Additions on top of the Codex patch (2026-09-23) ──────────────────────

from bot.paper import sweep


def ladder(bids, asks):
    return OrderBook([OrderBookLevel(p, q) for p, q in bids],
                     [OrderBookLevel(p, q) for p, q in asks])


def test_sweep_walks_levels_with_per_level_fees():
    from bot.strategies.fees import booked_fee_usd
    b = ladder([(.40, 30), (.39, 50), (.35, 500)], [(.42, 10)])
    fill = sweep(b, 'sell', 100)
    assert fill['filled'] == pytest.approx(100) and fill['levels'] == 3
    assert fill['vwap'] == pytest.approx((.40*30 + .39*50 + .35*20) / 100)
    assert fill['fees'] == pytest.approx(round(booked_fee_usd(30, .40) + booked_fee_usd(50, .39)
                                               + booked_fee_usd(20, .35), 2))
    short_cover = sweep(b, 'buy', 25)
    assert short_cover['filled'] == pytest.approx(10) and short_cover['vwap'] == pytest.approx(.42)
    assert sweep(ladder([(.6, 5)], [(.5, 5)]), 'sell', 1) is None   # crossed book


def test_loop_exit_uses_book_depth_beyond_top_level():
    exit_book = ladder([(.35, 40), (.34, 100)], [(.39, 100)])
    bot, pos = bot_with_position(exit_book)
    qty = pos.quantity
    bot.check_positions()
    assert pos.status == 'closed'
    assert pos.close_price == pytest.approx((.35*40 + .34*(qty-40)) / qty)
    row = trade_db.get_recent_trades()[0]
    assert row['close_reason'] == 'stop_loss' and row['exit_fees'] > 0


def test_partial_close_cash_and_restart_consistency():
    p, pos, trade = opened()
    start_cash = p.bankroll
    qty = pos.quantity
    pnl = p.close_partial(pos, 40, .45, 'stop_loss', exit_fees=0.74)
    frac = 40 / qty
    entry_part = round(trade.fees * frac, 2)
    assert pnl == pytest.approx((.45 - .52) * 40 - entry_part - 0.74)
    assert p.bankroll == pytest.approx(start_cash + .52 * 40 + (.45 - .52) * 40 - 0.74)
    assert pos.quantity == qty - 40 and pos.size_usd == pytest.approx(.52 * (qty - 40))
    restarted = Portfolio(initial_bankroll=1000, restore_state=True)
    assert restarted.bankroll == pytest.approx(p.bankroll)
    again = restarted.get_open_positions()[0]
    assert again.quantity == qty - 40 and again.entry_fees == pytest.approx(pos.entry_fees)
    final = restarted.close_position(again, .60, 'take_profit')
    rows = trade_db.get_recent_trades()
    assert {r['close_reason'] for r in rows} == {'stop_loss_partial', 'take_profit'}
    total_net = sum(r['realized_pnl'] for r in rows)
    assert restarted.bankroll == pytest.approx(1000 + total_net)


class FakeResolutions:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, slugs):
        self.calls.append(list(slugs))
        return {s: v for s, v in self.values.items() if s in slugs}


@pytest.mark.parametrize('side,settle,won', [('buy', 1.0, True), ('buy', 0.0, False), ('sell', 0.0, True)])
def test_finished_game_settles_at_exchange_result_without_exit_fee(side, settle, won):
    if side == 'buy':
        bot, pos = bot_with_position(book(.99, .995))
    else:
        p, pos, _ = opened('sell', .80, 20, .80, .82)
        bot, _ = bot_with_position(book(.004, .006))
        bot.portfolio = p
    slug = pos.slug
    bot.market_data.active_slugs = set()          # gone from a complete active list
    bot.market_data.active_slugs_complete = True
    bot.market_data.get_market_resolutions = FakeResolutions({slug: settle})
    bot.check_positions()
    assert pos.status == 'closed' and pos.close_reason == 'resolved'
    row = trade_db.get_recent_trades()[0]
    assert row['exit_fees'] == 0 and row['close_price'] == settle
    assert (row['realized_pnl'] > 0) == won


def test_settlement_is_not_polled_for_fresh_normal_positions():
    bot, pos = bot_with_position(book(.50, .52))
    bot.market_data.active_slugs = {pos.slug}
    bot.market_data.active_slugs_complete = True
    fake = FakeResolutions({})
    bot.market_data.get_market_resolutions = fake
    bot.check_positions()
    assert fake.calls == []
    # once the book disappears the next cycle asks, but at most once a minute
    bot.market_data.get_us_order_book.return_value = OrderBook()
    bot.check_positions(); bot.check_positions(); bot.check_positions()
    assert fake.calls == [[pos.slug]]


def test_market_resolution_lookup_parses_settlement_and_fallback():
    from bot.market_data import MarketDataClient
    from utils.config import load_config
    cfg = load_config()
    md = MarketDataClient(cfg.api, None, cfg.filters)
    listing = {"markets": [
        {"slug": "aec-nfl-a-b-2026-09-20", "status": "MARKET_STATUS_RESOLVED", "outcomePrices": '["1","0"]'},
        {"slug": "tsc-nfl-a-b-2026-09-20-total-40pt5", "status": "MARKET_STATUS_RESOLVED", "outcomePrices": '["0","1"]'},
        {"slug": "aec-nfl-c-d-2026-09-27", "status": "MARKET_STATUS_OPEN", "outcomePrices": '["0.4","0.6"]'},
    ]}
    calls = []
    def fake_get(url, params=None):
        calls.append((url, params))
        if url.endswith("/v1/markets"):
            return listing
        if "aec-nfl-a-b" in url:
            return {"slug": "aec-nfl-a-b-2026-09-20", "settlement": 1}
        return None   # settlement endpoint down → fall back to outcomePrices
    md._get = fake_get
    out = md.get_market_resolutions(["aec-nfl-a-b-2026-09-20", "tsc-nfl-a-b-2026-09-20-total-40pt5",
                                     "aec-nfl-c-d-2026-09-27"])
    assert out == {"aec-nfl-a-b-2026-09-20": 1.0, "tsc-nfl-a-b-2026-09-20-total-40pt5": 0.0}
    assert calls[0][1]["slug"] == ["aec-nfl-a-b-2026-09-20", "tsc-nfl-a-b-2026-09-20-total-40pt5",
                                   "aec-nfl-c-d-2026-09-27"]
