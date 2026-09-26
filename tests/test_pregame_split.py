"""2026-09-25: pregame hold-to-settlement profiles, the daily-loss pause that
froze position management, the live take-profit override that ignored a
disabled take-profit, and the NHL preseason gate."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils.config import TradingConfig, load_config
from utils.models import OrderBook, OrderBookLevel, Position

from tests.test_strategy_v2 import snap


def _iso(hours_from_now):
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bot(**trading):
    from bot.trading_loop import TradingBot
    from bot.market_data import MarketDataClient
    bot = TradingBot.__new__(TradingBot)
    bot.config = SimpleNamespace(trading=TradingConfig(**trading))
    bot.running = True
    bot.logger = Mock()
    bot.market_data = Mock()
    bot.market_data._parse_datetime = MarketDataClient._parse_datetime.__get__(bot.market_data)
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [])
    from bot.strategies.risk import RiskManager
    bot.risk = RiskManager(bot.config.trading)
    return bot


# ── daily-loss pause blocks new entries only ────────────────────────────────

def test_pause_blocks_entries_but_keeps_the_cycle_running(tmp_path):
    bot = _bot()
    bot._data_dir = str(tmp_path)
    assert bot._check_supervisor_flags() is True and bot.entries_paused_until is None
    (tmp_path / "pause_until").write_text(_iso(3))
    assert bot._check_supervisor_flags() is True          # cycle still runs
    assert bot.entries_paused_until is not None
    bot._detect_edge = Mock(return_value=None)
    bot.process_markets([snap()])
    bot._detect_edge.assert_not_called()                  # no entries evaluated
    held = SimpleNamespace(slug="aec-nfl-a-b-2026-09-27")
    bot.portfolio = SimpleNamespace(get_open_positions=lambda: [held])
    keep = bot._prescreen([{"slug": "aec-nfl-a-b-2026-09-27"}, {"slug": "aec-nfl-c-d-2026-09-27"}], "t")
    assert [m["slug"] for m in keep] == ["aec-nfl-a-b-2026-09-27"]   # held market still refreshed
    (tmp_path / "pause_until").write_text(_iso(-1))       # expired: removed, entries reopen
    assert bot._check_supervisor_flags() is True and bot.entries_paused_until is None
    assert not (tmp_path / "pause_until").exists()
    (tmp_path / "kill_switch").write_text("x")
    assert bot._check_supervisor_flags() is False         # the kill switch still halts everything


# ── live take-profit override ───────────────────────────────────────────────

@pytest.mark.parametrize("configured,expected_during", [(0.05, 0.03), (-1.0, -1.0)])
def test_live_take_profit_only_tightens_an_enabled_take_profit(configured, expected_during):
    from bot.strategies.risk import RiskManager
    bot = _bot(take_profit_threshold=configured)
    bot.risk = RiskManager(bot.config.trading)
    bot.live_take_profit = 0.03
    seen = []
    bot._check_open_positions = lambda live, cycle: seen.append(bot.risk.config.take_profit_threshold)
    bot.check_positions(has_live_games=True)
    assert seen == [expected_during] and bot.risk.config.take_profit_threshold == configured


def test_take_profit_is_restored_when_a_check_fails():
    from bot.strategies.risk import RiskManager
    bot = _bot(take_profit_threshold=0.05)
    bot.risk = RiskManager(bot.config.trading)
    bot._check_open_positions = Mock(side_effect=RuntimeError("book fetch blew up"))
    with pytest.raises(RuntimeError):
        bot.check_positions(has_live_games=True)
    assert bot.risk.config.take_profit_threshold == 0.05


def test_hold_profile_keeps_its_take_profit_setting_in_live_games():
    from bot.strategies.risk import RiskManager
    bot = _bot(take_profit_threshold=0.05, hold_to_settlement=True)
    bot.risk = RiskManager(bot.config.trading)
    seen = []
    bot._check_open_positions = lambda live, cycle: seen.append(bot.risk.config.take_profit_threshold)
    bot.check_positions(has_live_games=True)
    assert seen == [0.05]


# ── hold to settlement ──────────────────────────────────────────────────────

def _position(side="buy", entry=0.40, qty=100, hours_held=1.0):
    return Position(market_id="m", token_id="t", side=side, entry_price=entry,
                    size_usd=entry * qty, quantity=qty, estimated_prob=0.5,
                    entry_time=datetime.now(timezone.utc) - timedelta(hours=hours_held),
                    current_price=entry, slug="aec-cfb-army-templ-2026-09-26", peak_price=entry)


@pytest.mark.parametrize("price", [0.10, 0.39, 0.70, 0.95])
def test_hold_to_settlement_never_exits_on_price(price):
    from bot.strategies.risk import RiskManager
    hold = RiskManager(TradingConfig(hold_to_settlement=True))
    assert hold.check_position(_position(), price, 0.5, allow_quote_resolution=False) is None
    normal = RiskManager(TradingConfig())
    if price == 0.10:
        assert normal.check_position(_position(), price, 0.5, allow_quote_resolution=False) == "stop_loss"


def test_hold_profile_marks_without_rest_and_closes_only_at_settlement(tmp_path, monkeypatch):
    from tests.test_paper_validation import bot_with_position, book, FakeResolutions
    from bot import trade_db
    bot, pos = bot_with_position(book(.10, .12))          # would be a stop loss for a normal profile
    bot.config.trading.hold_to_settlement = True
    bot.risk.config = bot.config.trading
    bot.market_data.get_us_order_book.side_effect = AssertionError("REST book forbidden for hold marks")
    bot.market_data.get_cached_book.return_value = None
    bot._cached_markets = [{"slug": pos.slug, "outcomePrices": json.dumps(["0.11", "0.89"])}]
    bot.market_data.active_slugs = {pos.slug}
    bot.market_data.active_slugs_complete = True
    bot.market_data.get_market_resolutions = FakeResolutions({})
    bot.check_positions(has_live_games=True)
    assert pos.status == "open" and pos.current_price == pytest.approx(0.11)
    assert trade_db.get_recent_trades() == []
    bot.market_data.get_cached_book.return_value = book(.004, .006)   # finished game, pinned book
    bot.market_data.get_market_resolutions = FakeResolutions({pos.slug: 0.0})
    bot.check_positions()
    bot._settle_checked = {}
    bot.check_positions()
    assert pos.status == "closed" and pos.close_reason == "resolved"
    row = trade_db.get_recent_trades()[0]
    assert row["exit_fees"] == 0 and row["close_price"] == 0.0


# ── pregame entry window and market families ────────────────────────────────

def test_pregame_window_rules():
    bot = _bot(entry_window="pregame", pregame_cutoff_minutes=10)
    ml = "aec-cfb-army-templ-2026-09-26"
    assert bot._entry_allowed(ml, False, 2.0)
    assert not bot._entry_allowed(ml, True, 0.0)            # in-game
    assert not bot._entry_allowed(ml, False, 5 / 60)        # inside the 10-minute cutoff
    assert not bot._entry_allowed(ml, False, None)          # unknown kickoff: fail closed
    late = _bot(entry_window="pregame", pregame_max_hours=3)
    assert late._entry_allowed(ml, False, 2.5) and not late._entry_allowed(ml, False, 5.0)
    live_only = _bot(entry_window="live")
    assert live_only._entry_allowed(ml, True, 1.0) and not live_only._entry_allowed(ml, False, 2.0)
    anytime = _bot()
    assert anytime._entry_allowed(ml, True, 1.0) and anytime._entry_allowed(ml, False, 30.0)


def test_market_kinds_limit_the_families():
    bot = _bot(market_kinds="ml")
    assert bot._entry_allowed("aec-nfl-nyg-lar-2026-09-27", False, 5.0)
    assert not bot._entry_allowed("asc-nfl-nyg-lar-2026-09-27-pos-6pt5", False, 5.0)
    assert not bot._entry_allowed("tsc-nfl-nyg-lar-2026-09-27-44pt5", False, 5.0)


def test_pregame_profile_prescreen_skips_live_and_late_markets():
    bot = _bot(entry_window="pregame")
    bot._detect_edge = Mock(return_value=("sig", "snap"))
    bot.market_data.build_snapshot = Mock(return_value="light")
    markets = [{"slug": "aec-nfl-live-game-2026-09-25", "gameStartTime": _iso(-0.5)},
               {"slug": "aec-nfl-soon-game-2026-09-25", "gameStartTime": _iso(0.05)},
               {"slug": "aec-nfl-later-game-2026-09-26", "gameStartTime": _iso(6)},
               {"slug": "aec-nfl-no-time-2026-09-26"}]
    keep = bot._prescreen(markets, "full")
    assert [m["slug"] for m in keep] == ["aec-nfl-later-game-2026-09-26"]


def test_pipeline_rejects_a_live_snapshot_for_a_pregame_profile():
    bot = _bot(entry_window="pregame")
    bot._settled_slugs, bot._slug_cooldowns = set(), {}
    bot._detect_edge = Mock(return_value=None)
    bot.process_markets([snap(live=True)])
    bot._detect_edge.assert_not_called()
    pre = snap(live=False)
    pre.hours_to_expiry = 4.0
    bot.process_markets([pre])
    bot._detect_edge.assert_called_once()


# ── league start dates (NHL preseason) ──────────────────────────────────────

def test_league_start_date_drops_games_before_it():
    from bot.market_data import MarketDataClient
    from tests.test_strategy_v2 import _market
    cfg = load_config()
    soon = _iso(5)
    markets = [_market("aec-nhl-min-dal-2026-09-25", soon), _market("aec-nfl-nyg-lar-2026-09-27", soon),
               _market("asc-nhl-min-dal-2026-09-25-pos-1pt5", soon, price="0.50")]
    cfg.filters.league_start_dates = "nhl:2099-01-01, bogus, nfl:not-a-date"
    md = MarketDataClient(cfg.api, None, cfg.filters)
    assert [m["slug"] for m in md._filter_markets(markets, {"nfl", "nhl"})] == ["aec-nfl-nyg-lar-2026-09-27"]
    cfg.filters.league_start_dates = "nhl:2020-01-01"
    md = MarketDataClient(cfg.api, None, cfg.filters)
    kept = {m["slug"] for m in md._filter_markets(markets, {"nfl", "nhl"})}
    assert "aec-nhl-min-dal-2026-09-25" in kept


def test_shipped_config_keeps_nhl_preseason_out_until_sep_29():
    from bot.market_data import MarketDataClient
    cfg = load_config()
    starts = MarketDataClient(cfg.api, None, cfg.filters).league_start_dates()
    assert starts["nhl"].date().isoformat() == "2026-09-29" and "nfl" not in starts


# ── moneyline consensus: token-0 team mapping ───────────────────────────────

@pytest.mark.parametrize("slug,probs,expected", [
    ("aec-cfb-tx-tenn-2026-09-26", {"tennessee": 0.358, "texas": 0.642}, 0.642),
    ("aec-cfb-nd-pur-2026-09-26", {"purdue": 0.054, "notre dame": 0.946}, 0.946),
    ("aec-cfb-nw-ind-2026-09-25", {"indiana": 0.907, "northwestern": 0.093}, 0.093),
    ("aec-cfb-hawaii-wyom-2026-09-26", {"wyoming cowboys": 0.444, "hawai'i warriors": 0.556}, 0.556),
    ("aec-cfb-army-templ-2026-09-25", {"temple": 0.372, "army": 0.628}, 0.628),
    ("aec-nfl-kc-buf-2026-09-27", {"buffalo bills": 0.55, "kansas city chiefs": 0.45}, 0.45),
])
def test_moneyline_consensus_prices_the_away_team_token(slug, probs, expected):
    from bot.signals.odds_api import OddsCache
    oc = OddsCache(api_key="", cache_ttl=300)
    oc.get_consensus_odds = lambda s: {"probs": probs, "num_books": 4, "home_team": "x", "away_team": "y"}
    assert oc.get_probability_for_slug(slug) == (pytest.approx(expected), 4)


def test_three_way_slug_still_prices_the_named_outcome():
    from bot.signals.odds_api import OddsCache
    oc = OddsCache(api_key="", cache_ttl=300)
    probs = {"brighton": 0.30, "liverpool": 0.45, "draw": 0.25}
    assert oc.outcome_prob("atc-epl-bha-liv-2026-03-21-liv", probs) == pytest.approx(0.45)


# ── daily counters roll over without a trade event ──────────────────────────

def test_daily_caps_reset_on_a_new_utc_day_without_a_trade_event():
    from bot.strategies.risk import RiskManager
    risk = RiskManager(TradingConfig(max_daily_trades=15, daily_loss_limit_usd=75))
    risk.reset_daily_pnl("2026-09-25")
    risk.daily_trade_count, risk.daily_pnl = 15, -80.0
    assert not risk.is_daily_trade_limit_reached() and not risk.is_daily_limit_breached()
    assert risk.daily_trade_count == 0 and risk.daily_pnl == 0.0
    assert risk.can_open_position([])
    risk.daily_trade_count = 15                      # same day: still capped
    assert risk.is_daily_trade_limit_reached() and not risk.can_open_position([])


def test_capped_profile_logs_once_and_evaluates_nothing():
    from bot.strategies.risk import RiskManager
    bot = _bot(max_daily_trades=1)
    bot.risk = RiskManager(bot.config.trading)
    bot.risk.record_trade_opened()
    bot._settled_slugs, bot._slug_cooldowns = set(), {}
    bot._detect_edge = Mock(return_value=None)
    bot.process_markets([snap()]); bot.process_markets([snap()])
    bot._detect_edge.assert_not_called()
    events = [c.args[0] for c in bot.logger.warning.call_args_list]
    assert events == ["entries_capped"]
