"""Unit tests for all core components — rewritten for the current codebase.

Replaces the old file which referenced removed modules (bot.signals.unusual_whales,
price_momentum_signal, volume_signal, etc.) and a stale Portfolio/Estimator API.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from utils.models import (
    OrderBook, OrderBookLevel, MarketSnapshot, Signal,
    TradeSignal, Position, Trade, BacktestResult,
)
from utils.config import load_config, BotConfig, SignalConfig, TradingConfig
from bot.signals.signals import (
    order_book_imbalance_signal,
    liquidity_imbalance_signal,
    line_movement_signal,
    odds_value_signal,
)
from bot.signals.estimator import ProbabilityEstimator, detect_market_type, WEIGHTS
from bot.strategies.sizing import PositionSizer
from bot.strategies.risk import RiskManager
from bot.strategies.trade_filter import validate_trade
from bot.edge_log import extract_game_id
from bot.portfolio import Portfolio
from data.loader import (
    generate_synthetic_price_series,
    generate_synthetic_order_book,
    generate_synthetic_markets,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def signal_config():
    return SignalConfig()


@pytest.fixture
def trading_config():
    return TradingConfig()


@pytest.fixture
def sample_order_book():
    return OrderBook(
        bids=[
            OrderBookLevel(price=0.48, size=200),
            OrderBookLevel(price=0.47, size=300),
            OrderBookLevel(price=0.46, size=150),
        ],
        asks=[
            OrderBookLevel(price=0.52, size=180),
            OrderBookLevel(price=0.53, size=250),
            OrderBookLevel(price=0.54, size=120),
        ],
    )


@pytest.fixture
def sample_snapshot(sample_order_book):
    prices = [0.45 + i * 0.005 for i in range(50)]
    return MarketSnapshot(
        market_id="test_market",
        token_id="test_token",
        question="Will test pass?",
        price=0.50,
        volume_24h=5000.0,
        liquidity=2000.0,
        order_book=sample_order_book,
        price_history=prices,
        timestamp=datetime.now(timezone.utc),
        category="test",
        slug="test_market",
    )


@pytest.fixture
def sports_snapshot(sample_order_book):
    """A snapshot whose slug is detected as a sports market."""
    prices = [0.40 + i * 0.005 for i in range(50)]
    return MarketSnapshot(
        market_id="aec-nba-atl-hou-2026-03-20",
        token_id="aec-nba-atl-hou-2026-03-20",
        question="Will Atlanta Hawks beat Houston Rockets?",
        price=0.45,
        volume_24h=10000.0,
        liquidity=3000.0,
        order_book=sample_order_book,
        price_history=prices,
        timestamp=datetime.now(timezone.utc),
        category="sports",
        slug="aec-nba-atl-hou-2026-03-20",
    )


# ============================================================================
# OrderBook Tests
# ============================================================================

class TestOrderBook:
    def test_best_bid(self, sample_order_book):
        assert sample_order_book.best_bid == 0.48

    def test_best_ask(self, sample_order_book):
        assert sample_order_book.best_ask == 0.52

    def test_mid_price(self, sample_order_book):
        assert sample_order_book.mid_price == 0.50

    def test_bid_depth(self, sample_order_book):
        assert sample_order_book.bid_depth == 650  # 200 + 300 + 150

    def test_ask_depth(self, sample_order_book):
        assert sample_order_book.ask_depth == 550  # 180 + 250 + 120

    def test_empty_order_book(self):
        ob = OrderBook()
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.mid_price is None
        assert ob.bid_depth == 0


# ============================================================================
# Signal Tests (current signals in bot/signals/signals.py)
# ============================================================================

class TestCoreSignals:
    # --- order_book_imbalance_signal ---

    def test_order_book_imbalance_returns_signal(self, sample_snapshot, signal_config):
        sig = order_book_imbalance_signal(sample_snapshot, signal_config)
        assert isinstance(sig, Signal)
        assert sig.name == "order_book_imbalance"
        assert 0.0 <= sig.value <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

    def test_order_book_imbalance_value_above_half_when_more_bids(self, signal_config):
        # bid_depth 650 > ask_depth 550 → value > 0.5
        ob = OrderBook(
            bids=[OrderBookLevel(price=0.48, size=650)],
            asks=[OrderBookLevel(price=0.52, size=350)],
        )
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.50,
            volume_24h=100, liquidity=100, order_book=ob,
            price_history=[0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = order_book_imbalance_signal(snap, signal_config)
        assert sig.value > 0.5

    def test_order_book_imbalance_zero_confidence_when_empty(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.5,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = order_book_imbalance_signal(snap, signal_config)
        assert sig.confidence == 0.0

    # --- liquidity_imbalance_signal ---

    def test_liquidity_imbalance_returns_signal(self, sample_snapshot, signal_config):
        sig = liquidity_imbalance_signal(sample_snapshot, signal_config)
        assert isinstance(sig, Signal)
        assert sig.name == "liquidity_imbalance"
        assert 0.0 <= sig.value <= 1.0

    def test_liquidity_imbalance_zero_confidence_when_empty(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.5,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = liquidity_imbalance_signal(snap, signal_config)
        assert sig.confidence == 0.0

    # --- line_movement_signal ---

    def test_line_movement_neutral_near_half(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.50,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.50], timestamp=datetime.now(timezone.utc),
        )
        sig = line_movement_signal(snap, signal_config)
        assert isinstance(sig, Signal)
        assert sig.name == "line_movement"
        assert sig.direction == "neutral"
        assert sig.confidence == pytest.approx(0.1)

    def test_line_movement_bullish_above_half(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.70,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.70], timestamp=datetime.now(timezone.utc),
        )
        sig = line_movement_signal(snap, signal_config)
        assert sig.value > 0.5
        assert sig.direction == "bullish"

    # --- odds_value_signal ---

    def test_odds_value_zero_confidence_when_no_cache(self, sample_snapshot, signal_config):
        sig = odds_value_signal(sample_snapshot, signal_config, odds_cache=None)
        assert isinstance(sig, Signal)
        assert sig.name == "odds_value"
        assert sig.confidence == 0.0


# ============================================================================
# Market Type Detection (replaces TestEnrichmentSignals)
# ============================================================================

class TestMarketTypeDetection:
    def _snap(self, slug: str, question: str = "Will this happen?") -> MarketSnapshot:
        return MarketSnapshot(
            market_id=slug, token_id=slug, question=question,
            price=0.5, volume_24h=100, liquidity=100,
            order_book=OrderBook(), price_history=[0.5],
            timestamp=datetime.now(timezone.utc),
            slug=slug,
        )

    def test_sports_slug_prefix_aec(self):
        assert detect_market_type(self._snap("aec-nba-atl-hou-2026-03-20")) == "sports"

    def test_sports_slug_prefix_asc(self):
        assert detect_market_type(self._snap("asc-nba-atl-hou-pos-5pt5-2026-03-20")) == "sports"

    def test_sports_slug_prefix_tsc(self):
        assert detect_market_type(self._snap("tsc-nba-atl-hou-pt5-220-2026-03-20")) == "sports"

    def test_sports_slug_prefix_atc(self):
        assert detect_market_type(self._snap("atc-epl-bha-liv-2026-03-21-bha")) == "sports"

    def test_crypto_question_detected(self):
        snap = self._snap("btc-price-market", question="Will Bitcoin hit $100,000 by June?")
        assert detect_market_type(snap) == "crypto"

    def test_politics_question_detected(self):
        snap = self._snap("election-2026", question="Will the Democrat win the election?")
        assert detect_market_type(snap) == "politics"

    def test_other_is_default(self):
        snap = self._snap("random-market", question="Will this random event happen?")
        assert detect_market_type(snap) == "other"


# ============================================================================
# Probability Estimator Tests
# ============================================================================

class TestProbabilityEstimator:
    def test_compute_signals_other_type_returns_4(self, signal_config, sample_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        signals = estimator.compute_signals(sample_snapshot, "other")
        # other → order_book_imbalance, liquidity_imbalance, cross_market, line_movement
        assert len(signals) == 4

    def test_compute_signals_sports_type_returns_5(self, signal_config, sports_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        signals = estimator.compute_signals(sports_snapshot, "sports")
        # sports → order_book_imbalance, liquidity_imbalance, odds_value, line_movement, sports_context
        assert len(signals) == 5

    def test_estimate_probability_returns_valid_range(self, signal_config, sample_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        signals = estimator.compute_signals(sample_snapshot, "other")
        prob, conf = estimator.estimate_probability(signals, WEIGHTS["other"])
        assert 0.01 <= prob <= 0.99
        assert 0.0 <= conf <= 1.0

    def test_detect_edge_blocked_without_external_data(self, signal_config, sample_snapshot):
        """External validation gate: detect_edge returns None when the primary signal
        (cross_market for 'other') has confidence=0 because no cache is provided."""
        estimator = ProbabilityEstimator(signal_config)  # no predictit_cache
        result = estimator.detect_edge(sample_snapshot, min_edge=0.001, max_edge=0.40)
        assert result is None

    def test_detect_edge_blocked_for_sports_without_odds_cache(self, signal_config, sports_snapshot):
        """Sports markets require odds_value confidence > 0; no cache → blocked."""
        estimator = ProbabilityEstimator(signal_config)  # no odds_cache
        result = estimator.detect_edge(sports_snapshot, min_edge=0.001, max_edge=0.40)
        assert result is None


# ============================================================================
# Validate Trade Checklist
# ============================================================================

class TestValidateTrade:
    """Tests for the 7-point pre-trade validation checklist in trade_filter.py."""

    NHL_SLUG = "aec-nhl-tor-bos-2026-03-20"  # NHL min_edge=0.04

    def _signal(self, slug=None, edge=0.06, side="buy"):
        slug = slug or self.NHL_SLUG
        return TradeSignal(
            market_id="m", token_id="t", side=side,
            estimated_prob=0.50 + abs(edge),
            market_price=0.50,
            edge=edge, position_size_usd=20,
            slug=slug,
        )

    def _snapshot(self, price=0.50, bid_size=600, ask_size=600, slug=None):
        slug = slug or self.NHL_SLUG
        return MarketSnapshot(
            market_id="m", token_id="t",
            question="Will Toronto Maple Leafs beat Boston Bruins?",
            price=price, volume_24h=5000, liquidity=2000,
            order_book=OrderBook(
                bids=[OrderBookLevel(price=0.48, size=bid_size)],
                asks=[OrderBookLevel(price=0.52, size=ask_size)],
            ),
            price_history=[0.5],
            timestamp=datetime.now(timezone.utc),
            slug=slug,
        )

    def test_valid_trade_passes(self):
        sig = self._signal()
        snap = self._snapshot()
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=1, max_daily_trades=5,
        )
        assert result is None

    def test_rejects_insufficient_books(self):
        sig = self._signal()
        snap = self._snapshot()
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=1, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=1, max_daily_trades=5,
        )
        assert result is not None
        assert "books" in result

    def test_rejects_price_outside_range(self):
        sig = self._signal()
        snap = self._snapshot(price=0.90)  # > 0.85 upper bound
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=1, max_daily_trades=5,
        )
        assert result is not None
        assert "range" in result

    def test_rejects_low_liquidity(self):
        sig = self._signal()
        snap = self._snapshot(bid_size=100, ask_size=100)  # total 200 < 1000 minimum
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=1, max_daily_trades=5,
        )
        assert result is not None
        assert "liquidity" in result

    def test_rejects_correlated_game(self):
        sig = self._signal()
        snap = self._snapshot()
        game_id = extract_game_id(sig.slug)
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids={game_id},
            game_id=game_id,
            daily_trades=1, max_daily_trades=5,
        )
        assert result is not None
        assert "game" in result

    def test_rejects_daily_limit_reached(self):
        sig = self._signal()
        snap = self._snapshot()
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=5, max_daily_trades=5,
        )
        assert result is not None
        assert "limit" in result

    def test_rejects_last_5_minutes(self):
        sig = self._signal()
        snap = self._snapshot()
        result = validate_trade(
            signal=sig, snapshot=snap,
            num_books=3, open_game_ids=set(),
            game_id=extract_game_id(sig.slug),
            daily_trades=1, max_daily_trades=5,
            game_time_remaining=180,  # 3 minutes remaining — buzzer-beater risk
        )
        assert result is not None
        assert "last_5" in result


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizer:
    def test_kelly_sizing(self, trading_config):
        # TradingConfig() defaults to position_sizing_method="kelly"
        sizer = PositionSizer(trading_config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=0,
        )
        size = sizer.size_position(signal, bankroll=1000, current_exposure=0)
        assert size > 0
        assert size <= trading_config.max_position_size_usd

    def test_tiered_kelly_sizing(self):
        config = TradingConfig(position_sizing_method="tiered_kelly")
        sizer = PositionSizer(config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.59, market_price=0.50,
            edge=0.09, position_size_usd=0,
        )
        # 7-10% edge tier → $30
        size = sizer.size_position(signal, bankroll=1000, current_exposure=0)
        assert size == 30.0

    def test_fixed_fractional_sizing(self):
        config = TradingConfig(position_sizing_method="fixed_fractional", fixed_fraction=0.02)
        sizer = PositionSizer(config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.60, market_price=0.50,
            edge=0.10, position_size_usd=0,
        )
        size = sizer.size_position(signal, bankroll=1000, current_exposure=0)
        assert size == 20.0  # 2% of 1000

    def test_sizing_respects_max(self, trading_config):
        sizer = PositionSizer(trading_config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.95, market_price=0.10,
            edge=0.85, position_size_usd=0,
        )
        size = sizer.size_position(signal, bankroll=10000, current_exposure=0)
        assert size <= trading_config.max_position_size_usd

    def test_sizing_zero_when_full_exposure(self, trading_config):
        sizer = PositionSizer(trading_config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=0,
        )
        size = sizer.size_position(
            signal, bankroll=1000,
            current_exposure=trading_config.max_portfolio_exposure_usd,
        )
        assert size == 0.0


# ============================================================================
# Risk Manager Tests
# ============================================================================

class TestRiskManager:
    def test_stop_loss_triggered(self, trading_config):
        # stop_loss_threshold = 0.25 (25%)
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.60, entry_time=datetime.now(timezone.utc),
        )
        # Price dropped 26% from entry (0.37) — exceeds 25% stop-loss threshold.
        # Stop-loss is checked BEFORE the 10-minute hold gate.
        reason = rm.check_position(pos, current_price=0.37, estimated_prob=0.60)
        assert reason == "stop_loss"

    def test_take_profit_triggered(self, trading_config):
        # take_profit_threshold = 0.05, minimum_take_profit_usd = 5.0
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.60,
            # Set entry_time 15 min ago so we pass the 10-minute minimum hold gate.
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=15),
        )
        # edge_remaining = |0.60 - 0.59| = 0.01 ≤ 0.05 threshold
        # unrealized_pnl = 0.09 * 100 = $9 ≥ $5 minimum
        reason = rm.check_position(pos, current_price=0.59, estimated_prob=0.60)
        assert reason == "take_profit"

    def test_no_close_under_min_hold(self, trading_config):
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.65, entry_time=datetime.now(timezone.utc),
        )
        # Price slightly up, no stop-loss, but entry_time=now → <10 min held → None
        reason = rm.check_position(pos, current_price=0.52, estimated_prob=0.65)
        assert reason is None

    def test_daily_loss_limit(self):
        config = TradingConfig(daily_loss_limit_usd=75)
        rm = RiskManager(config)
        rm.reset_daily_pnl()
        rm.record_pnl(-50)
        assert not rm.is_daily_limit_breached()  # -50 > -75
        rm.record_pnl(-60)                        # total = -110 ≤ -75
        assert rm.is_daily_limit_breached()

    def test_max_open_positions(self, trading_config):
        rm = RiskManager(trading_config)
        positions = [
            Position(
                market_id=f"m_{i}", token_id=f"t_{i}", side="buy",
                entry_price=0.5, size_usd=10, quantity=20,
                estimated_prob=0.6, entry_time=datetime.now(timezone.utc),
            )
            for i in range(trading_config.max_open_positions)
        ]
        assert not rm.can_open_position(positions)


# ============================================================================
# Portfolio Tests
# ============================================================================

class TestPortfolio:
    def test_open_and_close_position(self):
        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=50,
        )
        trade = Trade(
            market_id="m", token_id="t", side="buy",
            price=0.50, quantity=100, size_usd=50,
            timestamp=datetime.now(timezone.utc),
            fees=1.0,
        )
        pos = portfolio.open_position(signal, trade)
        assert pos.status == "open"
        assert len(portfolio.get_open_positions()) == 1

        pnl = portfolio.close_position(pos, current_price=0.60, reason="take_profit")
        assert pos.status == "closed"
        assert len(portfolio.get_open_positions()) == 0
        assert pnl > 0  # buy at 0.50, close at 0.60 → profit

    def test_portfolio_stats(self):
        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        mock_trades = [{"realized_pnl": 10.0}]
        # Patch SQLite read so the test is deterministic regardless of DB contents
        with patch("bot.trade_db.get_trades_since", return_value=mock_trades):
            stats = portfolio.get_stats()
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1
        assert stats["win_rate"] == 1.0


# ============================================================================
# Data Generation Tests
# ============================================================================

class TestDataGeneration:
    def test_synthetic_price_series(self):
        prices = generate_synthetic_price_series(length=100, seed=42)
        assert len(prices) == 100
        assert all(0.01 <= p <= 0.99 for p in prices)

    def test_synthetic_price_series_deterministic(self):
        p1 = generate_synthetic_price_series(length=50, seed=123)
        p2 = generate_synthetic_price_series(length=50, seed=123)
        assert p1 == p2

    def test_synthetic_order_book(self):
        ob = generate_synthetic_order_book(0.50, seed=42)
        assert len(ob.bids) == 5
        assert len(ob.asks) == 5
        assert ob.best_bid < ob.best_ask

    def test_synthetic_markets(self):
        markets = generate_synthetic_markets(num_markets=3, num_snapshots=10, seed=42)
        assert len(markets) == 3
        assert len(markets[0]) == 10
        assert isinstance(markets[0][0], MarketSnapshot)


# ============================================================================
# Backtest Engine Tests
# ============================================================================

class TestBacktestEngine:
    def test_backtest_runs(self, config):
        from backtest.engine import BacktestEngine
        markets = generate_synthetic_markets(num_markets=5, num_snapshots=50, seed=42)
        engine = BacktestEngine(config)
        result = engine.run(markets)
        assert isinstance(result, BacktestResult)
        assert result.initial_bankroll == config.backtest.initial_bankroll_usd

    def test_backtest_metrics(self, config):
        from backtest.engine import BacktestEngine
        markets = generate_synthetic_markets(num_markets=5, num_snapshots=100, seed=42)
        engine = BacktestEngine(config)
        result = engine.run(markets)
        assert 0 <= result.win_rate <= 1
        assert result.max_drawdown >= 0
        assert len(result.equity_curve) > 0


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    def test_load_config(self, config):
        assert isinstance(config, BotConfig)
        assert config.trading.paper_trading is True
        assert config.trading.min_edge_threshold == 0.05

    def test_config_defaults(self):
        config = BotConfig()
        assert config.trading.kelly_fraction == 0.5
        # SignalConfig default is 0.15 (not 0.25 — was a stale assertion in the old file)
        assert config.signals.order_book_imbalance_weight == 0.15

    def test_config_signal_weights_sum_to_one(self):
        s = SignalConfig()
        total = (
            s.order_book_imbalance_weight   # 0.15
            + s.line_movement_weight         # 0.20
            + s.odds_value_weight            # 0.40
            + s.liquidity_imbalance_weight   # 0.10
            + s.sports_context_weight        # 0.15
        )
        assert abs(total - 1.0) < 0.01
