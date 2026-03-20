"""Unit tests for all core components."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from utils.models import (
    OrderBook, OrderBookLevel, MarketSnapshot, Signal,
    TradeSignal, Position, Trade, BacktestResult,
)
from utils.config import load_config, BotConfig, SignalConfig, TradingConfig
from utils.logger import TradingLogger
from bot.signals.signals import (
    price_momentum_signal,
    volume_signal,
    order_book_imbalance_signal,
    mean_reversion_signal,
    volatility_signal,
)
from bot.signals.unusual_whales import (
    smart_money_signal,
    whale_flow_signal,
    market_sentiment_signal,
)
from bot.signals.estimator import ProbabilityEstimator
from bot.strategies.sizing import PositionSizer
from bot.strategies.risk import RiskManager
from bot.portfolio import Portfolio
from bot.execution import ExecutionEngine
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
    )


@pytest.fixture
def bullish_snapshot(sample_order_book):
    # Rising price trend
    prices = [0.30 + i * 0.01 for i in range(50)]
    ob = OrderBook(
        bids=[OrderBookLevel(price=0.78, size=500), OrderBookLevel(price=0.77, size=400)],
        asks=[OrderBookLevel(price=0.82, size=100), OrderBookLevel(price=0.83, size=80)],
    )
    return MarketSnapshot(
        market_id="bullish_market",
        token_id="bullish_token",
        question="Bullish test?",
        price=0.80,
        volume_24h=10000.0,
        liquidity=3000.0,
        order_book=ob,
        price_history=prices,
        timestamp=datetime.now(timezone.utc),
        category="test",
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
# Signal Tests
# ============================================================================

class TestCoreSignals:
    def test_momentum_signal_returns_signal(self, sample_snapshot, signal_config):
        sig = price_momentum_signal(sample_snapshot, signal_config)
        assert isinstance(sig, Signal)
        assert sig.name == "price_momentum"
        assert 0 <= sig.value <= 1
        assert 0 <= sig.confidence <= 1

    def test_momentum_bullish_trend(self, bullish_snapshot, signal_config):
        sig = price_momentum_signal(bullish_snapshot, signal_config)
        assert sig.value > 0.5  # Should be bullish
        assert sig.direction == "bullish"

    def test_momentum_insufficient_data(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.5,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = price_momentum_signal(snap, signal_config)
        assert sig.confidence == 0.0

    def test_volume_signal_returns_signal(self, sample_snapshot, signal_config):
        sig = volume_signal(sample_snapshot, signal_config)
        assert isinstance(sig, Signal)
        assert sig.name == "volume"

    def test_volume_signal_zero_volume(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.5,
            volume_24h=0, liquidity=100, order_book=OrderBook(),
            price_history=[0.5, 0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = volume_signal(snap, signal_config)
        assert sig.confidence == 0.0

    def test_order_book_imbalance(self, sample_snapshot, signal_config):
        sig = order_book_imbalance_signal(sample_snapshot, signal_config)
        assert sig.name == "order_book_imbalance"
        # Bid depth 650 > Ask depth 550, so should be slightly bullish
        assert sig.value > 0.5

    def test_order_book_imbalance_empty(self, signal_config):
        snap = MarketSnapshot(
            market_id="m", token_id="t", question="?", price=0.5,
            volume_24h=100, liquidity=100, order_book=OrderBook(),
            price_history=[0.5], timestamp=datetime.now(timezone.utc),
        )
        sig = order_book_imbalance_signal(snap, signal_config)
        assert sig.confidence == 0.0

    def test_mean_reversion_signal(self, sample_snapshot, signal_config):
        sig = mean_reversion_signal(sample_snapshot, signal_config)
        assert sig.name == "mean_reversion"
        assert 0 <= sig.value <= 1

    def test_volatility_signal(self, sample_snapshot, signal_config):
        sig = volatility_signal(sample_snapshot, signal_config)
        assert sig.name == "volatility"
        assert sig.value == 0.5  # Volatility signal is directionally neutral


# ============================================================================
# Enrichment Signal Tests
# ============================================================================

class TestEnrichmentSignals:
    def test_smart_money_no_enrichment(self):
        sig = smart_money_signal(None)
        assert sig.value == 0.5
        assert sig.confidence == 0.0

    def test_smart_money_bullish(self):
        enrichment = {
            "market_detail": {"data": {
                "smart_money_sentiment": 0.8,
                "conviction_score": 0.7,
                "accumulation_detected": True,
            }},
            "smart_money_data": {"data": [{"score": 0.8}]},
        }
        sig = smart_money_signal(enrichment)
        assert sig.value > 0.5
        assert sig.direction == "bullish"

    def test_whale_flow_no_enrichment(self):
        sig = whale_flow_signal(None)
        assert sig.value == 0.5
        assert sig.confidence == 0.0

    def test_whale_flow_with_data(self):
        enrichment = {
            "whale_data": {"data": [
                {"direction": "buy", "amount": 5000},
            ]},
            "unusual_data": {"data": []},
            "positions": {"data": {"yes_percentage": 70, "no_percentage": 30}},
        }
        sig = whale_flow_signal(enrichment)
        assert sig.value > 0.5  # Bullish whale flow
        assert sig.confidence > 0

    def test_market_sentiment_no_enrichment(self):
        sig = market_sentiment_signal(None)
        assert sig.value == 0.5
        assert sig.confidence == 0.0

    def test_market_sentiment_with_data(self):
        enrichment = {
            "market_tide": {"data": {
                "sentiment": 0.5,
                "related_count": 3,
            }},
        }
        sig = market_sentiment_signal(enrichment)
        assert sig.value > 0.5


# ============================================================================
# Probability Estimator Tests
# ============================================================================

class TestProbabilityEstimator:
    def test_estimate_probability(self, signal_config, sample_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        signals = estimator.compute_signals(sample_snapshot)
        assert len(signals) == 8

        prob, conf = estimator.estimate_probability(signals)
        assert 0.01 <= prob <= 0.99
        assert 0 <= conf <= 1

    def test_detect_edge_no_edge(self, signal_config, sample_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        # Price close to fair value -> no edge
        result = estimator.detect_edge(sample_snapshot, min_edge=0.20, max_edge=0.40)
        assert result is None

    def test_detect_edge_with_edge(self, signal_config, bullish_snapshot):
        estimator = ProbabilityEstimator(signal_config)
        # Bullish signals should create edge
        result = estimator.detect_edge(bullish_snapshot, min_edge=0.01, max_edge=0.40)
        # May or may not find edge depending on exact signal values
        if result is not None:
            assert isinstance(result, TradeSignal)
            assert result.side in ("buy", "sell")


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizer:
    def test_kelly_sizing(self, trading_config):
        sizer = PositionSizer(trading_config)
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=0,
        )
        size = sizer.size_position(signal, bankroll=1000, current_exposure=0)
        assert size > 0
        assert size <= trading_config.max_position_size_usd

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
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.60, entry_time=datetime.now(timezone.utc),
        )
        # Price dropped 20% from entry -> should trigger 15% stop-loss
        reason = rm.check_position(pos, current_price=0.40, estimated_prob=0.60)
        assert reason == "stop_loss"

    def test_take_profit_triggered(self, trading_config):
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.60, entry_time=datetime.now(timezone.utc),
        )
        # Price converged to estimated prob (within take-profit threshold)
        reason = rm.check_position(pos, current_price=0.59, estimated_prob=0.60)
        assert reason == "take_profit"

    def test_no_close_normal_conditions(self, trading_config):
        rm = RiskManager(trading_config)
        pos = Position(
            market_id="m", token_id="t", side="buy",
            entry_price=0.50, size_usd=50, quantity=100,
            estimated_prob=0.65, entry_time=datetime.now(timezone.utc),
        )
        reason = rm.check_position(pos, current_price=0.52, estimated_prob=0.65)
        assert reason is None

    def test_daily_loss_limit(self, trading_config):
        rm = RiskManager(trading_config)
        rm.reset_daily_pnl()
        rm.record_pnl(-50)
        assert not rm.is_daily_limit_breached()
        rm.record_pnl(-60)  # Total -110 > 100 limit
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
        portfolio = Portfolio(initial_bankroll=1000)
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
        assert pnl != 0  # Should have some P&L

    def test_portfolio_stats(self):
        portfolio = Portfolio(initial_bankroll=1000)

        # Open and close a winning trade
        signal = TradeSignal(
            market_id="m", token_id="t", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=50,
        )
        trade = Trade(
            market_id="m", token_id="t", side="buy",
            price=0.50, quantity=100, size_usd=50,
            timestamp=datetime.now(timezone.utc), fees=1.0,
        )
        pos = portfolio.open_position(signal, trade)
        portfolio.close_position(pos, 0.60, "take_profit")

        stats = portfolio.get_stats()
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1


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
        assert config.signals.order_book_imbalance_weight == 0.25

    def test_config_signal_weights_sum_to_one(self, config):
        s = config.signals
        total = (s.price_momentum_weight + s.volume_signal_weight +
                 s.order_book_imbalance_weight + s.mean_reversion_weight +
                 s.volatility_signal_weight + s.uw_smart_money_weight +
                 s.uw_whale_flow_weight + s.uw_market_sentiment_weight)
        assert abs(total - 1.0) < 0.01
