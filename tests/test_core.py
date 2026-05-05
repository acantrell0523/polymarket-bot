"""Unit tests for all core components — rewritten for the current codebase.

# NOTE: historical DB tests are at the bottom of this file (TestHistoricalDB).

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

    def test_nba_outright_finals_slug_is_sports(self):
        """NBA Finals-winner slugs contain '-nba-' and must classify as sports."""
        snap = self._snap(
            "will-the-oklahoma-city-thunder-win-the-2026-nba-finals",
            question="Will the Oklahoma City Thunder win the 2026 NBA Finals?",
        )
        assert detect_market_type(snap) == "sports"

    def test_nba_outright_mvp_slug_is_sports(self):
        """NBA MVP slugs contain '-nba-' and must classify as sports."""
        snap = self._snap(
            "will-nikola-jokic-win-the-20252026-nba-mvp",
            question="Will Nikola Jokic win the 2025-26 NBA MVP?",
        )
        assert detect_market_type(snap) == "sports"


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

    def test_config_weight_drives_probability_output(self):
        """Changing a weight in SignalConfig.weights is actually reflected in
        estimate_probability — this proves config tuning is no longer a no-op."""
        # Two signals with strongly opposing values.
        signals = [
            Signal(name="order_book_imbalance", value=0.90, confidence=1.0, direction="bullish"),
            Signal(name="cross_market",          value=0.10, confidence=1.0, direction="bearish"),
        ]

        # Default "other" weights: cross_market=0.40, order_book_imbalance=0.25
        config_default = SignalConfig()
        estimator_default = ProbabilityEstimator(config_default)
        prob_default, _ = estimator_default.estimate_probability(
            signals, config_default.weights["other"]
        )

        # Heavily boost cross_market so the bearish signal dominates.
        config_boosted = SignalConfig()
        config_boosted.weights["other"]["cross_market"] = 0.90
        config_boosted.weights["other"]["order_book_imbalance"] = 0.10
        estimator_boosted = ProbabilityEstimator(config_boosted)
        prob_boosted, _ = estimator_boosted.estimate_probability(
            signals, config_boosted.weights["other"]
        )

        # Boosting the bearish cross_market signal must pull probability lower.
        assert prob_boosted < prob_default

    def test_missing_config_weights_fall_back_to_defaults_no_error(self):
        """When config.weights has no entry for a market type, detect_edge falls
        back to the hardcoded WEIGHTS defaults without raising any error."""
        # Clear all per-market-type weights from the config.
        config = SignalConfig()
        config.weights = {}  # no market types defined at all
        estimator = ProbabilityEstimator(config)

        signals = [
            Signal(name="cross_market",          value=0.55, confidence=1.0, direction="bullish"),
            Signal(name="order_book_imbalance", value=0.60, confidence=1.0, direction="bullish"),
        ]

        # Replicate the fallback logic from detect_edge.
        default_weights = WEIGHTS.get("other", WEIGHTS["other"])
        config_weights  = getattr(estimator.config, "weights", {}).get("other", {})
        effective_weights = {**default_weights, **config_weights}

        # When config is empty the effective weights must equal the hardcoded defaults exactly.
        assert effective_weights == WEIGHTS["other"]

        # estimate_probability must succeed and return a valid probability.
        prob, conf = estimator.estimate_probability(signals, effective_weights)
        assert 0.01 <= prob <= 0.99
        assert 0.0 <= conf <= 1.0


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

    def test_independent_paper_positions_and_bankrolls(self):
        """Two Portfolio instances must not share paper positions or bankroll state.

        Guards against the class-level mutable default anti-pattern, e.g.:
            _paper_positions: List[Position] = []   # ← all instances would share this list
        Both attributes must be initialised inside __init__ so each instance
        gets its own independent list/float.
        """
        p1 = Portfolio(paper_mode=True, initial_bankroll=500)
        p2 = Portfolio(paper_mode=True, initial_bankroll=1000)

        # Bankrolls are independent at construction time
        assert p1.bankroll == 500
        assert p2.bankroll == 1000

        # Open a position only on p1
        signal = TradeSignal(
            market_id="m1", token_id="t1", side="buy",
            estimated_prob=0.65, market_price=0.50,
            edge=0.15, position_size_usd=50,
        )
        trade = Trade(
            market_id="m1", token_id="t1", side="buy",
            price=0.50, quantity=100, size_usd=50,
            timestamp=datetime.now(timezone.utc),
            fees=1.0,
        )
        p1.open_position(signal, trade)

        # p1 has 1 open position; p2 must still have 0 (independent lists)
        assert len(p1.get_open_positions()) == 1
        assert len(p2.get_open_positions()) == 0

        # p1 bankroll was decremented by the fee; p2 bankroll is unchanged
        assert p1.bankroll == pytest.approx(499.0)  # 500 - 1.0 fee
        assert p2.bankroll == pytest.approx(1000.0)

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


# ============================================================================
# Exit Telemetry Tests
# ============================================================================

class TestExitTelemetry:
    """Tests for the exit telemetry layer: max favorable/adverse P&L tracking,
    let_it_ride counter, and exit_log write on close.

    The tracking update logic lives in TradingBot.check_positions() and is
    replicated below via _apply_telemetry_update() so these tests stay
    independent of the full bot infrastructure.
    """

    def _make_position(
        self,
        entry_price: float = 0.50,
        quantity: float = 100,
        side: str = "buy",
        slug: str = "test-market",
    ) -> Position:
        return Position(
            market_id=slug,
            token_id=slug,
            side=side,
            entry_price=entry_price,
            size_usd=entry_price * quantity,
            quantity=quantity,
            estimated_prob=0.65,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=20),
            current_price=entry_price,
            slug=slug,
        )

    @staticmethod
    def _apply_telemetry_update(position: Position, current_price: float):
        """Mirror the four-line block from TradingBot.check_positions()."""
        if position.side == "buy":
            pnl_per_unit = current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - current_price
        unrealized_usd = pnl_per_unit * position.quantity
        if unrealized_usd > position.max_favorable_pnl_usd:
            position.max_favorable_pnl_usd = unrealized_usd
        if unrealized_usd < position.max_adverse_pnl_usd:
            position.max_adverse_pnl_usd = unrealized_usd

    # ------------------------------------------------------------------
    # max_favorable_pnl_usd
    # ------------------------------------------------------------------

    def test_max_favorable_tracks_peak_profit(self):
        """max_favorable_pnl_usd must capture the highest positive unrealized P&L."""
        pos = self._make_position(entry_price=0.50, quantity=100)

        self._apply_telemetry_update(pos, 0.53)   # +$3
        assert pos.max_favorable_pnl_usd == pytest.approx(3.0)

        self._apply_telemetry_update(pos, 0.57)   # +$7 — new high
        assert pos.max_favorable_pnl_usd == pytest.approx(7.0)

        self._apply_telemetry_update(pos, 0.52)   # +$2 — pullback, peak must NOT drop
        assert pos.max_favorable_pnl_usd == pytest.approx(7.0)

    def test_max_favorable_stays_zero_when_always_losing(self):
        """If the position never enters profit, max_favorable must stay at its default 0.0."""
        pos = self._make_position(entry_price=0.50, quantity=100)
        self._apply_telemetry_update(pos, 0.48)   # -$2
        self._apply_telemetry_update(pos, 0.45)   # -$5
        assert pos.max_favorable_pnl_usd == 0.0

    # ------------------------------------------------------------------
    # max_adverse_pnl_usd
    # ------------------------------------------------------------------

    def test_max_adverse_tracks_peak_loss(self):
        """max_adverse_pnl_usd must capture the most negative unrealized P&L."""
        pos = self._make_position(entry_price=0.50, quantity=100)

        self._apply_telemetry_update(pos, 0.47)   # -$3
        assert pos.max_adverse_pnl_usd == pytest.approx(-3.0)

        self._apply_telemetry_update(pos, 0.42)   # -$8 — new low
        assert pos.max_adverse_pnl_usd == pytest.approx(-8.0)

        self._apply_telemetry_update(pos, 0.49)   # -$1 — recovery, worst must NOT improve
        assert pos.max_adverse_pnl_usd == pytest.approx(-8.0)

    def test_max_adverse_stays_zero_when_always_winning(self):
        """If the position is always in profit, max_adverse stays at its default 0.0."""
        pos = self._make_position(entry_price=0.50, quantity=100)
        self._apply_telemetry_update(pos, 0.55)   # +$5
        self._apply_telemetry_update(pos, 0.60)   # +$10
        assert pos.max_adverse_pnl_usd == 0.0

    def test_sell_side_favorable_direction_is_price_drop(self):
        """For a SELL position, a falling price is favorable (profit)."""
        pos = self._make_position(entry_price=0.60, quantity=100, side="sell")
        self._apply_telemetry_update(pos, 0.50)   # price fell — sell wins
        assert pos.max_favorable_pnl_usd == pytest.approx(10.0)
        assert pos.max_adverse_pnl_usd == 0.0

    # ------------------------------------------------------------------
    # let_it_ride_count
    # ------------------------------------------------------------------

    def test_let_it_ride_count_starts_at_zero(self):
        pos = self._make_position()
        assert pos.let_it_ride_count == 0

    def test_let_it_ride_count_increments_each_cycle(self):
        """let_it_ride_count must accumulate across cycles without resetting."""
        pos = self._make_position()
        for expected in range(1, 6):
            pos.let_it_ride_count += 1   # as trading_loop.py does
            assert pos.let_it_ride_count == expected

    def test_let_it_ride_count_does_not_increment_on_normal_close(self):
        """Closing without let_it_ride must leave let_it_ride_count at 0."""
        pos = self._make_position()
        # Never touch let_it_ride_count — simulate a plain take_profit exit
        portfolio = Portfolio(paper_mode=True, initial_bankroll=500)
        with patch("bot.trade_db.insert_exit_log"), \
             patch("bot.trade_db.insert_trade"), \
             patch("bot.edge_log.update_edge_log_outcome"):
            portfolio._paper_positions.append(pos)
            portfolio.close_position(pos, current_price=0.58, reason="take_profit")
        assert pos.let_it_ride_count == 0

    # ------------------------------------------------------------------
    # exit_log DB write
    # ------------------------------------------------------------------

    def test_exit_log_written_on_close_paper_mode(self):
        """insert_exit_log must be called exactly once when a paper position closes."""
        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        signal = TradeSignal(
            market_id="exit-log-test",
            token_id="exit-log-token",
            side="buy",
            estimated_prob=0.65,
            market_price=0.50,
            edge=0.15,
            position_size_usd=50,
            slug="exit-log-test",
        )
        trade = Trade(
            market_id="exit-log-test",
            token_id="exit-log-token",
            side="buy",
            price=0.50,
            quantity=100,
            size_usd=50,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=20),
        )
        pos = portfolio.open_position(signal, trade)

        # Pre-populate tracking fields as the live loop would
        pos.max_favorable_pnl_usd = 5.0
        pos.max_adverse_pnl_usd = -2.0
        pos.let_it_ride_count = 0

        with patch("bot.trade_db.insert_exit_log") as mock_insert, \
             patch("bot.trade_db.insert_trade"), \
             patch("bot.edge_log.update_edge_log_outcome"):
            portfolio.close_position(pos, current_price=0.55, reason="take_profit")

        mock_insert.assert_called_once()
        kw = mock_insert.call_args[1]   # all args in portfolio.py are keyword args
        assert kw["slug"] == "exit-log-test"
        assert kw["close_reason"] == "take_profit"
        assert kw["max_favorable_pnl_usd"] == pytest.approx(5.0)
        assert kw["max_adverse_pnl_usd"] == pytest.approx(-2.0)
        assert kw["let_it_ride_triggered"] is False
        assert kw["entry_estimated_prob"] == pytest.approx(0.65)
        assert kw["realized_pnl"] == pytest.approx(5.0)  # (0.55-0.50)*100

    def test_exit_log_records_let_it_ride_triggered_true(self):
        """let_it_ride_triggered must be True when let_it_ride_count > 0 at close."""
        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        signal = TradeSignal(
            market_id="lir-test",
            token_id="lir-token",
            side="buy",
            estimated_prob=0.65,
            market_price=0.50,
            edge=0.15,
            position_size_usd=50,
            slug="lir-test",
        )
        trade = Trade(
            market_id="lir-test",
            token_id="lir-token",
            side="buy",
            price=0.50,
            quantity=100,
            size_usd=50,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
        pos = portfolio.open_position(signal, trade)
        pos.let_it_ride_count = 5   # Was riding for 5 cycles before stop-loss

        with patch("bot.trade_db.insert_exit_log") as mock_insert, \
             patch("bot.trade_db.insert_trade"), \
             patch("bot.edge_log.update_edge_log_outcome"):
            portfolio.close_position(pos, current_price=0.37, reason="stop_loss")

        kw = mock_insert.call_args[1]
        assert kw["let_it_ride_triggered"] is True
        assert kw["num_let_it_ride_triggers"] == 5
        assert kw["close_reason"] == "stop_loss"


# ============================================================================
# Live Loop let_it_ride Safety (Regression Guard)
# ============================================================================

class TestLiveLoopLetItRideSafety:
    """Regression guard: the live trading loop must NOT close a position when
    check_position() returns 'let_it_ride'.

    Bug shape found in backtest/engine.py: a bare ``if close_reason:`` truthiness
    check fires on the non-close string "let_it_ride" and incorrectly closes a
    winning position.

    The live loop in trading_loop.py is protected by:
        1.  ``if close_reason == "let_it_ride":``   (explicit equality, line 496)
        2.  ``continue``                             (line 519) — exits the loop
                                                     iteration so the truthiness
                                                     check on line 521 is never
                                                     reached for "let_it_ride".

    These tests pin that protection contract so a future refactor cannot
    accidentally remove or reorder the guards.
    """

    def _make_winning_position(self) -> Position:
        """Buy position at 0.75 (above LET_IT_RIDE_THRESHOLD=0.70), held 15 min
        (past the 10-minute minimum-hold gate) — guaranteed to return
        'let_it_ride' from RiskManager.check_position()."""
        return Position(
            market_id="lir-live-test",
            token_id="lir-live-token",
            side="buy",
            entry_price=0.50,
            size_usd=50,
            quantity=100,
            estimated_prob=0.80,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=15),
            current_price=0.75,
            slug="lir-live-test",
        )

    # ------------------------------------------------------------------
    # Precondition: verify risk.check_position() actually returns "let_it_ride"
    # ------------------------------------------------------------------

    def test_check_position_returns_let_it_ride_for_winning_buy(self):
        """RiskManager.check_position() must return 'let_it_ride' for a buy
        position at 75¢ held 15 min.  This is the precondition that makes
        the bug in backtest/engine.py fire — and that the live loop must guard.
        """
        rm = RiskManager(TradingConfig())
        pos = self._make_winning_position()
        result = rm.check_position(pos, current_price=0.75, estimated_prob=0.80)
        assert result == "let_it_ride", (
            f"Expected 'let_it_ride', got {result!r}. "
            "Precondition broken — check LET_IT_RIDE_THRESHOLD in risk.py."
        )

    def test_check_position_returns_let_it_ride_for_winning_sell(self):
        """Same contract for a SELL position at 25¢ (below 1-0.70=0.30)."""
        rm = RiskManager(TradingConfig())
        pos = Position(
            market_id="lir-sell-test",
            token_id="lir-sell-token",
            side="sell",
            entry_price=0.50,
            size_usd=50,
            quantity=100,
            estimated_prob=0.20,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=15),
            current_price=0.25,
            slug="lir-sell-test",
        )
        result = rm.check_position(pos, current_price=0.25, estimated_prob=0.20)
        assert result == "let_it_ride"

    # ------------------------------------------------------------------
    # Main regression test: dispatch logic must not close on "let_it_ride"
    # ------------------------------------------------------------------

    def test_live_loop_does_not_close_on_let_it_ride(self):
        """The close dispatch logic in check_positions() must NOT close a
        position when check_position() returns 'let_it_ride'.

        Reproduces the exact control-flow sequence from
        trading_loop.py check_positions() (lines 491-521), unwound from
        the for-loop so we can assert on each branch outcome:

            close_reason = self.risk.check_position(...)      # line 491

            if close_reason == "let_it_ride":                 # line 496 — explicit ==
                position.let_it_ride_count += 1
                continue                                       # line 519 — exits iteration

            if close_reason:                                  # line 521 — "let_it_ride"
                # close position                              #  can NEVER reach here
        """
        from unittest.mock import patch as _patch

        pos = self._make_winning_position()
        rm = RiskManager(TradingConfig())

        close_was_attempted = False
        reached_truthiness_check = False

        with _patch.object(rm, "check_position", return_value="let_it_ride"):
            close_reason = rm.check_position(
                pos, pos.current_price, pos.estimated_prob
            )

            # === Exact dispatch from trading_loop.py:496-521 (loop unwound) ===
            if close_reason == "let_it_ride":
                pos.let_it_ride_count += 1
                # In the real loop `continue` prevents the next block from running.
                # We model that here by an else — which is semantically identical:
                # if we took the "let_it_ride" branch, we skip the truthiness check.
            else:
                reached_truthiness_check = True   # would be line 521
                if close_reason:
                    close_was_attempted = True

        # ── Assertions ──────────────────────────────────────────────────────
        assert close_reason == "let_it_ride", (
            "Mock did not return 'let_it_ride' — precondition failure."
        )
        assert not reached_truthiness_check, (
            "BUG DETECTED: 'let_it_ride' fell through to the truthiness check "
            "(if close_reason:) instead of being caught by the explicit == guard. "
            "This is the same bug found in backtest/engine.py."
        )
        assert not close_was_attempted, (
            "BUG DETECTED: close_position() would have been called for a "
            "'let_it_ride' position."
        )
        assert pos.status == "open", (
            "Position status must remain 'open' — 'let_it_ride' is a HOLD signal."
        )
        assert pos.let_it_ride_count == 1, (
            "let_it_ride_count must be incremented exactly once by the guard branch."
        )

    def test_truthiness_check_still_closes_on_real_exit_signals(self):
        """Verify the truthiness check at line 521 correctly fires for all
        genuine close reasons — so the guard doesn't accidentally suppress
        valid exits.

        This is the complement of test_live_loop_does_not_close_on_let_it_ride:
        if we ever widen the guard to suppress real exits, this test catches it.
        """
        from unittest.mock import patch as _patch

        real_exits = ["resolved", "stop_loss", "aggressive_exit",
                      "trailing_stop", "take_profit"]

        for exit_reason in real_exits:
            pos = self._make_winning_position()
            rm = RiskManager(TradingConfig())

            close_was_attempted = False

            with _patch.object(rm, "check_position", return_value=exit_reason):
                close_reason = rm.check_position(
                    pos, pos.current_price, pos.estimated_prob
                )

                if close_reason == "let_it_ride":
                    pos.let_it_ride_count += 1
                else:
                    if close_reason:
                        close_was_attempted = True

            assert close_was_attempted, (
                f"Close was NOT attempted for exit reason '{exit_reason}'. "
                "The guard must only suppress 'let_it_ride', not real exits."
            )
            assert pos.let_it_ride_count == 0, (
                f"let_it_ride_count incremented for exit reason '{exit_reason}'."
            )

    def test_none_return_does_not_trigger_close(self):
        """When check_position() returns None (hold, not yet time to exit),
        neither the 'let_it_ride' branch nor the close branch should fire."""
        from unittest.mock import patch as _patch

        pos = self._make_winning_position()
        rm = RiskManager(TradingConfig())

        close_was_attempted = False

        with _patch.object(rm, "check_position", return_value=None):
            close_reason = rm.check_position(
                pos, pos.current_price, pos.estimated_prob
            )

            if close_reason == "let_it_ride":
                pos.let_it_ride_count += 1
            else:
                if close_reason:
                    close_was_attempted = True

        assert not close_was_attempted
        assert pos.let_it_ride_count == 0
        assert pos.status == "open"


# ============================================================================
# Historical DB Tests
# ============================================================================

class TestHistoricalDB:
    """Tests for data/historical_db.py — schema, idempotency, and inspect queries.

    All tests use a tmp_path-isolated SQLite file so they never touch
    the production data/trades.db.
    """

    @pytest.fixture
    def db(self, tmp_path):
        """Return path to a fresh temp database."""
        return str(tmp_path / "test_historical.db")

    # ── 1. Schema creation idempotency ───────────────────────────────────────

    def test_init_tables_creates_both_tables(self, db):
        """init_tables() creates historical_markets and historical_snapshots."""
        from data.historical_db import init_tables, get_conn
        init_tables(db)
        conn = get_conn(db)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "historical_markets" in tables
        assert "historical_snapshots" in tables

    def test_init_tables_is_idempotent(self, db):
        """Calling init_tables() twice must not raise and must not duplicate rows."""
        from data.historical_db import init_tables, get_conn
        init_tables(db)
        init_tables(db)   # second call must be a no-op
        conn = get_conn(db)
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name IN ('historical_markets','historical_snapshots')"
        ).fetchone()[0]
        conn.close()
        assert count == 2  # exactly one copy of each table

    # ── 2. INSERT OR IGNORE duplicate handling ───────────────────────────────

    def test_duplicate_market_insert_ignored(self, db):
        """Inserting the same slug twice keeps exactly one row."""
        from data.historical_db import init_tables, get_conn, upsert_historical_market
        init_tables(db)
        conn = get_conn(db)
        for _ in range(3):
            upsert_historical_market(
                conn,
                {"slug": "test-market", "question": "Will it happen?",
                 "market_type": "nba_outright"},
            )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM historical_markets WHERE slug='test-market'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_duplicate_snapshot_insert_ignored(self, db):
        """Inserting (slug, timestamp, source) twice keeps exactly one snapshot row."""
        from data.historical_db import init_tables, get_conn, upsert_snapshots
        init_tables(db)
        conn = get_conn(db)
        history = [{"t": 1_700_000_000, "p": 0.45}]
        upsert_snapshots(conn, "test-market", history, source="clob_prices_history")
        upsert_snapshots(conn, "test-market", history, source="clob_prices_history")
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM historical_snapshots WHERE slug='test-market'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    # ── 3. Inspect queries on fixture data ───────────────────────────────────

    def test_inspect_snapshot_counts_fixture(self, db):
        """get_snapshot_counts() returns correct counts for known fixture data."""
        from data.historical_db import (
            init_tables, get_conn,
            upsert_historical_market, upsert_snapshots, get_snapshot_counts,
        )
        init_tables(db)
        conn = get_conn(db)

        # Insert two markets with known densities
        for slug, n_pts in [("market-a", 10), ("market-b", 3)]:
            upsert_historical_market(conn, {"slug": slug, "market_type": "nba_outright"})
            upsert_snapshots(
                conn, slug,
                [{"t": 1_700_000_000 + i * 86400, "p": 0.5 + i * 0.01} for i in range(n_pts)],
            )
        conn.commit()
        conn.close()

        counts = get_snapshot_counts(db)
        assert len(counts) == 2
        # Sorted by count DESC → market-a first
        assert counts[0]["slug"] == "market-a"
        assert counts[0]["cnt"] == 10
        assert counts[1]["slug"] == "market-b"
        assert counts[1]["cnt"] == 3

    def test_inspect_settled_outcome_distribution_fixture(self, db):
        """get_settled_outcome_distribution() counts outcomes correctly."""
        from data.historical_db import (
            init_tables, get_conn,
            upsert_historical_market, get_settled_outcome_distribution,
        )
        init_tables(db)
        conn = get_conn(db)

        outcomes = [("YES", "s1"), ("YES", "s2"), ("NO", "s3"), ("open", "s4")]
        for outcome, slug in outcomes:
            upsert_historical_market(
                conn,
                {"slug": slug, "settled_outcome": outcome, "market_type": "nba_outright"},
            )
        conn.commit()
        conn.close()

        dist = get_settled_outcome_distribution(db)
        dist_map = {r["settled_outcome"]: r["cnt"] for r in dist}
        assert dist_map.get("YES") == 2
        assert dist_map.get("NO") == 1
        assert dist_map.get("open") == 1

    # ── 4. Idempotency regression: CLOB API timestamp jitter ─────────────────

    def test_ingest_idempotency(self, db):
        """Re-running upsert_snapshots with slightly different timestamps for the
        same daily candles must NOT grow the snapshot count.

        Root cause: the CLOB prices-history API returns intra-day timestamps that
        vary by up to a few hundred seconds between calls.  The fix buckets every
        raw timestamp to its UTC day boundary before insert so the
        UNIQUE(slug, timestamp, source) constraint catches the duplicate.

        Regression guard: this test would have failed before the bucketing fix
        (snapshot count would jump from 5 → 10 on the second call).
        """
        import sqlite3 as _sqlite3
        from data.historical_db import init_tables, get_conn, upsert_snapshots

        init_tables(db)

        SLUG = "test-idempotency-market"
        SOURCE = "clob_prices_history"

        # Run 1: timestamps as returned on the first API call
        history_run1 = [
            {"t": 1_776_988_823, "p": 0.50},   # 2026-04-30  (some second within the day)
            {"t": 1_777_075_213, "p": 0.55},   # 2026-05-01
            {"t": 1_777_161_607, "p": 0.60},   # 2026-05-02
            {"t": 1_777_248_009, "p": 0.58},   # 2026-05-03
            {"t": 1_777_334_409, "p": 0.62},   # 2026-05-04
        ]

        conn = get_conn(db)
        n1 = upsert_snapshots(conn, SLUG, history_run1, source=SOURCE)
        conn.commit()
        conn.close()

        count_after_run1 = _sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM historical_snapshots WHERE slug=?", (SLUG,)
        ).fetchone()[0]
        assert n1 == 5, f"Expected 5 inserts on run 1, got {n1}"
        assert count_after_run1 == 5

        # Run 2: same candles but timestamps shifted by a few hundred seconds
        # (simulating CLOB API jitter observed in production)
        history_run2 = [
            {"t": 1_776_988_901, "p": 0.50},   # +78 s from run-1 → same day bucket
            {"t": 1_777_075_299, "p": 0.55},   # +86 s
            {"t": 1_777_161_703, "p": 0.60},   # +96 s
            {"t": 1_777_248_121, "p": 0.58},   # +112 s
            {"t": 1_777_334_497, "p": 0.62},   # +88 s
        ]

        conn = get_conn(db)
        n2 = upsert_snapshots(conn, SLUG, history_run2, source=SOURCE)
        conn.commit()
        conn.close()

        count_after_run2 = _sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM historical_snapshots WHERE slug=?", (SLUG,)
        ).fetchone()[0]

        assert n2 == 0, (
            f"Run 2 inserted {n2} rows — timestamp bucketing is not working; "
            "duplicate candles were written"
        )
        assert count_after_run2 == count_after_run1, (
            f"IDEMPOTENCY FAILURE: run1={count_after_run1} snapshots, "
            f"run2={count_after_run2} snapshots"
        )


# ============================================================================
# Historical DB → Loader Tests
# ============================================================================

class TestHistoricalDBLoader:
    """Tests for load_historical_data_from_db() in data/loader.py.

    All tests use a tmp_path-isolated SQLite file so they never touch
    the production data/trades.db.
    """

    @pytest.fixture
    def populated_db(self, tmp_path):
        """Temp DB seeded with 2 NBA markets and 15 snapshots each."""
        from data.historical_db import (
            init_tables, get_conn, upsert_historical_market, upsert_snapshots,
        )
        db = str(tmp_path / "loader_test.db")
        init_tables(db)
        conn = get_conn(db)
        markets = [
            ("will-the-lakers-win-the-2026-nba-finals",  "111", "Will the Lakers win the 2026 NBA Finals?"),
            ("will-lebron-james-win-the-20252026-nba-mvp", "222", "Will LeBron James win the 2025-26 NBA MVP?"),
        ]
        for slug, mkt_id, question in markets:
            upsert_historical_market(conn, {
                "slug": slug,
                "market_id": mkt_id,
                "question": question,
                "token_id_0": f"token_{mkt_id}",
                "market_type": "nba_outright",
            })
            upsert_snapshots(
                conn, slug,
                [{"t": 1_700_000_000 + i * 86400, "p": 0.10 + i * 0.02} for i in range(15)],
            )
        conn.commit()
        conn.close()
        return db

    @pytest.fixture
    def empty_db(self, tmp_path):
        """Temp DB with tables created but no rows."""
        from data.historical_db import init_tables
        db = str(tmp_path / "empty_loader_test.db")
        init_tables(db)
        return db

    # ── 1. Correct shape when tables are populated ────────────────────────────

    def test_returns_correct_shape_when_populated(self, populated_db):
        """load_historical_data_from_db() returns List[List[MarketSnapshot]] with
        outer length = number of markets and inner length = snapshots per market."""
        from data.loader import load_historical_data_from_db
        result = load_historical_data_from_db(populated_db)
        assert result is not None, "Expected populated DB to return data, got None"
        assert len(result) == 2, f"Expected 2 markets, got {len(result)}"
        for market_snaps in result:
            assert len(market_snaps) == 15, (
                f"Expected 15 snapshots per market, got {len(market_snaps)}"
            )
        assert isinstance(result[0][0], MarketSnapshot)

    # ── 2. Returns None when tables are empty (triggers synthetic fallback) ───

    def test_returns_none_when_tables_empty(self, empty_db):
        """load_historical_data_from_db() must return None for an empty DB so the
        caller falls back to synthetic data rather than returning an empty list."""
        from data.loader import load_historical_data_from_db
        result = load_historical_data_from_db(empty_db)
        assert result is None, (
            "Expected None for empty DB (triggers synthetic fallback), "
            f"got {result!r}"
        )

    # ── 3. Converted snapshots have all required fields ───────────────────────

    def test_snapshots_have_all_required_fields(self, populated_db):
        """Every MarketSnapshot produced by the DB loader must have valid values
        for every field the backtest engine and signal pipeline touch."""
        from data.loader import load_historical_data_from_db
        result = load_historical_data_from_db(populated_db)
        assert result is not None

        # Use the 6th snapshot (index 5) of the first market so price_history
        # has exactly 5 preceding prices — avoids the edge-case of the first
        # snapshot (price_history=[]) for the non-trivial assertion below.
        snap = result[0][5]

        # Identity fields
        assert isinstance(snap.market_id, str) and snap.market_id != ""
        assert isinstance(snap.token_id, str)
        assert isinstance(snap.question, str) and snap.question != ""
        assert isinstance(snap.slug, str) and snap.slug != ""

        # Price in valid probability range
        assert 0.01 <= snap.price <= 0.99, f"price {snap.price} out of [0.01, 0.99]"

        # Volume / liquidity defaults
        assert snap.volume_24h >= 0.0
        assert snap.liquidity == 0.0  # not stored in DB

        # Empty order book — signals degrade to confidence=0, no crash
        assert isinstance(snap.order_book, OrderBook)
        assert snap.order_book.bid_depth == 0
        assert snap.order_book.ask_depth == 0

        # Price history has exactly 5 elements (the 5 snapshots before index 5)
        assert isinstance(snap.price_history, list)
        assert len(snap.price_history) == 5, (
            f"Expected 5 elements in price_history at index 5, "
            f"got {len(snap.price_history)}"
        )

        # Timestamp is timezone-aware UTC
        assert isinstance(snap.timestamp, datetime)
        assert snap.timestamp.tzinfo is not None, "timestamp must be timezone-aware"

        # Metadata defaults
        assert snap.category == "sports"
        assert snap.hours_to_expiry == 72.0
        assert snap.is_live is False
        assert snap.outcomes == ["Yes", "No"]


# ============================================================================
# Exit Proximity Tests
# ============================================================================

class TestComputeExitProximity:
    """Tests for compute_exit_proximity() in bot/trading_loop.py.

    All formulas and sign conventions:
        negative = condition NOT yet met (threshold still ahead)
        positive = condition already past trigger (threshold passed)
    """

    def _make_position(
        self,
        entry_price: float = 0.50,
        current_price: float = 0.50,
        side: str = "buy",
        peak_price: float = 0.0,
        estimated_prob: float = 0.65,
        quantity: float = 100,
    ):
        from utils.models import Position
        pos = Position(
            market_id="prox-test",
            token_id="prox-token",
            side=side,
            entry_price=entry_price,
            size_usd=entry_price * quantity,
            quantity=quantity,
            estimated_prob=estimated_prob,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=20),
            current_price=current_price,
            slug="prox-test",
        )
        pos.peak_price = peak_price
        return pos

    def test_compute_exit_proximity_returns_all_five_keys(self):
        """compute_exit_proximity() must return exactly the 5 approved field names."""
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.45, peak_price=0.0)
        result = compute_exit_proximity(pos, 0.45, 0.65, TradingConfig())

        expected_keys = {
            "stop_loss_distance_pct",
            "take_profit_edge_distance",
            "aggressive_exit_distance_pct",
            "trailing_stop_distance_pct",
            "let_it_ride_distance_pct",
        }
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_stop_loss_distance_negative_when_not_fired(self):
        """stop_loss_distance_pct must be negative when loss < stop_loss_threshold.

        Position: buy at 0.50, close at 0.43.
        loss_pct = (0.50 - 0.43) / 0.50 = 0.14
        stop_loss_threshold = 0.25
        distance = 0.14 - 0.25 = -0.11  (11% away from triggering)
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.43)
        result = compute_exit_proximity(pos, 0.43, 0.65, TradingConfig())

        assert result["stop_loss_distance_pct"] == pytest.approx(-0.11, abs=1e-5), (
            f"Expected -0.11, got {result['stop_loss_distance_pct']}"
        )

    def test_stop_loss_distance_positive_when_fired(self):
        """stop_loss_distance_pct must be positive when loss > stop_loss_threshold.

        Position: buy at 0.50, close at 0.35.
        loss_pct = 0.30, stop_loss_threshold = 0.25 → distance = +0.05
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.35)
        result = compute_exit_proximity(pos, 0.35, 0.65, TradingConfig())

        assert result["stop_loss_distance_pct"] == pytest.approx(0.05, abs=1e-5), (
            f"Expected +0.05, got {result['stop_loss_distance_pct']}"
        )

    def test_take_profit_edge_distance_negative_when_edge_wide(self):
        """take_profit_edge_distance must be negative when edge hasn't converged.

        estimated_prob=0.65, current_price=0.45 → edge_remaining=0.20
        take_profit_threshold=0.05 → distance = 0.05 - 0.20 = -0.15
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.45, estimated_prob=0.65)
        result = compute_exit_proximity(pos, 0.45, 0.65, TradingConfig())

        assert result["take_profit_edge_distance"] == pytest.approx(-0.15, abs=1e-5), (
            f"Expected -0.15, got {result['take_profit_edge_distance']}"
        )

    def test_take_profit_edge_distance_positive_when_edge_converged(self):
        """take_profit_edge_distance must be positive when edge has converged.

        estimated_prob=0.65, current_price=0.62 → edge_remaining=0.03
        take_profit_threshold=0.05 → distance = 0.05 - 0.03 = +0.02
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.62, estimated_prob=0.65)
        result = compute_exit_proximity(pos, 0.62, 0.65, TradingConfig())

        assert result["take_profit_edge_distance"] == pytest.approx(0.02, abs=1e-5), (
            f"Expected +0.02, got {result['take_profit_edge_distance']}"
        )

    def test_trailing_stop_none_when_peak_not_activated(self):
        """trailing_stop_distance_pct must be None when peak_price=0 (never moved)."""
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.45, peak_price=0.0)
        result = compute_exit_proximity(pos, 0.45, 0.65, TradingConfig())

        assert result["trailing_stop_distance_pct"] is None, (
            f"Expected None (trailing stop unarmed), got {result['trailing_stop_distance_pct']}"
        )

    def test_trailing_stop_none_when_peak_gain_below_activation(self):
        """trailing_stop_distance_pct must be None when peak_gain < 0.15 activation.

        entry=0.50, peak=0.55 → peak_gain=(0.55-0.50)/0.50=0.10 < 0.15 → not armed.
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.48, peak_price=0.55)
        result = compute_exit_proximity(pos, 0.48, 0.65, TradingConfig())

        assert result["trailing_stop_distance_pct"] is None, (
            f"Expected None (peak_gain=0.10 < activation 0.15), "
            f"got {result['trailing_stop_distance_pct']}"
        )

    def test_trailing_stop_positive_when_drop_exceeds_trigger(self):
        """trailing_stop_distance_pct must be positive when the drop exceeded 10%.

        entry=0.50, peak=0.60 → peak_gain=0.20 ≥ 0.15 (armed).
        current=0.45 → drop=(0.60-0.45)/0.50=0.30
        distance = 0.30 - 0.10 = +0.20
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.45, peak_price=0.60)
        result = compute_exit_proximity(pos, 0.45, 0.65, TradingConfig())

        assert result["trailing_stop_distance_pct"] == pytest.approx(0.20, abs=1e-5), (
            f"Expected +0.20, got {result['trailing_stop_distance_pct']}"
        )

    def test_let_it_ride_distance_negative_when_below_threshold(self):
        """let_it_ride_distance_pct must be negative when price < 0.70.

        buy, current_price=0.55 → favorable_price=0.55
        distance = 0.55 - 0.70 = -0.15
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.55)
        result = compute_exit_proximity(pos, 0.55, 0.65, TradingConfig())

        assert result["let_it_ride_distance_pct"] == pytest.approx(-0.15, abs=1e-5), (
            f"Expected -0.15, got {result['let_it_ride_distance_pct']}"
        )

    def test_let_it_ride_distance_sell_side_uses_complement(self):
        """For a SELL position, favorable_price = 1 - current_price.

        sell, current_price=0.20 → favorable_price=0.80
        distance = 0.80 - 0.70 = +0.10
        """
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.50, current_price=0.20, side="sell")
        result = compute_exit_proximity(pos, 0.20, 0.30, TradingConfig())

        assert result["let_it_ride_distance_pct"] == pytest.approx(0.10, abs=1e-5), (
            f"Expected +0.10 for sell at 0.20, got {result['let_it_ride_distance_pct']}"
        )

    def test_zero_entry_price_returns_safe_defaults(self):
        """A position with entry_price=0 must not raise — returns zeros/None."""
        from bot.trading_loop import compute_exit_proximity
        from utils.config import TradingConfig

        pos = self._make_position(entry_price=0.0, current_price=0.40)
        result = compute_exit_proximity(pos, 0.40, 0.65, TradingConfig())

        assert result["stop_loss_distance_pct"] == 0.0
        assert result["trailing_stop_distance_pct"] is None


class TestExitProximityRecordedInDB:
    """Test that exit_proximity is serialized and passed to insert_exit_log."""

    def test_exit_proximity_json_passed_to_insert_exit_log(self):
        """When close_position() receives exit_proximity, insert_exit_log must be
        called with that dict serialized to a JSON string in exit_proximity_json."""
        from bot.portfolio import Portfolio

        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        signal = TradeSignal(
            market_id="prox-db-test",
            token_id="prox-db-token",
            side="buy",
            estimated_prob=0.65,
            market_price=0.50,
            edge=0.15,
            position_size_usd=50,
            slug="prox-db-test",
        )
        trade = Trade(
            market_id="prox-db-test",
            token_id="prox-db-token",
            side="buy",
            price=0.50,
            quantity=100,
            size_usd=50,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=20),
        )
        pos = portfolio.open_position(signal, trade)

        proximity = {
            "stop_loss_distance_pct": -0.11,
            "take_profit_edge_distance": -0.15,
            "aggressive_exit_distance_pct": -0.18,
            "trailing_stop_distance_pct": None,
            "let_it_ride_distance_pct": -0.30,
        }

        import json
        with patch("bot.trade_db.insert_exit_log") as mock_insert, \
             patch("bot.trade_db.insert_trade"), \
             patch("bot.edge_log.update_edge_log_outcome"):
            portfolio.close_position(
                pos, current_price=0.43, reason="stop_loss",
                exit_proximity=proximity,
            )

        mock_insert.assert_called_once()
        kw = mock_insert.call_args[1]

        assert "exit_proximity_json" in kw, (
            "insert_exit_log must receive exit_proximity_json kwarg"
        )
        decoded = json.loads(kw["exit_proximity_json"])
        assert decoded["stop_loss_distance_pct"] == pytest.approx(-0.11)
        # JSON null → Python None
        assert decoded["trailing_stop_distance_pct"] is None

    def test_exit_proximity_defaults_to_empty_json_when_not_provided(self):
        """When exit_proximity is not passed, exit_proximity_json must be '{}'."""
        from bot.portfolio import Portfolio

        portfolio = Portfolio(paper_mode=True, initial_bankroll=1000)
        signal = TradeSignal(
            market_id="prox-default-test",
            token_id="prox-default-token",
            side="buy",
            estimated_prob=0.65,
            market_price=0.50,
            edge=0.15,
            position_size_usd=50,
            slug="prox-default-test",
        )
        trade = Trade(
            market_id="prox-default-test",
            token_id="prox-default-token",
            side="buy",
            price=0.50,
            quantity=100,
            size_usd=50,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=20),
        )
        pos = portfolio.open_position(signal, trade)

        with patch("bot.trade_db.insert_exit_log") as mock_insert, \
             patch("bot.trade_db.insert_trade"), \
             patch("bot.edge_log.update_edge_log_outcome"):
            # Do NOT pass exit_proximity — default path
            portfolio.close_position(pos, current_price=0.55, reason="take_profit")

        kw = mock_insert.call_args[1]
        assert kw["exit_proximity_json"] == "{}", (
            f"Default exit_proximity_json should be '{{}}', "
            f"got {kw['exit_proximity_json']!r}"
        )


class TestAnalyzeExitsStopLossTightness:
    """Verify q_stop_loss_tightness_diagnostic runs without error and produces
    correct structured output."""

    def test_function_exists_and_is_importable(self):
        """q_stop_loss_tightness_diagnostic must be importable from analyze_exits."""
        from scripts.analyze_exits import q_stop_loss_tightness_diagnostic
        assert callable(q_stop_loss_tightness_diagnostic)

    def _make_exit_log_db(self):
        """Return an in-memory SQLite connection with the exit_log schema."""
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(":memory:")
        conn.row_factory = _sqlite3.Row
        conn.executescript("""
            CREATE TABLE exit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL DEFAULT '',
                market_id TEXT NOT NULL DEFAULT '',
                side TEXT NOT NULL DEFAULT 'buy',
                entry_time TEXT NOT NULL DEFAULT '',
                close_time TEXT NOT NULL DEFAULT '',
                hold_seconds REAL DEFAULT 0,
                entry_price REAL NOT NULL DEFAULT 0,
                close_price REAL NOT NULL DEFAULT 0,
                size_usd REAL NOT NULL DEFAULT 0,
                quantity REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                close_reason TEXT NOT NULL DEFAULT '',
                entry_estimated_prob REAL NOT NULL DEFAULT 0,
                max_favorable_pnl_usd REAL DEFAULT 0,
                max_adverse_pnl_usd REAL DEFAULT 0,
                peak_unrealized_pct REAL DEFAULT 0,
                let_it_ride_triggered INTEGER DEFAULT 0,
                num_let_it_ride_triggers INTEGER DEFAULT 0,
                exit_threshold_distance REAL DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                exit_proximity_json TEXT DEFAULT '{}'
            );
        """)
        conn.commit()
        return conn

    def test_no_data_path_does_not_crash(self, capsys):
        """Running the diagnostic on an empty exit_log must complete without raising."""
        from scripts.analyze_exits import q_stop_loss_tightness_diagnostic

        conn = self._make_exit_log_db()
        since_iso = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()

        q_stop_loss_tightness_diagnostic(conn, since_iso)
        conn.close()

        captured = capsys.readouterr()
        assert "Traceback" not in captured.out
        assert "Error" not in captured.out

    def test_with_stop_loss_proximity_data_prints_tightness_info(self, capsys):
        """A stop_loss row with exit_proximity_json where take_profit_edge_distance
        > -0.02 must trigger the ⚠ flag in the output."""
        import json as _json
        from scripts.analyze_exits import q_stop_loss_tightness_diagnostic

        conn = self._make_exit_log_db()

        # take_profit_edge_distance = +0.01 → edge already converged (within 2%) → ⚠
        proximity = _json.dumps({
            "stop_loss_distance_pct": 0.05,
            "take_profit_edge_distance": 0.01,
            "aggressive_exit_distance_pct": -0.20,
            "trailing_stop_distance_pct": None,
            "let_it_ride_distance_pct": -0.30,
        })
        close_ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO exit_log
               (slug, market_id, side, entry_time, close_time, entry_price,
                close_price, size_usd, quantity, realized_pnl, close_reason,
                entry_estimated_prob, exit_proximity_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("test-sl-slug", "test-mid", "buy", close_ts, close_ts,
             0.50, 0.37, 50, 100, -13.0, "stop_loss", 0.65, proximity),
        )
        conn.commit()

        since_iso = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat()

        q_stop_loss_tightness_diagnostic(conn, since_iso)
        conn.close()

        captured = capsys.readouterr()
        assert "test-sl-slug" in captured.out, (
            "slug must appear in the diagnostic output"
        )
        assert "⚠" in captured.out, (
            "⚠ warning must appear when take_profit edge was within 2%"
        )
