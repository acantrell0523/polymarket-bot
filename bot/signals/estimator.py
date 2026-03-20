"""Combines all signals into probability estimates and detects edge."""

from typing import List, Optional, Dict, Any, Tuple
from utils.models import MarketSnapshot, Signal, TradeSignal
from utils.config import SignalConfig
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


class ProbabilityEstimator:
    """
    Combines all 8 signals into a single probability estimate.
    Uses confidence-weighted averaging.
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        self.weights = {
            "price_momentum": config.price_momentum_weight,
            "volume": config.volume_signal_weight,
            "order_book_imbalance": config.order_book_imbalance_weight,
            "mean_reversion": config.mean_reversion_weight,
            "volatility": config.volatility_signal_weight,
            "smart_money": config.uw_smart_money_weight,
            "whale_flow": config.uw_whale_flow_weight,
            "market_sentiment": config.uw_market_sentiment_weight,
        }

    def compute_signals(
        self, snapshot: MarketSnapshot, enrichment: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """Compute all signals for a market snapshot."""
        signals = [
            price_momentum_signal(snapshot, self.config),
            volume_signal(snapshot, self.config),
            order_book_imbalance_signal(snapshot, self.config),
            mean_reversion_signal(snapshot, self.config),
            volatility_signal(snapshot, self.config),
        ]

        # Add enrichment signals
        signals.append(smart_money_signal(enrichment))
        signals.append(whale_flow_signal(enrichment))
        signals.append(market_sentiment_signal(enrichment))

        return signals

    def estimate_probability(self, signals: List[Signal]) -> Tuple[float, float]:
        """
        Combine signals into a single probability estimate.

        Returns (estimated_probability, overall_confidence).

        Formula: p = sum(w_i * c_i * v_i) / sum(w_i * c_i)
        """
        weighted_sum = 0.0
        weight_sum = 0.0

        for signal in signals:
            w = self.weights.get(signal.name, 0)
            if w <= 0:
                continue
            effective_weight = w * signal.confidence
            weighted_sum += effective_weight * signal.value
            weight_sum += effective_weight

        if weight_sum == 0:
            return 0.5, 0.0

        estimated_prob = weighted_sum / weight_sum
        estimated_prob = max(0.01, min(0.99, estimated_prob))

        # Overall confidence: weighted average of individual confidences
        conf_sum = 0.0
        w_total = 0.0
        for signal in signals:
            w = self.weights.get(signal.name, 0)
            if w > 0:
                conf_sum += w * signal.confidence
                w_total += w
        overall_confidence = conf_sum / w_total if w_total > 0 else 0.0

        return estimated_prob, overall_confidence

    def detect_edge(
        self,
        snapshot: MarketSnapshot,
        min_edge: float,
        max_edge: float,
        enrichment: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeSignal]:
        """
        Run all signals, estimate probability, and detect if there's a tradeable edge.

        Returns a TradeSignal if edge exceeds threshold, else None.
        """
        signals = self.compute_signals(snapshot, enrichment)
        estimated_prob, confidence = self.estimate_probability(signals)

        # Edge = estimated probability - market price
        edge = estimated_prob - snapshot.price

        # Check if edge exceeds thresholds
        if abs(edge) < min_edge or abs(edge) > max_edge:
            return None

        # Determine trade side
        if edge > 0:
            side = "buy"  # Underpriced -> buy
        else:
            side = "sell"  # Overpriced -> sell

        return TradeSignal(
            market_id=snapshot.market_id,
            token_id=snapshot.token_id,
            side=side,
            estimated_prob=estimated_prob,
            market_price=snapshot.price,
            edge=edge,
            position_size_usd=0.0,  # Filled by position sizer
            signals=signals,
            timestamp=snapshot.timestamp,
            slug=snapshot.slug,
        )
