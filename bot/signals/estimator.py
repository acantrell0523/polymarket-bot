"""Combines signals into probability estimates and detects edge."""

from typing import List, Optional, Tuple
from utils.models import MarketSnapshot, Signal, TradeSignal
from utils.config import SignalConfig
from bot.signals.signals import (
    order_book_imbalance_signal,
    line_movement_signal,
    odds_value_signal,
    liquidity_imbalance_signal,
)
from bot.signals.odds_api import OddsCache


class ProbabilityEstimator:
    """Combines 4 signals into a single probability estimate."""

    def __init__(self, config: SignalConfig, odds_cache: Optional[OddsCache] = None):
        self.config = config
        self.odds_cache = odds_cache
        self.weights = {
            "order_book_imbalance": config.order_book_imbalance_weight,
            "line_movement": config.line_movement_weight,
            "odds_value": config.odds_value_weight,
            "liquidity_imbalance": config.liquidity_imbalance_weight,
        }

    def compute_signals(self, snapshot: MarketSnapshot) -> List[Signal]:
        """Compute all signals for a market snapshot."""
        return [
            order_book_imbalance_signal(snapshot, self.config),
            line_movement_signal(snapshot, self.config),
            odds_value_signal(snapshot, self.config, self.odds_cache),
            liquidity_imbalance_signal(snapshot, self.config),
        ]

    def estimate_probability(self, signals: List[Signal]) -> Tuple[float, float]:
        """Combine signals into a single probability estimate.

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
    ) -> Optional[TradeSignal]:
        """Run all signals, estimate probability, and detect tradeable edge.

        CRITICAL rules:
        1. If odds_value has confidence=0 (no external odds), do NOT trade.
        2. If odds-based edge < 2%, penalize all other signal confidences by 50%.
        3. If odds-based edge < 3%, cap the combined edge at 3%.
        """
        signals = self.compute_signals(snapshot)

        # Gate: require external odds validation
        odds_signal = next((s for s in signals if s.name == "odds_value"), None)
        if odds_signal is None or odds_signal.confidence == 0:
            return None

        # Extract the raw odds edge
        odds_edge = abs(odds_signal.metadata.get("edge", 0))

        # Confidence penalty: if odds edge < 2%, halve confidence of non-odds signals
        if odds_edge < 0.02:
            for s in signals:
                if s.name != "odds_value":
                    s.confidence *= 0.5

        estimated_prob, confidence = self.estimate_probability(signals)

        # Edge = estimated probability - market price
        edge = estimated_prob - snapshot.price

        # Edge cap: combined edge cannot exceed odds edge + 1%
        # This ensures other signals can add a small boost but not create edge from nothing
        max_allowed_edge = odds_edge + 0.01
        if abs(edge) > max_allowed_edge:
            edge = max_allowed_edge if edge > 0 else -max_allowed_edge

        if abs(edge) < min_edge or abs(edge) > max_edge:
            return None

        side = "buy" if edge > 0 else "sell"

        # Recompute estimated_prob from capped edge
        estimated_prob = snapshot.price + edge

        return TradeSignal(
            market_id=snapshot.market_id,
            token_id=snapshot.token_id,
            side=side,
            estimated_prob=estimated_prob,
            market_price=snapshot.price,
            edge=edge,
            position_size_usd=0.0,
            signals=signals,
            timestamp=snapshot.timestamp,
            slug=snapshot.slug,
        )
