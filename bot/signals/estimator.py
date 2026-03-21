"""Combines signals into probability estimates and detects edge.

Auto-detects market type from slug/question and routes to the appropriate
signal sources:
  - Sports → odds_api (the-odds-api.com)
  - Crypto → crypto_model (CoinGecko)
  - Politics/Other → cross_market (PredictIt)

The external validation gate applies to ALL types: no external data = no trade.
"""

import re
from typing import List, Optional, Tuple, Dict
from utils.models import MarketSnapshot, Signal, TradeSignal
from utils.config import SignalConfig
from bot.signals.signals import (
    order_book_imbalance_signal,
    line_movement_signal,
    odds_value_signal,
    liquidity_imbalance_signal,
    cross_market_signal,
    crypto_model_signal,
    sports_context_signal,
)
from bot.signals.odds_api import OddsCache
from bot.signals.cross_market import PredictItCache
from bot.signals.crypto_api import CryptoCache
from bot.signals.sports_data import ESPNCache, GameContextAnalyzer


# Market type detection patterns
SPORTS_PREFIXES = ("aec-", "asc-", "tsc-", "atc-")
CRYPTO_KEYWORDS = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|xrp|crypto|cardano|ada|"
    r"avalanche|avax|polkadot|dot|chainlink|link|bnb)\b",
    re.IGNORECASE,
)
POLITICS_KEYWORDS = re.compile(
    r"\b(election|president|congress|senate|governor|democrat|republican|"
    r"vote|primary|nominee|caucus|poll|cabinet|impeach)\b",
    re.IGNORECASE,
)

# Weights per market type
WEIGHTS = {
    "sports": {
        "odds_value": 0.40,
        "sports_context": 0.15,
        "line_movement": 0.20,
        "order_book_imbalance": 0.15,
        "liquidity_imbalance": 0.10,
    },
    "crypto": {
        "crypto_model": 0.45,
        "cross_market": 0.25,
        "order_book_imbalance": 0.20,
        "liquidity_imbalance": 0.10,
    },
    "politics": {
        "cross_market": 0.45,
        "order_book_imbalance": 0.25,
        "line_movement": 0.15,
        "liquidity_imbalance": 0.15,
    },
    "other": {
        "cross_market": 0.40,
        "order_book_imbalance": 0.25,
        "line_movement": 0.20,
        "liquidity_imbalance": 0.15,
    },
}


def detect_market_type(snapshot: MarketSnapshot) -> str:
    """Classify a market into sports/crypto/politics/other."""
    slug = snapshot.slug.lower()

    if any(slug.startswith(p) for p in SPORTS_PREFIXES):
        return "sports"

    q = snapshot.question
    if CRYPTO_KEYWORDS.search(q) or CRYPTO_KEYWORDS.search(slug):
        return "crypto"
    if POLITICS_KEYWORDS.search(q):
        return "politics"

    return "other"


class ProbabilityEstimator:
    """Multi-market-type estimator with external validation gate."""

    def __init__(
        self,
        config: SignalConfig,
        odds_cache: Optional[OddsCache] = None,
        predictit_cache: Optional[PredictItCache] = None,
        crypto_cache: Optional[CryptoCache] = None,
        espn_cache: Optional[ESPNCache] = None,
        game_context_analyzer: Optional[GameContextAnalyzer] = None,
    ):
        self.config = config
        self.odds_cache = odds_cache
        self.predictit_cache = predictit_cache
        self.crypto_cache = crypto_cache
        self.espn_cache = espn_cache
        self.game_context_analyzer = game_context_analyzer

    def compute_signals(self, snapshot: MarketSnapshot, market_type: str) -> List[Signal]:
        """Compute signals appropriate for the market type."""
        signals = [
            order_book_imbalance_signal(snapshot, self.config),
            liquidity_imbalance_signal(snapshot, self.config),
        ]

        if market_type == "sports":
            signals.append(odds_value_signal(snapshot, self.config, self.odds_cache))
            signals.append(line_movement_signal(snapshot, self.config))
            signals.append(sports_context_signal(
                snapshot, self.config, self.espn_cache, self.game_context_analyzer
            ))
        elif market_type == "crypto":
            signals.append(crypto_model_signal(snapshot, self.config, self.crypto_cache))
            signals.append(cross_market_signal(snapshot, self.config, self.predictit_cache))
        elif market_type == "politics":
            signals.append(cross_market_signal(snapshot, self.config, self.predictit_cache))
            signals.append(line_movement_signal(snapshot, self.config))
        else:  # "other"
            signals.append(cross_market_signal(snapshot, self.config, self.predictit_cache))
            signals.append(line_movement_signal(snapshot, self.config))

        return signals

    def _get_primary_signal(self, signals: List[Signal], market_type: str) -> Optional[Signal]:
        """Get the primary external validation signal for this market type."""
        primary_name = {
            "sports": "odds_value",
            "crypto": "crypto_model",
            "politics": "cross_market",
            "other": "cross_market",
        }.get(market_type)

        return next((s for s in signals if s.name == primary_name), None)

    def estimate_probability(
        self, signals: List[Signal], weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Combine signals using confidence-weighted averaging.

        p = sum(w_i * c_i * v_i) / sum(w_i * c_i)
        """
        weighted_sum = 0.0
        weight_sum = 0.0

        for signal in signals:
            w = weights.get(signal.name, 0)
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
            w = weights.get(signal.name, 0)
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
        """Detect tradeable edge with external validation gate.

        Rules:
        1. If primary external signal has confidence=0, do NOT trade.
        2. If external edge < 2%, penalize other signals by 50%.
        3. Combined edge capped at external edge + 1%.
        """
        market_type = detect_market_type(snapshot)
        weights = WEIGHTS.get(market_type, WEIGHTS["other"])
        signals = self.compute_signals(snapshot, market_type)

        # Gate: require external validation
        primary = self._get_primary_signal(signals, market_type)
        if primary is None or primary.confidence == 0:
            return None

        # Extract the raw external edge
        ext_edge = abs(primary.metadata.get("edge", 0))

        # Confidence penalty: if external edge < 2%, halve non-primary confidences
        if ext_edge < 0.02:
            for s in signals:
                if s.name != primary.name:
                    s.confidence *= 0.5

        estimated_prob, confidence = self.estimate_probability(signals, weights)

        # Edge = estimated probability - market price
        edge = estimated_prob - snapshot.price

        # Edge cap: combined edge cannot exceed external edge + 1%
        max_allowed_edge = ext_edge + 0.01
        if abs(edge) > max_allowed_edge:
            edge = max_allowed_edge if edge > 0 else -max_allowed_edge

        if abs(edge) < min_edge or abs(edge) > max_edge:
            return None

        side = "buy" if edge > 0 else "sell"
        estimated_prob = snapshot.price + edge

        # Log signals to SQLite for the dashboard
        try:
            from bot.trade_db import log_signals_for_slug
            log_signals_for_slug(
                slug=snapshot.slug,
                market_type=market_type,
                signals=signals,
                polymarket_price=snapshot.price,
                edge=edge,
            )
        except Exception:
            pass  # DB errors must not block trading

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
