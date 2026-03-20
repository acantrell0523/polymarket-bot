"""
Unusual Whales signal functions - 3 enrichment signals.

These functions consume the enrichment dict produced by OnChainEnrichmentClient
(or the original UW API) and produce Signal objects. The signal layer doesn't
care where the data comes from - it just reads the dict keys.
"""

from typing import Dict, Any, Optional
from utils.models import Signal


def smart_money_signal(enrichment: Optional[Dict[str, Any]]) -> Signal:
    """
    Smart money signal: detects informed positioning from trade patterns.

    Reads from enrichment["market_detail"]["data"] for smart money metrics
    and enrichment["smart_money_data"]["data"] for accumulation flags.
    """
    if not enrichment:
        return Signal(name="smart_money", value=0.5, confidence=0.0, direction="neutral")

    market_detail = enrichment.get("market_detail", {})
    detail_data = market_detail.get("data", {})

    sentiment = detail_data.get("smart_money_sentiment", 0)
    conviction = detail_data.get("conviction_score", 0)
    accumulation = detail_data.get("accumulation_detected", False)

    # Smart money data (accumulation patterns)
    smart_data = enrichment.get("smart_money_data", {})
    smart_items = smart_data.get("data", [])

    # Base value from sentiment
    value = 0.5 + sentiment * 0.2  # [-1,1] -> [0.3, 0.7]
    value = max(0.1, min(0.9, value))

    # Boost confidence if accumulation detected
    confidence = conviction
    if accumulation or len(smart_items) > 0:
        confidence = min(confidence + 0.2, 1.0)

    direction = "bullish" if sentiment > 0.1 else "bearish" if sentiment < -0.1 else "neutral"

    return Signal(
        name="smart_money",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"sentiment": sentiment, "conviction": conviction, "accumulation": accumulation},
    )


def whale_flow_signal(enrichment: Optional[Dict[str, Any]]) -> Signal:
    """
    Whale flow signal: detects large position holders and unusual activity.

    Reads from enrichment["whale_data"], enrichment["unusual_data"],
    and enrichment["positions"].
    """
    if not enrichment:
        return Signal(name="whale_flow", value=0.5, confidence=0.0, direction="neutral")

    whale_data = enrichment.get("whale_data", {})
    whale_items = whale_data.get("data", [])

    unusual_data = enrichment.get("unusual_data", {})
    unusual_items = unusual_data.get("data", [])

    positions = enrichment.get("positions", {})
    pos_data = positions.get("data", {})
    yes_pct = pos_data.get("yes_percentage", 50)

    # Direction from whale trades
    whale_direction = 0.0
    whale_volume = 0.0
    for item in whale_items:
        d = item.get("direction", "")
        amt = item.get("amount", 0)
        if d == "buy":
            whale_direction += amt
        elif d == "sell":
            whale_direction -= amt
        whale_volume += amt

    # Normalize direction
    if whale_volume > 0:
        net_direction = whale_direction / whale_volume
    else:
        net_direction = 0.0

    # Combine whale direction with position skew
    position_skew = (yes_pct - 50) / 50  # [-1, 1]
    combined = net_direction * 0.7 + position_skew * 0.3

    value = 0.5 + combined * 0.2
    value = max(0.1, min(0.9, value))

    # Confidence from volume and unusual activity
    confidence = 0.0
    if whale_volume > 0:
        confidence = min(whale_volume / 5000, 0.8)
    if unusual_items:
        confidence = min(confidence + 0.2, 1.0)

    direction = "bullish" if combined > 0.1 else "bearish" if combined < -0.1 else "neutral"

    return Signal(
        name="whale_flow",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"net_direction": net_direction, "whale_volume": whale_volume, "yes_pct": yes_pct},
    )


def market_sentiment_signal(enrichment: Optional[Dict[str, Any]]) -> Signal:
    """
    Market sentiment signal: cross-market directional bias.

    Reads from enrichment["market_tide"] for macro sentiment.
    """
    if not enrichment:
        return Signal(name="market_sentiment", value=0.5, confidence=0.0, direction="neutral")

    tide = enrichment.get("market_tide", {})
    tide_data = tide.get("data", {})

    sentiment_score = tide_data.get("sentiment", 0)
    related_count = tide_data.get("related_count", 0)

    value = 0.5 + sentiment_score * 0.15
    value = max(0.2, min(0.8, value))

    # Confidence based on how many related markets we have data for
    confidence = min(related_count / 5, 0.8) if related_count > 0 else 0.0

    direction = "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral"

    return Signal(
        name="market_sentiment",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"sentiment_score": sentiment_score, "related_count": related_count},
    )
