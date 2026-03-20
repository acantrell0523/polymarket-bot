"""Signal functions for edge detection on Polymarket US."""

from typing import Optional, Tuple
from utils.models import MarketSnapshot, Signal
from utils.config import SignalConfig
from bot.signals.odds_api import OddsCache


def order_book_imbalance_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """Order book imbalance: ratio of bid depth to ask depth.

    More bids than asks = buying pressure = bullish.
    Uses order book data already fetched from the US API.
    """
    ob = snapshot.order_book
    bid_depth = ob.bid_depth
    ask_depth = ob.ask_depth

    total_depth = bid_depth + ask_depth
    if total_depth == 0:
        return Signal(name="order_book_imbalance", value=0.5, confidence=0.0, direction="neutral")

    # Imbalance ratio: 1.0 = all bids, 0.0 = all asks, 0.5 = balanced
    imbalance = bid_depth / total_depth

    # Value: shift from 0.5 based on imbalance
    value = 0.3 + imbalance * 0.4  # Maps [0,1] to [0.3, 0.7]

    # Confidence: higher when total depth is meaningful
    confidence = min(total_depth / 1000, 1.0)

    direction = "bullish" if imbalance > 0.55 else "bearish" if imbalance < 0.45 else "neutral"

    return Signal(
        name="order_book_imbalance",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"bid_depth": float(bid_depth), "ask_depth": float(ask_depth), "imbalance": float(imbalance)},
    )


def line_movement_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """Line movement: detect significant price changes from opening line.

    If the price has moved significantly in one direction, that movement
    carries information — the market is pricing in new information.
    Uses the difference between current price and mid-range (0.5) as a proxy
    for line movement, since we don't have historical opening prices.
    Markets far from 0.5 have already moved and carry strong consensus.
    """
    price = snapshot.price

    # How far from 50/50 — measures how much the line has moved
    distance_from_even = abs(price - 0.5)

    if distance_from_even < 0.05:
        # Near 50/50 — no strong line movement signal
        return Signal(name="line_movement", value=0.5, confidence=0.1, direction="neutral")

    # Strong favorites (price > 0.7 or < 0.3) have had significant line movement
    # The signal says: trust the direction the line has moved
    if price > 0.5:
        # Line has moved toward YES — bullish
        value = 0.5 + min(distance_from_even * 0.4, 0.2)
        direction = "bullish"
    else:
        # Line has moved toward NO — bearish
        value = 0.5 - min(distance_from_even * 0.4, 0.2)
        direction = "bearish"

    confidence = min(distance_from_even * 2, 0.9)

    return Signal(
        name="line_movement",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"price": float(price), "distance_from_even": float(distance_from_even)},
    )


def odds_value_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    odds_cache: Optional[OddsCache] = None,
) -> Signal:
    """Compare Polymarket price to consensus odds from major sportsbooks.

    This is the most important signal. If external books have a team at 55%
    but Polymarket has them at 40%, that's real edge.

    Returns confidence=0 if no external odds are available, which blocks the trade.
    """
    if not odds_cache or not odds_cache.enabled:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_odds_api_key"})

    result = odds_cache.get_probability_for_slug(snapshot.slug)
    if result is None:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_matching_event"})

    consensus_prob, num_books = result

    if num_books < 3:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": f"only_{num_books}_books", "consensus": consensus_prob})

    polymarket_price = snapshot.price
    edge = consensus_prob - polymarket_price

    # The signal value IS the consensus probability — what the market "should" be priced at
    value = max(0.01, min(0.99, consensus_prob))

    # Confidence scales with number of books and size of edge
    confidence = min(num_books / 5, 1.0) * min(abs(edge) * 5, 1.0)
    confidence = max(0.1, min(confidence, 1.0))

    direction = "bullish" if edge > 0.02 else "bearish" if edge < -0.02 else "neutral"

    return Signal(
        name="odds_value",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "consensus_prob": float(consensus_prob),
            "polymarket_price": float(polymarket_price),
            "edge": float(edge),
            "num_books": num_books,
        },
    )


def liquidity_imbalance_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """Liquidity imbalance: asymmetry in order book depth suggests informed money.

    Heavy liquidity on the bid side with thin asks suggests smart money
    is accumulating YES. The opposite suggests smart money is selling.

    Different from order_book_imbalance: this focuses on the ratio of
    top-of-book depth (top 3 levels) vs total depth to detect concentrated
    informed orders vs distributed retail flow.
    """
    ob = snapshot.order_book
    bids = ob.bids
    asks = ob.asks

    if not bids or not asks:
        return Signal(name="liquidity_imbalance", value=0.5, confidence=0.0, direction="neutral")

    # Top-of-book depth (first 3 levels)
    top_bid_depth = sum(level.size for level in bids[:3])
    top_ask_depth = sum(level.size for level in asks[:3])
    total_top = top_bid_depth + top_ask_depth

    if total_top == 0:
        return Signal(name="liquidity_imbalance", value=0.5, confidence=0.0, direction="neutral")

    # Full depth
    full_bid = ob.bid_depth
    full_ask = ob.ask_depth
    total_full = full_bid + full_ask

    # Concentration ratio: what fraction of total depth is at the top?
    # High concentration at top of book = likely informed/institutional
    bid_concentration = top_bid_depth / full_bid if full_bid > 0 else 0
    ask_concentration = top_ask_depth / full_ask if full_ask > 0 else 0

    # Imbalance of top-of-book
    top_imbalance = top_bid_depth / total_top

    # Value: shift based on where concentrated liquidity sits
    if top_imbalance > 0.6 and bid_concentration > 0.5:
        # Heavy concentrated bids = informed buying
        value = 0.5 + min((top_imbalance - 0.5) * 0.5, 0.2)
        direction = "bullish"
    elif top_imbalance < 0.4 and ask_concentration > 0.5:
        # Heavy concentrated asks = informed selling
        value = 0.5 - min((0.5 - top_imbalance) * 0.5, 0.2)
        direction = "bearish"
    else:
        value = 0.5
        direction = "neutral"

    confidence = min(total_top / 500, 0.8)

    return Signal(
        name="liquidity_imbalance",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "top_bid": float(top_bid_depth),
            "top_ask": float(top_ask_depth),
            "top_imbalance": float(top_imbalance),
            "bid_concentration": float(bid_concentration),
            "ask_concentration": float(ask_concentration),
        },
    )
