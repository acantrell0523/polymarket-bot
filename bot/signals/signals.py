"""5 core signal functions for edge detection."""

import numpy as np
from utils.models import MarketSnapshot, Signal
from utils.config import SignalConfig


def price_momentum_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """
    Compute price momentum using linear regression slope of recent prices.
    Positive slope = bullish momentum, negative = bearish.
    """
    lookback = config.momentum_lookback
    prices = snapshot.price_history[-lookback:] if len(snapshot.price_history) >= lookback else snapshot.price_history

    if len(prices) < 3:
        return Signal(name="price_momentum", value=0.5, confidence=0.0, direction="neutral")

    x = np.arange(len(prices), dtype=float)
    y = np.array(prices, dtype=float)

    # Linear regression
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()

    if ss_xx == 0:
        return Signal(name="price_momentum", value=0.5, confidence=0.0, direction="neutral")

    slope = ss_xy / ss_xx

    # R-squared for confidence
    y_pred = slope * (x - x_mean) + y_mean
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    r_squared = max(0, min(1, r_squared))

    # Normalize slope to a probability adjustment
    # Positive slope -> value > 0.5 (bullish), negative -> < 0.5 (bearish)
    normalized_slope = np.clip(slope * 10, -0.3, 0.3)
    value = 0.5 + normalized_slope

    direction = "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral"

    return Signal(
        name="price_momentum",
        value=float(value),
        confidence=float(r_squared),
        direction=direction,
        metadata={"slope": float(slope), "r_squared": float(r_squared)},
    )


def volume_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """
    Volume signal: compare current volume to average.
    High relative volume suggests informed trading activity.
    """
    if snapshot.volume_24h <= 0:
        return Signal(name="volume", value=0.5, confidence=0.0, direction="neutral")

    # Use price history length as a proxy for average volume periods
    # In production, we'd track volume history separately
    # Higher volume + price rising = bullish volume confirmation
    prices = snapshot.price_history
    if len(prices) < 2:
        return Signal(name="volume", value=0.5, confidence=0.3, direction="neutral")

    # Volume-price relationship
    recent_return = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0

    # Volume factor: assume average volume = liquidity * 5 as a proxy
    avg_volume_estimate = max(snapshot.liquidity * 5, 100)
    volume_ratio = snapshot.volume_24h / avg_volume_estimate
    volume_ratio = min(volume_ratio, 5.0)  # Cap at 5x

    # High volume + positive return = bullish, high volume + negative return = bearish
    if recent_return > 0:
        value = 0.5 + min(volume_ratio * 0.05, 0.2)
    elif recent_return < 0:
        value = 0.5 - min(volume_ratio * 0.05, 0.2)
    else:
        value = 0.5

    confidence = min(volume_ratio / 3, 1.0)
    direction = "bullish" if value > 0.5 else "bearish" if value < 0.5 else "neutral"

    return Signal(
        name="volume",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"volume_ratio": float(volume_ratio), "recent_return": float(recent_return)},
    )


def order_book_imbalance_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """
    Order book imbalance: ratio of bid depth to ask depth.
    More bids than asks = buying pressure = bullish.
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


def mean_reversion_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """
    Mean reversion: z-score of current price vs rolling mean.
    Extreme z-scores suggest price will revert toward the mean.
    """
    window = config.mean_reversion_window
    prices = snapshot.price_history[-window:] if len(snapshot.price_history) >= window else snapshot.price_history

    if len(prices) < 5:
        return Signal(name="mean_reversion", value=0.5, confidence=0.0, direction="neutral")

    arr = np.array(prices, dtype=float)
    mean = arr.mean()
    std = arr.std()

    if std < 1e-8:
        return Signal(name="mean_reversion", value=0.5, confidence=0.0, direction="neutral")

    z_score = (snapshot.price - mean) / std

    # If price is above mean (positive z), expect reversion down -> bearish
    # If price is below mean (negative z), expect reversion up -> bullish
    # Only signal when z exceeds threshold
    z_threshold = config.mean_reversion_z_threshold

    if abs(z_score) < z_threshold * 0.5:
        value = 0.5
        confidence = 0.1
        direction = "neutral"
    elif z_score > 0:
        # Price above mean, expect down
        value = 0.5 - min(z_score * 0.1, 0.25)
        confidence = min(abs(z_score) / (z_threshold * 2), 1.0)
        direction = "bearish"
    else:
        # Price below mean, expect up
        value = 0.5 + min(abs(z_score) * 0.1, 0.25)
        confidence = min(abs(z_score) / (z_threshold * 2), 1.0)
        direction = "bullish"

    return Signal(
        name="mean_reversion",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={"z_score": float(z_score), "mean": float(mean), "std": float(std)},
    )


def volatility_signal(snapshot: MarketSnapshot, config: SignalConfig) -> Signal:
    """
    Volatility signal: recent return volatility.
    High volatility = uncertainty = lower confidence in current price.
    """
    lookback = config.volatility_lookback
    prices = snapshot.price_history[-lookback:] if len(snapshot.price_history) >= lookback else snapshot.price_history

    if len(prices) < 3:
        return Signal(name="volatility", value=0.5, confidence=0.0, direction="neutral")

    arr = np.array(prices, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return Signal(name="volatility", value=0.5, confidence=0.0, direction="neutral")

    vol = float(np.std(returns))

    # High volatility -> price is uncertain -> stay closer to 0.5
    # Low volatility -> price is stable -> current price is more likely fair
    # This signal doesn't have a directional bias, it modulates confidence
    value = 0.5  # Neutral directional value

    # Confidence is inverse of volatility (stable markets = more confident in other signals)
    confidence = float(np.clip(1.0 - vol * 5, 0.1, 1.0))

    return Signal(
        name="volatility",
        value=float(value),
        confidence=float(confidence),
        direction="neutral",
        metadata={"volatility": float(vol)},
    )
