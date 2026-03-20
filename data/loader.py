"""Historical data loader + synthetic data generator."""

import os
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from utils.models import MarketSnapshot, OrderBook, OrderBookLevel


def generate_synthetic_price_series(
    length: int = 500,
    start_price: float = 0.5,
    volatility: float = 0.02,
    drift: float = 0.0,
    seed: Optional[int] = None,
) -> List[float]:
    """Generate a synthetic price series using geometric Brownian motion, clamped to [0.01, 0.99]."""
    if seed is not None:
        np.random.seed(seed)

    prices = [start_price]
    for _ in range(length - 1):
        ret = drift + volatility * np.random.randn()
        new_price = prices[-1] * (1 + ret)
        new_price = max(0.01, min(0.99, new_price))
        prices.append(round(new_price, 4))

    return prices


def generate_synthetic_order_book(
    price: float,
    depth_levels: int = 5,
    base_size: float = 100.0,
    spread_bps: int = 100,
    seed: Optional[int] = None,
) -> OrderBook:
    """Generate a synthetic order book around a price."""
    if seed is not None:
        np.random.seed(seed)

    spread = price * spread_bps / 10000
    half_spread = spread / 2

    bids = []
    asks = []

    for i in range(depth_levels):
        bid_price = max(0.01, price - half_spread - i * spread * 0.5)
        ask_price = min(0.99, price + half_spread + i * spread * 0.5)
        bid_size = base_size * (1 + np.random.random()) * (depth_levels - i) / depth_levels
        ask_size = base_size * (1 + np.random.random()) * (depth_levels - i) / depth_levels

        bids.append(OrderBookLevel(price=round(bid_price, 4), size=round(bid_size, 2)))
        asks.append(OrderBookLevel(price=round(ask_price, 4), size=round(ask_size, 2)))

    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)

    return OrderBook(bids=bids, asks=asks)


def generate_synthetic_markets(
    num_markets: int = 20,
    history_length: int = 200,
    num_snapshots: int = 500,
    seed: int = 42,
) -> List[List[MarketSnapshot]]:
    """
    Generate synthetic market data for backtesting.

    Returns a list of markets, each containing a time series of snapshots.
    """
    np.random.seed(seed)
    markets = []

    categories = ["politics", "sports", "crypto", "science", "entertainment"]
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for m in range(num_markets):
        market_seed = seed + m
        np.random.seed(market_seed)

        start_price = 0.2 + np.random.random() * 0.6  # [0.2, 0.8]
        vol = 0.005 + np.random.random() * 0.03  # [0.005, 0.035]
        drift = (np.random.random() - 0.5) * 0.001  # Small drift

        prices = generate_synthetic_price_series(
            length=num_snapshots + history_length,
            start_price=start_price,
            volatility=vol,
            drift=drift,
            seed=market_seed,
        )

        market_id = f"market_{m:03d}"
        token_id = f"token_{m:03d}"
        category = categories[m % len(categories)]

        snapshots = []
        for i in range(num_snapshots):
            idx = i + history_length
            price = prices[idx]
            history = prices[max(0, idx - history_length):idx]

            volume = max(100, 1000 + np.random.randn() * 500)
            liquidity = max(50, 500 + np.random.randn() * 200)

            ob = generate_synthetic_order_book(
                price, seed=market_seed + i
            )

            timestamp = base_time + timedelta(hours=i)

            snapshot = MarketSnapshot(
                market_id=market_id,
                token_id=token_id,
                question=f"Will event {m} happen?",
                price=price,
                volume_24h=round(volume, 2),
                liquidity=round(liquidity, 2),
                order_book=ob,
                price_history=history,
                timestamp=timestamp,
                category=category,
                hours_to_expiry=max(6, 720 - i),
            )
            snapshots.append(snapshot)

        markets.append(snapshots)

    return markets


def load_historical_data(data_dir: str) -> Optional[List[List[MarketSnapshot]]]:
    """Load historical data from disk. Returns None if no data found."""
    if not os.path.exists(data_dir):
        return None

    # Look for JSON files
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        return None

    markets = []
    for fname in sorted(files):
        path = os.path.join(data_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        # Parse into snapshots (simplified)
        snapshots = []
        for entry in data:
            ob = OrderBook()
            snapshot = MarketSnapshot(
                market_id=entry.get("market_id", ""),
                token_id=entry.get("token_id", ""),
                question=entry.get("question", ""),
                price=entry.get("price", 0.5),
                volume_24h=entry.get("volume_24h", 0),
                liquidity=entry.get("liquidity", 0),
                order_book=ob,
                price_history=entry.get("price_history", []),
                timestamp=datetime.fromisoformat(entry.get("timestamp", "2024-01-01")),
                category=entry.get("category", ""),
            )
            snapshots.append(snapshot)
        if snapshots:
            markets.append(snapshots)

    return markets if markets else None
