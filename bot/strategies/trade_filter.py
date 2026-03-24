"""Pre-trade validation and strategic filtering.

Every trade must pass ALL checks before execution.
This is the "sniper not machine gun" philosophy.
"""

from typing import Optional, Dict
from utils.models import TradeSignal, MarketSnapshot


# Per-league minimum edge thresholds
# NHL: less efficient, bigger edges → lower threshold
# NCAA: mispriced underdogs → moderate threshold
# NBA: most efficient → higher threshold
LEAGUE_MIN_EDGE = {
    "nhl": 0.04,
    "cbb": 0.05,
    "nba": 0.07,
    "epl": 0.05,
    "nfl": 0.05,
    "mlb": 0.05,
    "ufc": 0.06,
}
DEFAULT_MIN_EDGE = 0.05

# Price range: avoid extreme prices where resolution risk is high
MIN_PRICE = 0.15
MAX_PRICE = 0.85

# Minimum liquidity to ensure we can fill
MIN_LIQUIDITY_USD = 1000.0

# Short bias: shorts on favorites get a small edge bonus
# because overpriced favorites that lose pay out huge
SHORT_BIAS_BONUS = 0.005  # 0.5% edge bonus for shorts


def get_league_from_slug(slug: str) -> str:
    """Extract league code from slug."""
    parts = slug.split("-")
    return parts[1] if len(parts) >= 2 else ""


def get_league_min_edge(slug: str) -> float:
    """Get the minimum edge threshold for this league."""
    league = get_league_from_slug(slug)
    return LEAGUE_MIN_EDGE.get(league, DEFAULT_MIN_EDGE)


def apply_short_bias(edge: float, side: str) -> float:
    """Apply short bias bonus. Shorts on favorites get a slight edge boost."""
    if side == "sell":
        return abs(edge) + SHORT_BIAS_BONUS
    return abs(edge)


def validate_trade(
    signal: TradeSignal,
    snapshot: MarketSnapshot,
    num_books: int,
    open_game_ids: set,
    game_id: str,
    daily_trades: int,
    max_daily_trades: int,
    game_time_remaining: Optional[float] = None,
) -> Optional[str]:
    """Validate a trade against ALL pre-trade checks.

    Returns None if trade is valid, or a string reason why it was rejected.
    """
    league = get_league_from_slug(signal.slug)

    # 1. Minimum 2 books
    if num_books < 2:
        return f"only_{num_books}_books"

    # 2. Per-league minimum edge
    min_edge = get_league_min_edge(signal.slug)
    effective_edge = apply_short_bias(signal.edge, signal.side)
    if effective_edge < min_edge:
        return f"edge_{effective_edge*100:.1f}pct_below_{league}_{min_edge*100:.0f}pct_min"

    # 3. Price range check (15% to 85%)
    price = snapshot.price
    if price < MIN_PRICE or price > MAX_PRICE:
        return f"price_{price:.2f}_outside_15-85_range"

    # 4. Liquidity check
    ob = snapshot.order_book
    total_depth = ob.bid_depth + ob.ask_depth if ob else 0
    if total_depth < MIN_LIQUIDITY_USD:
        return f"liquidity_{total_depth:.0f}_below_{MIN_LIQUIDITY_USD:.0f}_min"

    # 5. Correlated game check
    if game_id in open_game_ids:
        return f"already_have_position_on_game_{game_id}"

    # 6. Daily trade limit
    if daily_trades >= max_daily_trades:
        return f"daily_limit_{max_daily_trades}_reached"

    # 7. Last 5 minutes block (buzzer beater risk)
    if game_time_remaining is not None and game_time_remaining < 300:
        return f"last_5_minutes_block_{game_time_remaining:.0f}s_remaining"

    return None  # All checks passed


def rank_opportunities(opportunities: list) -> list:
    """Rank trade opportunities. Best first.

    Scoring: absolute edge + short bias bonus.
    Shorts get a slight preference because our biggest winners were shorts.
    """
    def score(item):
        signal, snapshot = item
        edge = abs(signal.edge)
        # Short bias: shorts score slightly higher
        if signal.side == "sell":
            edge += SHORT_BIAS_BONUS
        return edge

    return sorted(opportunities, key=score, reverse=True)
