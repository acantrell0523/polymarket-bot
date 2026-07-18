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
    # WNBA: STARVED 2026-07-18 on measured evidence — avg CLV -1.57%,
    # 1 of 17 entries beat the close, -$20 realized. The market reads us,
    # not the reverse. 15% bar = effectively delisted without code removal;
    # revisit only if a future signal source demonstrates positive CLV.
    "wnba": 0.15,
    "mls": 0.05,
    "ufc": 0.06,
    # ITF tennis: thin, high-vig, notoriously noisy markets — demand more
    "itfme": 0.07,
    "itfwo": 0.07,
    # College football: soft market (the thesis), but hold the standard floor
    "cfb": 0.05,
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
    """Extract league code from slug (bare or prefixed slug family)."""
    from bot.leagues import league_from_slug
    league = league_from_slug(slug)
    if league:
        return league
    # Unregistered league (e.g. ufc): keep the prefixed-slug convention
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
    # Config-driven thresholds. Defaults preserve the historical hardcoded
    # behavior; the trading loop passes values from TradingConfig so all of
    # these are tunable without code changes.
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
    min_liquidity_usd: float = MIN_LIQUIDITY_USD,
    max_spread: Optional[float] = None,
    min_net_edge: Optional[float] = None,
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

    # 3. Price range check (avoid extreme favorites/longshots)
    price = snapshot.price
    if price < min_price or price > max_price:
        return f"price_{price:.2f}_outside_{min_price:.2f}-{max_price:.2f}_range"

    # 4. Liquidity check
    ob = snapshot.order_book
    total_depth = ob.bid_depth + ob.ask_depth if ob else 0
    if total_depth < min_liquidity_usd:
        return f"liquidity_{total_depth:.0f}_below_{min_liquidity_usd:.0f}_min"

    # 5. Correlated game check
    if game_id in open_game_ids:
        return f"already_have_position_on_game_{game_id}"

    # 6. Daily trade limit
    if daily_trades >= max_daily_trades:
        return f"daily_limit_{max_daily_trades}_reached"

    # 7. Last 5 minutes block (buzzer beater risk)
    if game_time_remaining is not None and game_time_remaining < 300:
        return f"last_5_minutes_block_{game_time_remaining:.0f}s_remaining"

    # 8. Spread check — a wide book can't be exited cleanly. The spread is set
    # on the signal by compute_edge_breakdown; 0.0 means "book missing", which
    # we let through (the liquidity check above already caught empty books).
    if max_spread is not None and signal.spread > max_spread:
        return f"spread_{signal.spread:.3f}_above_{max_spread:.3f}_max"

    # 9. Net edge check — edge must survive fees + crossing the spread.
    # signal.net_edge is set by compute_edge_breakdown (executable price with
    # taker fee); if the caller never computed it, skip the check.
    if min_net_edge is not None and signal.exec_price > 0:
        if signal.net_edge < min_net_edge:
            return (f"net_edge_{signal.net_edge*100:.1f}pct_below_"
                    f"{min_net_edge*100:.1f}pct_min_after_costs")

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


def get_live_signal(signal: TradeSignal):
    """The fresh live_win_prob signal on a TradeSignal, or None.

    Presence (confidence > 0) is what routes an opportunity through
    validate_live_trade instead of the pregame checklist.
    """
    for s in signal.signals:
        if s.name == "live_win_prob" and s.confidence > 0:
            return s
    return None


def validate_live_trade(
    signal: TradeSignal,
    snapshot: MarketSnapshot,
    live_signal,
    open_game_ids: set,
    game_id: str,
    daily_trades: int,
    max_daily_trades: int,
    game_time_remaining: Optional[float] = None,
    game_period: Optional[int] = None,
    game_total_seconds: Optional[float] = None,
    # thresholds (config-driven; defaults mirror TradingConfig)
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
    min_liquidity_usd: float = MIN_LIQUIDITY_USD,
    max_spread: Optional[float] = None,
    min_net_edge: Optional[float] = None,
    max_model_age_seconds: float = 45.0,
    max_divergence: float = 0.15,
    min_elapsed_seconds: float = 300.0,
    clockless_max_period: int = 7,
) -> Optional[str]:
    """Validation checklist for MID-GAME trades.

    The pregame checklist can't work here: sportsbooks pull their lines at
    tip-off, so its >=2-books rule rejected every live opportunity. For live
    trades the ESPN win-probability model IS the external validation, held
    to live-specific standards:

      1. Model freshness — data older than max_model_age_seconds is
         misinformation during a live game.
      2. Divergence sanity cap — a model-vs-market gap ABOVE max_divergence
         is a red flag, not an opportunity (Sky@Wings Q4: model 24.9% vs
         market 13%; the market was right). Real repricing lag is single
         digits.
      3. Game-phase window — skip the first min_elapsed_seconds (model
         warming up / chaotic lines) and the last 5 minutes (buzzer-beater
         risk); clockless sports (MLB) block from clockless_max_period on.
      4. The usual market-quality gates: per-league min edge, price band,
         liquidity, spread, net edge, correlated game, daily limit.

    Returns None if valid, else a rejection reason string.
    """
    league = get_league_from_slug(signal.slug)

    # 1. Model freshness
    age = float(live_signal.metadata.get("age_seconds", 9999))
    if age > max_model_age_seconds:
        return f"live_model_stale_{age:.0f}s_above_{max_model_age_seconds:.0f}s"

    # 2. Divergence sanity cap (model vs market, from the live signal itself)
    divergence = abs(float(live_signal.metadata.get("edge", 0)))
    if divergence > max_divergence:
        return (f"live_divergence_{divergence*100:.1f}pct_above_"
                f"{max_divergence*100:.0f}pct_red_flag")

    # 3. Game-phase window
    if game_time_remaining is not None:
        if game_time_remaining < 300:
            return f"last_5_minutes_block_{game_time_remaining:.0f}s_remaining"
        if game_total_seconds and game_total_seconds > 0:
            elapsed = game_total_seconds - game_time_remaining
            if elapsed < min_elapsed_seconds:
                return f"live_too_early_{elapsed:.0f}s_elapsed"
    elif game_period is not None:
        # Clockless sport: inning-based endgame block
        if game_period >= clockless_max_period:
            return f"live_period_{game_period}_at_or_past_{clockless_max_period}_block"
    else:
        # Live trade with NO game-phase information: fail closed. We cannot
        # rule out the endgame, which is exactly when we must not enter.
        return "live_no_game_phase_info"

    # 4a. Per-league minimum edge
    min_edge = get_league_min_edge(signal.slug)
    effective_edge = apply_short_bias(signal.edge, signal.side)
    if effective_edge < min_edge:
        return f"edge_{effective_edge*100:.1f}pct_below_{league}_{min_edge*100:.0f}pct_min"

    # 4b. Price band
    price = snapshot.price
    if price < min_price or price > max_price:
        return f"price_{price:.2f}_outside_{min_price:.2f}-{max_price:.2f}_range"

    # 4c. Liquidity
    ob = snapshot.order_book
    total_depth = ob.bid_depth + ob.ask_depth if ob else 0
    if total_depth < min_liquidity_usd:
        return f"liquidity_{total_depth:.0f}_below_{min_liquidity_usd:.0f}_min"

    # 4d. Spread
    if max_spread is not None and signal.spread > max_spread:
        return f"spread_{signal.spread:.3f}_above_{max_spread:.3f}_max"

    # 4e. Net edge after costs
    if min_net_edge is not None and signal.exec_price > 0:
        if signal.net_edge < min_net_edge:
            return (f"net_edge_{signal.net_edge*100:.1f}pct_below_"
                    f"{min_net_edge*100:.1f}pct_min_after_costs")

    # 4f. Correlated game
    if game_id in open_game_ids:
        return f"already_have_position_on_game_{game_id}"

    # 4g. Daily trade limit
    if daily_trades >= max_daily_trades:
        return f"daily_limit_{max_daily_trades}_reached"

    return None  # All live checks passed
