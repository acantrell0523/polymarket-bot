"""Signal functions for edge detection on Polymarket US."""

from typing import Optional, Tuple
from utils.models import MarketSnapshot, Signal
from utils.config import SignalConfig
from bot.signals.odds_api import OddsCache
from bot.signals.cross_market import PredictItCache
from bot.signals.crypto_api import CryptoCache
from bot.signals.sports_data import ESPNCache, GameContextAnalyzer


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
    # Guard: only strict GAME slugs may match a moneyline consensus. Spread/
    # total slugs (asc-mlb-tor-sd-...-neg-1pt5, tsc-...-8pt5) share the team
    # tokens, so slug matching handed them the MONEYLINE probability — 8-13%
    # phantom "edges" against the wrong reference, observed 2026-07-12 and
    # stopped only by the 3-book rule. Sports-family non-game slugs get no
    # external consensus, which blocks them at the gate.
    from bot.leagues import league_from_slug
    from bot.signals.live_win_prob import slug_game_teams
    if league_from_slug(snapshot.slug) and slug_game_teams(snapshot.slug) is None:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "non_game_market_no_moneyline_match"})

    if not odds_cache or not odds_cache.enabled:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_odds_api_key"})

    result = odds_cache.get_probability_for_slug(snapshot.slug)
    if result is None:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_matching_event"})

    consensus_prob, num_books = result

    # Minimum 2 books for any trade, 3 books for edges over 7%
    if num_books < 2:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": f"only_{num_books}_books", "consensus": consensus_prob})

    polymarket_price = snapshot.price

    # Check for sharp book consensus (Pinnacle, Circa)
    # If sharp books disagree with overall consensus, weight sharp books more heavily
    sharp_consensus = 0.0
    books_used = ""
    consensus_data = odds_cache.get_consensus_odds(snapshot.slug)
    if consensus_data:
        sharp_probs = consensus_data.get("sharp_probs", {})
        books_used = consensus_data.get("books_used", "")
        if sharp_probs:
            # Find the team we care about
            for team_name, prob in sharp_probs.items():
                if team_name.lower() == "draw":
                    continue
                parts = snapshot.slug.split("-")
                outcome_abbr = parts[-1] if len(parts) > 5 else (parts[2] if len(parts) > 2 else "")
                if odds_cache._team_matches(outcome_abbr, team_name):
                    sharp_consensus = prob
                    break
            # If sharp books differ from overall by 3%+, blend toward sharp
            if sharp_consensus > 0 and abs(sharp_consensus - consensus_prob) >= 0.03:
                # Weight sharp 60%, overall 40%
                consensus_prob = sharp_consensus * 0.6 + consensus_prob * 0.4

    edge = consensus_prob - polymarket_price

    # Record line movement for tracking.
    # Skipped for historical caches (backtest replays) so backtests never
    # write time-series rows into the live trades.db.
    if not getattr(odds_cache, "is_historical", False):
        try:
            from bot.edge_log import record_line_movement
            record_line_movement(
                slug=snapshot.slug,
                consensus_prob=consensus_prob,
                polymarket_price=polymarket_price,
                num_books=num_books,
                sharp_consensus=sharp_consensus,
                overall_consensus=consensus_prob,
            )
        except Exception:
            pass

    # Require 3 books for edges over 7% — large edges from a single source are unreliable
    if abs(edge) > 0.07 and num_books < 3:
        return Signal(name="odds_value", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": f"edge_{abs(edge)*100:.0f}pct_needs_3_books_have_{num_books}",
                                "consensus": consensus_prob, "edge": edge})

    # The signal value IS the consensus probability — what the market "should" be priced at
    value = max(0.01, min(0.99, consensus_prob))

    # Confidence scales with number of books and size of edge
    confidence = min(num_books / 5, 1.0) * min(abs(edge) * 5, 1.0)
    confidence = max(0.1, min(confidence, 1.0))

    # Boost confidence if sharp books agree with the direction
    if sharp_consensus > 0:
        sharp_edge = sharp_consensus - polymarket_price
        if (sharp_edge > 0 and edge > 0) or (sharp_edge < 0 and edge < 0):
            confidence = min(confidence * 1.15, 1.0)

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
            "sharp_consensus": float(sharp_consensus),
            "books_used": books_used,
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


def derivative_line_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    line_aggregator=None,
) -> Signal:
    """Book consensus for SPREAD/TOTAL markets — the primary external signal
    for derivative slugs (asc-/tsc- families).

    Verified semantics (2026-07-14): spread YES = first-listed team covers
    the signed line; total YES = over. The moneyline consensus is the WRONG
    reference for these markets (it produced 8-13% phantom edges on Jul 12);
    this signal compares against the book line at the SAME points instead.

    Confidence mirrors odds_value conventions: scales with distinct books
    quoting this exact line and with edge size, so the >=2 books validation
    rule keeps single-book lines from trading large.
    """
    neutral = Signal(name="derivative_line", value=0.5, confidence=0.0,
                     direction="neutral", metadata={"reason": "no_line_data"})
    if line_aggregator is None:
        return neutral

    from bot.leagues import parse_derivative_slug, LEAGUES
    parsed = parse_derivative_slug(snapshot.slug)
    if not parsed:
        neutral.metadata["reason"] = "not_derivative_slug"
        return neutral

    sport_key = (LEAGUES.get(parsed["league"]) or {}).get("odds_api_key")
    if not sport_key:
        return neutral

    try:
        result = line_aggregator.find_line(
            sport_key, parsed["away"], parsed["home"],
            parsed["kind"], parsed["line"],
        )
    except Exception as e:
        neutral.metadata["reason"] = f"line_lookup_failed: {e}"
        return neutral
    if result is None:
        neutral.metadata["reason"] = "line_not_quoted"
        return neutral

    prob, num_books = result
    prob = max(0.01, min(0.99, prob))
    edge = prob - snapshot.price
    confidence = min(num_books / 2, 1.0) * min(abs(edge) * 5, 1.0)
    confidence = max(0.1, min(confidence, 0.9))

    return Signal(
        name="derivative_line",
        value=float(prob),
        confidence=float(confidence),
        direction="bullish" if edge > 0 else "bearish" if edge < 0 else "neutral",
        metadata={"edge": edge, "num_books": num_books,
                  "kind": parsed["kind"], "line": parsed["line"]},
    )


def live_win_prob_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    live_cache=None,
) -> Signal:
    """ESPN live win-probability model vs the Polymarket price — the PRIMARY
    external signal for live games.

    While a game is in progress, the pregame sportsbook consensus is stale
    the moment anything happens; ESPN's per-play model reprices within
    seconds. Edge = model prob − market price captures Polymarket's repricing
    lag, which is where in-game edges live.

    Confidence is freshness-driven: 0.90 for data under 30s old, decaying
    linearly to 0 at 150s. A stale model output is misinformation during a
    live game, so it gets no vote rather than a reduced one.
    """
    neutral = Signal(name="live_win_prob", value=0.5, confidence=0.0,
                     direction="neutral", metadata={"reason": "no_live_data"})
    if live_cache is None or not getattr(snapshot, "is_live", False):
        return neutral

    try:
        result = live_cache.get_live_prob(snapshot.slug)
    except Exception as e:
        neutral.metadata["reason"] = f"live_lookup_failed: {e}"
        return neutral
    if result is None:
        return neutral

    prob, age = result
    prob = max(0.01, min(0.99, prob))

    if age <= 30:
        confidence = 0.90
    elif age >= 150:
        return Signal(name="live_win_prob", value=prob, confidence=0.0,
                      direction="neutral", metadata={"reason": "stale", "age": age})
    else:
        confidence = 0.90 * (150 - age) / 120

    edge = prob - snapshot.price
    return Signal(
        name="live_win_prob",
        value=float(prob),
        confidence=float(confidence),
        direction="bullish" if edge > 0 else "bearish" if edge < 0 else "neutral",
        metadata={"edge": edge, "age_seconds": age, "source": "espn_win_probability"},
    )


def cross_market_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    predictit_cache: Optional[PredictItCache] = None,
) -> Signal:
    """Compare Polymarket price to PredictIt price on the same event.

    Pure cross-market arbitrage: if PredictIt has 55% and Polymarket has 40%,
    that's real edge.

    Returns confidence=0 if no PredictIt match found.
    """
    if not predictit_cache:
        return Signal(name="cross_market", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_predictit_cache"})

    # Extract keywords from question for matching
    question = snapshot.question
    keywords = [w for w in question.replace("?", "").split() if len(w) > 3]

    result = predictit_cache.get_probability(question, keywords)
    if result is None:
        return Signal(name="cross_market", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_predictit_match"})

    predictit_prob, matched_name = result
    polymarket_price = snapshot.price
    edge = predictit_prob - polymarket_price

    value = max(0.01, min(0.99, predictit_prob))
    confidence = min(abs(edge) * 5, 0.9)  # Scale confidence with edge size

    direction = "bullish" if edge > 0.02 else "bearish" if edge < -0.02 else "neutral"

    return Signal(
        name="cross_market",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "predictit_prob": float(predictit_prob),
            "polymarket_price": float(polymarket_price),
            "edge": float(edge),
            "matched_contract": matched_name,
        },
    )


def sports_context_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    espn_cache=None,
    game_context_analyzer=None,
) -> Signal:
    """Game context signal using ESPN data: home advantage, fatigue, tournament context.

    Modifies confidence in the direction the game context suggests.
    """
    if not espn_cache or not game_context_analyzer:
        return Signal(name="sports_context", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_espn_cache"})

    # Parse sport and teams from slug
    parts = snapshot.slug.split("-")
    if len(parts) < 4:
        return Signal(name="sports_context", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "unparseable_slug"})

    sport_abbr = parts[1]
    sport_map = {
        "nba": "basketball_nba",
        "cbb": "basketball_ncaab",
        "wnba": "basketball_wnba",
    }
    sport_key = sport_map.get(sport_abbr)
    if not sport_key:
        return Signal(name="sports_context", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": f"unsupported_sport_{sport_abbr}"})

    home_abbr = parts[2]
    away_abbr = parts[3]

    context = game_context_analyzer.analyze(sport_key, home_abbr, away_abbr)

    modifier = context.get("context_modifier", 0)
    # Value: 0.5 + modifier (home advantage pushes toward home team winning)
    value = max(0.01, min(0.99, 0.5 + modifier))

    # Confidence based on how much context data we have
    confidence = 0.0
    if abs(modifier) > 0.01:
        confidence = min(abs(modifier) * 5, 0.7)

    # Boost confidence if we detected fatigue
    if context.get("fatigue_home") or context.get("fatigue_away"):
        confidence = max(confidence, 0.5)

    direction = "bullish" if modifier > 0.01 else "bearish" if modifier < -0.01 else "neutral"

    return Signal(
        name="sports_context",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "home_advantage": context.get("home_advantage", 0),
            "fatigue_home": context.get("fatigue_home", False),
            "fatigue_away": context.get("fatigue_away", False),
            "is_conference_tourney": context.get("is_conference_tourney", False),
            "home_record": context.get("home_record", ""),
            "away_record": context.get("away_record", ""),
            "context_modifier": modifier,
            "neutral_site": context.get("neutral_site", False),
        },
    )


def onchain_flow_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    onchain_client=None,
) -> Signal:
    """Whale / smart-money flow from free Polymarket CLOB trade data.

    Supporting signal only: it expresses a small directional TILT away from the
    current market price rather than an independent probability estimate,
    because trade-flow data tells you which way informed money is leaning but
    not what the fair probability is.

        value = price + 0.05 * whale_net_direction + 0.05 * smart_money_sentiment

    Both components are in [-1, 1], so the tilt is capped at ±10 points of
    probability. Confidence is capped at 0.5 so this can never dominate the
    external validation signals, and scales with how much real volume backs
    the reading.

    Returns confidence=0 (no-op) when no client is wired in or there is no
    trade activity to analyze.
    """
    if not onchain_client:
        return Signal(name="onchain_flow", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_onchain_client"})

    try:
        enrichment = onchain_client.get_enrichment_for_market(snapshot)
    except Exception as e:
        # On-chain data is best-effort — a failed fetch must never block a scan.
        return Signal(name="onchain_flow", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": f"enrichment_failed: {e}"})

    smart = enrichment.get("market_detail", {}).get("data", {})
    smart_sentiment = float(smart.get("smart_money_sentiment", 0) or 0)

    whale_items = enrichment.get("whale_data", {}).get("data", [])
    whale_net = 0.0
    whale_volume = 0.0
    whale_count = 0
    if whale_items:
        item = whale_items[0]
        whale_volume = float(item.get("amount", 0) or 0)
        whale_count = int(item.get("whale_count", 0) or 0)
        whale_net = 1.0 if item.get("direction") == "buy" else -1.0

    if whale_count == 0 and abs(smart_sentiment) < 0.05:
        return Signal(name="onchain_flow", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_flow_activity"})

    tilt = 0.05 * whale_net + 0.05 * smart_sentiment
    value = max(0.01, min(0.99, snapshot.price + tilt))

    # Confidence: needs real money behind it. $2k of whale volume or strong
    # smart-money sentiment reaches the 0.5 cap.
    volume_conf = min(whale_volume / 2000.0, 1.0)
    sentiment_conf = min(abs(smart_sentiment), 1.0)
    confidence = min(0.5, 0.5 * max(volume_conf, sentiment_conf))

    direction = "bullish" if tilt > 0.005 else "bearish" if tilt < -0.005 else "neutral"

    return Signal(
        name="onchain_flow",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "whale_net_direction": whale_net,
            "whale_volume_usd": whale_volume,
            "whale_count": whale_count,
            "smart_money_sentiment": smart_sentiment,
            "tilt": tilt,
        },
    )


def crypto_model_signal(
    snapshot: MarketSnapshot,
    config: SignalConfig,
    crypto_cache: Optional[CryptoCache] = None,
) -> Signal:
    """Estimate crypto target probability using log-normal model and compare to Polymarket.

    Uses current price, volatility, and time remaining to compute a model probability
    for "Will X hit $Y by date Z?" markets.

    Returns confidence=0 if we can't parse the market or fetch price data.
    """
    if not crypto_cache:
        return Signal(name="crypto_model", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_crypto_cache"})

    result = crypto_cache.estimate_probability(
        snapshot.question, snapshot.price, slug=snapshot.slug)
    if result is None:
        return Signal(name="crypto_model", value=0.5, confidence=0.0, direction="neutral",
                      metadata={"reason": "no_crypto_match"})

    model_prob, meta = result
    polymarket_price = snapshot.price
    edge = model_prob - polymarket_price

    value = max(0.01, min(0.99, model_prob))

    # Confidence — the external gate for crypto.
    #  * time_confidence: long-dated barriers are inherently uncertain (vol
    #    compounds), so trust them less.
    #  * parse_confidence: a slug-parsed deadline is authoritative; a
    #    question with NO date fell back to a 30-day GUESS (parse_source=
    #    "question" + days_remaining==30) — that guess must NOT drive a
    #    confident trade, so it's heavily discounted.
    #  * A barrier probability that is essentially 0/1 (target already
    #    touched, or wildly far) carries little tradeable signal — the edge
    #    term handles that, but cap so a 90-point "edge" can't max out.
    days = meta.get("days_remaining", 30)
    time_confidence = max(0.3, 1.0 - (days / 730))   # decay over ~2yr, not 1yr
    parse_confidence = 1.0
    if meta.get("parse_source") == "question" and days == 30:
        parse_confidence = 0.3   # 30-day default is a guess, not a deadline
    confidence = min(abs(edge) * 4, 0.8) * time_confidence * parse_confidence
    # Fat-tail guard: a GBM barrier model is LEAST reliable at the extremes
    # (0/1 rails) — crypto's fat tails routinely violate log-normality there,
    # so a model screaming "impossible/certain" is exactly where we should
    # NOT bet big. Halve confidence when the model pins the rails.
    if value <= 0.03 or value >= 0.97:
        confidence *= 0.5

    direction = "bullish" if edge > 0.02 else "bearish" if edge < -0.02 else "neutral"

    return Signal(
        name="crypto_model",
        value=float(value),
        confidence=float(confidence),
        direction=direction,
        metadata={
            "model_prob": float(model_prob),
            "polymarket_price": float(polymarket_price),
            "edge": float(edge),
            **{k: v for k, v in meta.items() if k != "model_prob"},
        },
    )
