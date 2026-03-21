"""Signal testing framework.

Three test modes:
1. Pseudo-backtest: Pull resolved markets and test what signals would have said
2. Morning scan: Show today's top edge opportunities with full breakdown
3. Signal analyzer: Deep-dive into specific markets showing full decision tree
"""

import sys
import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.models import MarketSnapshot, Signal, TradeSignal, OrderBook, OrderBookLevel
from bot.market_data import MarketDataClient
from bot.signals.estimator import ProbabilityEstimator, detect_market_type, WEIGHTS
from bot.signals.odds_api import OddsCache
from bot.signals.cross_market import PredictItCache
from bot.signals.crypto_api import CryptoCache
from bot.signals.sports_data import ESPNCache, GameContextAnalyzer
from bot.edge_log import extract_game_id, classify_edge_pattern, build_edge_snapshot_for_signal
from utils.logger import TradingLogger


def _make_clients(config):
    """Create all the data clients."""
    logger = TradingLogger(level="WARNING", console=True)
    market_data = MarketDataClient(config.api, logger, config.filters)
    odds_cache = OddsCache(api_key=config.odds_api_key, cache_ttl=300)
    predictit_cache = PredictItCache(cache_ttl=300)
    crypto_cache = CryptoCache(cache_ttl=300)
    espn_cache = ESPNCache(cache_ttl=300)
    game_context = GameContextAnalyzer(espn_cache)
    estimator = ProbabilityEstimator(
        config.signals, odds_cache, predictit_cache, crypto_cache,
        espn_cache, game_context,
    )
    return market_data, odds_cache, estimator, espn_cache, game_context


# ============================================================================
# TEST 1: Pseudo-backtest on resolved markets
# ============================================================================

def run_pseudo_backtest(config, limit: int = 50):
    """Pull recently resolved markets and test what our signals would have said.

    Since we can't reconstruct full historical order books, this tests:
    - Whether our odds_api signal would have detected edge
    - What the consensus was vs the resolution price
    - Simulated P&L if we had traded every 5%+ edge
    """
    print("\n" + "=" * 70)
    print("  PSEUDO-BACKTEST: Testing signals on resolved markets")
    print("=" * 70)

    market_data, odds_cache, estimator, espn_cache, game_context = _make_clients(config)

    if not odds_cache.enabled:
        print("\n  [ERROR] THE_ODDS_API_KEY not set — cannot run backtest")
        print("  Set the environment variable and retry.")
        return

    # Fetch active markets (we'll test against current prices as proxy)
    print("\n  Fetching markets...")
    markets = market_data.get_active_markets(limit=500)
    print(f"  Found {len(markets)} active markets")

    # Build snapshots and test signal pipeline
    results = []
    tested = 0
    errors = 0

    for market in markets:
        if tested >= limit:
            break

        snapshot = market_data.build_snapshot(market)
        if not snapshot:
            continue

        mtype = detect_market_type(snapshot)
        if mtype != "sports":
            continue

        # Get consensus from odds API
        prob_result = odds_cache.get_probability_for_slug(snapshot.slug)
        if not prob_result:
            continue

        consensus_prob, num_books = prob_result
        if num_books < 3:
            continue

        tested += 1
        edge = consensus_prob - snapshot.price

        # Run full signal pipeline
        try:
            trade_signal = estimator.detect_edge(
                snapshot,
                min_edge=0.03,  # Lower threshold for testing
                max_edge=0.40,
            )
        except Exception as e:
            errors += 1
            continue

        # Simulate: if we had traded at current price with the detected edge
        would_trade = trade_signal is not None and abs(edge) >= 0.05
        sim_side = "buy" if edge > 0 else "sell"

        # For the pseudo-backtest, assume markets near 0.9+ or 0.1- are "resolved"
        # and use the consensus as ground truth
        is_near_resolved = snapshot.price >= 0.85 or snapshot.price <= 0.15

        result = {
            "slug": snapshot.slug,
            "question": snapshot.question[:60],
            "poly_price": snapshot.price,
            "consensus": consensus_prob,
            "edge": edge,
            "abs_edge": abs(edge),
            "num_books": num_books,
            "would_trade": would_trade,
            "side": sim_side,
            "is_near_resolved": is_near_resolved,
            "trade_signal": trade_signal,
        }
        results.append(result)

    print(f"  Tested {tested} sports markets ({errors} errors)")

    # Analysis
    tradeable = [r for r in results if r["would_trade"]]
    non_tradeable = [r for r in results if not r["would_trade"]]

    print(f"\n  Markets with 5%+ edge: {len(tradeable)}")
    print(f"  Markets without edge: {len(non_tradeable)}")

    if not tradeable:
        print("\n  No tradeable edges found. This is expected if markets are efficient.")
        _print_all_edges(results)
        return

    # Simulate P&L: assume we buy at poly price, true value is consensus
    # Simulated P&L = (consensus - poly_price) * $25 position for buys
    sim_trades = []
    for r in tradeable:
        edge = r["edge"]
        size = 25.0
        # Simplified: P&L = edge * quantity (quantity = size / price)
        qty = size / r["poly_price"] if r["poly_price"] > 0 else 0
        if r["side"] == "buy":
            sim_pnl = edge * qty
        else:
            sim_pnl = -edge * qty
        # Subtract 4% fees (2% entry + 2% exit)
        sim_pnl -= size * 0.04
        sim_trades.append({**r, "sim_pnl": sim_pnl, "size": size})

    total_pnl = sum(t["sim_pnl"] for t in sim_trades)
    wins = [t for t in sim_trades if t["sim_pnl"] > 0]
    losses = [t for t in sim_trades if t["sim_pnl"] <= 0]

    print(f"\n  {'SIMULATED RESULTS':=^50}")
    print(f"  Total trades: {len(sim_trades)}")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"  Win rate: {len(wins)/len(sim_trades)*100:.0f}%")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    if wins:
        print(f"  Avg win: ${sum(t['sim_pnl'] for t in wins)/len(wins):+.2f}")
    if losses:
        print(f"  Avg loss: ${sum(t['sim_pnl'] for t in losses)/len(losses):+.2f}")

    # Edge bucket analysis
    print(f"\n  {'EDGE BUCKET ANALYSIS':=^50}")
    buckets = {"3-5%": [], "5-7%": [], "7-10%": [], "10%+": []}
    for t in sim_trades:
        pct = t["abs_edge"] * 100
        if pct < 5:
            buckets["3-5%"].append(t)
        elif pct < 7:
            buckets["5-7%"].append(t)
        elif pct < 10:
            buckets["7-10%"].append(t)
        else:
            buckets["10%+"].append(t)

    print(f"  {'Bucket':<10} {'Count':>6} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>10}")
    print(f"  {'-'*46}")
    for bucket, trades in buckets.items():
        if trades:
            w = len([t for t in trades if t["sim_pnl"] > 0])
            avg = sum(t["sim_pnl"] for t in trades) / len(trades)
            total = sum(t["sim_pnl"] for t in trades)
            print(f"  {bucket:<10} {len(trades):>6} {w/len(trades)*100:>5.0f}% {avg:>+10.2f} {total:>+10.2f}")

    # Best and worst trades
    sim_trades.sort(key=lambda t: t["sim_pnl"], reverse=True)
    print(f"\n  {'TOP 5 BEST TRADES':=^50}")
    for t in sim_trades[:5]:
        print(f"  ${t['sim_pnl']:+6.2f}  edge={t['edge']*100:+5.1f}%  {t['slug']}")

    print(f"\n  {'TOP 5 WORST TRADES':=^50}")
    for t in sim_trades[-5:]:
        print(f"  ${t['sim_pnl']:+6.2f}  edge={t['edge']*100:+5.1f}%  {t['slug']}")

    _print_all_edges(results)


def _print_all_edges(results):
    """Print all detected edges sorted by size."""
    sorted_results = sorted(results, key=lambda r: r["abs_edge"], reverse=True)
    print(f"\n  {'ALL EDGES DETECTED':=^60}")
    print(f"  {'Slug':<40} {'Poly':>6} {'Books':>6} {'Edge':>7} {'Trade?':>7}")
    print(f"  {'-'*70}")
    for r in sorted_results[:30]:
        trade_flag = "YES" if r["would_trade"] else "no"
        print(
            f"  {r['slug'][:40]:<40} "
            f"{r['poly_price']:>6.3f} "
            f"{r['consensus']:>6.3f} "
            f"{r['edge']*100:>+6.1f}% "
            f"{trade_flag:>7}"
        )


# ============================================================================
# TEST 2: Morning scan — today's top edges
# ============================================================================

def run_morning_scan(config):
    """Run the morning scan and display full signal breakdown for top opportunities."""
    print("\n" + "=" * 70)
    print("  MORNING EDGE SCAN: Today's top opportunities")
    print("=" * 70)

    market_data, odds_cache, estimator, espn_cache, game_context = _make_clients(config)

    if not odds_cache.enabled:
        print("\n  [ERROR] THE_ODDS_API_KEY not set")
        return

    print("\n  Fetching all active markets...")
    markets = market_data.get_active_markets(limit=500)
    print(f"  Found {len(markets)} active markets")

    opportunities = []

    for market in markets:
        snapshot = market_data.build_snapshot(market)
        if not snapshot:
            continue

        mtype = detect_market_type(snapshot)
        if mtype != "sports":
            continue

        # Get consensus
        consensus_data = odds_cache.get_consensus_odds(snapshot.slug)
        if not consensus_data:
            continue

        prob_result = odds_cache.get_probability_for_slug(snapshot.slug)
        if not prob_result:
            continue

        consensus_prob, num_books = prob_result
        if num_books < 3:
            continue

        edge = consensus_prob - snapshot.price

        # Run full signal pipeline
        weights = WEIGHTS.get(mtype, WEIGHTS["other"])
        signals = estimator.compute_signals(snapshot, mtype)

        # Game context
        parts = snapshot.slug.split("-")
        league = parts[1].upper() if len(parts) >= 2 else "?"
        game_time = market.get("gameStartTime", "")

        # Sharp books
        sharp_probs = consensus_data.get("sharp_probs", {})
        sharp_consensus = 0.0
        for team_name, prob in sharp_probs.items():
            if team_name.lower() == "draw":
                continue
            outcome_abbr = parts[-1] if len(parts) > 5 else (parts[2] if len(parts) > 2 else "")
            if odds_cache._team_matches(outcome_abbr, team_name):
                sharp_consensus = prob
                break

        opportunities.append({
            "slug": snapshot.slug,
            "question": snapshot.question,
            "poly_price": snapshot.price,
            "consensus": consensus_prob,
            "edge": edge,
            "abs_edge": abs(edge),
            "num_books": num_books,
            "league": league,
            "game_time": game_time,
            "is_live": snapshot.is_live,
            "signals": signals,
            "sharp_consensus": sharp_consensus,
            "books_used": consensus_data.get("books_used", ""),
            "spread": consensus_data.get("spread", 0),
            "order_book": snapshot.order_book,
        })

    opportunities.sort(key=lambda x: x["abs_edge"], reverse=True)

    if not opportunities:
        print("\n  No sports markets with consensus odds found.")
        return

    print(f"\n  Found {len(opportunities)} sports markets with odds data")
    print(f"  Markets with 5%+ edge: {len([o for o in opportunities if o['abs_edge'] >= 0.05])}")

    # Display top 10 with full signal breakdown
    print(f"\n  {'TOP 10 EDGE OPPORTUNITIES':=^70}")
    for i, opp in enumerate(opportunities[:10], 1):
        _print_opportunity(i, opp)

    return opportunities


def _print_opportunity(rank: int, opp: Dict):
    """Print a single opportunity with full signal breakdown."""
    edge_pct = opp["edge"] * 100
    direction = "BUY" if opp["edge"] > 0 else "SELL"

    game_time_str = ""
    if opp.get("game_time"):
        try:
            gt = datetime.fromisoformat(opp["game_time"].replace("Z", "+00:00"))
            game_time_str = gt.strftime("%I:%M %p ET")
        except (ValueError, TypeError):
            pass

    live_tag = " [LIVE]" if opp.get("is_live") else ""

    print(f"\n  {'─' * 66}")
    print(f"  #{rank}  {opp['league']}{live_tag}  {opp['question'][:55]}")
    print(f"  Slug: {opp['slug']}")
    if game_time_str:
        print(f"  Game time: {game_time_str}")

    print(f"\n  Polymarket:  {opp['poly_price']:.3f}")
    print(f"  Consensus:   {opp['consensus']:.3f}  ({opp['num_books']} books)")
    if opp.get("sharp_consensus"):
        print(f"  Sharp books: {opp['sharp_consensus']:.3f}")
    print(f"  Edge:        {edge_pct:+.1f}%  → {direction}")
    print(f"  Book spread: {opp.get('spread', 0)*100:.1f}%")

    ob = opp.get("order_book")
    if ob:
        print(f"  Order book:  bid_depth={ob.bid_depth:.0f}  ask_depth={ob.ask_depth:.0f}")

    print(f"\n  Signal breakdown:")
    for sig in opp.get("signals", []):
        conf_bar = "█" * int(sig.confidence * 10) + "░" * (10 - int(sig.confidence * 10))
        print(
            f"    {sig.name:<25} "
            f"value={sig.value:.3f}  "
            f"conf={sig.confidence:.2f} [{conf_bar}]  "
            f"{sig.direction}"
        )
        if sig.metadata:
            meta_keys = ["consensus_prob", "edge", "num_books", "sharp_consensus",
                         "context_modifier", "fatigue_home", "fatigue_away",
                         "home_advantage", "is_conference_tourney"]
            meta_items = {k: v for k, v in sig.metadata.items() if k in meta_keys and v}
            if meta_items:
                print(f"    {'':25} {meta_items}")

    # Would we trade this?
    would_trade = opp["abs_edge"] >= 0.05
    if would_trade:
        # Tiered sizing
        if opp["abs_edge"] >= 0.10:
            size = "$35"
        elif opp["abs_edge"] >= 0.07:
            size = "$25"
        else:
            size = "$15"
        print(f"\n  → WOULD TRADE: {direction} @ {opp['poly_price']:.3f}, size {size}")
    else:
        print(f"\n  → SKIP: edge {edge_pct:+.1f}% below 5% threshold")


# ============================================================================
# TEST 3: Signal analyzer — deep-dive on specific markets
# ============================================================================

def run_signal_analyzer(config, slugs: Optional[List[str]] = None):
    """Analyze signals on specific markets showing full decision tree."""
    print("\n" + "=" * 70)
    print("  SIGNAL ANALYZER: Full decision tree for specific markets")
    print("=" * 70)

    market_data, odds_cache, estimator, espn_cache, game_context = _make_clients(config)

    if not odds_cache.enabled:
        print("\n  [ERROR] THE_ODDS_API_KEY not set")
        return

    # If no slugs provided, pick markets from different leagues
    if not slugs:
        print("\n  No slugs specified — picking from active markets...")
        markets = market_data.get_active_markets(limit=500)

        # Pick a mix: try NBA, NCAA, NHL
        target_leagues = {"nba": None, "cbb": None, "nhl": None}
        other_picks = []

        for market in markets:
            snapshot = market_data.build_snapshot(market)
            if not snapshot:
                continue
            mtype = detect_market_type(snapshot)
            if mtype != "sports":
                continue

            parts = snapshot.slug.split("-")
            league = parts[1] if len(parts) >= 2 else ""

            if league in target_leagues and target_leagues[league] is None:
                target_leagues[league] = market
            elif len(other_picks) < 2:
                other_picks.append(market)

        selected_markets = [m for m in target_leagues.values() if m is not None]
        remaining = 5 - len(selected_markets)
        selected_markets.extend(other_picks[:remaining])

        if not selected_markets:
            print("  No sports markets found!")
            return

        print(f"  Selected {len(selected_markets)} markets for analysis")
        slugs_to_analyze = [m.get("slug", "") for m in selected_markets]
    else:
        slugs_to_analyze = slugs
        selected_markets = None

    for slug in slugs_to_analyze:
        if not slug:
            continue

        # Build snapshot
        if selected_markets:
            market = next((m for m in selected_markets if m.get("slug") == slug), None)
        else:
            # Need to fetch the market by slug
            market = None
            all_markets = market_data.get_active_markets(limit=500)
            market = next((m for m in all_markets if m.get("slug") == slug), None)

        if not market:
            print(f"\n  [SKIP] Market not found: {slug}")
            continue

        snapshot = market_data.build_snapshot(market)
        if not snapshot:
            print(f"\n  [SKIP] Could not build snapshot: {slug}")
            continue

        _analyze_single_market(snapshot, market, estimator, odds_cache, espn_cache, game_context, config)


def _analyze_single_market(snapshot, market, estimator, odds_cache, espn_cache, game_context, config):
    """Full decision tree analysis for a single market."""
    slug = snapshot.slug
    parts = slug.split("-")
    league = parts[1].upper() if len(parts) >= 2 else "?"

    print(f"\n  {'═' * 66}")
    print(f"  MARKET: {snapshot.question[:60]}")
    print(f"  Slug: {slug}")
    print(f"  League: {league}  |  Live: {snapshot.is_live}  |  Price: {snapshot.price:.3f}")
    print(f"  {'─' * 66}")

    # Step 1: Market type detection
    mtype = detect_market_type(snapshot)
    print(f"\n  1. MARKET TYPE DETECTION")
    print(f"     Detected: {mtype}")
    print(f"     Weights: {WEIGHTS.get(mtype, {})}")

    # Step 2: External validation gate
    print(f"\n  2. EXTERNAL VALIDATION GATE")
    consensus_data = odds_cache.get_consensus_odds(slug)
    if consensus_data:
        print(f"     Home: {consensus_data.get('home_team', '?')}")
        print(f"     Away: {consensus_data.get('away_team', '?')}")
        print(f"     Books: {consensus_data.get('num_books', 0)}")
        print(f"     Book spread: {consensus_data.get('spread', 0)*100:.1f}%")
        print(f"     Books used: {consensus_data.get('books_used', 'N/A')}")

        prob_result = odds_cache.get_probability_for_slug(slug)
        if prob_result:
            consensus, num_books = prob_result
            print(f"     Consensus prob: {consensus:.3f}")
            print(f"     Polymarket:     {snapshot.price:.3f}")
            print(f"     Raw edge:       {(consensus - snapshot.price)*100:+.1f}%")
            print(f"     → GATE: PASSED ({num_books} books ≥ 3)")

            # Sharp books
            sharp_probs = consensus_data.get("sharp_probs", {})
            if sharp_probs:
                print(f"     Sharp book probs: {sharp_probs}")
        else:
            print(f"     → GATE: FAILED (no matching probability)")
    else:
        print(f"     → GATE: FAILED (no consensus data found)")

    # Step 3: Signal computation
    print(f"\n  3. SIGNAL COMPUTATION")
    signals = estimator.compute_signals(snapshot, mtype)
    for sig in signals:
        emoji = "✓" if sig.confidence > 0.3 else "○"
        print(f"     {emoji} {sig.name:<25} val={sig.value:.3f}  conf={sig.confidence:.2f}  dir={sig.direction}")
        if sig.metadata:
            # Print interesting metadata
            for k, v in sig.metadata.items():
                if k not in ("reason",) and v:
                    print(f"       └─ {k}: {v}")

    # Step 4: Probability estimation
    print(f"\n  4. PROBABILITY ESTIMATION")
    weights = WEIGHTS.get(mtype, WEIGHTS["other"])
    est_prob, confidence = estimator.estimate_probability(signals, weights)
    print(f"     Estimated prob: {est_prob:.3f}")
    print(f"     Confidence:     {confidence:.3f}")
    print(f"     Market price:   {snapshot.price:.3f}")
    print(f"     Implied edge:   {(est_prob - snapshot.price)*100:+.1f}%")

    # Step 5: Edge detection with caps
    print(f"\n  5. EDGE DETECTION")
    trade_signal = estimator.detect_edge(
        snapshot,
        min_edge=config.trading.min_edge_threshold,
        max_edge=config.trading.max_edge_threshold,
    )

    if trade_signal:
        print(f"     Final edge:  {trade_signal.edge*100:+.1f}%")
        print(f"     Side:        {trade_signal.side}")
        print(f"     → DECISION: TRADE")

        # Pattern classification
        sig_snapshot = build_edge_snapshot_for_signal(trade_signal, odds_cache, slug)
        pattern = classify_edge_pattern(sig_snapshot)
        print(f"     Pattern:     {pattern}")

        # Sizing
        abs_edge = abs(trade_signal.edge)
        if abs_edge >= 0.10:
            tier = "$35 (10%+ edge)"
        elif abs_edge >= 0.07:
            tier = "$25 (7-10% edge)"
        else:
            tier = "$15 (5-7% edge)"
        print(f"     Tier size:   {tier}")

        # Correlation check
        game_id = extract_game_id(slug)
        print(f"     Game ID:     {game_id}")
    else:
        print(f"     → DECISION: NO TRADE")
        print(f"     Reason: Edge below {config.trading.min_edge_threshold*100:.0f}% threshold or gate failed")

    # Step 6: Game context (ESPN)
    if len(parts) >= 4:
        sport_map = {"nba": "basketball_nba", "cbb": "basketball_ncaab"}
        sport_key = sport_map.get(parts[1], "")
        if sport_key:
            print(f"\n  6. GAME CONTEXT (ESPN)")
            ctx = game_context.analyze(sport_key, parts[2], parts[3])
            if ctx.get("home_advantage"):
                print(f"     Home advantage: {ctx['home_advantage']*100:+.1f}%")
            if ctx.get("home_record"):
                print(f"     Home record: {ctx['home_record']}")
            if ctx.get("away_record"):
                print(f"     Away record: {ctx['away_record']}")
            print(f"     Fatigue home: {ctx.get('fatigue_home', False)}")
            print(f"     Fatigue away: {ctx.get('fatigue_away', False)}")
            print(f"     Conference tourney: {ctx.get('is_conference_tourney', False)}")
            print(f"     Neutral site: {ctx.get('neutral_site', False)}")
            print(f"     Context modifier: {ctx.get('context_modifier', 0)*100:+.1f}%")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    config = load_config()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("backtest", "all"):
        run_pseudo_backtest(config, limit=50)

    if mode in ("scan", "all"):
        run_morning_scan(config)

    if mode in ("analyze", "all"):
        slugs = sys.argv[2:] if len(sys.argv) > 2 else None
        run_signal_analyzer(config, slugs)

    print(f"\n{'=' * 70}")
    print(f"  Tests complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
