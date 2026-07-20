"""Main orchestrator loop with dual-speed scanning."""

import os
import sys
import time
import signal
import copy
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from dateutil import parser as dateutil_parser

from utils.config import load_config, BotConfig
from utils.logger import TradingLogger
from utils.models import MarketSnapshot
from bot.signals.estimator import ProbabilityEstimator
from bot.strategies.sizing import PositionSizer
from bot.strategies.risk import RiskManager, LET_IT_RIDE_THRESHOLD
from bot.market_data import MarketDataClient
from bot.execution import ExecutionEngine
from bot.portfolio import Portfolio
from bot.alerts import SlackAlerter
from bot.health import HealthMonitor
from bot.strategies.edge import compute_edge_breakdown


def compute_exit_proximity(position, current_price: float, estimated_prob: float, config) -> dict:
    """Compute signed distance to every exit threshold at the moment of close.

    Returns a dict with 5 fields — one per exit condition defined in RiskManager.
    All distances share the same sign convention:
        negative = condition has NOT yet been met (threshold still ahead)
        positive = condition HAS been met (threshold already passed)

    This makes queries like "was take_profit within 2% when stop_loss fired?"
    expressible as a single SQL inequality on exit_proximity_json fields.

    Args:
        position    : Position object at close time (peak_price must be set).
        current_price: The price at which the position is closing.
        estimated_prob: The probability estimate used for take-profit edge calc.
        config      : TradingConfig instance (supplies threshold constants).

    Returns dict keys:
        stop_loss_distance_pct       – loss_pct − stop_loss_threshold
        take_profit_edge_distance    – take_profit_threshold − edge_remaining
        aggressive_exit_distance_pct – gain_pct − aggressive_exit_pct
        trailing_stop_distance_pct   – drop_from_peak − trailing_stop_pct
                                       (None / JSON null when trailing stop unarmed)
        let_it_ride_distance_pct     – favorable_price − LET_IT_RIDE_THRESHOLD
    """
    entry_price = position.entry_price

    # Degenerate guard: no valid entry price → return zeros
    if not entry_price or entry_price <= 0:
        return {
            "stop_loss_distance_pct": 0.0,
            "take_profit_edge_distance": 0.0,
            "aggressive_exit_distance_pct": 0.0,
            "trailing_stop_distance_pct": None,
            "let_it_ride_distance_pct": 0.0,
        }

    if position.side == "buy":
        pnl_per_unit = current_price - entry_price
    else:
        pnl_per_unit = entry_price - current_price

    gain_pct = pnl_per_unit / entry_price
    loss_pct = -gain_pct  # positive when losing

    # ── 1. stop_loss ──────────────────────────────────────────────────────────
    # Fires when loss_pct >= stop_loss_threshold (0.25).
    # Distance = loss_pct − threshold  →  negative = not fired, positive = fired.
    stop_loss_distance_pct = loss_pct - config.stop_loss_threshold

    # ── 2. take_profit ────────────────────────────────────────────────────────
    # Fires when edge_remaining <= take_profit_threshold (0.05) AND profit ≥ $5.
    # Edge distance = threshold − edge_remaining  → negative = not converged yet,
    # positive = edge has converged past the threshold.
    edge_remaining = abs(estimated_prob - current_price)
    take_profit_edge_distance = config.take_profit_threshold - edge_remaining

    # ── 3. aggressive_exit ────────────────────────────────────────────────────
    # Fires when gain_pct >= aggressive_exit_pct (0.30).
    # Distance = gain_pct − threshold  →  negative = not there yet.
    aggressive_exit_distance_pct = gain_pct - config.aggressive_exit_pct

    # ── 4. trailing_stop ─────────────────────────────────────────────────────
    # Two-stage: first peak_gain must reach activation (0.15), then drop_from_peak
    # must reach trailing_stop_pct (0.10).
    # Returns None when the trailing stop was never armed (peak never activated).
    trailing_stop_distance_pct = None  # default: unarmed / not applicable
    peak_price = getattr(position, "peak_price", 0)
    if peak_price and peak_price > 0 and entry_price > 0:
        if position.side == "buy":
            peak_gain = (peak_price - entry_price) / entry_price
            drop_from_peak = (peak_price - current_price) / entry_price
        else:
            peak_gain = (entry_price - peak_price) / entry_price
            drop_from_peak = (current_price - peak_price) / entry_price

        if peak_gain >= config.trailing_stop_activation_pct:
            # Trailing stop is armed — compute distance to trigger.
            trailing_stop_distance_pct = drop_from_peak - config.trailing_stop_pct

    # ── 5. let_it_ride ────────────────────────────────────────────────────────
    # Fires when favorable_price >= LET_IT_RIDE_THRESHOLD (0.70).
    # For buys favorable_price = current_price; for sells it's 1 − current_price.
    if position.side == "buy":
        favorable_price = current_price
    else:
        favorable_price = 1.0 - current_price
    let_it_ride_distance_pct = favorable_price - LET_IT_RIDE_THRESHOLD

    return {
        "stop_loss_distance_pct": round(stop_loss_distance_pct, 6),
        "take_profit_edge_distance": round(take_profit_edge_distance, 6),
        "aggressive_exit_distance_pct": round(aggressive_exit_distance_pct, 6),
        "trailing_stop_distance_pct": (
            round(trailing_stop_distance_pct, 6)
            if trailing_stop_distance_pct is not None else None
        ),
        "let_it_ride_distance_pct": round(let_it_ride_distance_pct, 6),
    }


class TradingBot:
    """Main trading bot orchestrator with dual-speed scanning."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = TradingLogger(
            level=config.logging.level,
            log_file=config.logging.file,
            console=config.logging.console,
        )
        self.market_data = MarketDataClient(config.api, self.logger, config.filters)
        self.sizer = PositionSizer(config.trading)
        self.risk = RiskManager(config.trading)
        self.executor = ExecutionEngine(config, self.logger)
        self.alerter = SlackAlerter(
            webhook_url=config.alerts.slack_webhook_url,
            enabled=config.alerts.enabled,
        )

        # Initialize external data caches
        from bot.signals.odds_api import OddsCache
        from bot.signals.cross_market import PredictItCache
        from bot.signals.crypto_api import CryptoCache
        from bot.signals.sports_data import ESPNCache, GameContextAnalyzer

        self.odds_cache = OddsCache(api_key=config.odds_api_key, cache_ttl=300)
        self.predictit_cache = PredictItCache(cache_ttl=300)
        self.crypto_cache = CryptoCache(cache_ttl=300)
        self.espn_cache = ESPNCache(cache_ttl=300)
        self.game_context = GameContextAnalyzer(self.espn_cache)

        # Live odds tracker for NCAA in-game edges
        from bot.signals.live_odds import LiveOddsTracker
        self.live_odds_tracker = LiveOddsTracker(cache_ttl=30)

        # Game schedule awareness
        from bot.game_schedule import GameSchedule
        self.game_schedule = GameSchedule(cache_ttl=120)

        # On-chain enrichment (free CLOB/Gamma whale + smart-money data).
        # Feeds the supplementary onchain_flow signal; disabled via config.
        self.onchain_client = None
        if getattr(config, "onchain", None) and config.onchain.enabled:
            from bot.signals.onchain import OnChainEnrichmentClient
            self.onchain_client = OnChainEnrichmentClient(
                clob_url=config.api.clob_url,
                gamma_url=config.api.gamma_url,
                logger=self.logger,
                max_rps=config.onchain.max_requests_per_second,
                cache_ttl=config.onchain.cache_ttl_seconds,
            )

        # Live in-game win probabilities (ESPN model) — the live-edge engine
        from bot.signals.live_win_prob import LiveWinProbCache
        self.live_cache = LiveWinProbCache()

        # Spread/total book lines (FanDuel + Pinnacle) for derivative markets
        from bot.signals.book_scrapers import MultiBookAggregator
        self.line_aggregator = MultiBookAggregator(cache_ttl=300)

        # Kalshi cross-exchange prices (same-instrument validation)
        from bot.signals.kalshi import KalshiCache
        self.kalshi_cache = KalshiCache(cache_ttl=120)

        # Liveness + API health tracking (heartbeat file read by supervisor)
        self._data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.health = HealthMonitor(
            data_dir=self._data_dir,
            logger=self.logger,
            alerter=self.alerter,
        )

        if not self.odds_cache.enabled:
            self.logger.warning("odds_api_disabled", {
                "message": "THE_ODDS_API_KEY not set — sports markets will not trade"
            })

        self.logger.info("slack_alerter_status", {
            "enabled": self.alerter.enabled,
            "webhook_set": bool(self.alerter.webhook_url),
            "webhook_len": len(self.alerter.webhook_url) if self.alerter.webhook_url else 0,
        })

        # Estimator with all caches
        self.estimator = ProbabilityEstimator(
            config.signals, self.odds_cache, self.predictit_cache, self.crypto_cache,
            self.espn_cache, self.game_context, onchain_client=self.onchain_client,
            live_cache=self.live_cache,
        )

        self.portfolio = Portfolio(
            exchange_client=self.executor._client,
            logger=self.logger,
            alerter=self.alerter,
            alert_config=config.alerts,
            paper_mode=config.trading.paper_trading,
            initial_bankroll=config.backtest.initial_bankroll_usd,
            restore_state=True,  # survive restarts: entry_time/prob/telemetry
        )
        self.logger.info("portfolio_initialized", {
            "paper_mode": config.trading.paper_trading,
            "exchange_connected": self.executor._client is not None,
            "bankroll": round(self.portfolio.bankroll, 2),
            "equity": round(self.portfolio.get_equity(), 2),
        })
        self.running = True

        # Cooldown tracker: slug → earliest time we can reopen
        self._slug_cooldowns: Dict[str, float] = {}
        self._cooldown_seconds = 600  # 10 minutes

        # Live discipline (Jul 13 postmortem): per-game entry/stop counters.
        # game_id embeds the date, so counters never need a daily reset.
        # Stop counts are restored from today's trades so a restart can't
        # forget that a game already stopped us out twice.
        self._game_entry_counts: Dict[str, int] = {}
        self._game_stop_counts: Dict[str, int] = {}
        self._pending_live_edges: Dict[str, tuple] = {}  # slug -> (side, first_seen_ts)
        # decision_log dedup: (slug,decision,reason) -> last-logged ts. A
        # locked-out or 1-book market re-evaluates every 3s and would flood
        # the table with identical rows (231 of 265 today), making CLV/edge
        # analysis impossible. Log a repeat only after this cooldown.
        self._decision_dedup: Dict[tuple, float] = {}
        self._decision_dedup_seconds = 900  # 15 min
        self._restore_game_discipline_counts()

        # Slugs we've already auto-settled — persisted to file so it survives restarts
        self._settled_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "settled_slugs.txt")
        self._settled_slugs: set = self._load_settled_slugs()

        # Live-game estimator with aggressive weights
        live_signal_config = copy.deepcopy(config.signals)
        live_signal_config.odds_value_weight = 0.40
        live_signal_config.order_book_imbalance_weight = 0.25
        live_signal_config.line_movement_weight = 0.15
        live_signal_config.liquidity_imbalance_weight = 0.20
        self.live_estimator = ProbabilityEstimator(
            live_signal_config, self.odds_cache, self.predictit_cache, self.crypto_cache,
            self.espn_cache, self.game_context, onchain_client=self.onchain_client,
            live_cache=self.live_cache, line_aggregator=self.line_aggregator,
            kalshi_cache=self.kalshi_cache,
        )

        # Live-game trading overrides — use same edge threshold as config
        self.live_min_edge = config.trading.min_edge_threshold
        self.live_take_profit = 0.03
        self.live_scan_interval = 3
        self.full_scan_interval = 60

        # Cached market list from last full scan
        self._cached_markets: List[Dict] = []
        self._last_full_scan = 0.0

    def _check_supervisor_flags(self) -> bool:
        """Check if supervisor has halted or paused trading. Returns True if OK to trade."""
        kill_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kill_switch")
        pause_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pause_until")

        if os.path.exists(kill_path):
            self.logger.warning("trading_halted_by_kill_switch", {})
            return False

        if os.path.exists(pause_path):
            try:
                with open(pause_path, "r") as f:
                    resume_at = dateutil_parser.isoparse(f.read().strip())
                if datetime.now(timezone.utc) < resume_at:
                    self.logger.info("trading_paused", {"resume_at": resume_at.isoformat()})
                    return False
                else:
                    os.remove(pause_path)
            except Exception:
                os.remove(pause_path)
        return True

    def _is_live_market(self, market: Dict) -> bool:
        """Check if a market's game is currently in progress."""
        game_start_str = market.get("gameStartTime")
        if not game_start_str:
            return False
        game_start = self.market_data._parse_datetime(game_start_str)
        if game_start is None:
            return False
        return game_start <= datetime.now(timezone.utc)

    def _split_markets(self, markets: List[Dict]):
        """Split markets into live and pre-game lists.

        The live list is restricted to GAME markets in registered leagues —
        markets our signals can actually price. Measured live during UFC 329:
        476 markets carried a started gameStartTime, ~140 of them UFC props
        (method-of-victory, round props) that no signal can value; scanning
        them stretched the intended 3-second live pass to 5+ minutes of API
        calls for zero tradeable output.
        """
        from bot.signals.live_win_prob import slug_game_teams

        live = []
        pregame = []
        for m in markets:
            if self._is_live_market(m):
                if slug_game_teams(m.get("slug", "")) is not None:
                    live.append(m)
                # non-game live markets (props etc.): unpriceable — drop from
                # the fast path; the 60s full scan still sees them.
            else:
                pregame.append(m)
        return live, pregame

    def _detect_edge(self, snapshot: MarketSnapshot):
        """Detect edge for a single market. Uses per-league minimum edge."""
        from bot.strategies.trade_filter import get_league_min_edge

        if snapshot.is_live:
            estimator = self.live_estimator
        else:
            estimator = self.estimator

        # Per-league minimum edge (NHL=4%, NCAA=5%, NBA=7%)
        min_edge = get_league_min_edge(snapshot.slug)

        trade_signal = estimator.detect_edge(
            snapshot,
            min_edge=min_edge,
            max_edge=self.config.trading.max_edge_threshold,
        )
        if trade_signal is None:
            return None
        return (trade_signal, snapshot)

    def process_markets(self, snapshots: list):
        """Sniper strategy: rank ALL opportunities, take only the best 1-2.

        Before any trade, every opportunity must pass the full validation
        checklist in trade_filter.validate_trade().
        """
        import time as _time
        from bot.edge_log import extract_game_id, get_open_game_ids
        from bot.strategies.trade_filter import (
            validate_trade, validate_live_trade, get_live_signal,
            rank_opportunities, get_league_from_slug,
        )

        # Collect all edges
        opportunities = []
        open_positions = self.portfolio.get_open_positions()
        open_slugs = {p.slug for p in open_positions}
        open_games = get_open_game_ids(open_positions)

        for snapshot in snapshots:
            if not snapshot:
                continue
            slug = snapshot.slug

            if slug in open_slugs:
                continue
            if slug in self._settled_slugs:
                continue

            cooldown_until = self._slug_cooldowns.get(slug, 0)
            if _time.time() < cooldown_until:
                continue

            result = self._detect_edge(snapshot)
            if result:
                opportunities.append(result)

        if not opportunities:
            return

        # Rank with short bias (shorts score slightly higher)
        opportunities = rank_opportunities(opportunities)

        slots = self.config.trading.max_open_positions - len(open_slugs)
        if slots <= 0:
            return

        games_opening = set()

        tcfg = self.config.trading

        for trade_signal, snapshot in opportunities:
            if not self.risk.can_open_position(self.portfolio.get_open_positions()):
                break

            game_id = extract_game_id(trade_signal.slug)

            # PER-GAME DISCIPLINE (applies to both validation paths):
            # after max_stops_per_game stop-losses the game is locked out,
            # and no game takes more than max_entries_per_game entries.
            # Jul 13: 6 of 12 trades were stop-outs from re-entering the
            # same two games — these caps would have saved ~$30.
            if self._game_stop_counts.get(game_id, 0) >= tcfg.max_stops_per_game:
                self._log_decision(trade_signal, snapshot, "rejected",
                                   f"game_lockout_{self._game_stop_counts[game_id]}_stops")
                continue
            if self._game_entry_counts.get(game_id, 0) >= tcfg.max_entries_per_game:
                self._log_decision(trade_signal, snapshot, "rejected",
                                   f"max_entries_per_game_{tcfg.max_entries_per_game}_reached")
                continue

            # Get number of books for this market. Derivative markets carry
            # their own book count (books quoting that exact line) on the
            # derivative_line signal — the moneyline consensus lookup would
            # return 0 for their slugs and wrongly block them.
            deriv_sig = next((sg for sg in trade_signal.signals
                              if sg.name == "derivative_line" and sg.confidence > 0),
                             None)
            kalshi_sig = next((sg for sg in trade_signal.signals
                               if sg.name == "kalshi_value" and sg.confidence > 0),
                              None)
            if deriv_sig is not None:
                num_books = int(deriv_sig.metadata.get("num_books", 0))
            elif kalshi_sig is not None:
                # A matched same-instrument Kalshi market IS the external
                # validation — the >=2 books rule is satisfied by a regulated
                # exchange quoting the identical contract.
                num_books = 2
            else:
                consensus = self.odds_cache.get_consensus_odds(trade_signal.slug)
                num_books = consensus.get("num_books", 0) if consensus else 0

            # Get game time remaining for last-5-minutes block
            league = get_league_from_slug(trade_signal.slug)
            parts = trade_signal.slug.split("-")
            game_time_remaining = None
            if len(parts) >= 4 and hasattr(self, 'game_schedule'):
                game_time_remaining = self.game_schedule.get_game_time_remaining(
                    league, parts[2], parts[3]
                )

            # Cost-aware edge: what the edge is worth at the executable price
            # (ask for buys, bid for sells) after the taker fee. Attached to
            # the signal so the trade filter and Kelly sizing use net numbers.
            breakdown = compute_edge_breakdown(
                trade_signal.estimated_prob, snapshot, trade_signal.side,
                fee_coefficient=tcfg.taker_fee_coefficient,
            )
            trade_signal.net_edge = breakdown.net_edge
            trade_signal.exec_price = breakdown.exec_price
            trade_signal.spread = breakdown.spread
            trade_signal.fee_rate = breakdown.fee_rate

            # get_open_game_ids returns a dict (game_id -> slugs); union
            # with the set of games opened this cycle needs its keys.
            # (dict | set raised TypeError and killed every cycle that
            # found an opportunity — 1,433 scan_cycle_errors on Jul 12.)
            merged_open_games = set(open_games) | games_opening

            # ROUTE: mid-game opportunities validated by the live model get
            # the live checklist (books pull lines at tip-off, so the pregame
            # >=2-books rule rejected every live trade); everything else uses
            # the pregame checklist unchanged.
            live_signal = get_live_signal(trade_signal)
            if live_signal is not None and getattr(snapshot, "is_live", False):
                validation_path = "live"
                # Total game seconds from the league registry (elapsed-time
                # window); None for clockless sports, which gate on period.
                from bot.leagues import LEAGUES as _LEAGUES
                clock_info = (_LEAGUES.get(league) or {}).get("clock")
                game_total_seconds = (
                    clock_info["periods"] * clock_info["minutes"] * 60
                    if clock_info else None
                )
                game_period = None
                if clock_info is None and len(parts) >= 4:
                    game_period = self.game_schedule.get_game_period(
                        league, parts[2], parts[3]
                    )
                rejection = validate_live_trade(
                    signal=trade_signal,
                    snapshot=snapshot,
                    live_signal=live_signal,
                    open_game_ids=merged_open_games,
                    game_id=game_id,
                    daily_trades=self.risk.daily_trade_count,
                    max_daily_trades=tcfg.max_daily_trades,
                    game_time_remaining=game_time_remaining,
                    game_period=game_period,
                    game_total_seconds=game_total_seconds,
                    min_price=tcfg.min_price,
                    max_price=tcfg.max_price,
                    min_liquidity_usd=tcfg.min_book_liquidity_usd,
                    max_spread=tcfg.max_spread,
                    min_net_edge=tcfg.min_net_edge,
                    max_model_age_seconds=tcfg.max_live_model_age_seconds,
                    max_divergence=tcfg.max_live_divergence,
                    min_elapsed_seconds=tcfg.live_min_elapsed_seconds,
                    clockless_max_period=tcfg.live_clockless_max_period,
                )
            else:
                validation_path = "pregame"
                rejection = validate_trade(
                    signal=trade_signal,
                    snapshot=snapshot,
                    num_books=num_books,
                    open_game_ids=merged_open_games,
                    game_id=game_id,
                    daily_trades=self.risk.daily_trade_count,
                    max_daily_trades=tcfg.max_daily_trades,
                    game_time_remaining=game_time_remaining,
                    min_price=tcfg.min_price,
                    max_price=tcfg.max_price,
                    min_liquidity_usd=tcfg.min_book_liquidity_usd,
                    max_spread=tcfg.max_spread,
                    min_net_edge=tcfg.min_net_edge,
                )

            if rejection:
                self.logger.info("trade_rejected", {
                    "slug": trade_signal.slug,
                    "edge": round(trade_signal.edge * 100, 1),
                    "net_edge": round(trade_signal.net_edge * 100, 1),
                    "side": trade_signal.side,
                    "reason": rejection,
                })
                self._log_decision(trade_signal, snapshot, "rejected",
                                   f"[{validation_path}] {rejection}")
                continue

            # TWO-SCAN CONFIRMATION (live only): a live edge must persist
            # across scans before we act — a single model spike (one play,
            # one stale book pull) is not an edge. First sighting arms the
            # slug; a second sighting between confirm_min and confirm_max
            # seconds later trades; older sightings re-arm.
            if validation_path == "live":
                import time as _time
                now_ts = _time.time()
                prev = self._pending_live_edges.get(trade_signal.slug)
                window_ok = (prev is not None and prev[0] == trade_signal.side
                             and tcfg.live_confirm_min_seconds
                             <= now_ts - prev[1]
                             <= tcfg.live_confirm_max_seconds)
                if not window_ok:
                    self._pending_live_edges[trade_signal.slug] = (trade_signal.side, now_ts)
                    self._log_decision(trade_signal, snapshot, "rejected",
                                       "[live] pending_confirmation_second_scan")
                    continue
                del self._pending_live_edges[trade_signal.slug]

            exposure = self.portfolio.get_total_exposure()
            size = self.sizer.size_position(trade_signal, self.portfolio.bankroll, exposure)
            # Live positions run wider stops (0.35 vs 0.25), so they carry
            # smaller size: similar $ risk, fewer noise whipsaws.
            if validation_path == "live":
                size = min(size, tcfg.live_max_position_size_usd)
            if size <= 0:
                self._log_decision(trade_signal, snapshot, "skipped_sizing",
                                   "size_below_minimum_or_no_exposure_room")
                continue

            trade_signal.position_size_usd = size
            trade_signal._question = snapshot.question
            trade_signal._is_live = snapshot.is_live

            trade = self.executor.execute_trade(trade_signal)
            if trade:
                self.portfolio.open_position(trade_signal, trade)
                self.risk.record_trade_opened()
                games_opening.add(game_id)
                self._game_entry_counts[game_id] = self._game_entry_counts.get(game_id, 0) + 1
                self._log_edge_entry(trade_signal, snapshot)
                self._log_decision(trade_signal, snapshot, "executed",
                                   f"[{validation_path}] all_checks_passed")

                self.logger.info("sniper_trade_executed", {
                    "slug": trade_signal.slug,
                    "side": trade_signal.side,
                    "edge": round(trade_signal.edge * 100, 1),
                    "net_edge": round(trade_signal.net_edge * 100, 1),
                    "size": size,
                    "league": league,
                    "books": num_books,
                    "daily_trade": self.risk.daily_trade_count,
                })
            else:
                self._log_decision(trade_signal, snapshot, "execution_failed",
                                   "order_not_filled_or_error")

    def _log_decision(self, trade_signal, snapshot, decision: str, reason: str):
        """Write one decision audit row. Never allowed to break trading.

        Deduplicated: an identical (slug, decision, reason) is written at most
        once per _decision_dedup_seconds, so repeated per-scan rejections
        (game_lockout, only_1_books) don't drown the executed/novel decisions
        the paper-validation analysis actually needs. 'executed' is NEVER
        deduped — every real trade is always recorded.
        """
        try:
            import time as _time
            if decision != "executed":
                key = (trade_signal.slug, decision, reason)
                last = self._decision_dedup.get(key, 0)
                now_ts = _time.time()
                if now_ts - last < self._decision_dedup_seconds:
                    return
                self._decision_dedup[key] = now_ts
            import json as _json
            from bot.trade_db import insert_decision
            from bot.signals.estimator import detect_market_type
            insert_decision(
                slug=trade_signal.slug,
                decision=decision,
                reason=reason,
                market_type=detect_market_type(snapshot),
                side=trade_signal.side,
                polymarket_price=trade_signal.market_price,
                exec_price=trade_signal.exec_price,
                estimated_prob=trade_signal.estimated_prob,
                gross_edge=trade_signal.edge,
                net_edge=trade_signal.net_edge,
                spread=trade_signal.spread,
                fee_rate=trade_signal.fee_rate,
                position_size_usd=trade_signal.position_size_usd,
                metadata_json=_json.dumps({
                    "is_live": bool(getattr(snapshot, "is_live", False)),
                    "paper_mode": self.config.trading.paper_trading,
                }),
            )
        except Exception:
            pass  # audit logging must never block trading

    def _refresh_position_estimates(self, snapshots: list):
        """Re-estimate probability for open positions as new information arrives.

        Position.estimated_prob drives the take-profit ("edge converged") exit.
        Without refreshing it, exits are judged against the entry-time belief,
        which goes stale the moment the game starts or the line moves. We only
        overwrite when the estimator has live external validation (returns
        non-None); otherwise the last good estimate stands.
        """
        open_positions = self.portfolio.get_open_positions()
        if not open_positions:
            return
        by_slug = {s.slug: s for s in snapshots if s}
        for pos in open_positions:
            snap = by_slug.get(pos.slug)
            if snap is None:
                continue
            estimator = self.live_estimator if snap.is_live else self.estimator
            try:
                new_prob = estimator.estimate_for_snapshot(snap)
            except Exception:
                continue  # estimation failure keeps the previous belief
            if new_prob is None:
                continue
            if abs(new_prob - pos.estimated_prob) >= 0.01:
                self.logger.info("position_estimate_refreshed", {
                    "slug": pos.slug,
                    "old_prob": round(pos.estimated_prob, 3),
                    "new_prob": round(new_prob, 3),
                })
            pos.estimated_prob = new_prob

    def _enforce_daily_loss_pause(self):
        """When the daily loss limit is breached, pause durably until next UTC day.

        The in-memory RiskManager check already blocks new entries, but it
        resets on restart. Writing data/pause_until makes the halt survive
        crashes/redeploys — the loop's supervisor-flag check enforces it.
        """
        if not self.risk.is_daily_limit_breached():
            return
        pause_path = os.path.join(self._data_dir, "pause_until")
        if os.path.exists(pause_path):
            return  # already paused
        now = datetime.now(timezone.utc)
        resume_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        try:
            with open(pause_path, "w") as f:
                f.write(resume_at.isoformat())
        except OSError:
            return
        self.logger.error("daily_loss_limit_breached", {
            "daily_pnl": round(self.risk.daily_pnl, 2),
            "limit": self.config.trading.daily_loss_limit_usd,
            "paused_until": resume_at.isoformat(),
        })
        if self.alerter:
            try:
                self.alerter._post("#ff0000",
                    f":octagonal_sign: *Daily loss limit hit* "
                    f"(${self.risk.daily_pnl:.2f} ≤ -${self.config.trading.daily_loss_limit_usd:.2f})\n"
                    f"Trading paused until `{resume_at.isoformat()}`")
            except Exception:
                pass

    def _restore_game_discipline_counts(self):
        """Rebuild today's per-game stop/entry counts from the trades table."""
        try:
            from bot.trade_db import get_trades_since
            from bot.edge_log import extract_game_id
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0)
            for t in get_trades_since(today):
                gid = extract_game_id(t.get("slug", ""))
                if not gid:
                    continue
                self._game_entry_counts[gid] = self._game_entry_counts.get(gid, 0) + 1
                if t.get("close_reason") == "stop_loss":
                    self._game_stop_counts[gid] = self._game_stop_counts.get(gid, 0) + 1
        except Exception as e:
            self.logger.warning("discipline_count_restore_failed", {"error": str(e)})

    def _load_settled_slugs(self) -> set:
        try:
            with open(self._settled_file, "r") as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            return set()

    def _save_settled_slug(self, slug: str):
        self._settled_slugs.add(slug)
        try:
            with open(self._settled_file, "a") as f:
                f.write(slug + "\n")
        except Exception:
            pass

    def _get_exchange_pnl(self, slug: str, position) -> Optional[float]:
        """Get actual P&L from the exchange for a closed position."""
        if not self.executor._client:
            return None
        try:
            positions = self.portfolio.get_exchange_positions()
            if slug in positions:
                ex_pos = positions[slug]
                cost = float(ex_pos.get("cost", {}).get("value", "0"))
                cash_value = float(ex_pos.get("cashValue", {}).get("value", "0"))
                net = int(ex_pos.get("netPosition", "0"))
                if net == 0:
                    return cash_value - cost
            return None
        except Exception:
            return None

    def _check_live_ncaa_edges(self):
        """Check live NCAA game odds for fast-moving edges.

        During March Madness, sportsbook lines move fast on momentum swings.
        If ESPN's live line moved 5%+ but Polymarket hasn't caught up, trade.
        """
        try:
            # Build current Polymarket prices for NCAA markets
            poly_prices = {}
            for pos in self.portfolio.get_open_positions():
                if "cbb" in (pos.slug or ""):
                    poly_prices[pos.slug] = pos.current_price

            # Also include markets we could open
            for market in self._cached_markets:
                slug = market.get("slug", "")
                if "cbb" not in slug:
                    continue
                snapshot = self.market_data.build_snapshot(market)
                if snapshot and snapshot.is_live:
                    poly_prices[slug] = snapshot.price

            if not poly_prices:
                return

            edges = self.live_odds_tracker.detect_live_edges(
                poly_prices, min_edge=0.05
            )

            for edge in edges:
                self.logger.info("live_ncaa_edge_detected", {
                    "slug": edge["slug"],
                    "poly": edge["poly_price"],
                    "espn": edge["espn_prob"],
                    "edge": round(edge["edge"] * 100, 1),
                    "status": edge["status"],
                    "score": f"{edge['away_score']}-{edge['home_score']}",
                })
        except Exception as e:
            self.logger.error("live_ncaa_edge_check_failed", {"error": str(e)})

    def _record_closing_lines(self, live_markets: List[Dict]):
        """When a game transitions to live, record the closing line for CLV tracking."""
        try:
            from bot.edge_log import record_closing_line
            for market in live_markets:
                slug = market.get("slug", "")
                if not slug:
                    continue
                # Get consensus at tip-off. The odds_cache path fails at
                # exactly the wrong moment (books pull lines at tip-off),
                # which left every Jul 17 MLB entry unstamped — fall back to
                # the 4-book aggregator, whose caches still hold the pregame
                # close. YES side = first-listed (away) team.
                consensus = None
                result = self.odds_cache.get_probability_for_slug(slug)
                if result:
                    consensus, _ = result
                if not consensus:
                    try:
                        from bot.signals.live_win_prob import slug_game_teams
                        from bot.leagues import LEAGUES, parse_derivative_slug
                        parsed = slug_game_teams(slug)
                        if parsed:
                            league, away, home = parsed
                            sk = (LEAGUES.get(league) or {}).get("odds_api_key")
                            game = (self.line_aggregator.find_game(sk, home, away)
                                    if sk else None)
                            if game:
                                consensus = game.get("away_prob")
                        else:
                            # Derivative (spread/total) slugs — the weekend's
                            # dominant trades — never stamped: they don't
                            # parse as games. Use the book line at the same
                            # points, exactly as the entry signal does.
                            d = parse_derivative_slug(slug)
                            if d:
                                sk = (LEAGUES.get(d["league"]) or {}).get("odds_api_key")
                                r = (self.line_aggregator.find_line(
                                        sk, d["away"], d["home"], d["kind"], d["line"])
                                     if sk else None)
                                if r:
                                    consensus = r[0]
                    except Exception:
                        pass
                if consensus:
                    live_price = self.market_data.get_live_price(slug)
                    if live_price:
                        record_closing_line(slug, consensus, live_price)
        except Exception:
            pass  # CLV recording must not break trading

    def _log_edge_entry(self, trade_signal, snapshot):
        """Log full edge snapshot when a trade is opened."""
        try:
            from bot.edge_log import (
                insert_edge_log, build_edge_snapshot_for_signal, classify_edge_pattern,
                check_resolution_flag,
            )
            sig_snapshot = build_edge_snapshot_for_signal(
                trade_signal, self.odds_cache, trade_signal.slug
            )
            pattern = classify_edge_pattern(sig_snapshot)

            # Check for resolution ambiguity
            res_flag = check_resolution_flag(trade_signal.slug, getattr(trade_signal, '_question', ''))

            # Extract consensus and book info from odds_value signal metadata
            odds_meta = sig_snapshot["signals"].get("odds_value", {}).get("metadata", {})
            consensus = odds_meta.get("consensus_prob", 0)
            books_used = odds_meta.get("books_used", "")
            num_books = odds_meta.get("num_books", 0)

            # Extract league from slug
            parts = trade_signal.slug.split("-")
            league = parts[1].upper() if len(parts) >= 2 else ""

            insert_edge_log(
                slug=trade_signal.slug,
                polymarket_price=trade_signal.market_price,
                consensus_price=consensus,
                books_used=books_used,
                num_books=num_books,
                edge_at_entry=trade_signal.edge,
                signal_snapshot=sig_snapshot,
                edge_pattern=pattern,
                is_live_game=getattr(trade_signal, "_is_live", False),
                league=league,
                market_type="sports" if league else "",
                resolution_flag=res_flag,
            )
        except Exception:
            pass  # Edge logging must not break trading

    def check_positions(self, has_live_games: bool = False, cycle: int = 0):
        """Check all open positions: fetch live prices, check risk, close via API."""
        original_tp = self.risk.config.take_profit_threshold
        if has_live_games:
            self.risk.config.take_profit_threshold = self.live_take_profit

        open_positions = self.portfolio.get_open_positions()
        log_diagnostics = (cycle % 10 == 0) and cycle > 0 and open_positions

        for position in open_positions:
            # Fetch live price from the exchange
            slug = position.slug or position.market_id

            # Skip positions we've already auto-settled (waiting for exchange cleanup)
            if slug in self._settled_slugs:
                continue

            live_price = self.market_data.get_live_price(slug)

            if live_price is not None:
                position.current_price = live_price
            # If we can't get a price, keep the last known price

            # Compute unrealized P&L for diagnostics
            if position.side == "buy":
                pnl_per_unit = position.current_price - position.entry_price
            else:
                pnl_per_unit = position.entry_price - position.current_price
            pnl_pct = pnl_per_unit / position.entry_price if position.entry_price > 0 else 0
            edge_remaining = abs(position.estimated_prob - position.current_price)

            # --- Exit telemetry: update running P&L extremes each cycle ---
            # unrealized_usd is positive when we're winning, negative when losing
            unrealized_usd = pnl_per_unit * position.quantity
            telemetry_changed = False
            if unrealized_usd > position.max_favorable_pnl_usd:
                position.max_favorable_pnl_usd = unrealized_usd
                telemetry_changed = True
            if unrealized_usd < position.max_adverse_pnl_usd:
                position.max_adverse_pnl_usd = unrealized_usd
                telemetry_changed = True

            # Log diagnostics every 10th cycle
            if log_diagnostics:
                would_sl = pnl_pct <= -self.risk.config.stop_loss_threshold
                would_tp = edge_remaining <= self.risk.config.take_profit_threshold
                self.logger.info("position_check", {
                    "slug": slug,
                    "side": position.side,
                    "entry": position.entry_price,
                    "current": position.current_price,
                    "pnl_pct": round(pnl_pct * 100, 1),
                    "edge_remaining": round(edge_remaining * 100, 1),
                    "sl_threshold": self.risk.config.stop_loss_threshold,
                    "tp_threshold": self.risk.config.take_profit_threshold,
                    "would_stop_loss": would_sl,
                    "would_take_profit": would_tp,
                })

            # Live positions get the wider live stop (0.35 vs 0.25): a 25%
            # stop on a 25c live contract is 6c — inside normal in-game noise
            # (Jul 13: six noise stop-outs). Size is capped smaller for live
            # entries, so dollar risk stays comparable.
            pos_is_live = False
            try:
                from bot.signals.live_win_prob import slug_game_teams as _sgt2
                if _sgt2(slug) is not None:
                    pos_is_live = self.live_cache.get_live_prob(slug) is not None
            except Exception:
                pos_is_live = False

            _orig_sl = self.risk.config.stop_loss_threshold
            if pos_is_live:
                self.risk.config.stop_loss_threshold = \
                    self.config.trading.live_stop_loss_threshold
            try:
                # Check risk thresholds (also updates position.peak_price)
                close_reason = self.risk.check_position(
                    position, position.current_price, position.estimated_prob
                )
            finally:
                self.risk.config.stop_loss_threshold = _orig_sl

            # Persist bot-owned state so peak/extremes/estimated_prob survive
            # scan reconstruction and restarts (audit: state used to reset
            # every cycle, silently disabling min-hold/trailing/take-profit).
            if telemetry_changed or close_reason == "let_it_ride" or position.peak_price > 0:
                self.portfolio.persist_position_state(position)

            # Let winners ride — don't close, alert instead
            if close_reason == "let_it_ride":
                # Track every cycle the position is held in let_it_ride state
                position.let_it_ride_count += 1
                if not getattr(position, '_ride_alerted', False):
                    potential_payout = position.quantity * 1.0  # $1/share at resolution
                    self.logger.info("let_it_ride", {
                        "slug": slug,
                        "side": position.side,
                        "current_price": position.current_price,
                        "entry_price": position.entry_price,
                        "potential_payout": round(potential_payout, 2),
                    })
                    if self.alerter:
                        self.alerter._post("#36a64f",
                            f":rocket: *Position Riding*\n"
                            f"`{slug}`\n"
                            f"Side: `{position.side.upper()}`  |  "
                            f"Entry: `{position.entry_price:.3f}`  |  "
                            f"Now: `{position.current_price:.3f}`\n"
                            f"Holding for full payout. "
                            f"Potential: `${potential_payout:.2f}`"
                        )
                    position._ride_alerted = True
                continue

            if close_reason:
                # Submit close order to the exchange
                closed_on_exchange = self.executor.close_position(position)
                if closed_on_exchange:
                    # Check if this was an auto-settle (no real close happened)
                    was_auto_settled = getattr(self.executor, '_last_close_was_auto_settle', False)
                    self.executor._last_close_was_auto_settle = False

                    if was_auto_settled:
                        # Position is settling on exchange — don't write to DB again
                        self._save_settled_slug(slug)
                        self.logger.info("position_auto_settled", {
                            "slug": slug,
                            "reason": close_reason,
                            "message": "Awaiting exchange settlement, skipping DB write",
                        })
                        continue

                    # Try to get exchange-reported P&L as source of truth
                    self.portfolio.invalidate_cache()
                    exchange_pnl = self._get_exchange_pnl(slug, position)

                    # Distance from the stop-loss boundary at exit time.
                    # Expressed as a fraction of entry_price so it's scale-free.
                    # e.g. 0.02 means the close price was 2% of entry_price above the SL floor.
                    # Useful for "were stop-losses too tight?" analysis in exit_log.
                    try:
                        sl_frac = self.risk.config.stop_loss_threshold
                        if position.side == "buy":
                            sl_boundary = position.entry_price * (1.0 - sl_frac)
                            _etd = (position.current_price - sl_boundary) / position.entry_price
                        else:
                            sl_boundary = position.entry_price * (1.0 + sl_frac)
                            _etd = (sl_boundary - position.current_price) / position.entry_price
                        exit_threshold_distance = max(_etd, 0.0)
                    except Exception:
                        exit_threshold_distance = 0.0

                    # Compute full proximity dict — signed distance to every exit
                    # threshold at the moment of close. Stored as JSON in exit_log
                    # for post-hoc queries like "was take_profit close when SL fired?"
                    try:
                        exit_proximity = compute_exit_proximity(
                            position,
                            position.current_price,
                            position.estimated_prob,
                            self.risk.config,
                        )
                    except Exception:
                        exit_proximity = {}

                    self.portfolio.close_position(
                        position, position.current_price, close_reason,
                        exchange_pnl=exchange_pnl,
                        exit_threshold_distance=exit_threshold_distance,
                        exit_proximity=exit_proximity,
                    )
                    self.risk.record_pnl(position.realized_pnl)
                    # Durable halt if this loss pushed us past the daily limit
                    self._enforce_daily_loss_pause()
                    self.logger.info("position_exit_complete", {
                        "slug": slug,
                        "reason": close_reason,
                        "side": position.side,
                        "entry": position.entry_price,
                        "exit": position.current_price,
                        "pnl": round(position.realized_pnl, 2),
                        "exchange_pnl": round(exchange_pnl, 2) if exchange_pnl is not None else None,
                    })
                    # Per-game stop lockout bookkeeping
                    if close_reason == "stop_loss":
                        from bot.edge_log import extract_game_id as _egid
                        gid = _egid(slug)
                        if gid:
                            self._game_stop_counts[gid] = self._game_stop_counts.get(gid, 0) + 1

                    # Re-entry cooldown. Jul 13 postmortem: the old "win > $2
                    # => immediate re-entry" exception bought the 52c top
                    # seconds after a +$18 exit. GAME markets now always
                    # cool down; the exception survives only for non-game
                    # markets (crypto/politics), which don't whipsaw plays.
                    import time as _time
                    from bot.signals.live_win_prob import slug_game_teams as _sgt
                    if _sgt(slug) is not None or position.realized_pnl < 2.0:
                        self._slug_cooldowns[slug] = _time.time() + self._cooldown_seconds
                else:
                    self.logger.error("position_exit_failed", {
                        "slug": slug,
                        "reason": close_reason,
                        "message": "Close order failed on exchange, position remains open",
                    })

        if has_live_games:
            self.risk.config.take_profit_threshold = original_tp

    def _do_full_scan(self) -> List[Dict]:
        """Full market scan — fetches all markets, processes everything.

        Feeds the health monitor: an empty market list (API down OR nothing
        tradeable) counts as a market-data failure so repeated empties trigger
        degraded-mode backoff instead of a tight retry loop. Backing off when
        there is genuinely nothing to trade is harmless.
        """
        markets = self.market_data.get_active_markets()
        if markets:
            self.health.record_success("market_data")
        self._cached_markets = markets
        self._last_full_scan = time.time()
        return markets

    def _do_live_scan(self, live_markets: List[Dict]):
        """Fast scan — only builds snapshots for live game markets."""
        snapshots = []
        for market in live_markets:
            if not self.running:
                break
            snapshot = self.market_data.build_snapshot(market)
            if snapshot:
                snapshots.append(snapshot)
        self.process_markets(snapshots)

    def _log_open_positions(self):
        """Log all currently open positions at startup."""
        open_pos = self.portfolio.get_open_positions()
        if not open_pos:
            self.logger.info("open_positions", {"count": 0, "positions": []})
            print("  Open positions: none")
            return

        pos_list = []
        for p in open_pos:
            pos_list.append({
                "market_id": p.market_id,
                "side": p.side,
                "entry_price": p.entry_price,
                "size_usd": p.size_usd,
            })

        self.logger.info("open_positions", {
            "count": len(open_pos),
            "positions": pos_list,
        })
        print(f"  Open positions: {len(open_pos)}")
        for p in open_pos:
            print(f"    {p.side.upper():4s}  ${p.size_usd:.0f}  @ {p.entry_price:.3f}  {p.market_id}")

    def run(self):
        """Main trading loop with dual-speed scanning.

        Fast loop (every 3s): only scans live in-game markets (~10 markets).
        Full scan (every 60s): scans all filtered markets (~68 markets) for new opportunities.
        """
        mode = "PAPER" if self.config.trading.paper_trading else "LIVE"
        self.logger.info("bot_starting", {
            "mode": mode,
            "bankroll": self.portfolio.bankroll,
            "live_scan_interval": self.live_scan_interval,
            "full_scan_interval": self.full_scan_interval,
        })

        equity = self.portfolio.get_equity()

        print(f"\n{'='*60}")
        print(f"  Polymarket Trading Bot - {mode} MODE")
        print(f"  Cash balance: ${self.portfolio.bankroll:.2f}")
        print(f"  Account value: ${equity:.2f}")
        print(f"  Fast scan: {self.live_scan_interval}s (live games only)")
        print(f"  Full scan: {self.full_scan_interval}s (all markets)")
        print(f"  Edge threshold: {self.config.trading.min_edge_threshold*100:.1f}% "
              f"(live: {self.live_min_edge*100:.1f}%)")

        # Log open positions (from exchange)
        self._log_open_positions()

        # Show today's game schedule
        schedule = self.game_schedule.format_schedule()
        if schedule:
            print(f"\n  Today's Games:")
            for line in schedule.split("\n"):
                if line.strip():
                    print(f"  {line}")

        should_scan, reason = self.game_schedule.should_be_scanning()
        print(f"\n  Scanning: {'YES' if should_scan else 'SLEEPING'} ({reason})")
        print(f"{'='*60}\n")

        # Initial full scan + startup alert
        markets = self._do_full_scan()
        self.alerter.bot_started(mode, self.portfolio.bankroll, len(markets))

        def signal_handler(sig, frame):
            print("\nShutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        cycle = 0
        last_summary_date = None

        while self.running:
            cycle += 1
            try:
                # Liveness heartbeat — read by the supervisor's stale check
                self.health.beat({
                    "cycle": cycle,
                    "mode": mode,
                    "open_positions": len(self.portfolio.get_open_positions()),
                    "degraded_sources": [
                        s for s in ("market_data",) if self.health.is_degraded(s)
                    ],
                })

                # Check supervisor kill switch / pause
                if not self._check_supervisor_flags():
                    time.sleep(10)
                    continue

                # Game schedule awareness: sleep when no games within 2 hours
                if not self.config.trading.paper_trading and cycle % 60 == 1:
                    should_scan, reason = self.game_schedule.should_be_scanning()
                    if not should_scan:
                        self.logger.info("bot_sleeping", {"reason": reason})
                        time.sleep(120)  # Check again in 2 minutes
                        continue

                now = time.time()
                time_since_full = now - self._last_full_scan
                is_full_scan = time_since_full >= self.full_scan_interval

                if is_full_scan:
                    # === FULL SCAN: all markets ===
                    self.logger.info("full_scan_start", {"cycle": cycle})

                    # Force fresh exchange data for this cycle
                    self.portfolio.invalidate_cache()

                    markets = self._do_full_scan()
                    if not markets:
                        # Degraded-mode backoff: repeated empty fetches slow the
                        # loop down exponentially (up to 5 min) instead of
                        # hammering a dead or empty API every few seconds.
                        backoff = self.health.record_failure("market_data")
                        self.logger.warning("no_markets_found", {
                            "cycle": cycle,
                            "consecutive": self.health.consecutive_failures("market_data"),
                            "backoff_seconds": backoff,
                        })
                        time.sleep(max(backoff, self.live_scan_interval))
                        continue

                    live_markets, pregame_markets = self._split_markets(markets)

                    # Build snapshots ONLY for the priceable universe. Each
                    # snapshot costs rate-limited API calls; the raw in-window
                    # list (~2,000 on a Sunday: tennis, NPB, props) took 10+
                    # minutes per full scan and froze the heartbeat. Priceable:
                    # game markets in registered leagues, plus non-sports
                    # markets (crypto/politics — no gameStartTime).
                    from bot.signals.live_win_prob import slug_game_teams
                    from bot.leagues import parse_derivative_slug
                    scannable = [
                        m for m in markets
                        if slug_game_teams(m.get("slug", "")) is not None
                        or parse_derivative_slug(m.get("slug", "")) is not None
                        or m.get("slug", "").startswith(("cpc-", "tc-temp-"))
                        or not m.get("gameStartTime")
                    ]
                    self.logger.info("full_scan_universe", {
                        "in_window": len(markets),
                        "priceable": len(scannable),
                    })

                    snapshots = []
                    for market in scannable:
                        if not self.running:
                            break
                        snapshot = self.market_data.build_snapshot(market)
                        if snapshot:
                            snapshots.append(snapshot)

                    # Refresh open positions' probability estimates BEFORE the
                    # exit checks so take-profit sees current beliefs.
                    self._refresh_position_estimates(snapshots)
                    self.process_markets(snapshots)

                    has_live = len(live_markets) > 0
                    self.check_positions(has_live, cycle=cycle)
                    self.portfolio.record_equity()

                    stats = self.portfolio.get_stats()
                    self.logger.info("full_scan_complete", {
                        "cycle": cycle,
                        "total_markets": len(markets),
                        "live_markets": len(live_markets),
                        "pregame_markets": len(pregame_markets),
                        "open_positions": len(self.portfolio.get_open_positions()),
                        "total_trades": stats["total_trades"],
                        "total_pnl": round(stats["total_pnl"], 2),
                        "cash_balance": round(self.portfolio.bankroll, 2),
                        "account_value": round(self.portfolio.get_equity(), 2),
                    })

                else:
                    # === FAST SCAN: live games only ===
                    live_markets, pregame_markets = self._split_markets(self._cached_markets)

                    # Stamp closing lines PREGAME (start <=15 min out):
                    # books still quote the game then, whereas at the live
                    # transition they've already pulled it — that race left
                    # the entire weekend unstamped.
                    try:
                        now_utc = datetime.now(timezone.utc)
                        imminent = []
                        for pm in pregame_markets:
                            gs = pm.get("gameStartTime")
                            if not gs:
                                continue
                            try:
                                start = dateutil_parser.isoparse(gs)
                            except Exception:
                                continue
                            if timedelta(0) <= start - now_utc <= timedelta(minutes=15):
                                imminent.append(pm)
                        if imminent:
                            self._record_closing_lines(imminent)
                    except Exception:
                        pass

                    if live_markets:
                        # Beat inside the fast path too: a long live pass must
                        # not look like a dead loop to the supervisor.
                        self.health.beat({
                            "cycle": cycle,
                            "mode": mode,
                            "phase": "live_scan",
                            "live_count": len(live_markets),
                            "open_positions": len(self.portfolio.get_open_positions()),
                        })
                        self.logger.info("live_scan_start", {
                            "cycle": cycle,
                            "live_count": len(live_markets),
                        })

                        # Record closing lines for CLV tracking (once per game start)
                        if cycle % 20 == 1:  # every ~60s during live scans
                            self._record_closing_lines(live_markets)

                        # Check live NCAA odds for fast-moving edges
                        if cycle % 10 == 0:  # every ~30s
                            self._check_live_ncaa_edges()

                        self._do_live_scan(live_markets)
                        self.check_positions(has_live_games=True, cycle=cycle)

                        self.logger.info("live_scan_complete", {
                            "cycle": cycle,
                            "live_markets": len(live_markets),
                            "open_positions": len(self.portfolio.get_open_positions()),
                        })

                # Daily summary alert
                now_utc = datetime.now(timezone.utc)
                today_str = now_utc.strftime("%Y-%m-%d")
                if (now_utc.hour == self.config.alerts.daily_summary_hour
                        and last_summary_date != today_str
                        and self.config.alerts.on_daily_summary):
                    stats_for_summary = self.portfolio.get_stats()
                    self.alerter.daily_summary(
                        total_trades=stats_for_summary.get("total_trades", 0),
                        wins=stats_for_summary.get("winning_trades", 0),
                        losses=stats_for_summary.get("losing_trades", 0),
                        daily_pnl=stats_for_summary.get("total_pnl", 0),
                        open_positions=len(self.portfolio.get_open_positions()),
                        bankroll=self.portfolio.bankroll,
                    )
                    last_summary_date = today_str

                time.sleep(self.live_scan_interval)

            except Exception as e:
                self.logger.error("scan_cycle_error", {"cycle": cycle, "error": str(e)})
                if self.config.alerts.on_error:
                    self.alerter.error(f"Scan cycle {cycle} failed", str(e))
                time.sleep(self.config.api.scan_interval_seconds)

        # Final summary
        stats = self.portfolio.get_stats()
        print(f"\n{'='*60}")
        print(f"  BOT STOPPED")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Win rate: {stats['win_rate']*100:.1f}%")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Final bankroll: ${self.portfolio.bankroll:.2f}")
        print(f"{'='*60}\n")


def main():
    """Entry point for the trading loop."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    config = load_config(config_path)
    bot = TradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()
