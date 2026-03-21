"""Main orchestrator loop with dual-speed scanning."""

import os
import sys
import time
import signal
import copy
from datetime import datetime, timezone
from typing import Optional, List, Dict
from dateutil import parser as dateutil_parser

from utils.config import load_config, BotConfig
from utils.logger import TradingLogger
from utils.models import MarketSnapshot
from bot.signals.estimator import ProbabilityEstimator
from bot.strategies.sizing import PositionSizer
from bot.strategies.risk import RiskManager
from bot.market_data import MarketDataClient
from bot.execution import ExecutionEngine
from bot.portfolio import Portfolio
from bot.alerts import SlackAlerter


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

        if not self.odds_cache.enabled:
            self.logger.warning("odds_api_disabled", {
                "message": "THE_ODDS_API_KEY not set — sports markets will not trade"
            })

        # Estimator with all caches
        self.estimator = ProbabilityEstimator(
            config.signals, self.odds_cache, self.predictit_cache, self.crypto_cache,
            self.espn_cache, self.game_context,
        )

        # Fetch real balance from exchange for live mode
        initial_bankroll = config.backtest.initial_bankroll_usd
        if not config.trading.paper_trading and self.executor._client:
            try:
                bal = self.executor._client.account.balances()
                balances = bal.get("balances", [])
                if balances:
                    initial_bankroll = float(balances[0].get("currentBalance", initial_bankroll))
                    self.logger.info("bankroll_from_exchange", {"balance": initial_bankroll})
            except Exception as e:
                self.logger.warning("bankroll_fetch_failed", {"error": str(e)})

        self.portfolio = Portfolio(
            initial_bankroll=initial_bankroll,
            logger=self.logger,
            alerter=self.alerter,
            alert_config=config.alerts,
        )
        self.running = True

        # Cooldown tracker: slug → earliest time we can reopen
        self._slug_cooldowns: Dict[str, float] = {}
        self._cooldown_seconds = 600  # 10 minutes

        # Live-game estimator with aggressive weights
        live_signal_config = copy.deepcopy(config.signals)
        live_signal_config.odds_value_weight = 0.40
        live_signal_config.order_book_imbalance_weight = 0.25
        live_signal_config.line_movement_weight = 0.15
        live_signal_config.liquidity_imbalance_weight = 0.20
        self.live_estimator = ProbabilityEstimator(
            live_signal_config, self.odds_cache, self.predictit_cache, self.crypto_cache,
            self.espn_cache, self.game_context,
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
        """Split markets into live and pre-game lists."""
        live = []
        pregame = []
        for m in markets:
            if self._is_live_market(m):
                live.append(m)
            else:
                pregame.append(m)
        return live, pregame

    def _detect_edge(self, snapshot: MarketSnapshot):
        """Detect edge for a single market. Returns (signal, snapshot) or None."""
        if snapshot.is_live:
            estimator = self.live_estimator
            min_edge = self.live_min_edge
        else:
            estimator = self.estimator
            min_edge = self.config.trading.min_edge_threshold

        trade_signal = estimator.detect_edge(
            snapshot,
            min_edge=min_edge,
            max_edge=self.config.trading.max_edge_threshold,
        )
        if trade_signal is None:
            return None
        return (trade_signal, snapshot)

    def process_markets(self, snapshots: list):
        """Rank all opportunities by edge size and execute the best ones."""
        import time as _time
        from bot.edge_log import extract_game_id, get_open_game_ids

        # Collect all edges
        opportunities = []
        open_positions = self.portfolio.get_open_positions()
        open_slugs = {p.slug for p in open_positions}

        # Build game correlation map — one position per game max
        open_games = get_open_game_ids(open_positions)

        for snapshot in snapshots:
            if not snapshot:
                continue
            slug = snapshot.slug

            # Skip if already holding this exact market
            if slug in open_slugs:
                continue

            # Skip if we already have a position on the same game
            # (correlated market detection: moneyline + spread on same game = bad)
            game_id = extract_game_id(slug)
            if game_id in open_games:
                continue

            # Skip if in cooldown
            cooldown_until = self._slug_cooldowns.get(slug, 0)
            if _time.time() < cooldown_until:
                continue

            result = self._detect_edge(snapshot)
            if result:
                opportunities.append(result)

        if not opportunities:
            return

        # Rank by absolute edge, descending — take only the best
        opportunities.sort(key=lambda x: abs(x[0].edge), reverse=True)

        slots = self.config.trading.max_open_positions - len(open_slugs)
        if slots <= 0:
            return

        # Track games we're about to open in this batch to avoid duplicates
        games_opening = set()

        for trade_signal, snapshot in opportunities[:slots]:
            if not self.risk.can_open_position(self.portfolio.positions):
                break

            # One position per game within the same batch
            game_id = extract_game_id(trade_signal.slug)
            if game_id in games_opening or game_id in open_games:
                continue

            exposure = self.portfolio.get_total_exposure()
            size = self.sizer.size_position(trade_signal, self.portfolio.bankroll, exposure)
            if size <= 0:
                continue

            trade_signal.position_size_usd = size
            trade_signal._question = snapshot.question
            trade_signal._is_live = snapshot.is_live

            trade = self.executor.execute_trade(trade_signal)
            if trade:
                self.portfolio.open_position(trade_signal, trade)
                self.risk.record_trade_opened()
                games_opening.add(game_id)
                # Log edge snapshot for this trade
                self._log_edge_entry(trade_signal, snapshot)

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
                # Get consensus at tip-off
                result = self.odds_cache.get_probability_for_slug(slug)
                if result:
                    consensus, _ = result
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

            # Check risk thresholds
            close_reason = self.risk.check_position(
                position, position.current_price, position.estimated_prob
            )
            if close_reason:
                # Submit close order to the exchange
                closed_on_exchange = self.executor.close_position(position)
                if closed_on_exchange:
                    self.portfolio.close_position(
                        position, position.current_price, close_reason
                    )
                    self.risk.record_pnl(position.realized_pnl)
                    self.logger.info("position_exit_complete", {
                        "slug": slug,
                        "reason": close_reason,
                        "entry": position.entry_price,
                        "exit": position.current_price,
                        "pnl": round(position.realized_pnl, 2),
                    })
                    # Smart re-entry: loss = 10min cooldown, win (>$2) = immediate
                    import time as _time
                    if position.realized_pnl < 2.0:
                        self._slug_cooldowns[slug] = _time.time() + self._cooldown_seconds
                else:
                    self.logger.error("position_exit_failed", {
                        "slug": slug,
                        "reason": close_reason,
                        "message": "Close order failed on exchange, position remains open",
                    })

        if has_live_games:
            self.risk.config.take_profit_threshold = original_tp

    def sync_from_exchange(self):
        """Rebuild internal position state from the exchange's actual positions."""
        exchange_positions = self.executor.get_exchange_positions()
        if not exchange_positions:
            self.logger.warning("sync_no_exchange_positions", {
                "message": "No positions returned from exchange"
            })
            return

        # Clear existing internal positions
        old_count = len(self.portfolio.get_open_positions())
        self.portfolio.positions = []

        from utils.models import Position
        for slug, p_data in exchange_positions.items():
            net = int(p_data.get("netPosition", "0"))
            if net == 0:
                continue

            cost_val = float(p_data.get("cost", {}).get("value", "0"))
            cash_val = float(p_data.get("cashValue", {}).get("value", "0"))
            qty = abs(net)
            entry_price = cost_val / qty if qty > 0 else 0
            current_price_est = cash_val / qty if qty > 0 else entry_price
            side = "buy" if net > 0 else "sell"

            meta = p_data.get("marketMetadata", {})
            title = meta.get("title", slug)

            position = Position(
                market_id=slug,
                token_id=slug,
                side=side,
                entry_price=entry_price,
                size_usd=cost_val,
                quantity=qty,
                estimated_prob=0.5,
                entry_time=datetime.now(timezone.utc),
                current_price=current_price_est,
                slug=slug,
            )
            self.portfolio.positions.append(position)

        new_count = len(self.portfolio.get_open_positions())
        self.logger.info("sync_complete", {
            "old_internal_positions": old_count,
            "exchange_positions": len(exchange_positions),
            "synced_positions": new_count,
        })

    def _do_full_scan(self) -> List[Dict]:
        """Full market scan — fetches all markets, processes everything."""
        markets = self.market_data.get_active_markets()
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

        print(f"\n{'='*60}")
        print(f"  Polymarket Trading Bot - {mode} MODE")
        print(f"  Bankroll: ${self.portfolio.bankroll:.2f}")
        print(f"  Fast scan: {self.live_scan_interval}s (live games only)")
        print(f"  Full scan: {self.full_scan_interval}s (all markets)")
        print(f"  Edge threshold: {self.config.trading.min_edge_threshold*100:.1f}% "
              f"(live: {self.live_min_edge*100:.1f}%)")

        # Sync internal state from exchange
        if not self.config.trading.paper_trading:
            print("  Syncing positions from exchange...")
            self.sync_from_exchange()

        # Log open positions
        self._log_open_positions()
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
                # Check supervisor kill switch / pause
                if not self._check_supervisor_flags():
                    time.sleep(10)
                    continue

                now = time.time()
                time_since_full = now - self._last_full_scan
                is_full_scan = time_since_full >= self.full_scan_interval

                if is_full_scan:
                    # === FULL SCAN: all markets ===
                    self.logger.info("full_scan_start", {"cycle": cycle})

                    markets = self._do_full_scan()
                    if not markets:
                        self.logger.warning("no_markets_found", {"cycle": cycle})
                        time.sleep(self.live_scan_interval)
                        continue

                    live_markets, pregame_markets = self._split_markets(markets)

                    # Build all snapshots, then rank and process best opportunities
                    snapshots = []
                    for market in markets:
                        if not self.running:
                            break
                        snapshot = self.market_data.build_snapshot(market)
                        if snapshot:
                            snapshots.append(snapshot)
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
                        "bankroll": round(self.portfolio.bankroll, 2),
                    })

                else:
                    # === FAST SCAN: live games only ===
                    live_markets, _ = self._split_markets(self._cached_markets)

                    if live_markets:
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
