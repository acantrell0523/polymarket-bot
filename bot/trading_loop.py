"""Main orchestrator loop with on-chain enrichment pipeline."""

import sys
import time
import signal
import copy
from datetime import datetime, timezone
from typing import Optional

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
    """Main trading bot orchestrator."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = TradingLogger(
            level=config.logging.level,
            log_file=config.logging.file,
            console=config.logging.console,
        )
        self.market_data = MarketDataClient(config.api, self.logger, config.filters)
        self.estimator = ProbabilityEstimator(config.signals)
        self.sizer = PositionSizer(config.trading)
        self.risk = RiskManager(config.trading)
        self.executor = ExecutionEngine(config, self.logger)
        self.alerter = SlackAlerter(
            webhook_url=config.alerts.slack_webhook_url,
            enabled=config.alerts.enabled,
        )
        self.portfolio = Portfolio(
            initial_bankroll=config.backtest.initial_bankroll_usd,
            logger=self.logger,
            alerter=self.alerter,
            alert_config=config.alerts,
        )
        self.running = True

        # Live-game estimator with aggressive signal weights
        live_signal_config = copy.deepcopy(config.signals)
        live_signal_config.order_book_imbalance_weight = 0.35
        live_signal_config.price_momentum_weight = 0.30
        # Redistribute remaining 0.35 across other signals proportionally
        live_signal_config.volume_signal_weight = 0.10
        live_signal_config.mean_reversion_weight = 0.08
        live_signal_config.volatility_signal_weight = 0.02
        live_signal_config.uw_smart_money_weight = 0.07
        live_signal_config.uw_whale_flow_weight = 0.05
        live_signal_config.uw_market_sentiment_weight = 0.03
        self.live_estimator = ProbabilityEstimator(live_signal_config)

        # Live-game trading overrides
        self.live_min_edge = 0.015       # 1.5% edge threshold
        self.live_take_profit = 0.005    # 0.5% take-profit
        self.live_scan_interval = 3      # 3 second scan for live games

        # On-chain enrichment client (optional)
        self.onchain_client = None
        if config.onchain.enabled:
            try:
                # Import from the project root onchain.py
                sys.path.insert(0, ".")
                from onchain import OnChainEnrichmentClient
                self.onchain_client = OnChainEnrichmentClient(
                    clob_url=config.api.clob_url,
                    gamma_url=config.api.gamma_url,
                    logger=self.logger,
                    max_rps=config.onchain.max_requests_per_second,
                    cache_ttl=config.onchain.cache_ttl_seconds,
                )
            except ImportError:
                self.logger.warning("onchain_import_failed", {
                    "message": "OnChainEnrichmentClient not available, running without enrichment"
                })

    def _should_enrich(self, has_edge: bool) -> bool:
        """Decide whether to enrich a market with on-chain data."""
        if not self.onchain_client:
            return False
        if self.config.onchain.enrich_all_markets:
            return True
        if self.config.onchain.enrich_on_edge_only and has_edge:
            return True
        return False

    def process_market(self, snapshot: MarketSnapshot):
        """Process a single market snapshot through the full pipeline."""
        # Select estimator and thresholds based on live status
        if snapshot.is_live:
            estimator = self.live_estimator
            min_edge = self.live_min_edge
        else:
            estimator = self.estimator
            min_edge = self.config.trading.min_edge_threshold

        # Phase 1: Core signal edge detection
        trade_signal = estimator.detect_edge(
            snapshot,
            min_edge=min_edge,
            max_edge=self.config.trading.max_edge_threshold,
        )

        # Phase 2: Enrich with on-chain data if edge detected
        if self._should_enrich(trade_signal is not None) and trade_signal is not None:
            enrichment = self.onchain_client.get_enrichment_for_market(snapshot)
            # Re-run with enrichment
            trade_signal = estimator.detect_edge(
                snapshot,
                min_edge=min_edge,
                max_edge=self.config.trading.max_edge_threshold,
                enrichment=enrichment,
            )

        if trade_signal is None:
            return

        # Phase 3: Risk checks
        if not self.risk.can_open_position(self.portfolio.positions):
            return

        # Phase 4: Position sizing
        exposure = self.portfolio.get_total_exposure()
        size = self.sizer.size_position(trade_signal, self.portfolio.bankroll, exposure)
        if size <= 0:
            return

        trade_signal.position_size_usd = size

        # Attach snapshot metadata for alerts
        trade_signal._question = snapshot.question
        trade_signal._is_live = snapshot.is_live

        # Phase 5: Execute
        trade = self.executor.execute_trade(trade_signal)
        if trade:
            self.portfolio.open_position(trade_signal, trade)

    def check_positions(self, has_live_games: bool = False):
        """Check all open positions for stop-loss/take-profit."""
        # For live games, temporarily use tighter take-profit
        original_tp = self.risk.config.take_profit_threshold
        if has_live_games:
            self.risk.config.take_profit_threshold = self.live_take_profit

        for position in self.portfolio.get_open_positions():
            close_reason = self.risk.check_position(
                position, position.current_price, position.estimated_prob
            )
            if close_reason:
                self.portfolio.close_position(
                    position, position.current_price, close_reason
                )
                self.risk.record_pnl(position.realized_pnl)

        # Restore original threshold
        if has_live_games:
            self.risk.config.take_profit_threshold = original_tp

    def run(self):
        """Main trading loop."""
        mode = "PAPER" if self.config.trading.paper_trading else "LIVE"
        self.logger.info("bot_starting", {
            "mode": mode,
            "bankroll": self.portfolio.bankroll,
            "scan_interval": self.config.api.scan_interval_seconds,
        })

        print(f"\n{'='*60}")
        print(f"  Polymarket Trading Bot - {mode} MODE")
        print(f"  Bankroll: ${self.portfolio.bankroll:.2f}")
        print(f"  Scan interval: {self.config.api.scan_interval_seconds}s")
        print(f"  Edge threshold: {self.config.trading.min_edge_threshold*100:.1f}%")
        print(f"{'='*60}\n")

        # Send startup alert
        markets_preview = self.market_data.get_active_markets()
        self.alerter.bot_started(mode, self.portfolio.bankroll, len(markets_preview))

        def signal_handler(sig, frame):
            print("\nShutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        cycle = 0
        last_summary_date = None
        while self.running:
            cycle += 1
            try:
                self.logger.info("scan_cycle_start", {"cycle": cycle})

                # Fetch active markets
                markets = self.market_data.get_active_markets()

                if not markets:
                    self.logger.warning("no_markets_found", {"cycle": cycle})
                    time.sleep(self.config.api.scan_interval_seconds)
                    continue

                # Build snapshots and process
                has_live_games = False
                for market in markets:
                    if not self.running:
                        break
                    snapshot = self.market_data.build_snapshot(market)
                    if snapshot:
                        if snapshot.is_live:
                            has_live_games = True
                        self.process_market(snapshot)

                # Check existing positions (tighter take-profit for live games)
                self.check_positions(has_live_games)

                # Record equity
                self.portfolio.record_equity()

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

                # Log status
                stats = self.portfolio.get_stats()
                open_count = len(self.portfolio.get_open_positions())
                # Use faster scan interval when live games are active
                scan_interval = self.live_scan_interval if has_live_games else self.config.api.scan_interval_seconds

                self.logger.info("scan_cycle_complete", {
                    "cycle": cycle,
                    "markets_scanned": len(markets),
                    "live_games": has_live_games,
                    "open_positions": open_count,
                    "total_trades": stats["total_trades"],
                    "total_pnl": round(stats["total_pnl"], 2),
                    "bankroll": round(self.portfolio.bankroll, 2),
                    "next_scan_seconds": scan_interval,
                })

                time.sleep(scan_interval)

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
