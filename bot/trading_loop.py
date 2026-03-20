"""Main orchestrator loop with dual-speed scanning."""

import sys
import time
import signal
import copy
from datetime import datetime, timezone
from typing import Optional, List, Dict

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
        self.estimator = ProbabilityEstimator(config.signals)
        self.sizer = PositionSizer(config.trading)
        self.risk = RiskManager(config.trading)
        self.executor = ExecutionEngine(config, self.logger)
        self.alerter = SlackAlerter(
            webhook_url=config.alerts.slack_webhook_url,
            enabled=config.alerts.enabled,
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

        # Live-game estimator with aggressive signal weights
        live_signal_config = copy.deepcopy(config.signals)
        live_signal_config.order_book_imbalance_weight = 0.35
        live_signal_config.price_momentum_weight = 0.30
        live_signal_config.volume_signal_weight = 0.10
        live_signal_config.mean_reversion_weight = 0.08
        live_signal_config.volatility_signal_weight = 0.02
        live_signal_config.uw_smart_money_weight = 0.07
        live_signal_config.uw_whale_flow_weight = 0.05
        live_signal_config.uw_market_sentiment_weight = 0.03
        self.live_estimator = ProbabilityEstimator(live_signal_config)

        # Live-game trading overrides
        self.live_min_edge = 0.015
        self.live_take_profit = 0.03
        self.live_scan_interval = 3
        self.full_scan_interval = 60  # full market scan every 60s

        # Cached market list from last full scan
        self._cached_markets: List[Dict] = []
        self._last_full_scan = 0.0

        # On-chain enrichment client (optional)
        self.onchain_client = None
        if config.onchain.enabled:
            try:
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
        if not self.onchain_client:
            return False
        if self.config.onchain.enrich_all_markets:
            return True
        if self.config.onchain.enrich_on_edge_only and has_edge:
            return True
        return False

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

    def process_market(self, snapshot: MarketSnapshot):
        """Process a single market snapshot through the full pipeline."""
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

        if self._should_enrich(trade_signal is not None) and trade_signal is not None:
            enrichment = self.onchain_client.get_enrichment_for_market(snapshot)
            trade_signal = estimator.detect_edge(
                snapshot,
                min_edge=min_edge,
                max_edge=self.config.trading.max_edge_threshold,
                enrichment=enrichment,
            )

        if trade_signal is None:
            return

        if not self.risk.can_open_position(self.portfolio.positions):
            return

        exposure = self.portfolio.get_total_exposure()
        size = self.sizer.size_position(trade_signal, self.portfolio.bankroll, exposure)
        if size <= 0:
            return

        trade_signal.position_size_usd = size
        trade_signal._question = snapshot.question
        trade_signal._is_live = snapshot.is_live

        trade = self.executor.execute_trade(trade_signal)
        if trade:
            self.portfolio.open_position(trade_signal, trade)

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
        for market in live_markets:
            if not self.running:
                break
            snapshot = self.market_data.build_snapshot(market)
            if snapshot:
                self.process_market(snapshot)

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

                    # Process all markets (live + pregame)
                    for market in markets:
                        if not self.running:
                            break
                        snapshot = self.market_data.build_snapshot(market)
                        if snapshot:
                            self.process_market(snapshot)

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
