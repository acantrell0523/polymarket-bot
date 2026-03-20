"""Replay engine with slippage & fee simulation."""

import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from utils.models import (
    MarketSnapshot, TradeSignal, Trade, Position, BacktestResult,
)
from utils.config import BotConfig, BacktestConfig, TradingConfig, SignalConfig
from bot.signals.estimator import ProbabilityEstimator
from bot.strategies.sizing import PositionSizer
from bot.strategies.risk import RiskManager
from bot.portfolio import Portfolio


class BacktestEngine:
    """Replays historical market data through the trading pipeline."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.estimator = ProbabilityEstimator(config.signals)
        self.sizer = PositionSizer(config.trading)
        self.risk = RiskManager(config.trading)

    def apply_slippage(self, price: float, side: str, snapshot: MarketSnapshot) -> float:
        """Apply slippage to execution price."""
        if self.config.backtest.slippage_model == "fixed":
            slippage = self.config.backtest.fixed_slippage_bps / 10000
        else:
            # Depth-based slippage
            ob = snapshot.order_book
            total_depth = ob.bid_depth + ob.ask_depth
            if total_depth > 0:
                slippage = self.config.backtest.depth_slippage_multiplier / (total_depth / 1000)
                slippage = min(slippage, 0.05)  # Cap at 5%
            else:
                slippage = 0.01  # Default 1% if no depth

        if side == "buy":
            return min(price * (1 + slippage), 0.99)
        else:
            return max(price * (1 - slippage), 0.01)

    def apply_fees(self, size_usd: float, is_taker: bool = True) -> float:
        """Calculate fees for a trade."""
        if is_taker:
            return size_usd * self.config.backtest.taker_fee_bps / 10000
        return size_usd * self.config.backtest.maker_fee_bps / 10000

    def run(self, market_data: List[List[MarketSnapshot]]) -> BacktestResult:
        """
        Run backtest on historical market data.

        Args:
            market_data: List of markets, each containing time-ordered snapshots.

        Returns:
            BacktestResult with all metrics.
        """
        portfolio = Portfolio(self.config.backtest.initial_bankroll_usd)

        # Flatten and sort all snapshots by time
        all_snapshots = []
        for market_snapshots in market_data:
            for snap in market_snapshots:
                all_snapshots.append(snap)

        all_snapshots.sort(key=lambda s: s.timestamp)

        # Track the latest snapshot per market for position management
        latest_by_market: Dict[str, MarketSnapshot] = {}

        for snapshot in all_snapshots:
            latest_by_market[snapshot.market_id] = snapshot

            # Skip if insufficient history
            if len(snapshot.price_history) < 10:
                continue

            # Check existing positions for this market
            for pos in portfolio.get_open_positions():
                if pos.market_id == snapshot.market_id:
                    close_reason = self.risk.check_position(
                        pos, snapshot.price, pos.estimated_prob
                    )
                    if close_reason:
                        exit_price = self.apply_slippage(
                            snapshot.price,
                            "sell" if pos.side == "buy" else "buy",
                            snapshot,
                        )
                        pnl = portfolio.close_position(pos, exit_price, close_reason, snapshot.timestamp)
                        self.risk.record_pnl(pnl)

            # Detect edge
            trade_signal = self.estimator.detect_edge(
                snapshot,
                min_edge=self.config.trading.min_edge_threshold,
                max_edge=self.config.trading.max_edge_threshold,
            )

            if trade_signal is None:
                continue

            # Check if we already have a position in this market
            existing = [p for p in portfolio.get_open_positions()
                       if p.market_id == snapshot.market_id]
            if existing:
                continue

            # Risk checks
            if not self.risk.can_open_position(portfolio.positions):
                continue

            # Position sizing
            exposure = portfolio.get_total_exposure()
            size = self.sizer.size_position(trade_signal, portfolio.bankroll, exposure)
            if size <= 0:
                continue

            trade_signal.position_size_usd = size

            # Apply slippage
            exec_price = self.apply_slippage(snapshot.price, trade_signal.side, snapshot)
            fees = self.apply_fees(size)

            quantity = size / exec_price if exec_price > 0 else 0

            trade = Trade(
                market_id=trade_signal.market_id,
                token_id=trade_signal.token_id,
                side=trade_signal.side,
                price=exec_price,
                quantity=quantity,
                size_usd=size,
                timestamp=snapshot.timestamp,
                trade_type="entry",
                fees=fees,
                slippage=abs(exec_price - snapshot.price),
                is_paper=True,
            )

            portfolio.open_position(trade_signal, trade)
            portfolio.record_equity(snapshot.timestamp)

        # Close all remaining positions at last known price
        for pos in portfolio.get_open_positions():
            last_snap = latest_by_market.get(pos.market_id)
            if last_snap:
                portfolio.close_position(pos, last_snap.price, "backtest_end", last_snap.timestamp)

        portfolio.record_equity()

        return self._compute_results(portfolio)

    def _compute_results(self, portfolio: Portfolio) -> BacktestResult:
        """Compute backtest metrics from portfolio state."""
        closed = [p for p in portfolio.positions if p.status == "closed"]

        total_trades = len(closed)
        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = sum(p.realized_pnl for p in closed)
        total_return_pct = (total_pnl / portfolio.initial_bankroll) * 100

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = sum(p.realized_pnl for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p.realized_pnl for p in losses) / len(losses) if losses else 0

        # Max drawdown
        equity = portfolio.equity_curve
        max_dd, max_dd_pct = self._compute_max_drawdown(equity)

        # Sharpe ratio (annualized, assuming hourly data)
        sharpe = self._compute_sharpe(equity)

        # Profit factor
        gross_profit = sum(p.realized_pnl for p in wins)
        gross_loss = abs(sum(p.realized_pnl for p in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average duration
        durations = []
        for p in closed:
            if p.close_time and p.entry_time:
                dur = (p.close_time - p.entry_time).total_seconds() / 3600
                durations.append(dur)
        avg_duration = sum(durations) / len(durations) if durations else 0

        return BacktestResult(
            trades=portfolio.trades,
            positions=portfolio.positions,
            equity_curve=equity,
            timestamps=portfolio.timestamps,
            initial_bankroll=portfolio.initial_bankroll,
            final_bankroll=portfolio.bankroll,
            total_return=total_pnl,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_duration_hours=avg_duration,
            config={
                "min_edge_threshold": self.config.trading.min_edge_threshold,
                "position_sizing_method": self.config.trading.position_sizing_method,
                "kelly_fraction": self.config.trading.kelly_fraction,
                "fixed_fraction": self.config.trading.fixed_fraction,
                "stop_loss_threshold": self.config.trading.stop_loss_threshold,
            },
        )

    def _compute_max_drawdown(self, equity: List[float]) -> tuple:
        """Compute max drawdown in $ and %."""
        if len(equity) < 2:
            return 0.0, 0.0

        peak = equity[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return round(max_dd, 2), round(max_dd_pct * 100, 2)

    def _compute_sharpe(self, equity: List[float], periods_per_year: float = 8760) -> float:
        """Compute annualized Sharpe ratio."""
        if len(equity) < 3:
            return 0.0

        arr = np.array(equity, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)
        return round(float(sharpe), 2)
