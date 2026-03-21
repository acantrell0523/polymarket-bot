"""Risk management: stop-loss, take-profit, trailing stop, aggressive exit."""

from typing import List, Optional
from datetime import datetime, timezone
from utils.models import Position
from utils.config import TradingConfig

MIN_HOLD_SECONDS = 600  # 10 minutes — no exit under 10min except 25% stop-loss


class RiskManager:
    """Manages risk controls for open positions."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.last_reset_date: Optional[str] = None

    def reset_daily_pnl(self, date_str: Optional[str] = None):
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.last_reset_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def record_pnl(self, pnl: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.reset_daily_pnl(today)
        self.daily_pnl += pnl

    def record_trade_opened(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.reset_daily_pnl(today)
        self.daily_trade_count += 1

    def is_daily_trade_limit_reached(self) -> bool:
        return self.daily_trade_count >= self.config.max_daily_trades

    def is_daily_limit_breached(self) -> bool:
        return self.daily_pnl <= -self.config.daily_loss_limit_usd

    def check_position(self, position: Position, current_price: float, estimated_prob: float) -> Optional[str]:
        """Check if a position should be closed.

        Exit hierarchy:
        1. Position resolved (price at 0 or 1)
        2. Stop-loss (down 25%)
        3. Aggressive exit (up 30%+)
        4. Trailing stop (was up 15%, dropped back)
        5. Take-profit (edge converged, min $5 profit, held 60s+)
        """
        if position.status != "open":
            return None

        position.current_price = current_price

        if position.side == "buy":
            pnl_per_unit = current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - current_price

        position.unrealized_pnl = pnl_per_unit * position.quantity
        gain_pct = pnl_per_unit / position.entry_price if position.entry_price > 0 else 0

        # Track peak favorable price for trailing stop
        if position.side == "buy":
            if current_price > position.peak_price:
                position.peak_price = current_price
        else:
            # For shorts, "peak" means lowest price (most favorable)
            if position.peak_price == 0 or current_price < position.peak_price:
                position.peak_price = current_price

        # 1. Resolved: price hit 0 or 1 — always exit immediately
        if current_price <= 0.01 or current_price >= 0.99:
            return "resolved"

        # 2. Stop-loss (25%) — always exit immediately
        loss_pct = -pnl_per_unit / position.entry_price if position.entry_price > 0 else 0
        if loss_pct >= self.config.stop_loss_threshold:
            return "stop_loss"

        # --- Minimum hold time gate for all other exits ---
        # No position closes in under 10 minutes unless it hits stop-loss or resolves.
        held_seconds = 0.0
        if position.entry_time:
            now = datetime.now(timezone.utc)
            entry = position.entry_time
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            held_seconds = (now - entry).total_seconds()

        if held_seconds < MIN_HOLD_SECONDS:
            return None

        # 3. Aggressive exit: up 30%+ from entry
        if gain_pct >= self.config.aggressive_exit_pct:
            return "aggressive_exit"

        # 4. Trailing stop: was up 15%+, dropped 10% from peak
        if position.peak_price > 0 and position.entry_price > 0:
            if position.side == "buy":
                peak_gain = (position.peak_price - position.entry_price) / position.entry_price
                drop_from_peak = (position.peak_price - current_price) / position.entry_price
            else:
                peak_gain = (position.entry_price - position.peak_price) / position.entry_price
                drop_from_peak = (current_price - position.peak_price) / position.entry_price

            if peak_gain >= self.config.trailing_stop_activation_pct:
                if drop_from_peak >= self.config.trailing_stop_pct:
                    return "trailing_stop"

        # 5. Take-profit: edge converged, min $5 profit
        edge_remaining = abs(estimated_prob - current_price)
        if edge_remaining <= self.config.take_profit_threshold:
            if position.unrealized_pnl < self.config.minimum_take_profit_usd:
                pass  # Under $5 profit — hold
            elif pnl_per_unit <= 0:
                pass  # Not in profit
            else:
                return "take_profit"

        return None

    def can_open_position(self, current_positions: List[Position]) -> bool:
        open_positions = [p for p in current_positions if p.status == "open"]
        if len(open_positions) >= self.config.max_open_positions:
            return False
        if self.is_daily_limit_breached():
            return False
        if self.is_daily_trade_limit_reached():
            return False
        return True

    def get_portfolio_exposure(self, positions: List[Position]) -> float:
        return sum(p.size_usd for p in positions if p.status == "open")
