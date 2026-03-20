"""Risk management: stop-loss, take-profit, daily loss limit."""

from typing import List, Optional
from datetime import datetime, timezone
from utils.models import Position
from utils.config import TradingConfig

MIN_HOLD_SECONDS = 60  # Minimum time before take-profit can trigger


class RiskManager:
    """Manages risk controls for open positions."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.last_reset_date: Optional[str] = None

    def reset_daily_pnl(self, date_str: Optional[str] = None):
        """Reset daily P&L tracker (called at start of each trading day)."""
        self.daily_pnl = 0.0
        self.last_reset_date = date_str or datetime.utcnow().strftime("%Y-%m-%d")

    def record_pnl(self, pnl: float):
        """Record realized P&L."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.reset_daily_pnl(today)
        self.daily_pnl += pnl

    def is_daily_limit_breached(self) -> bool:
        """Check if daily loss limit has been breached."""
        return self.daily_pnl <= -self.config.daily_loss_limit_usd

    def check_position(self, position: Position, current_price: float, estimated_prob: float) -> Optional[str]:
        """
        Check if a position should be closed.

        Returns close reason string, or None to keep position open.
        """
        if position.status != "open":
            return None

        # Update current price and unrealized P&L
        position.current_price = current_price

        if position.side == "buy":
            pnl_per_unit = current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - current_price

        position.unrealized_pnl = pnl_per_unit * position.quantity

        # Stop-loss check
        loss_pct = -pnl_per_unit / position.entry_price if position.entry_price > 0 else 0
        if loss_pct >= self.config.stop_loss_threshold:
            return "stop_loss"

        # Take-profit: close when price converges to estimated fair value
        # Guards: must be held 60s+ AND position must be in actual profit
        edge_remaining = abs(estimated_prob - current_price)
        if edge_remaining <= self.config.take_profit_threshold:
            # Only take profit if the position is actually profitable
            if pnl_per_unit <= 0:
                pass  # Not in profit — don't close
            elif position.entry_time:
                now = datetime.now(timezone.utc)
                entry = position.entry_time
                if entry.tzinfo is None:
                    entry = entry.replace(tzinfo=timezone.utc)
                held_seconds = (now - entry).total_seconds()
                if held_seconds >= MIN_HOLD_SECONDS:
                    return "take_profit"

        # Position resolved: price hit 0 or 1
        if current_price <= 0.01 or current_price >= 0.99:
            return "resolved"

        return None

    def can_open_position(self, current_positions: List[Position]) -> bool:
        """Check if we're allowed to open a new position."""
        open_positions = [p for p in current_positions if p.status == "open"]

        if len(open_positions) >= self.config.max_open_positions:
            return False

        if self.is_daily_limit_breached():
            return False

        return True

    def get_portfolio_exposure(self, positions: List[Position]) -> float:
        """Calculate total portfolio exposure in USD."""
        return sum(p.size_usd for p in positions if p.status == "open")
