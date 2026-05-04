"""Backtest-only in-memory portfolio.

Zero exchange calls, zero SQLite writes, zero side effects from the live trading path.
All financial state is tracked purely in memory for replay/simulation purposes.
"""

from datetime import datetime, timezone
from typing import List, Optional

from utils.models import Position, Trade, TradeSignal


class BacktestPortfolio:
    """Pure in-memory portfolio for backtesting.

    Provides the same surface area that :class:`~backtest.engine.BacktestEngine`
    needs, but with correct bankroll accounting and no live-trading side effects:

    * ``bankroll`` — available cash; decreases on open, recovers + P&L on close
    * ``positions`` — all positions (open **and** closed)
    * ``trades`` — every entry :class:`~utils.models.Trade`
    * ``equity_curve`` / ``timestamps`` — mark-to-market snapshots
    * ``get_open_positions()`` — open positions only
    * ``get_total_exposure()`` — sum of cost-basis for open positions
    * ``open_position(signal, trade)`` — deducts size + fees from bankroll
    * ``close_position(pos, price, reason, ts)`` — returns cost + P&L to bankroll
    * ``record_equity(ts)`` — append mark-to-market snapshot
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll: float = initial_bankroll
        self._bankroll: float = initial_bankroll
        self._positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bankroll(self) -> float:
        """Available cash.  Cost of open positions has already been deducted."""
        return self._bankroll

    @property
    def positions(self) -> List[Position]:
        """All positions — open **and** closed."""
        return self._positions

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> List[Position]:
        """Return only positions that are still open."""
        return [p for p in self._positions if p.status == "open"]

    def get_total_exposure(self) -> float:
        """Total USD cost-basis of all open positions."""
        return sum(p.size_usd for p in self.get_open_positions())

    def get_equity(self) -> float:
        """Mark-to-market equity: available cash + current value of open positions.

        Uses ``position.current_price`` if it has been set by the risk manager,
        otherwise falls back to entry price (cost basis).
        """
        pos_value = sum(
            (p.current_price if p.current_price > 0 else p.entry_price) * p.quantity
            for p in self.get_open_positions()
        )
        return self._bankroll + pos_value

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_position(self, signal: TradeSignal, trade: Trade) -> Position:
        """Create a new position and deduct its full cost + fees from bankroll.

        This gives correct bankroll accounting: after opening a $50 position
        the sizer sees $50 less available cash, preventing over-allocation.
        """
        position = Position(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            entry_price=trade.price,
            size_usd=trade.size_usd,
            quantity=trade.quantity,
            estimated_prob=signal.estimated_prob,
            entry_time=trade.timestamp,
            current_price=trade.price,
            slug=signal.slug,
        )
        self._positions.append(position)
        # Deduct full cost AND transaction fees so bankroll reflects real cash available
        self._bankroll -= trade.size_usd + trade.fees
        self.trades.append(trade)
        return position

    def close_position(
        self,
        position: Position,
        current_price: float,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Close a position and credit cost-basis + realized P&L back to bankroll.

        Returns the realized P&L (positive = profit, negative = loss).
        Calling this on an already-closed position is a no-op that returns 0.
        """
        if position.status != "open":
            return 0.0

        close_time = timestamp or datetime.now(timezone.utc)
        pnl = self._calculate_pnl(position, current_price, reason)

        position.status = "closed"
        position.close_reason = reason
        position.close_price = current_price
        position.close_time = close_time
        position.realized_pnl = pnl

        # Return original cost + realized gain (or minus loss) to available cash
        self._bankroll += position.size_usd + pnl

        return pnl

    def _calculate_pnl(self, position: Position, current_price: float, reason: str) -> float:
        """Compute realized P&L for a position being closed.

        Mirrors the same logic as :meth:`bot.portfolio.Portfolio._calculate_pnl`.
        """
        if reason == "resolved":
            if current_price <= 0.01:
                # Market resolved NO
                if position.side == "sell":
                    return position.entry_price * position.quantity
                else:
                    return -position.entry_price * position.quantity
            elif current_price >= 0.99:
                # Market resolved YES
                if position.side == "buy":
                    return (1.0 - position.entry_price) * position.quantity
                else:
                    return -(1.0 - position.entry_price) * position.quantity

        if position.side == "buy":
            return (current_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - current_price) * position.quantity

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def record_equity(self, timestamp: Optional[datetime] = None) -> None:
        """Append a mark-to-market equity snapshot to the equity curve."""
        self.equity_curve.append(self.get_equity())
        self.timestamps.append(timestamp or datetime.now(timezone.utc))
