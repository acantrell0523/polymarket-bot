"""Position tracking and P&L accounting."""

from datetime import datetime, timezone
from typing import List, Optional, Dict
from utils.models import Position, Trade, TradeSignal
from utils.logger import TradingLogger
from bot import trade_db


class Portfolio:
    """Tracks positions, trades, and P&L."""

    def __init__(self, initial_bankroll: float, logger: Optional[TradingLogger] = None,
                 alerter=None, alert_config=None):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_bankroll]
        self.timestamps: List[datetime] = [datetime.now(timezone.utc)]
        self.logger = logger
        self.alerter = alerter
        self.alert_config = alert_config

    def open_position(self, signal: TradeSignal, trade: Trade) -> Position:
        """Open a new position from a trade."""
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

        self.positions.append(position)
        self.trades.append(trade)

        # Deduct fees from bankroll
        self.bankroll -= trade.fees

        if self.logger:
            self.logger.info("position_opened", {
                "market_id": signal.market_id,
                "side": signal.side,
                "entry_price": trade.price,
                "size_usd": trade.size_usd,
                "edge": signal.edge,
            })

        if self.alerter and self.alert_config and self.alert_config.on_trade_open:
            self.alerter.trade_opened(
                question=getattr(signal, '_question', signal.market_id),
                slug=signal.slug,
                side=signal.side,
                price=trade.price,
                size_usd=trade.size_usd,
                edge=signal.edge,
                is_live=getattr(signal, '_is_live', False),
            )

        return position

    def close_position(self, position: Position, current_price: float,
                       reason: str, timestamp: Optional[datetime] = None) -> float:
        """
        Close a position and realize P&L.

        Returns realized P&L.
        """
        if position.status != "open":
            return 0.0

        close_time = timestamp or datetime.now(timezone.utc)

        if position.side == "buy":
            pnl_per_unit = current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - current_price

        realized_pnl = pnl_per_unit * position.quantity

        # Close fees
        close_value = current_price * position.quantity
        close_fees = close_value * 0.02  # 2% taker fee
        realized_pnl -= close_fees

        position.status = "closed"
        position.close_reason = reason
        position.close_price = current_price
        position.close_time = close_time
        position.realized_pnl = realized_pnl

        self.bankroll += realized_pnl

        # Record exit trade
        exit_trade = Trade(
            market_id=position.market_id,
            token_id=position.token_id,
            side="sell" if position.side == "buy" else "buy",
            price=current_price,
            quantity=position.quantity,
            size_usd=close_value,
            timestamp=close_time,
            trade_type="exit",
            fees=close_fees,
            is_paper=True,
        )
        self.trades.append(exit_trade)

        if self.logger:
            self.logger.info("position_closed", {
                "market_id": position.market_id,
                "reason": reason,
                "entry_price": position.entry_price,
                "close_price": current_price,
                "pnl": realized_pnl,
            })

        if self.alerter and self.alert_config and self.alert_config.on_trade_close:
            self.alerter.trade_closed(
                question=position.market_id,
                slug=getattr(position, 'slug', position.market_id),
                side=position.side,
                entry_price=position.entry_price,
                exit_price=current_price,
                pnl=realized_pnl,
                close_reason=reason,
            )

        # Persist to SQLite for supervisor analysis
        try:
            slug = getattr(position, 'slug', position.market_id)
            market_type = ""
            if slug.startswith("asc-") or "pos-" in slug:
                market_type = "spread"
            elif slug.startswith("tsc-") or "pt5" in slug:
                market_type = "totals"
            else:
                market_type = "moneyline"
            trade_db.insert_trade(
                slug=slug,
                market_id=position.market_id,
                side=position.side,
                entry_price=position.entry_price,
                close_price=current_price,
                quantity=position.quantity,
                size_usd=position.size_usd,
                realized_pnl=realized_pnl,
                close_reason=reason,
                market_type=market_type,
                entry_time=position.entry_time,
                close_time=close_time,
            )
        except Exception:
            pass  # DB errors must not break trading

        return realized_pnl

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == "open"]

    def get_total_exposure(self) -> float:
        """Get total USD exposure across open positions."""
        return sum(p.size_usd for p in self.get_open_positions())

    def get_total_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(p.realized_pnl for p in self.positions if p.status == "closed")

    def get_equity(self) -> float:
        """Get current equity (bankroll + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self.get_open_positions())
        return self.bankroll + unrealized

    def record_equity(self, timestamp: Optional[datetime] = None):
        """Record current equity for the equity curve."""
        self.equity_curve.append(self.get_equity())
        self.timestamps.append(timestamp or datetime.now(timezone.utc))

    def get_stats(self) -> Dict:
        """Compute portfolio statistics."""
        closed = [p for p in self.positions if p.status == "closed"]

        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl <= 0]

        return {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(p.realized_pnl for p in closed),
            "avg_pnl": sum(p.realized_pnl for p in closed) / len(closed),
            "avg_win": sum(p.realized_pnl for p in wins) / len(wins) if wins else 0,
            "avg_loss": sum(p.realized_pnl for p in losses) / len(losses) if losses else 0,
        }
