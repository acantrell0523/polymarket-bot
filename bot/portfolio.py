"""Portfolio manager — thin wrapper around the Polymarket exchange API.

The exchange is the SINGLE source of truth for:
  - Cash balance (bankroll)
  - Open positions and their current value
  - Unrealized P&L
  - Total account value

Internal state is only used for:
  - Tracking which positions the BOT opened (vs manual)
  - Logging trades to SQLite for edge analysis
  - Alerting on trade events
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict
from utils.models import Position, Trade, TradeSignal
from utils.logger import TradingLogger
from bot import trade_db


class Portfolio:
    """Exchange-backed portfolio tracker. All financial data comes from the API."""

    def __init__(self, exchange_client=None, logger: Optional[TradingLogger] = None,
                 alerter=None, alert_config=None, paper_mode: bool = True,
                 initial_bankroll: float = 0.0):
        self._client = exchange_client
        self.logger = logger
        self.alerter = alerter
        self.alert_config = alert_config
        self.paper_mode = paper_mode

        # Paper mode fallback — only used when no exchange client
        self._paper_bankroll = initial_bankroll

        # Internal tracking for bot-opened positions (slug set)
        self._bot_positions: set = set()

        # Equity curve for charting
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

        # Cache exchange data to avoid hammering API within the same cycle
        self._balance_cache: Optional[float] = None
        self._positions_cache: Optional[Dict] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 2.0  # seconds

    # ------------------------------------------------------------------
    # Exchange API calls (source of truth)
    # ------------------------------------------------------------------

    def _refresh_cache(self):
        """Refresh cached exchange data if stale."""
        import time
        now = time.time()
        if now - self._cache_time < self._cache_ttl:
            return
        self._cache_time = now

        if not self._client:
            return

        try:
            bal = self._client.account.balances()
            balances = bal.get("balances", [])
            if balances:
                b = balances[0]
                # buyingPower = actual available cash (what the app shows)
                # currentBalance includes margin held for SHORT positions
                self._balance_cache = float(b.get("buyingPower", b.get("currentBalance", 0)))
                self._margin_held = float(b.get("marginRequirement", 0))
        except Exception:
            pass

        try:
            result = self._client.portfolio.positions()
            self._positions_cache = result.get("positions", {})
        except Exception:
            pass

    def invalidate_cache(self):
        """Force next access to re-fetch from exchange."""
        self._cache_time = 0

    @property
    def bankroll(self) -> float:
        """Cash balance from the exchange."""
        if self.paper_mode or not self._client:
            return self._paper_bankroll
        self._refresh_cache()
        return self._balance_cache if self._balance_cache is not None else 0.0

    @bankroll.setter
    def bankroll(self, value: float):
        """Paper mode bankroll setter."""
        self._paper_bankroll = value

    def get_exchange_positions(self) -> Dict:
        """Get raw position data from exchange."""
        if self.paper_mode or not self._client:
            return {}
        self._refresh_cache()
        return self._positions_cache or {}

    def get_open_positions(self) -> List[Position]:
        """Get open positions from the exchange."""
        if self.paper_mode or not self._client:
            return self._get_paper_positions()

        self._refresh_cache()
        positions = []
        for slug, p_data in (self._positions_cache or {}).items():
            net = int(p_data.get("netPosition", "0"))
            if net == 0:
                continue

            cost_val = float(p_data.get("cost", {}).get("value", "0"))
            cash_val = float(p_data.get("cashValue", {}).get("value", "0"))
            qty = abs(net)
            entry_price = cost_val / qty if qty > 0 else 0
            side = "buy" if net > 0 else "sell"

            pos = Position(
                market_id=slug,
                token_id=slug,
                side=side,
                entry_price=entry_price,
                size_usd=cost_val,
                quantity=qty,
                estimated_prob=0.5,
                entry_time=datetime.now(timezone.utc),
                current_price=cash_val / qty if qty > 0 else entry_price,
                slug=slug,
            )
            pos.unrealized_pnl = cash_val - cost_val
            positions.append(pos)

        return positions

    def get_total_exposure(self) -> float:
        """Total USD cost of open positions, from the exchange."""
        if self.paper_mode or not self._client:
            return sum(p.size_usd for p in self._get_paper_positions())

        self._refresh_cache()
        total = 0.0
        for slug, p_data in (self._positions_cache or {}).items():
            net = int(p_data.get("netPosition", "0"))
            if net != 0:
                total += float(p_data.get("cost", {}).get("value", "0"))
        return total

    def get_equity(self) -> float:
        """Total account value: cash + position value, from the exchange."""
        if self.paper_mode or not self._client:
            return self._paper_bankroll

        self._refresh_cache()
        cash = self._balance_cache or 0
        pos_value = 0.0
        for slug, p_data in (self._positions_cache or {}).items():
            pos_value += float(p_data.get("cashValue", {}).get("value", "0"))
        return cash + pos_value

    # ------------------------------------------------------------------
    # Paper mode helpers
    # ------------------------------------------------------------------

    _paper_positions: List[Position] = []

    def _get_paper_positions(self) -> List[Position]:
        return [p for p in self._paper_positions if p.status == "open"]

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_position(self, signal: TradeSignal, trade: Trade) -> Position:
        """Record a new position opened by the bot."""
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

        self._bot_positions.add(signal.slug)
        self.invalidate_cache()

        if self.paper_mode:
            self._paper_positions.append(position)
            self._paper_bankroll -= trade.fees

        if self.logger:
            self.logger.info("position_opened", {
                "market_id": signal.market_id,
                "side": signal.side,
                "entry_price": trade.price,
                "size_usd": trade.size_usd,
                "edge": signal.edge,
                "bankroll": round(self.bankroll, 2),
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
                       reason: str, timestamp: Optional[datetime] = None,
                       exchange_pnl: Optional[float] = None) -> float:
        """Record a position close. P&L comes from the exchange when available."""
        if position.status != "open":
            return 0.0

        close_time = timestamp or datetime.now(timezone.utc)
        self.invalidate_cache()

        # Get P&L from exchange (source of truth)
        if exchange_pnl is not None:
            realized_pnl = exchange_pnl
        else:
            # Fallback calculation for resolved markets
            realized_pnl = self._calculate_pnl(position, current_price, reason)

        position.status = "closed"
        position.close_reason = reason
        position.close_price = current_price
        position.close_time = close_time
        position.realized_pnl = realized_pnl

        if self.paper_mode:
            self._paper_bankroll += realized_pnl

        if self.logger:
            self.logger.info("position_closed", {
                "market_id": position.market_id,
                "reason": reason,
                "side": position.side,
                "entry_price": position.entry_price,
                "close_price": current_price,
                "pnl": round(realized_pnl, 2),
                "bankroll": round(self.bankroll, 2),
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

        # Persist to SQLite for edge analysis (not P&L source of truth)
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
            pass

        # Update edge log
        try:
            from bot.edge_log import update_edge_log_outcome
            slug = getattr(position, 'slug', position.market_id)
            time_held = 0.0
            if position.entry_time:
                entry = position.entry_time
                if entry.tzinfo is None:
                    entry = entry.replace(tzinfo=timezone.utc)
                time_held = (close_time - entry).total_seconds()
            update_edge_log_outcome(
                slug=slug,
                entry_time_iso=position.entry_time.isoformat() if position.entry_time else "",
                actual_pnl=realized_pnl,
                time_held_seconds=time_held,
                price_at_close=current_price,
                close_reason=reason,
                final_outcome="win" if realized_pnl > 0 else "loss",
            )
        except Exception:
            pass

        return realized_pnl

    def _calculate_pnl(self, position: Position, current_price: float, reason: str) -> float:
        """Fallback P&L calculation when exchange data unavailable."""
        if reason == "resolved":
            if current_price <= 0.01:
                if position.side == "sell":
                    return position.entry_price * position.quantity
                else:
                    return -position.entry_price * position.quantity
            elif current_price >= 0.99:
                if position.side == "buy":
                    return (1.0 - position.entry_price) * position.quantity
                else:
                    return -(1.0 - position.entry_price) * position.quantity

        if position.side == "buy":
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity
        return pnl

    # ------------------------------------------------------------------
    # Stats (for logging — NOT source of truth for P&L)
    # ------------------------------------------------------------------

    def record_equity(self, timestamp: Optional[datetime] = None):
        """Record current equity for the equity curve."""
        self.equity_curve.append(self.get_equity())
        self.timestamps.append(timestamp or datetime.now(timezone.utc))

    def get_stats(self) -> Dict:
        """Get stats from trade_db (exchange-persisted trades)."""
        from datetime import timedelta
        try:
            since = datetime.now(timezone.utc) - timedelta(days=1)
            trades = trade_db.get_trades_since(since)
            if not trades:
                return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                        "avg_pnl": 0.0, "winning_trades": 0, "losing_trades": 0,
                        "avg_win": 0, "avg_loss": 0}

            wins = [t for t in trades if t["realized_pnl"] > 0]
            losses = [t for t in trades if t["realized_pnl"] <= 0]
            total_pnl = sum(t["realized_pnl"] for t in trades)

            return {
                "total_trades": len(trades),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(trades),
                "avg_win": sum(t["realized_pnl"] for t in wins) / len(wins) if wins else 0,
                "avg_loss": sum(t["realized_pnl"] for t in losses) / len(losses) if losses else 0,
            }
        except Exception:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                    "avg_pnl": 0.0, "winning_trades": 0, "losing_trades": 0,
                    "avg_win": 0, "avg_loss": 0}
