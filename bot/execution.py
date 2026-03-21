"""Order execution (live via Polymarket US SDK, or paper)."""

import os
from datetime import datetime, timezone
from typing import Optional
from utils.models import Trade, TradeSignal, Position
from utils.config import BotConfig, TradingConfig
from utils.logger import TradingLogger


class ExecutionEngine:
    """Executes trades in paper or live mode."""

    def __init__(self, config, logger: Optional[TradingLogger] = None):
        self.logger = logger

        # Support both BotConfig and TradingConfig for backwards compatibility
        if isinstance(config, BotConfig):
            self.config = config.trading
            self._bot_config = config
        else:
            self.config = config
            self._bot_config = None

        # Initialize live client if not in paper mode and credentials exist
        self._client = None
        if not self.config.paper_trading and self._bot_config:
            key_id = self._bot_config.wallet.key_id
            secret_key = self._bot_config.wallet.secret_key
            if key_id and secret_key:
                try:
                    from polymarket_us import PolymarketUS
                    self._client = PolymarketUS(
                        key_id=key_id,
                        secret_key=secret_key,
                    )
                    if self.logger:
                        self.logger.info("polymarket_us_sdk_initialized", {})
                except Exception as e:
                    if self.logger:
                        self.logger.error("polymarket_us_sdk_init_failed", {
                            "error": str(e),
                        })
            else:
                if self.logger:
                    self.logger.warning("live_trading_missing_credentials", {
                        "message": "POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY required for live trading"
                    })

    def execute_trade(self, signal: TradeSignal, trade_type: str = "entry") -> Optional[Trade]:
        """
        Execute a trade based on a signal.

        In paper mode, simulates execution at the signal price.
        In live mode, submits an order via Polymarket US SDK.
        """
        if self.config.paper_trading:
            return self._paper_execute(signal, trade_type)
        else:
            return self._live_execute(signal, trade_type)

    def _paper_execute(self, signal: TradeSignal, trade_type: str) -> Trade:
        """Simulate trade execution."""
        price = signal.market_price
        size_usd = signal.position_size_usd

        if size_usd <= 0:
            size_usd = self.config.min_position_size_usd

        quantity = size_usd / price if price > 0 else 0

        # Simulate fees (taker fee)
        fee_rate = 0.02  # 2% taker fee
        fees = size_usd * fee_rate

        trade = Trade(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            price=price,
            quantity=quantity,
            size_usd=size_usd,
            timestamp=signal.timestamp or datetime.now(timezone.utc),
            trade_type=trade_type,
            fees=fees,
            slippage=0.0,
            is_paper=True,
        )

        if self.logger:
            self.logger.info("paper_trade_executed", {
                "market_id": signal.market_id,
                "side": signal.side,
                "price": price,
                "size_usd": size_usd,
                "edge": signal.edge,
            })

        return trade

    def _live_execute(self, signal: TradeSignal, trade_type: str) -> Optional[Trade]:
        """Live trade execution via Polymarket US SDK."""
        if not self._client:
            if self.logger:
                self.logger.error("live_trading_no_client", {
                    "message": "Polymarket US SDK not initialized — check API credentials"
                })
            return None

        price = signal.market_price
        size_usd = signal.position_size_usd
        if size_usd <= 0:
            size_usd = self.config.min_position_size_usd

        quantity = int(size_usd / price) if price > 0 else 0
        if quantity <= 0:
            if self.logger:
                self.logger.warning("live_trade_zero_quantity", {
                    "market_id": signal.market_id,
                    "price": price,
                    "size_usd": size_usd,
                })
            return None

        # Use slug from signal; fall back to market_id
        market_slug = signal.slug or signal.market_id

        if signal.side == "buy":
            intent = "ORDER_INTENT_BUY_LONG"
        else:
            intent = "ORDER_INTENT_SELL_LONG"

        try:
            result = self._client.orders.create({
                "marketSlug": market_slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": str(price), "currency": "USD"},
                "quantity": quantity,
                "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
            })

            # Check if the order actually filled
            order_id = None
            executions = []
            if isinstance(result, dict):
                order_id = result.get("id", "")
                executions = result.get("executions", [])

            if not executions:
                # Order submitted but nothing filled — do NOT create a position
                if self.logger:
                    self.logger.warning("live_trade_no_fill", {
                        "market_id": signal.market_id,
                        "slug": market_slug,
                        "side": signal.side,
                        "price": price,
                        "quantity": quantity,
                        "order_id": order_id,
                        "message": "IOC order got no fills, no position created",
                    })
                return None

            # Order filled (fully or partially)
            filled_qty = sum(int(e.get("quantity", e.get("qty", 0))) for e in executions)
            if filled_qty <= 0:
                filled_qty = quantity  # fallback if execution format differs

            filled_size = filled_qty * price
            fee_rate = 0.02
            fees = filled_size * fee_rate

            trade = Trade(
                market_id=signal.market_id,
                token_id=signal.token_id,
                side=signal.side,
                price=price,
                quantity=filled_qty,
                size_usd=filled_size,
                timestamp=signal.timestamp or datetime.now(timezone.utc),
                trade_type=trade_type,
                fees=fees,
                slippage=0.0,
                is_paper=False,
            )

            if self.logger:
                self.logger.info("live_trade_filled", {
                    "market_id": signal.market_id,
                    "slug": market_slug,
                    "side": signal.side,
                    "intent": intent,
                    "price": price,
                    "requested_qty": quantity,
                    "filled_qty": filled_qty,
                    "size_usd": filled_size,
                    "edge": signal.edge,
                    "order_id": order_id,
                    "executions": len(executions),
                })

            return trade

        except Exception as e:
            if self.logger:
                self.logger.error("live_trade_failed", {
                    "market_id": signal.market_id,
                    "slug": market_slug,
                    "error": str(e),
                })
            return None

    def get_exchange_positions(self) -> dict:
        """Fetch actual positions from the exchange. Returns {slug: position_data}."""
        if not self._client:
            return {}
        try:
            result = self._client.portfolio.positions()
            return result.get("positions", {}) if isinstance(result, dict) else {}
        except Exception as e:
            if self.logger:
                self.logger.error("fetch_exchange_positions_failed", {"error": str(e)})
            return {}

    def close_position(self, position: Position) -> bool:
        """Close a position by selling via the Polymarket US SDK.

        Returns True if the close order was submitted successfully.
        """
        slug = position.slug or position.market_id

        if self.config.paper_trading:
            if self.logger:
                self.logger.info("paper_position_closed", {
                    "market_id": position.market_id,
                    "slug": slug,
                    "side": position.side,
                })
            return True

        if not self._client:
            if self.logger:
                self.logger.error("close_position_no_client", {
                    "message": "Polymarket US SDK not initialized"
                })
            return False

        try:
            # Get the actual position size from the exchange
            exchange_positions = self.get_exchange_positions()
            if slug not in exchange_positions:
                if self.logger:
                    self.logger.warning("close_position_not_on_exchange", {
                        "slug": slug,
                        "message": "Position not on exchange, marking as abandoned",
                    })
                return True

            ex_pos = exchange_positions[slug]
            net_qty = int(ex_pos.get("netPosition", "0"))
            qty_available = int(ex_pos.get("qtyAvailable", str(abs(net_qty))))

            if net_qty == 0:
                return True

            # Determine intent: if we're long (net > 0), sell long. If short, buy short.
            if net_qty > 0:
                intent = "ORDER_INTENT_SELL_LONG"
                sell_qty = qty_available
            else:
                intent = "ORDER_INTENT_BUY_SHORT"
                sell_qty = abs(qty_available)

            # Submit aggressive IOC sell at $0.01 to sweep the book
            try:
                result = self._client.orders.create({
                    "marketSlug": slug,
                    "intent": intent,
                    "type": "ORDER_TYPE_LIMIT",
                    "price": {"value": "0.01", "currency": "USD"},
                    "quantity": sell_qty,
                    "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                })
            except Exception as order_err:
                err_msg = str(order_err).lower()
                # Market resolved/closed — position will settle automatically
                if any(k in err_msg for k in ("resolved", "closed", "settled", "not found", "inactive")):
                    if self.logger:
                        self.logger.info("close_position_market_resolved", {
                            "slug": slug,
                            "message": "Market resolved, position will auto-settle",
                            "error": str(order_err),
                        })
                    return True
                raise

            if self.logger:
                self.logger.info("close_position_sell_submitted", {
                    "slug": slug,
                    "intent": intent,
                    "quantity": sell_qty,
                    "order_result": str(result),
                })

            # Verify the position is gone
            import time
            time.sleep(1)
            remaining = self.get_exchange_positions()
            if slug in remaining:
                remaining_net = int(remaining[slug].get("netPosition", "0"))
                if remaining_net != 0:
                    # Check if the order had no executions — likely a resolved market
                    executions = result.get("executions", []) if isinstance(result, dict) else []
                    if not executions:
                        self._close_failures = getattr(self, "_close_failures", {})
                        self._close_failures[slug] = self._close_failures.get(slug, 0) + 1
                        if self._close_failures[slug] >= 3:
                            # 3 consecutive failed closes — market is resolved or illiquid
                            if self.logger:
                                self.logger.info("close_position_auto_settle", {
                                    "slug": slug,
                                    "attempts": self._close_failures[slug],
                                    "message": "3 failed closes, marking as auto-settling",
                                })
                            del self._close_failures[slug]
                            self._last_close_was_auto_settle = True
                            return True
                    if self.logger:
                        self.logger.warning("close_position_partial", {
                            "slug": slug,
                            "original_net": net_qty,
                            "remaining_net": remaining_net,
                            "message": "Position partially closed",
                        })
                    return False

            if self.logger:
                self.logger.info("close_position_verified", {
                    "slug": slug,
                    "message": "Position confirmed closed on exchange",
                })
            return True

        except Exception as e:
            err_str = str(e)
            if self.logger:
                self.logger.error("close_position_failed", {
                    "market_id": position.market_id,
                    "slug": slug,
                    "error": err_str,
                })
            if "not found" in err_str.lower() or "no position" in err_str.lower():
                return True
            return False
