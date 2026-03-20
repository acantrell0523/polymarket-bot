"""Order execution (live via Polymarket US SDK, or paper)."""

import os
from datetime import datetime, timezone
from typing import Optional
from utils.models import Trade, TradeSignal
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
                "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
            })

            # Estimate fees (taker)
            fee_rate = 0.02
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
                is_paper=False,
            )

            if self.logger:
                self.logger.info("live_trade_executed", {
                    "market_id": signal.market_id,
                    "slug": market_slug,
                    "side": signal.side,
                    "intent": intent,
                    "price": price,
                    "size_usd": size_usd,
                    "quantity": quantity,
                    "edge": signal.edge,
                    "order_result": str(result),
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
