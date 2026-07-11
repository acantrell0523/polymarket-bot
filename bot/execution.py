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

        # Simulate fees on the US quadratic schedule, banker's-rounded like
        # the exchange books them (single source: bot/strategies/fees.py)
        from bot.strategies.fees import booked_fee_usd
        coef = getattr(self.config, "taker_fee_coefficient", 0.06)
        fees = booked_fee_usd(quantity, price, coef)

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

    # Execution payloads vary across SDK versions — try these quantity and
    # price field names in order. A fill we cannot parse is treated as NOT
    # filled for position-building purposes (never assume-full).
    _EXEC_QTY_FIELDS = ("quantity", "qty", "filledQuantity", "filled_quantity",
                        "size", "amount")
    _EXEC_PRICE_FIELDS = ("price", "executionPrice", "execution_price",
                          "fillPrice", "fill_price", "avgPrice")

    @staticmethod
    def _parse_number(value) -> Optional[float]:
        """Numbers arrive as float, int, str, or {'value': '0.52', ...}."""
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def reconcile_executions(cls, executions, limit_price: float):
        """Turn an SDK execution list into (filled_qty, vwap_price).

        Rules (each one is an audited failure mode of the old code):
          * Quantities are FLOATS — int() truncated a 2.5-share fill to 2.
          * Unknown field names contribute NOTHING — the old code summed to 0
            and then fell back to the FULL requested quantity, turning an
            unparseable partial fill into a phantom full fill.
          * Fill price is the execution VWAP, falling back to the limit price
            per-execution only when the payload carries no price.

        Returns (0.0, limit_price) when nothing parseable filled.
        """
        filled_qty = 0.0
        notional = 0.0
        for e in executions or []:
            if not isinstance(e, dict):
                continue
            qty = None
            for f in cls._EXEC_QTY_FIELDS:
                if f in e:
                    qty = cls._parse_number(e[f])
                    if qty is not None:
                        break
            if not qty or qty <= 0:
                continue
            px = None
            for f in cls._EXEC_PRICE_FIELDS:
                if f in e:
                    px = cls._parse_number(e[f])
                    if px is not None:
                        break
            if px is None or px <= 0:
                px = limit_price
            filled_qty += qty
            notional += qty * px

        vwap = notional / filled_qty if filled_qty > 0 else limit_price
        return filled_qty, vwap

    def _live_execute(self, signal: TradeSignal, trade_type: str) -> Optional[Trade]:
        """Live trade execution via Polymarket US SDK.

        Order construction rules:
          * Limit price is the EXECUTABLE price (signal.exec_price = best ask
            for buys / best bid for sells, set by compute_edge_breakdown).
            The old code submitted the mid — a buy IOC limit below the ask
            mostly can't fill, and the intended price protection was fiction.
          * IOC so we never rest in the book.
          * automaticOrder: true — exchange-required indicator that this
            order was placed by an automated system.

        Position building rules: the Trade reflects ONLY parseable fills
        (see reconcile_executions). Zero parseable fills => no Trade.
        """
        if not self._client:
            if self.logger:
                self.logger.error("live_trading_no_client", {
                    "message": "Polymarket US SDK not initialized — check API credentials"
                })
            return None

        # Executable price, never the mid. No exec_price on the signal =>
        # the caller skipped compute_edge_breakdown; refuse rather than guess.
        price = signal.exec_price if signal.exec_price > 0 else 0.0
        if price <= 0:
            if self.logger:
                self.logger.error("live_trade_no_exec_price", {
                    "market_id": signal.market_id,
                    "message": "signal missing executable price; refusing to submit at mid",
                })
            return None

        size_usd = signal.position_size_usd
        if size_usd <= 0:
            size_usd = self.config.min_position_size_usd

        quantity = int(size_usd / price)
        if quantity <= 0:
            if self.logger:
                self.logger.warning("live_trade_zero_quantity", {
                    "market_id": signal.market_id,
                    "price": price,
                    "size_usd": size_usd,
                })
            return None

        market_slug = signal.slug or signal.market_id
        intent = "ORDER_INTENT_BUY_LONG" if signal.side == "buy" else "ORDER_INTENT_SELL_LONG"

        try:
            result = self._client.orders.create({
                "marketSlug": market_slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": f"{price:.2f}", "currency": "USD"},
                "quantity": quantity,
                "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                # CFTC/exchange requirement: flag orders placed by automated
                # trading systems. Do not remove.
                "automaticOrder": True,
            })
        except Exception as e:
            if self.logger:
                self.logger.error("live_trade_failed", {
                    "market_id": signal.market_id,
                    "slug": market_slug,
                    "error": str(e),
                })
            return None

        order_id = ""
        executions = []
        if isinstance(result, dict):
            order_id = result.get("id", "")
            executions = result.get("executions", [])

        filled_qty, fill_vwap = self.reconcile_executions(executions, price)

        if filled_qty <= 0:
            # No parseable fills — NO position. If the payload had executions
            # we couldn't read, log loudly: that's an SDK contract change.
            if self.logger:
                event = ("live_trade_unparseable_executions" if executions
                         else "live_trade_no_fill")
                self.logger.warning(event, {
                    "market_id": signal.market_id,
                    "slug": market_slug,
                    "side": signal.side,
                    "price": price,
                    "requested_qty": quantity,
                    "order_id": order_id,
                    "raw_executions": str(executions)[:500],
                    "message": "no position created",
                })
            return None

        filled_qty = min(filled_qty, float(quantity))  # never exceed request
        filled_size = filled_qty * fill_vwap

        from bot.strategies.fees import booked_fee_usd
        coef = getattr(self.config, "taker_fee_coefficient", 0.06)
        fees = booked_fee_usd(filled_qty, fill_vwap, coef)

        trade = Trade(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            price=fill_vwap,
            quantity=filled_qty,
            size_usd=filled_size,
            timestamp=signal.timestamp or datetime.now(timezone.utc),
            trade_type=trade_type,
            fees=fees,
            slippage=abs(fill_vwap - price),
            is_paper=False,
        )

        if self.logger:
            self.logger.info("live_trade_filled", {
                "market_id": signal.market_id,
                "slug": market_slug,
                "side": signal.side,
                "intent": intent,
                "limit_price": price,
                "fill_vwap": round(fill_vwap, 4),
                "requested_qty": quantity,
                "filled_qty": filled_qty,
                "partial": filled_qty < quantity,
                "size_usd": round(filled_size, 2),
                "fees": fees,
                "edge": signal.edge,
                "order_id": order_id,
                "executions": len(executions),
            })

        return trade

    def get_exchange_positions(self) -> Optional[dict]:
        """Fetch actual positions from the exchange.

        Returns {slug: position_data} on success ({} = genuinely no positions),
        or None on API FAILURE. Callers must treat None as "cannot verify" —
        the old contract returned {} for both, which made a transient API
        error indistinguishable from "position settled" and let the bot
        silently abandon real open positions.
        """
        if not self._client:
            return None
        try:
            result = self._client.portfolio.positions()
            return result.get("positions", {}) if isinstance(result, dict) else {}
        except Exception as e:
            if self.logger:
                self.logger.error("fetch_exchange_positions_failed", {"error": str(e)})
            return None

    def _is_market_resolved(self, slug: str) -> bool:
        """Check with the exchange whether a market is closed/resolved.

        Used before EVER concluding a position will "auto-settle" — the old
        code inferred resolution from error-message keywords and unfilled
        IOC attempts, both of which also match plain API trouble. Unknown
        (network failure, unexpected payload) => False: keep managing the
        position rather than orphaning it.
        """
        try:
            import requests as _requests
            resp = _requests.get(
                f"https://gateway.polymarket.us/v1/markets/{slug}", timeout=10
            )
            if resp.status_code == 404:
                return True  # market no longer exists — settled/removed
            if resp.status_code != 200:
                return False
            data = resp.json()
            market = data.get("market", data) if isinstance(data, dict) else {}
            status = str(market.get("status", "")).lower()
            return bool(
                market.get("closed") or market.get("resolved")
                or status in ("resolved", "closed", "settled", "finalized")
            )
        except Exception:
            return False

    # Seconds to wait before verifying a close order took effect.
    # Class attribute so tests can zero it.
    CLOSE_VERIFY_DELAY_SECONDS = 1.0

    # Close intents mirror our OPEN intents: this bot opens longs with
    # BUY_LONG and opens shorts with SELL_LONG, so closing is the inverse
    # order on the same (LONG) instrument. The old code sent BUY_SHORT to
    # close a short — a different instrument — at $0.01, a buy limit that
    # can never take the ask: shorts were unclosable.
    _CLOSE_LONG = {"intent": "ORDER_INTENT_SELL_LONG", "price": "0.01"}   # take any bid
    _CLOSE_SHORT = {"intent": "ORDER_INTENT_BUY_LONG", "price": "0.99"}   # take any ask

    def close_position(self, position: Position) -> bool:
        """Close a position via the Polymarket US SDK.

        Returns True ONLY when one of these is verified:
          * the exchange confirms the position is gone after our close order,
          * the exchange reports we hold no such position (successful fetch),
          * the market is confirmed resolved (position settles on its own).
        API errors NEVER count as success — False means "still ours to manage"
        and the trading loop retries next cycle.
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

        # 1. What does the exchange say we hold?
        exchange_positions = self.get_exchange_positions()
        if exchange_positions is None:
            if self.logger:
                self.logger.warning("close_position_fetch_failed", {
                    "slug": slug,
                    "message": "cannot verify holdings; will retry (NOT abandoning)",
                })
            return False

        if slug not in exchange_positions:
            # Successful fetch, position genuinely absent — settled or never
            # existed. Safe to stop managing.
            if self.logger:
                self.logger.info("close_position_not_on_exchange", {
                    "slug": slug,
                    "message": "exchange reports no position; treating as settled",
                })
            return True

        ex_pos = exchange_positions[slug]
        net_qty = self._parse_number(ex_pos.get("netPosition")) or 0.0
        qty_available = self._parse_number(ex_pos.get("qtyAvailable"))
        if qty_available is None:
            qty_available = abs(net_qty)

        if net_qty == 0:
            return True

        params = self._CLOSE_LONG if net_qty > 0 else self._CLOSE_SHORT
        close_qty = int(abs(qty_available))
        if close_qty <= 0:
            if self.logger:
                self.logger.warning("close_position_nothing_available", {
                    "slug": slug, "net": net_qty,
                    "message": "netPosition nonzero but qtyAvailable 0; retrying later",
                })
            return False

        # 2. Submit the aggressive IOC close
        try:
            result = self._client.orders.create({
                "marketSlug": slug,
                "intent": params["intent"],
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": params["price"], "currency": "USD"},
                "quantity": close_qty,
                "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                "automaticOrder": True,
            })
        except Exception as order_err:
            # Error text often hints at resolution, but hints are not proof —
            # confirm with the market status endpoint before auto-settling.
            if self._is_market_resolved(slug):
                if self.logger:
                    self.logger.info("close_position_market_resolved", {
                        "slug": slug,
                        "message": "market confirmed resolved; position will settle",
                        "error": str(order_err),
                    })
                self._last_close_was_auto_settle = True
                return True
            if self.logger:
                self.logger.error("close_position_order_failed", {
                    "slug": slug, "error": str(order_err),
                    "message": "close order failed and market NOT resolved; retrying",
                })
            return False

        if self.logger:
            self.logger.info("close_position_submitted", {
                "slug": slug,
                "intent": params["intent"],
                "quantity": close_qty,
            })

        # 3. Verify the position is actually gone
        import time
        if self.CLOSE_VERIFY_DELAY_SECONDS > 0:
            time.sleep(self.CLOSE_VERIFY_DELAY_SECONDS)
        remaining = self.get_exchange_positions()
        if remaining is None:
            if self.logger:
                self.logger.warning("close_position_verify_failed", {
                    "slug": slug,
                    "message": "close submitted but verification fetch failed; retrying",
                })
            return False

        remaining_net = 0.0
        if slug in remaining:
            remaining_net = self._parse_number(remaining[slug].get("netPosition")) or 0.0

        if remaining_net == 0:
            self._close_failures = getattr(self, "_close_failures", {})
            self._close_failures.pop(slug, None)
            if self.logger:
                self.logger.info("close_position_verified", {
                    "slug": slug,
                    "message": "position confirmed closed on exchange",
                })
            return True

        # Still holding some or all of it.
        filled_any = abs(remaining_net) < abs(net_qty)
        self._close_failures = getattr(self, "_close_failures", {})
        self._close_failures[slug] = 0 if filled_any else self._close_failures.get(slug, 0) + 1

        # Repeated zero-fill closes suggest a halted/resolving market — but
        # ONLY the market-status endpoint gets to make that call.
        if self._close_failures.get(slug, 0) >= 3:
            if self._is_market_resolved(slug):
                if self.logger:
                    self.logger.info("close_position_auto_settle", {
                        "slug": slug,
                        "attempts": self._close_failures[slug],
                        "message": "market confirmed resolved after repeated no-fills",
                    })
                self._close_failures.pop(slug, None)
                self._last_close_was_auto_settle = True
                return True
            if self.logger:
                self.logger.error("close_position_stuck", {
                    "slug": slug,
                    "attempts": self._close_failures[slug],
                    "remaining_net": remaining_net,
                    "message": "repeated no-fill closes on an ACTIVE market — "
                               "position still ours; check liquidity/halts",
                })

        if self.logger:
            self.logger.warning("close_position_incomplete", {
                "slug": slug,
                "original_net": net_qty,
                "remaining_net": remaining_net,
                "message": "position not fully closed; will retry",
            })
        return False
