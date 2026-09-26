"""Maker strategy: quote inside wide pregame books, priced off the sportsbooks.

Polymarket US pays makers (rebate 0.0125 * C * p * (1 - p), credited at the
fill) and charges takers 0.0695 on the same curve, so at 50c every taker
strategy starts about 3.5% of notional behind per side while a maker starts
0.3% ahead. The venue's own market maker keeps every liquid market at
0.5c to 2c wide (pregame NFL moneylines: half a cent, six figures a side),
so there is no room there; the room is in live alternate spreads and totals,
where the median book is 5c wide and a quarter are 30c or wider. A quote
resting inside the model's fair value on each side is the best price in
those books, fills whenever a taker crosses, and earns the spread plus the
rebate. A fill is inventory held to settlement unless the opposite quote
fills first, which books the spread and flattens the position.

Paper fill model (no order is ever sent): every streamed book snapshot
carries the market's last trade (price, size, time) and cumulative shares
traded. Our resting bid at B was the best bid in the market, so a trade
printing at or below B was a seller who would have hit us first; a print at
or above our ask A was a buyer who would have lifted us. A new best ask at
or below B (a seller posting through us) fills the bid the same way. Filled
size is at most the shares the tape shows. Nothing is credited that the tape
does not show, and no fill is credited between our two quotes.

Pregame quotes refresh on every full scan; live quotes refresh every 30 s off
the live lines model (clock-scaled sigma) and are pulled with 5 minutes left
or when the game ends. Fills are polled every loop cycle from the leader's
mirrored books, so this process never spends the REST book budget.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from bot.signals.lines import parse_line_slug
from bot.strategies.fees import booked_fee_usd
from utils.models import Trade, TradeSignal

TICK = 0.01
MIRROR_MAX_AGE = 6 * 3600.0   # the leader's feed status decides freshness, not the file age


@dataclass
class Quote:
    slug: str
    kind: str
    market: dict
    fair: float
    num_books: int
    source: str
    bid: Optional[float]
    ask: Optional[float]
    bid_contracts: int
    ask_contracts: int
    size_usd: float
    game_start: Optional[datetime]
    shares_traded: float
    last_trade_time: str
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    best_bid: float = 0.0
    best_ask: float = 0.0
    is_live: bool = False


def quote_prices(fair: float, best_bid: float, best_ask: float, half_spread: float,
                 min_edge: float) -> Tuple[Optional[float], Optional[float]]:
    """Our bid and ask: fair -/+ half_spread, moved inside the existing best
    when the book is already tighter, and dropped on a side where that
    leaves less than min_edge between the quote and fair value."""
    bid = round(fair - half_spread, 2)
    if bid <= best_bid:
        bid = round(best_bid + TICK, 2)
    if bid >= best_ask or fair - bid < min_edge - 1e-9 or bid < 0.02:
        bid = None
    ask = round(fair + half_spread, 2)
    if ask >= best_ask:
        ask = round(best_ask - TICK, 2)
    if ask <= best_bid or ask - fair < min_edge - 1e-9 or ask > 0.98:
        ask = None
    if bid is not None and ask is not None and bid >= ask:
        ask = None
    return bid, ask


def tape(payload: dict) -> Tuple[float, Optional[float], float, str]:
    """(shares traded so far, last trade price, last trade size, last trade time)."""
    stats = payload.get("stats") or {}

    def num(x):
        try:
            return float(x["value"] if isinstance(x, dict) else x)
        except (TypeError, ValueError, KeyError):
            return None
    shares = num(stats.get("sharesTraded")) or 0.0
    return (shares, num(stats.get("lastTradePx")), num(stats.get("lastTradeQty")) or 0.0,
            str(stats.get("lastTradeSetTime") or ""))


class Maker:
    def __init__(self, bot):
        self.bot = bot
        self.quotes: Dict[str, Quote] = {}
        self._wide_since: Dict[str, float] = {}

    # ── fair value ────────────────────────────────────────────────────────

    def fair_value(self, market: dict, is_live: bool = False) -> Optional[Tuple[float, int, str]]:
        slug = market.get("slug", "") or ""
        if slug.startswith(("asc-", "tsc-")):
            parsed = parse_line_slug(slug)
            if not parsed:
                return None
            priced = self.bot.lines_cache.price_market(parsed, is_live)
            if not priced or (is_live and not priced.get("clock_known", True)):
                return None
            return float(priced["prob"]), int(priced["num_books"]), "lines_live" if is_live else "lines"
        if is_live:
            return None     # live moneylines: the maker keeps them 1c wide, and pregame odds are stale
        if slug.startswith("aec-"):
            got = self.bot.odds_cache.get_probability_for_slug(slug)
            if not got:
                return None
            return float(got[0]), int(got[1]), "consensus"
        return None

    # ── mirrored books ────────────────────────────────────────────────────

    def raw_book(self, slug: str):
        """(OrderBook, payload) from the leader's mirror, or None when the
        leader is not streaming this market (a stale mirror shows no trades
        and must not be read as a quiet market)."""
        md = self.bot.market_data
        feed = getattr(md, "book_feed", None)
        payload = feed.get_book(slug) if feed is not None else None
        if payload is None:
            if not md._leader_streams(slug):
                return None
            data = md._shared_read(os.path.join("books", f"{slug}.json"), MIRROR_MAX_AGE)
            payload = (data or {}).get("marketData") if isinstance(data, dict) else None
        if not payload:
            return None
        return md._parse_us_book({"marketData": payload}), payload

    # ── quoting ───────────────────────────────────────────────────────────

    def _live_pull_due(self, slug: str) -> bool:
        """Live quotes come off with maker_live_pull_seconds left or once the game ends."""
        from bot.certainty import seconds_left
        game_state = getattr(self.bot, "game_state", None)
        if game_state is None:
            return True
        try:
            gs = game_state.state_for(slug)
        except Exception:
            return True
        if not gs or gs["state"] != "in":
            return True
        left = seconds_left(gs)
        return left is None or left <= float(self.bot.config.trading.maker_live_pull_seconds)

    def refresh(self, markets: List[dict], live_only: bool = False) -> None:
        """Re-place quotes for `markets`. live_only=True touches only in-game
        quotes (called every 30 s); the full refresh runs on every full scan."""
        bot, cfg = self.bot, self.bot.config.trading
        now = datetime.now(timezone.utc)
        if getattr(bot, "entries_paused_until", None) is not None or bot._entries_capped():
            self._pull_all("entries_blocked")
            return
        open_by_slug = {p.slug: p for p in bot.portfolio.get_open_positions()}
        kinds = {k.strip().lower() for k in str(getattr(cfg, "market_kinds", "") or "").split(",")
                 if k.strip()}
        pull_before = timedelta(minutes=float(cfg.maker_pull_minutes))
        candidates = []
        for m in markets:
            slug = m.get("slug", "") or ""
            kind = bot._market_kind(slug)
            if not slug or (kinds and kind not in kinds):
                continue
            start = bot.market_data._parse_datetime(m.get("gameStartTime")) if m.get("gameStartTime") else None
            if start is None:
                continue
            is_live = start <= now
            if live_only and not is_live:
                continue
            if is_live:
                if not getattr(cfg, "maker_live_quotes", False) or self._live_pull_due(slug):
                    continue
            elif start - now <= pull_before:
                continue
            fv = self.fair_value(m, is_live)
            if not fv or fv[1] < int(cfg.maker_min_books):
                continue
            fair, num_books, source = fv
            if not (cfg.maker_min_fair <= fair <= cfg.maker_max_fair):
                continue
            raw = self.raw_book(slug)
            if not raw:
                continue
            book, payload = raw
            if not book.bids or not book.asks:
                continue
            best_bid, best_ask = book.bids[0].price, book.asks[0].price
            # Only books that have stayed wide: a gap the market maker left
            # for one play closes on the far side of whoever quoted into it.
            wide = best_ask - best_bid >= 2 * float(cfg.maker_half_spread) + 2 * TICK
            if not wide:
                self._wide_since.pop(slug, None)
                continue
            since = self._wide_since.setdefault(slug, now.timestamp())
            if slug not in self.quotes and now.timestamp() - since < float(cfg.maker_wide_seconds):
                continue
            bid, ask = quote_prices(fair, best_bid, best_ask, float(cfg.maker_half_spread),
                                    float(cfg.maker_min_edge))
            pos = open_by_slug.get(slug)
            if pos is not None:            # holding: only work the side that flattens
                if pos.side == "buy":
                    bid = None
                else:
                    ask = None
            if bid is None and ask is None:
                continue
            candidates.append((best_ask - best_bid, slug, kind, m, fair, num_books, source,
                               bid, ask, best_bid, best_ask, start, payload, is_live))
        candidates.sort(key=lambda c: -c[0])
        keep: Dict[str, Quote] = {s: q for s, q in self.quotes.items() if live_only and not q.is_live}
        size = float(cfg.maker_size_usd)
        room = max(0, int(cfg.maker_max_markets) - len(keep))
        for width, slug, kind, m, fair, num_books, source, bid, ask, bb, ba, start, payload, is_live in \
                candidates[:room]:
            q = self.quotes.get(slug)
            shares, _, _, last_time = tape(payload)
            if q is not None and q.bid == bid and q.ask == ask and abs(q.fair - fair) < 0.005:
                q.best_bid, q.best_ask, q.market = bb, ba, m
                keep[slug] = q
                continue
            q = Quote(slug=slug, kind=kind, market=m, fair=fair, num_books=num_books, source=source,
                      bid=bid, ask=ask, bid_contracts=int(size / bid) if bid else 0,
                      ask_contracts=int(size / (1.0 - ask)) if ask else 0, size_usd=size,
                      game_start=start, shares_traded=shares, last_trade_time=last_time,
                      best_bid=bb, best_ask=ba, is_live=is_live)
            keep[slug] = q
            bot.logger.info("maker_quote", {"slug": slug, "fair": round(fair, 4), "books": num_books,
                                            "source": source, "bid": bid, "ask": ask, "live": is_live,
                                            "best_bid": bb, "best_ask": ba, "size_usd": size})
            self._log(q, "quote")
        for slug, q in self.quotes.items():
            if slug not in keep:
                bot.logger.info("maker_quote_pulled", {"slug": slug, "bid": q.bid, "ask": q.ask, "live": q.is_live})
                self._log(q, "pull")
        self.quotes = keep

    def _pull_all(self, why: str) -> None:
        if self.quotes:
            self.bot.logger.info("maker_quotes_pulled", {"count": len(self.quotes), "why": why})
            for q in self.quotes.values():
                self._log(q, "pull")
        self.quotes = {}

    # ── fills ─────────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Credit fills the tape shows since the last poll."""
        for slug, q in list(self.quotes.items()):
            if not q.is_live and q.game_start is not None and q.game_start <= datetime.now(timezone.utc):
                self.bot.logger.info("maker_quote_pulled", {"slug": slug, "why": "kickoff"})
                self._log(q, "pull")
                del self.quotes[slug]
                continue
            raw = self.raw_book(slug)
            if not raw:
                continue
            book, payload = raw
            shares, last_px, last_qty, last_time = tape(payload)
            if last_time and last_time != q.last_trade_time:
                traded = max(shares - q.shares_traded, last_qty, 0.0)
                q.shares_traded, q.last_trade_time = shares, last_time
                if last_px is not None:
                    if q.bid is not None and last_px <= q.bid:
                        self._fill(q, "buy", traded, "trade_at_or_below_bid", last_px)
                        continue
                    if q.ask is not None and last_px >= q.ask:
                        self._fill(q, "sell", traded, "trade_at_or_above_ask", last_px)
                        continue
            else:
                q.shares_traded = max(q.shares_traded, shares)
            if q.bid is not None and book.asks and book.asks[0].price <= q.bid:
                self._fill(q, "buy", book.asks[0].size, "ask_crossed_bid", book.asks[0].price)
            elif q.ask is not None and book.bids and book.bids[0].price >= q.ask:
                self._fill(q, "sell", book.bids[0].size, "bid_crossed_ask", book.bids[0].price)

    def _fill(self, q: Quote, side: str, available: float, why: str, print_px: float) -> None:
        bot, cfg = self.bot, self.bot.config.trading
        price = q.bid if side == "buy" else q.ask
        wanted = q.bid_contracts if side == "buy" else q.ask_contracts
        qty = int(min(wanted, available))
        if price is None or qty < 1:
            return
        now = datetime.now(timezone.utc)
        rebate = booked_fee_usd(qty, price, float(cfg.maker_fee_coefficient))   # negative: paid to us
        pos = next((p for p in bot.portfolio.get_open_positions() if p.slug == q.slug), None)
        if pos is not None and pos.side == side:
            return                                          # never add to inventory
        if pos is not None:                                 # flip: the opposite quote flattens us
            close_qty = min(qty, pos.quantity)
            if close_qty < pos.quantity - 0.5:
                pnl = bot.portfolio.close_partial(pos, close_qty, price, "maker_flip",
                                                  booked_fee_usd(close_qty, price, float(cfg.maker_fee_coefficient)))
            else:
                pnl = bot.portfolio.close_position(pos, price, "maker_flip", exit_fees=rebate)
            bot.risk.record_pnl(pnl)
            bot.logger.info("maker_fill", {"slug": q.slug, "side": side, "price": price, "contracts": close_qty,
                                           "why": why, "print": print_px, "flip_pnl": round(pnl, 2),
                                           "rebate": round(-rebate, 2)})
            self._log(q, f"flip_{side}", price, close_qty)
        else:
            if not bot.risk.can_open_position(bot.portfolio.get_open_positions()):
                return
            collateral = price if side == "buy" else 1.0 - price
            room = float(cfg.max_portfolio_exposure_usd) - float(bot.portfolio.get_total_exposure())
            cash = float(bot.portfolio.bankroll) - 1.0
            qty = int(min(qty, room / collateral, cash / collateral))
            if qty < 1:
                return
            rebate = booked_fee_usd(qty, price, float(cfg.maker_fee_coefficient))
            size_usd = round(qty * collateral, 2)
            sig = TradeSignal(market_id=str(q.market.get("id") or q.slug), token_id=q.slug, side=side,
                              estimated_prob=q.fair, market_price=price,
                              edge=(q.fair - price) if side == "buy" else (price - q.fair),
                              position_size_usd=size_usd, slug=q.slug, timestamp=now)
            sig.exec_price = price
            sig.spread = round(q.best_ask - q.best_bid, 4)
            sig._question = q.market.get("question", "")
            sig._is_live = False
            trade = Trade(market_id=sig.market_id, token_id=q.slug, side=side, price=price, quantity=qty,
                          size_usd=size_usd, timestamp=now, trade_type="entry", fees=rebate, is_paper=True)
            try:
                bot.portfolio.open_position(sig, trade)
            except ValueError as e:
                bot.logger.warning("maker_fill_refused", {"slug": q.slug, "error": str(e)[:120]})
                return
            bot.risk.record_trade_opened()
            snapshot = None
            try:
                snapshot = bot.market_data.build_snapshot(q.market, fetch_book=False)
            except Exception:
                snapshot = None
            if snapshot:
                bot._log_decision(sig, snapshot, "executed", f"maker_{side}")
            bot.logger.info("maker_fill", {"slug": q.slug, "side": side, "price": price, "contracts": qty,
                                           "size_usd": size_usd, "rebate": round(-rebate, 2), "why": why,
                                           "print": print_px, "fair": round(q.fair, 4),
                                           "daily_trade": bot.risk.daily_trade_count})
            self._log(q, f"fill_{side}", price, qty)
        if side == "buy":
            q.bid = None
        else:
            q.ask = None
        if q.bid is None and q.ask is None:
            self.quotes.pop(q.slug, None)

    # ── measurement ───────────────────────────────────────────────────────

    def _log(self, q: Quote, event: str, price: Optional[float] = None, contracts: float = 0.0) -> None:
        try:
            from bot.trade_db import insert_maker_log
            insert_maker_log(datetime.now(timezone.utc).isoformat(), q.slug, q.kind, q.fair, q.num_books,
                             q.best_bid, q.best_ask, q.bid, q.ask, event, price, contracts)
        except Exception:
            pass
