"""Cost-aware edge accounting: what the edge is worth AFTER fees and spread.

The estimator produces a "gross" edge against the snapshot price (usually the
mid/last price). That number overstates the real opportunity because:

  1. We execute with IOC taker orders, so a BUY fills at the best ask and a
     SELL fills at the best bid — we always pay the spread, never earn it.
  2. The exchange charges a quadratic taker fee per contract:
     fee = Θ·contracts·price·(1−price), Θ = trading.taker_fee_coefficient
     (Polymarket US schedule effective 2026-07-01; see bot/strategies/fees.py).

This module converts a gross probability estimate into a NET edge measured at
the executable price with fees included. Everything downstream (trade filter,
Kelly sizing, decision log) should reason about the net numbers.

Definitions (YES-share prices in [0,1], p = our probability estimate):

  BUY  (long YES):  cost per share = ask * (1 + fee)
                    net_edge = p - ask * (1 + fee)
                    A share pays $1 with probability p, so this is the expected
                    profit per share per $1 of payout.

  SELL (short YES): proceeds per share = bid * (1 - fee)
                    net_edge = bid * (1 - fee) - p
                    Shorting a YES share at `bid` means we keep the proceeds and
                    pay $1 with probability p; equivalently it's buying NO at
                    (1 - bid).

Using the executable side of the book instead of the mid automatically charges
us half the spread; the fee term charges the exchange's cut.
"""

from dataclasses import dataclass
from typing import Optional

from utils.models import MarketSnapshot


@dataclass
class EdgeBreakdown:
    """Full cost decomposition for one candidate trade."""
    side: str                 # "buy" or "sell"
    estimated_prob: float     # our probability estimate for YES
    mid_price: float          # snapshot price the gross edge was computed against
    exec_price: float         # price we would actually fill at (ask for buy, bid for sell)
    spread: float             # best_ask - best_bid (absolute price units); 0 if book empty
    fee_rate: float           # quadratic fee coefficient Θ (see fees.py)
    gross_edge: float         # |estimated_prob - mid_price|
    net_edge: float           # edge after crossing the spread and paying the fee


def executable_price(snapshot: MarketSnapshot, side: str) -> float:
    """The price an IOC taker order would fill at, falling back to snapshot price.

    Buys lift the best ask; sells hit the best bid. When the order book is
    missing (e.g. historical replays with no book data) we fall back to the
    snapshot price, which makes net_edge degrade gracefully to gross - fee.
    """
    ob = snapshot.order_book
    if side == "buy":
        ask = ob.best_ask if ob else None
        return ask if ask and ask > 0 else snapshot.price
    bid = ob.best_bid if ob else None
    return bid if bid and bid > 0 else snapshot.price


def book_spread(snapshot: MarketSnapshot) -> float:
    """Absolute bid-ask spread, or 0.0 when either side of the book is empty."""
    ob = snapshot.order_book
    if not ob:
        return 0.0
    bid, ask = ob.best_bid, ob.best_ask
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 0.0
    return max(0.0, ask - bid)


def compute_edge_breakdown(
    estimated_prob: float,
    snapshot: MarketSnapshot,
    side: str,
    fee_coefficient: float,
) -> EdgeBreakdown:
    """Compute the net, executable edge for a candidate trade.

    net_edge > 0 means the trade has positive expected value at the price we
    would actually pay, after fees. This is the number the trade filter gates
    on and the number Kelly sizing should be driven by.

    Fees use the Polymarket US quadratic schedule (bot/strategies/fees.py):
    fee per contract = Θ·p·(1−p), NOT a flat percentage of notional. The
    difference is material: at a 50¢ price the taker fee is 3% of notional,
    at 85¢ it's ~0.9% — a flat 2% model misprices both.
    """
    from bot.strategies.fees import fee_per_contract

    exec_px = executable_price(snapshot, side)
    exec_px = min(max(exec_px, 0.01), 0.99)
    spread = book_spread(snapshot)
    fee_share = fee_per_contract(exec_px, fee_coefficient)

    if side == "buy":
        # Pay (ask + fee) per share, receive $1 with probability p.
        net = estimated_prob - (exec_px + fee_share)
    else:
        # Receive (bid - fee) per share, pay $1 with probability p.
        net = (exec_px - fee_share) - estimated_prob

    return EdgeBreakdown(
        side=side,
        estimated_prob=estimated_prob,
        mid_price=snapshot.price,
        exec_price=exec_px,
        spread=spread,
        fee_rate=fee_coefficient,   # carries the coefficient Θ (see fees.py)
        gross_edge=abs(estimated_prob - snapshot.price),
        net_edge=net,
    )
