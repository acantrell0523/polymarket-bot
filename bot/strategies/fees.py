"""Polymarket US fee schedule (effective 2026-07-01).

    Fee = Θ × C × p × (1 − p)

where C = contracts, p = trade price in dollars (0.01–0.99), and Θ is the
fee coefficient: takers pay Θ = 0.06, makers receive a rebate Θ = −0.0125.
Verified against docs.polymarket.us/fees on 2026-07-11.

Properties that matter for strategy math:
  * Quadratic in price, symmetric around $0.50 — the fee peaks at
    $1.50 per 100 contracts (p = 0.50) and shrinks toward the extremes
    (p = 0.90 → $0.54 per 100). The old flat 2%-of-notional model
    overcharged favorites and undercharged mid-priced markets, which
    misranked which trades are +EV.
  * Per-trade amounts use banker's rounding (half-to-even) to the cent,
    so tiny trades can round to $0.00. Rounding applies to BOOKED fees
    (execution, paper fills, backtest fills) — edge/Kelly math stays
    unrounded so sizing is smooth.

This bot always takes (IOC orders), so the taker coefficient is what the
edge and sizing math uses. The maker rebate constant is provided for
completeness / future passive-order support.
"""

from decimal import Decimal, ROUND_HALF_EVEN

# Exchange-published coefficients (docs.polymarket.us/fees)
TAKER_FEE_COEFFICIENT = 0.06
MAKER_REBATE_COEFFICIENT = -0.0125


def _clamp_price(price: float) -> float:
    return min(max(price, 0.01), 0.99)


def fee_per_contract(price: float, coefficient: float = TAKER_FEE_COEFFICIENT) -> float:
    """Fee in dollars for ONE contract at `price` (unrounded).

    This is the number strategy math should subtract per share:
    cost of a bought share = price + fee_per_contract(price);
    proceeds of a sold share = price - fee_per_contract(price).
    """
    p = _clamp_price(price)
    return coefficient * p * (1.0 - p)


def fee_usd(contracts: float, price: float,
            coefficient: float = TAKER_FEE_COEFFICIENT) -> float:
    """Total fee in dollars for a trade of `contracts` at `price` (unrounded)."""
    return contracts * fee_per_contract(price, coefficient)


def booked_fee_usd(contracts: float, price: float,
                   coefficient: float = TAKER_FEE_COEFFICIENT) -> float:
    """Fee as the exchange books it: banker's-rounded to the nearest cent.

    Use for actual/simulated fills so P&L matches the exchange to the penny;
    use the unrounded functions for edge and sizing math.
    """
    raw = fee_usd(contracts, price, coefficient)
    return float(Decimal(str(raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
