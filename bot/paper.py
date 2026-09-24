"""Conservative paper-book helpers. These do not submit exchange orders."""

import math


def executable_level(book, side):
    """Return best executable price and contracts at that price, or None.

    Refuse malformed, empty, one-sided and crossed books. Size is contracts,
    not dollars. The simulation never assumes fills beyond the IOC limit.
    """
    if book is None or side not in ("buy", "sell"):
        return None
    if not book.bids or not book.asks:
        return None
    for level in book.bids + book.asks:
        if (not math.isfinite(level.price) or not math.isfinite(level.size)
                or not 0 < level.price < 1 or level.size <= 0):
            return None
    bid = max(level.price for level in book.bids)
    ask = min(level.price for level in book.asks)
    if bid > ask:
        return None
    price = ask if side == "buy" else bid
    levels = book.asks if side == "buy" else book.bids
    return price, sum(level.size for level in levels if level.price == price)


def _validated_levels(book):
    """(bids best-first, asks best-first) as (price, size) lists, or None.

    Same validity rules as executable_level: no empty, one-sided, crossed or
    malformed books.
    """
    if book is None or not book.bids or not book.asks:
        return None
    for level in book.bids + book.asks:
        if (not math.isfinite(level.price) or not math.isfinite(level.size)
                or not 0 < level.price < 1 or level.size <= 0):
            return None
    bids = sorted(((lv.price, lv.size) for lv in book.bids), key=lambda x: -x[0])
    asks = sorted(((lv.price, lv.size) for lv in book.asks), key=lambda x: x[0])
    if bids[0][0] > asks[0][0]:
        return None
    return bids, asks


def sweep(book, side, quantity, coefficient=0.06):
    """Simulate a marketable IOC EXIT that walks the visible book.

    side "sell" hits bids (closing a long); "buy" lifts asks (covering a
    short). This mirrors the live close path, which sends a sweeping IOC
    (sell @ $0.01 / buy @ $0.99). Fees are booked per level, since each level
    is a separate fill. Returns {"filled", "vwap", "fees", "levels"} or None
    for an unusable book. `filled` can be below `quantity` when visible depth
    runs out; the caller closes that part and keeps the remainder open, as a
    real IOC would. Entries still use executable_level (top of book only,
    limited to the signal's price).
    """
    from bot.strategies.fees import booked_fee_usd
    levels = _validated_levels(book)
    if levels is None or quantity <= 0:
        return None
    ladder = levels[0] if side == "sell" else levels[1]
    remaining, filled, notional, fees, used = float(quantity), 0.0, 0.0, 0.0, 0
    for price, size in ladder:
        if remaining <= 1e-9:
            break
        take = min(size, remaining)
        filled += take
        notional += take * price
        fees += booked_fee_usd(take, price, coefficient)
        remaining -= take
        used += 1
    if filled <= 0:
        return None
    return {"filled": filled, "vwap": notional / filled, "fees": round(fees, 2), "levels": used}
