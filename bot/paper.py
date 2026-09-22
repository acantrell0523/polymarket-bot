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
