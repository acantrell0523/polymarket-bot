"""Comeback strategy: buy a heavy favorite that fell behind early.

Austin's thesis (2026-09-26): when a strong pregame favorite goes down
early, the market and the books overreact, and the favorite's moneyline is
cheap relative to how often it comes back. The entry needs all of:

  * pregame favorite of comeback_min_favorite or better, taken from
    Polymarket's own last pregame price (the anchor the trading loop stores
    on every full scan), with the sportsbook pregame consensus as fallback;
  * in progress, inside the first comeback_max_period periods, never overtime;
  * the favorite trailing by 1 to comeback_max_deficit points;
  * ESPN's live model still giving the favorite comeback_min_live_prob;
  * the score settled for score_quiet_seconds (no entry right after a score);
  * the favorite priced at least comeback_min_drop below its pregame price.

Buy the favorite's moneyline at the executable price (never above pregame
minus comeback_min_drop plus 5c) and hold to settlement. Every qualifying
moment is logged to comeback_log, taken or not, so the recovery rate can be
measured against the price.
"""
from typing import Optional, Tuple

from bot.certainty import in_overtime, score_settled, seconds_left


def favorite_side(pregame_token0_prob: Optional[float], min_favorite: float) -> Optional[Tuple[str, float]]:
    """("away"|"home", favorite's pregame probability) or None when neither
    side was a big enough favorite."""
    if pregame_token0_prob is None:
        return None
    p = float(pregame_token0_prob)
    if p >= min_favorite:
        return "away", p
    if 1.0 - p >= min_favorite:
        return "home", 1.0 - p
    return None


def decide(slug: str, gs: Optional[dict], pregame_token0_prob: Optional[float], win_prob_away, cfg) -> Optional[dict]:
    """The comeback entry for a moneyline slug, or None. Returns
    {side, limit, token0_prob, why, favorite, deficit, period, seconds_left,
    pregame_prob, live_prob}."""
    if not gs or gs.get("state") != "in" or not slug.startswith("aec-"):
        return None
    fav = favorite_side(pregame_token0_prob, float(cfg.comeback_min_favorite))
    if fav is None:
        return None
    favorite, p0 = fav
    if in_overtime(gs) or int(gs.get("period") or 0) > int(cfg.comeback_max_period):
        return None
    a, h = gs["away_score"], gs["home_score"]
    deficit = (h - a) if favorite == "away" else (a - h)
    if not (1 <= deficit <= int(cfg.comeback_max_deficit)):
        return None
    if not score_settled(gs, float(cfg.score_quiet_seconds)):
        return None
    if not win_prob_away:
        return None
    p_away, age = win_prob_away
    if age > 60.0:
        return None
    live_prob = float(p_away) if favorite == "away" else 1.0 - float(p_away)
    if live_prob < float(cfg.comeback_min_live_prob):
        return None
    max_pay = round(p0 - float(cfg.comeback_min_drop) + 0.05, 4)   # the price must have dropped
    side = "buy" if favorite == "away" else "sell"
    limit = max_pay if side == "buy" else round(1.0 - max_pay, 4)
    return {"side": side, "limit": limit, "token0_prob": float(p_away), "why": "comeback",
            "favorite": favorite, "deficit": deficit, "period": int(gs.get("period") or 0),
            "seconds_left": seconds_left(gs), "pregame_prob": p0, "live_prob": live_prob}
