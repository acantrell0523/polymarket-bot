"""Certainty strategy: buy near-certain outcomes late in a game and at the final.

The value strategy lost on both sides of the game clock. In-game, Polymarket
reprices on every play while the sportsbook quotes it was compared against
run 20 s to 2.5 min behind, so the "edge" was the market being right first
(24 of 82 in-game picks won against 31 expected). Pregame, the gap to the
books is about 1%, under the fee. Neither zone has an edge for a bot fed by
scraped quotes.

Thin prediction markets leak money in the tails instead: a team up 17 with
five minutes left is a 99% proposition that a lazy maker still offers at
93-97c, and after the final whistle the winner trades below $1 until the
market settles. This module decides those entries. Every position is held
to settlement (no stop can help a sure thing; only the result can), and the
trading loop closes it at the exchange's settlement value with no exit fee.

Two entry types:
  final      ESPN says the game is over: buy the side that won, moneyline
             or a spread/total whose result follows from the final score.
  late_lead  In progress, inside certainty_max_seconds_left, the leader is
             two scores up (>= 9 points inside 4:00, >= 17 inside 8:00),
             ESPN's win probability for the leader is >= certainty_min_win_prob
             and the quote is <= certainty_max_price. Moneylines only:
             backdoor covers make late spreads and totals the wrong tail.
             Overtime is never entered live (CFB overtime has no clock).

Token 0 of every slug is the away side (moneyline: away team; spread: away
team's line; total: Over). Buying token 0 backs the away side; selling it
backs the home side with (1 - price) collateral. Both fill at the executable
top of book within visible depth like every other paper trade.
"""
from typing import Dict, Optional, Tuple

from bot.http_cache import get_json
from bot.leagues import LEAGUES, game_seconds_remaining, normalize_abbr, scoreboard_url
from bot.signals.lines import parse_line_slug
from bot.signals.live_win_prob import slug_game_teams

SCOREBOARD_MAX_AGE = 12.0     # seconds; one ESPN fetch per league shared by every profile
WIN_PROB_MAX_AGE = 60.0       # ESPN model output older than this is not evidence
# Margin floors: a lead this size with this little time left is what the
# ESPN number must agree with before a live entry.
MARGIN_FLOORS = ((240.0, 9), (480.0, 17))


def _clock_seconds(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        text = str(raw).replace("'", "").replace("+", "").strip()
        parts = text.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(float(parts[1]))
        return float(parts[0])
    except (TypeError, ValueError):
        return None


class GameStateCache:
    """Today's ESPN scoreboard per league: state, scores, period and clock,
    keyed by the Polymarket team codes a slug uses."""

    def __init__(self, max_age: float = SCOREBOARD_MAX_AGE):
        self.max_age = max_age

    def games(self, league: str) -> Dict[Tuple[str, str], dict]:
        url = scoreboard_url(league)
        data = get_json(url, max_age=self.max_age) if url else None
        out: Dict[Tuple[str, str], dict] = {}
        for ev in (data or {}).get("events", []) or []:
            status = ev.get("status") or {}
            stype = status.get("type") or {}
            comp = (ev.get("competitions") or [{}])[0]
            scores: Dict[str, int] = {}
            away = home = ""
            for c in comp.get("competitors", []) or []:
                code = normalize_abbr(league, str((c.get("team") or {}).get("abbreviation", "")).lower())
                if not code:
                    continue
                try:
                    scores[code] = int(c.get("score") or 0)
                except (TypeError, ValueError):
                    scores[code] = 0
                if c.get("homeAway") == "home":
                    home = code
                else:
                    away = code
            if not (away and home):
                continue
            out[(away, home)] = {
                "state": stype.get("state", ""), "completed": bool(stype.get("completed")),
                "detail": stype.get("shortDetail", ""), "period": int(status.get("period") or 0),
                "clock_seconds": _clock_seconds(status.get("displayClock")),
                "scores": scores, "espn_away": away, "espn_home": home,
            }
        return out

    def state_for(self, slug: str) -> Optional[dict]:
        parsed = slug_game_teams(slug)
        if not parsed:
            return None
        league, away, home = parsed
        games = self.games(league)
        gs = games.get((away, home)) or games.get((home, away))
        if not gs or away not in gs["scores"] or home not in gs["scores"]:
            return None
        return {**gs, "league": league, "away": away, "home": home,
                "away_score": gs["scores"][away], "home_score": gs["scores"][home]}


def token0_final_value(slug: str, gs: dict) -> Optional[float]:
    """1.0 / 0.0 for what token 0 pays given the final score, None for a
    tie, a push or an unsupported slug."""
    a, h = gs["away_score"], gs["home_score"]
    if slug.startswith(("aec-",)):
        if a == h:
            return None
        return 1.0 if a > h else 0.0
    parsed = parse_line_slug(slug)
    if not parsed:
        return None
    if parsed["kind"] == "spread":
        value = (a - h) + parsed["line"]
    else:
        value = (a + h) - parsed["line"]
    if value == 0:
        return None
    return 1.0 if value > 0 else 0.0


def seconds_left(gs: dict) -> Optional[float]:
    if gs.get("clock_seconds") is None:
        return None
    return game_seconds_remaining(gs["league"], gs["period"], gs["clock_seconds"])


def in_overtime(gs: dict) -> bool:
    clock = (LEAGUES.get(gs["league"]) or {}).get("clock") or {}
    periods = clock.get("periods")
    return bool(periods) and gs["period"] > periods


def decide(slug: str, gs: Optional[dict], win_prob_away, cfg) -> Optional[dict]:
    """The entry for `slug` under the certainty rules, or None.

    win_prob_away: (P(away wins), age_seconds) from ESPN's live model, or
    None. cfg is TradingConfig. Returns {side, limit, token0_prob, why,
    leader, margin, seconds_left, kind}.
    """
    if not gs:
        return None
    kind = "spread" if slug.startswith("asc-") else "total" if slug.startswith("tsc-") else "ml"
    if gs["state"] == "post" or gs.get("completed"):
        value = token0_final_value(slug, gs)
        if value is None:
            return None
        side = "buy" if value == 1.0 else "sell"
        limit = cfg.finals_max_price if side == "buy" else round(1.0 - cfg.finals_max_price, 4)
        return {"side": side, "limit": limit, "token0_prob": value, "why": "final", "kind": kind,
                "leader": "away" if gs["away_score"] > gs["home_score"] else "home",
                "margin": abs(gs["away_score"] - gs["home_score"]), "seconds_left": 0.0}
    if gs["state"] != "in" or kind != "ml" or not getattr(cfg, "certainty_live_entries", True):
        return None
    if in_overtime(gs):
        return None
    left = seconds_left(gs)
    if left is None or left > cfg.certainty_max_seconds_left:
        return None
    margin = abs(gs["away_score"] - gs["home_score"])
    if margin == 0:
        return None
    if not any(left <= max_left and margin >= floor for max_left, floor in MARGIN_FLOORS):
        return None
    if not win_prob_away:
        return None
    p_away, age = win_prob_away
    if age > WIN_PROB_MAX_AGE:
        return None
    leader = "away" if gs["away_score"] > gs["home_score"] else "home"
    leader_prob = p_away if leader == "away" else 1.0 - p_away
    if leader_prob < cfg.certainty_min_win_prob:
        return None
    side = "buy" if leader == "away" else "sell"
    limit = cfg.certainty_max_price if side == "buy" else round(1.0 - cfg.certainty_max_price, 4)
    return {"side": side, "limit": limit, "token0_prob": p_away, "why": "late_lead", "kind": kind,
            "leader": leader, "margin": margin, "seconds_left": left}


def leader_quote(book, leader: str) -> Tuple[Optional[float], float]:
    """(price to back the leader, contracts at that price): token 0's ask
    for the away side, 1 - token 0's bid for the home side."""
    from bot.paper import executable_level
    if leader == "away":
        level = executable_level(book, "buy")
        return (level[0], level[1]) if level else (None, 0.0)
    level = executable_level(book, "sell")
    return (round(1.0 - level[0], 4), level[1]) if level else (None, 0.0)
