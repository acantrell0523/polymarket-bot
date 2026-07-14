"""Game schedule awareness.

Fetches today's game schedule, determines when to activate/sleep,
and tracks game clock for last-5-minutes blocking.
"""

import time
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from bot.signals.odds_api import TEAM_ABBREVS
from bot.leagues import LEAGUES, scoreboard_url, game_seconds_remaining


# All registered leagues. Off-season leagues return zero games from ESPN and
# cost one cached call each — this is what keeps the bot focused on whatever
# sports are ACTUALLY happening today, year-round, with no seasonal edits.
ESPN_ENDPOINTS = {
    league: scoreboard_url(league) for league in LEAGUES
}


class GameSchedule:
    """Tracks today's game schedule and game clocks."""

    def __init__(self, cache_ttl: int = 120):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, list]] = {}

    def _fetch_schedule(self, sport: str) -> list:
        url = ESPN_ENDPOINTS.get(sport)
        if not url:
            return []
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("events", [])
        except Exception:
            pass
        return []

    def _get_events(self, sport: str) -> list:
        now = time.time()
        if sport in self._cache:
            ts, data = self._cache[sport]
            if now - ts < self.cache_ttl:
                return data
        data = self._fetch_schedule(sport)
        self._cache[sport] = (now, data)
        return data

    def get_todays_games(self) -> List[Dict]:
        """Get all games for today across all sports."""
        games = []
        for sport in ESPN_ENDPOINTS:
            events = self._get_events(sport)
            for event in events:
                status = event.get("status", {}).get("type", {})
                comps = event.get("competitions", [{}])[0]
                competitors = comps.get("competitors", [])

                if len(competitors) < 2:
                    continue

                home = None
                away = None
                for c in competitors:
                    t = c.get("team", {})
                    info = {
                        "name": t.get("displayName", ""),
                        "abbr": t.get("abbreviation", "").lower(),
                        "seed": c.get("curatedRank", {}).get("current", 0),
                        "record": c.get("records", [{}])[0].get("summary", "") if c.get("records") else "",
                        "score": int(c.get("score", "0") or "0"),
                    }
                    if c.get("homeAway") == "home":
                        home = info
                    else:
                        away = info

                if not home or not away:
                    continue

                games.append({
                    "sport": sport,
                    "name": event.get("name", ""),
                    "date": event.get("date", ""),
                    "status_name": status.get("name", ""),
                    "status_detail": status.get("detail", ""),
                    "completed": status.get("completed", False),
                    "home": home,
                    "away": away,
                    "venue": comps.get("venue", {}).get("fullName", ""),
                    "neutral_site": comps.get("neutralSite", False),
                    "clock": status.get("displayClock", ""),
                    "period": status.get("period", 0),
                })

        return games

    def get_next_game_time(self) -> Optional[datetime]:
        """Get the start time of the next scheduled game."""
        now = datetime.now(timezone.utc)
        earliest = None
        for game in self.get_todays_games():
            if game["status_name"] != "STATUS_SCHEDULED":
                continue
            try:
                dt = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
                if dt > now:
                    if earliest is None or dt < earliest:
                        earliest = dt
            except (ValueError, TypeError):
                continue
        return earliest

    def should_be_scanning(self) -> Tuple[bool, str]:
        """Should the bot be actively scanning right now?

        Returns (should_scan, reason).
        Active when: any game is live, OR a game starts within 2 hours.
        """
        games = self.get_todays_games()
        if not games:
            return False, "no_games_today"

        now = datetime.now(timezone.utc)

        # Check for live games
        live = [g for g in games if g["status_name"] in
                ("STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD")]
        if live:
            return True, f"{len(live)}_games_live"

        # Check for games starting within 2 hours
        for game in games:
            if game["status_name"] != "STATUS_SCHEDULED":
                continue
            try:
                dt = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
                if dt - now < timedelta(hours=2):
                    mins = int((dt - now).total_seconds() / 60)
                    return True, f"game_in_{mins}_minutes"
            except (ValueError, TypeError):
                continue

        # No live games and nothing within 2 hours
        next_game = self.get_next_game_time()
        if next_game:
            mins = int((next_game - now).total_seconds() / 60)
            return False, f"next_game_in_{mins}_minutes"

        return False, "all_games_finished"

    def _find_live_event(self, sport: str, home_abbr: str, away_abbr: str):
        """Find the in-progress ESPN event for a team pair, or None.

        Caller abbrs are in the recorder's NORMALIZED space (slug abbrs like
        gsv/conn/cws); ESPN scoreboards carry raw abbrs (gs/con/chw), so both
        sides are normalized before comparing. Ordering-insensitive.
        """
        from bot.leagues import normalize_abbr

        events = self._get_events(sport)
        for event in events:
            status = event.get("status", {}).get("type", {})
            if status.get("name") not in ("STATUS_IN_PROGRESS", "STATUS_END_PERIOD"):
                continue

            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])

            abbrs = [
                normalize_abbr(sport, c.get("team", {}).get("abbreviation", "").lower())
                for c in competitors
            ]
            h_known = TEAM_ABBREVS.get(home_abbr, "")
            a_known = TEAM_ABBREVS.get(away_abbr, "")
            names = [c.get("team", {}).get("displayName", "").lower() for c in competitors]

            matched = False
            if home_abbr in abbrs and away_abbr in abbrs:
                matched = True
            elif h_known and a_known:
                if any(h_known in n for n in names) and any(a_known in n for n in names):
                    matched = True

            if matched:
                return event
        return None

    def get_game_period(self, sport: str, home_abbr: str, away_abbr: str) -> Optional[int]:
        """Current period/quarter/inning of a LIVE game, or None.

        The clockless-sport analog of the game clock: MLB live entries are
        gated on inning (see trading.live_clockless_max_period).
        """
        event = self._find_live_event(sport, home_abbr, away_abbr)
        if event is None:
            return None
        period = int(event.get("status", {}).get("period") or 0)
        return period if period > 0 else None

    def get_game_time_remaining(self, sport: str, home_abbr: str, away_abbr: str) -> Optional[float]:
        """Get seconds remaining in a game. Returns None if not live or unknown.

        Used for last-5-minutes blocking.
        """
        event = self._find_live_event(sport, home_abbr, away_abbr)
        if event is not None:

            # AUDIT #7 FIX: displayClock and period live on event["status"],
            # NOT on status["type"] (verified against a live payload
            # 2026-07-11). Reading them off the type object always defaulted
            # to "0:00"/period 0, which computed FULL-GAME time remaining
            # (e.g. 3:07 left in Q2 read as 2,400s) and silently defeated the
            # last-5-minutes safety gate.
            status_obj = event.get("status", {})
            clock_raw = status_obj.get("displayClock")
            period = int(status_obj.get("period") or 0)

            clock_seconds: Optional[float] = None
            if clock_raw is not None:
                try:
                    # Soccer clocks render as "67'" (minutes elapsed, counting
                    # up); stoppage time as "90'+". Strip markers before parsing.
                    clock = str(clock_raw)
                    clean = clock.replace("'", "").replace("+", "").strip()
                    parts = clean.split(":")
                    if len(parts) == 2:
                        clock_seconds = int(parts[0]) * 60 + int(float(parts[1]))
                    elif "'" in clock:
                        clock_seconds = int(float(parts[0])) * 60  # minutes elapsed
                    else:
                        clock_seconds = float(parts[0])
                except (ValueError, TypeError):
                    clock_seconds = None

            # Clockless sport (baseball)? The registry says so — no block.
            if game_seconds_remaining(sport, 1, 0) is None:
                return None

            # FAIL-CLOSED: a clocked sport that is LIVE but whose clock we
            # cannot read gets 0 seconds remaining, which BLOCKS new entries.
            # The old behavior (default to full game) failed open.
            if clock_seconds is None or (clock_seconds == 0 and period == 0):
                return 0.0

            remaining = game_seconds_remaining(sport, period, clock_seconds)
            if remaining is None:
                return None
            return float(remaining)

        return None

    def format_schedule(self) -> str:
        """Format today's schedule for logging/Slack."""
        games = self.get_todays_games()
        if not games:
            return "No games scheduled today."

        lines = []
        # Only leagues that actually have games today show up
        for sport in sorted({g["sport"] for g in games}):
            sport_games = [g for g in games if g["sport"] == sport]
            if not sport_games:
                continue

            lines.append(f"\n{sport.upper()} ({len(sport_games)} games):")
            for g in sport_games:
                try:
                    dt = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
                    time_str = dt.strftime("%I:%M %p ET")
                except (ValueError, TypeError):
                    time_str = "TBD"

                seed_a = f"({g['away']['seed']})" if g['away'].get('seed') else ""
                seed_h = f"({g['home']['seed']})" if g['home'].get('seed') else ""

                if g["completed"]:
                    status = f"FINAL {g['away']['score']}-{g['home']['score']}"
                elif g["status_name"] in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME"):
                    status = f"LIVE {g['status_detail']} {g['away']['score']}-{g['home']['score']}"
                else:
                    status = time_str

                lines.append(
                    f"  {seed_a:>4s} {g['away']['name'][:20]:<20s} vs "
                    f"{seed_h:>4s} {g['home']['name'][:20]:<20s} — {status}"
                )

        return "\n".join(lines)
