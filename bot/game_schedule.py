"""Game schedule awareness.

Fetches today's game schedule, determines when to activate/sleep,
and tracks game clock for last-5-minutes blocking.
"""

import time
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from bot.signals.odds_api import TEAM_ABBREVS


ESPN_ENDPOINTS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "cbb": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
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

    def get_game_time_remaining(self, sport: str, home_abbr: str, away_abbr: str) -> Optional[float]:
        """Get seconds remaining in a game. Returns None if not live or unknown.

        Used for last-5-minutes blocking.
        """
        events = self._get_events(sport)
        for event in events:
            status = event.get("status", {}).get("type", {})
            if status.get("name") not in ("STATUS_IN_PROGRESS", "STATUS_END_PERIOD"):
                continue

            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])

            # Match teams
            abbrs = [c.get("team", {}).get("abbreviation", "").lower() for c in competitors]
            h_known = TEAM_ABBREVS.get(home_abbr, "")
            a_known = TEAM_ABBREVS.get(away_abbr, "")
            names = [c.get("team", {}).get("displayName", "").lower() for c in competitors]

            matched = False
            if home_abbr in abbrs and away_abbr in abbrs:
                matched = True
            elif h_known and a_known:
                if any(h_known in n for n in names) and any(a_known in n for n in names):
                    matched = True

            if not matched:
                continue

            # Parse clock and period to estimate time remaining
            clock = status.get("displayClock", "0:00")
            period = status.get("period", 0)

            try:
                parts = clock.split(":")
                if len(parts) == 2:
                    clock_seconds = int(parts[0]) * 60 + int(float(parts[1]))
                else:
                    clock_seconds = int(float(parts[0]))
            except (ValueError, TypeError):
                clock_seconds = 0

            # Calculate total remaining based on sport
            if sport == "nba":
                # 4 quarters, 12 min each = 48 min total
                quarters_left = max(0, 4 - period)
                remaining = quarters_left * 12 * 60 + clock_seconds
            elif sport == "cbb":
                # 2 halves, 20 min each = 40 min total
                halves_left = max(0, 2 - period)
                remaining = halves_left * 20 * 60 + clock_seconds
            elif sport == "nhl":
                # 3 periods, 20 min each = 60 min total
                periods_left = max(0, 3 - period)
                remaining = periods_left * 20 * 60 + clock_seconds
            else:
                remaining = clock_seconds

            return float(remaining)

        return None

    def format_schedule(self) -> str:
        """Format today's schedule for logging/Slack."""
        games = self.get_todays_games()
        if not games:
            return "No games scheduled today."

        lines = []
        for sport in ("cbb", "nhl", "nba"):
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
