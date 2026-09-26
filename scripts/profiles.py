#!/usr/bin/env python3
"""Paper strategy profiles: single source of truth.

Every profile shares configs/config.yaml (v2 paper accounting, NFL/CFB, NHL
from its Sep 29 opener). Profile settings are POLYBOT_* env overrides in
each launchd plist. Since 2026-09-26 15:xx ET:

    certainty     bot/certainty.py: buys near-certain outcomes late in a game
                  (two-score lead, ESPN >= 97%, quote <= 96c) and decided
                  outcomes after the final whistle (<= 99c), holds to
                  settlement. Also the websocket order-book leader.
    finals        the same, finals only: the zero-game-risk floor
    pregame       value entries before kickoff (24 h out to 10 min before),
                  3% gross edge and 2% after entry costs, held to settlement
    pregame_late  pregame, entries only in the last 3 hours before kickoff

Install/refresh the launchd plists (does not start them):
    python scripts/profiles.py --install
"""
import os
import plistlib
import sys

HOME = os.path.expanduser("~")
MAIN = f"{HOME}/Projects/polymarket-bot"
PROFILE_ROOT = f"{HOME}/Projects/polybot-profiles"
SHARED = f"{MAIN}/data/shared"

LEADER = "certainty"   # runs in the main checkout and streams order books for everyone

# Shared by the pregame profiles; each then changes ONE thing.
PREGAME = {
    "POLYBOT_TRADING__ENTRY_WINDOW": "pregame",
    "POLYBOT_TRADING__HOLD_TO_SETTLEMENT": "true",
    "POLYBOT_TRADING__REQUIRE_ROUND_TRIP_EDGE": "false",   # no exit fee or exit spread
    "POLYBOT_TRADING__LEAGUE_MIN_EDGE_OVERRIDE": "0.03",   # 3% gross (net >= 2% after entry costs)
    "POLYBOT_FILTERS__MIN_HOURS_TO_EXPIRY": "0",           # scan up to the 10-min cutoff
    "POLYBOT_TRADING__MAX_OPEN_POSITIONS": "20",           # holds last hours: room for a full slate
    "POLYBOT_TRADING__MAX_DAILY_TRADES": "30",
    "POLYBOT_TRADING__MAX_PORTFOLIO_EXPOSURE_USD": "750",
    "POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD": "300",
}

CERTAINTY = {
    "POLYBOT_TRADING__STRATEGY": "certainty",
    "POLYBOT_TRADING__HOLD_TO_SETTLEMENT": "true",
    "POLYBOT_TRADING__MAX_OPEN_POSITIONS": "20",
    "POLYBOT_TRADING__MAX_DAILY_TRADES": "40",
    "POLYBOT_TRADING__MAX_PORTFOLIO_EXPOSURE_USD": "900",
    "POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD": "300",
}

PROFILES = {
    "certainty": {
        "description": "Buys near-certain outcomes: two-score leads late (ESPN 97%+, at most 96c) and decided games after the final (at most 99c), held to settlement.",
        "env": {"POLYBOT_BOOK_FEED": "1", **CERTAINTY},
    },
    "finals": {
        "description": "Finals only: buys the decided side after the final whistle (at most 99c) and waits for settlement. No game risk.",
        "env": {**CERTAINTY, "POLYBOT_TRADING__CERTAINTY_LIVE_ENTRIES": "false"},
    },
    "pregame": {
        "description": "Enters before kickoff only, up to 24 hours out, 3% edge over the books, and holds every position to settlement.",
        "env": dict(PREGAME),
    },
    "pregame_late": {
        "description": "Pregame and held to settlement, but enters only in the last 3 hours before kickoff.",
        "env": {**PREGAME, "POLYBOT_TRADING__PREGAME_MAX_HOURS": "3"},
    },
}


def root(name: str) -> str:
    return MAIN if name == LEADER else f"{PROFILE_ROOT}/{name}"


def label(name: str) -> str:
    return "com.polymarket.bot" if name == LEADER else f"com.polymarket.bot.{name}"


def plist_path(name: str) -> str:
    return f"{HOME}/Library/LaunchAgents/{label(name)}.plist"


def install() -> None:
    for name, spec in PROFILES.items():
        r = root(name)
        os.makedirs(f"{r}/reports", exist_ok=True)
        os.makedirs(f"{r}/data", exist_ok=True)
        env = {"POLYBOT_SHARED_DIR": SHARED, "POLYBOT_PROFILE": name, **spec["env"]}
        pl = {
            "Label": label(name),
            "ProgramArguments": [f"{r}/venv/bin/python", "-m", "bot.trading_loop"],
            "WorkingDirectory": r, "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 15,
            "StandardOutPath": f"{r}/reports/launchd_bot.log",
            "StandardErrorPath": f"{r}/reports/launchd_bot.log",
            "EnvironmentVariables": env,
        }
        with open(plist_path(name), "wb") as fh:
            plistlib.dump(pl, fh)
        print(f"wrote {plist_path(name)}")


if __name__ == "__main__":
    if "--install" in sys.argv:
        install()
    elif "--names" in sys.argv:
        print(" ".join(n for n in PROFILES if n != LEADER))
    else:
        for n, spec in PROFILES.items():
            print(f"{n:10} {spec['description']}")
