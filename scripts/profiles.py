#!/usr/bin/env python3
"""Paper strategy profiles: single source of truth.

Every profile shares configs/config.yaml (books-primary in-game signal, no
re-entry after a stop loss, NFL/CFB/NHL with NHL off until its regular season
opens Sep 29, v2 paper accounting). Profile settings are POLYBOT_* env
overrides in each launchd plist. Since 2026-09-25:

    baseline      in-game arm: the 2026-09-23 rules (stops, profit-taking,
                  round-trip edge gate); also streams order books for everyone
    pregame       pregame entries only (from 24 h out to 10 min before
                  kickoff), every position held to settlement
    pregame_late  pregame, entries only in the last 3 hours before kickoff
    pregame_ml    pregame, moneylines only (no spreads or totals)

The three pregame profiles pay no exit fee or exit spread, so their edge gate
covers entry costs only, and they get room for a full Saturday slate.

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

# Shared by the three pregame profiles; each then changes ONE thing.
PREGAME = {
    "POLYBOT_TRADING__ENTRY_WINDOW": "pregame",
    "POLYBOT_TRADING__HOLD_TO_SETTLEMENT": "true",
    "POLYBOT_TRADING__REQUIRE_ROUND_TRIP_EDGE": "false",   # no exit fee or exit spread
    "POLYBOT_FILTERS__MIN_HOURS_TO_EXPIRY": "0",           # scan up to the 10-min cutoff
    "POLYBOT_TRADING__MAX_OPEN_POSITIONS": "20",           # holds last hours: room for a full slate
    "POLYBOT_TRADING__MAX_DAILY_TRADES": "30",
    "POLYBOT_TRADING__MAX_PORTFOLIO_EXPOSURE_USD": "750",
    "POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD": "300",
}

PROFILES = {
    "baseline": {
        "description": "In-game arm: live sportsbook quotes, stops and profit-taking (the Sep 23 rules).",
        "env": {"POLYBOT_BOOK_FEED": "1"},
    },
    "pregame": {
        "description": "Enters before kickoff only, up to 24 hours out, and holds every position to settlement.",
        "env": dict(PREGAME),
    },
    "pregame_late": {
        "description": "Pregame and held to settlement, but enters only in the last 3 hours before kickoff.",
        "env": {**PREGAME, "POLYBOT_TRADING__PREGAME_MAX_HOURS": "3"},
    },
    "pregame_ml": {
        "description": "Pregame and held to settlement, moneylines only (no spreads or totals).",
        "env": {**PREGAME, "POLYBOT_TRADING__MARKET_KINDS": "ml"},
    },
}


def root(name: str) -> str:
    return MAIN if name == "baseline" else f"{PROFILE_ROOT}/{name}"


def label(name: str) -> str:
    return "com.polymarket.bot" if name == "baseline" else f"com.polymarket.bot.{name}"


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
        print(" ".join(n for n in PROFILES if n != "baseline"))
    else:
        for n, spec in PROFILES.items():
            print(f"{n:10} {spec['description']}")
