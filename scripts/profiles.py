#!/usr/bin/env python3
"""Paper strategy profiles: single source of truth.

Every profile shares configs/config.yaml (books-primary in-game signal, no
re-entry after a stop loss, round-trip edge gate, NFL/CFB/NHL only, v2 paper
accounting). Each profile changes ONE thing, as POLYBOT_* env overrides in
its launchd plist, so results compare cleanly:

    baseline   control; also streams order books for everyone (websocket leader)
    no_trail   trailing stop off
    ride       hold to settlement: no profit-taking exits, 35% stop,
               let winners ride from 55c
    espn_live  ESPN's in-game win-probability model as the primary signal
               (the pre-2026-09-23 behavior, kept as the control arm)

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

PROFILES = {
    "baseline": {
        "description": "Control. All 2026-09-23 rules, standard exits.",
        "env": {"POLYBOT_BOOK_FEED": "1"},
    },
    "no_trail": {
        "description": "Baseline with the trailing stop turned off.",
        "env": {"POLYBOT_TRADING__TRAILING_STOP_ENABLED": "false"},
    },
    "ride": {
        "description": "Baseline entries held to settlement: no profit-taking exits, 35% stop.",
        "env": {
            "POLYBOT_TRADING__AGGRESSIVE_EXIT_PCT": "5.0",
            "POLYBOT_TRADING__TRAILING_STOP_ENABLED": "false",
            "POLYBOT_TRADING__TAKE_PROFIT_THRESHOLD": "-1.0",
            "POLYBOT_TRADING__STOP_LOSS_THRESHOLD": "0.35",
            "POLYBOT_TRADING__LET_IT_RIDE_THRESHOLD": "0.55",
        },
    },
    "espn_live": {
        "description": "Baseline with ESPN's live win-probability model as the in-game primary signal.",
        "env": {"POLYBOT_SIGNALS__LIVE_PRIMARY": "espn"},
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
