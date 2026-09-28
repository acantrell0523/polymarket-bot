#!/usr/bin/env python3
"""Paper strategy profiles: single source of truth.

Every profile shares configs/config.yaml (v2 paper accounting, NFL/CFB, NHL
from its Sep 29 opener). Profile settings are POLYBOT_* env overrides in
each launchd plist. Since 2026-09-26 (finals retired 22:40 ET: Polymarket US expires a market at the final whistle):

    certainty     bot/certainty.py: buys near-certain outcomes late in a game
                  (two-score lead, ESPN >= 97%, quote <= 96c) and decided
                  outcomes after the final whistle (<= 99c), holds to
                  settlement. Also the websocket order-book leader.
    comeback      bot/comeback.py: a 75%+ pregame favorite trailing by <= 14
                  in the first half, ESPN still >= 50%, priced 20+ points
                  below its pregame price: buy and hold (Austin's thesis)
    maker         bot/maker.py: rests paper quotes 3c inside the model's fair
                  value in wide books (live alternate lines; pregame lines when
                  they are wide), earns the spread and the maker rebate,
                  holds fills to settlement or flips them
    pregame_late  the taker control: value entries in the last 3 hours before
                  kickoff, 3% gross edge, held to settlement. The venue's own
                  market maker keeps liquid pregame books 0.5c wide, so this
                  arm measures model error, not mispricing.
    live          in-game value betting (the Sep 23 rules) with a 4% net edge
                  bar and a whole-game block after a stop loss; the measured
                  live-betting arm Austin asked to keep

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
        "description": "Buys near-certain outcomes late in a game: a two-score lead with ESPN at 97%+ (at most 96c), 98.5%+ (97.5c) or 99.5%+ (98.5c), and totals already past their line (99.5c), held to settlement.",
        "env": {"POLYBOT_BOOK_FEED": "1", "POLYBOT_FILTERS__MIN_HOURS_TO_EXPIRY": "0", **CERTAINTY},
    },
    "comeback": {
        "description": "Buys a pregame favorite of 70%+ that trails by 17 or fewer in the first half (2 goals in hockey) while ESPN still gives it 50%+, at a price at least 20 points below its pregame price, and holds to settlement.",
        "env": {
            "POLYBOT_TRADING__STRATEGY": "comeback",
            "POLYBOT_TRADING__HOLD_TO_SETTLEMENT": "true",
            "POLYBOT_FILTERS__MIN_HOURS_TO_EXPIRY": "0",
            "POLYBOT_TRADING__MAX_OPEN_POSITIONS": "20",
            "POLYBOT_TRADING__MAX_DAILY_TRADES": "40",
            "POLYBOT_TRADING__MAX_PORTFOLIO_EXPOSURE_USD": "900",
            "POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD": "300",
        },
    },
    "maker": {
        "description": "Market maker: rests quotes 3c inside the model's fair value in books that have stayed wide for a minute (mostly live alternate lines), earns the spread plus the maker rebate, and holds fills to settlement unless the other side fills first.",
        "env": {
            "POLYBOT_TRADING__STRATEGY": "maker",
            "POLYBOT_TRADING__HOLD_TO_SETTLEMENT": "true",
            "POLYBOT_TRADING__MAX_OPEN_POSITIONS": "40",
            "POLYBOT_TRADING__MAX_DAILY_TRADES": "80",
            "POLYBOT_TRADING__MAX_PORTFOLIO_EXPOSURE_USD": "900",
            "POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD": "300",
            "POLYBOT_FILTERS__MIN_HOURS_TO_EXPIRY": "0",
            # Live quoting back on 2026-09-26 22:20 ET as a measured arm: the
            # first 22 live fills settled 11-11, +92.93 (one 4-to-1 short).
            # A book must have been wide for 60 s first.
            "POLYBOT_TRADING__MAKER_LIVE_QUOTES": "true",
            "POLYBOT_TRADING__MAKER_WIDE_SECONDS": "60",
            # College only (2026-09-28): NFL alternate lines are 0.5-4c wide,
            # nothing to quote inside; its 4 NFL fills lost 44.91.
            "POLYBOT_FILTERS__LEAGUES": "cfb",
        },
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
