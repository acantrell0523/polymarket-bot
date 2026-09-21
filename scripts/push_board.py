#!/usr/bin/env python3
"""Push the live strategy scoreboard to the Vercel board every 30s (launchd).

Reads each profile's heartbeat.json (equity, open positions with marks) and
trades.db (closed trades, decisions), plus the ESPN NFL scoreboard, and POSTs
one JSON document to https://<board>/api/board. Config in ~/.polybot-board.env:
    BOARD_URL=https://....vercel.app
    BOARD_WRITE_KEY=...
"""
import json
import os
import plistlib
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
PROFILES = {
    "baseline":   f"{HOME}/Projects/polymarket-bot",
    "aggressive": f"{HOME}/Projects/polybot-profiles/aggressive",
    "ride":       f"{HOME}/Projects/polybot-profiles/ride",
    "balanced":       f"{HOME}/Projects/polybot-profiles/balanced",
}
KNOBS = ["MIN_EDGE_THRESHOLD", "LEAGUE_MIN_EDGE_OVERRIDE", "MIN_NET_EDGE", "KELLY_FRACTION",
         "MAX_POSITION_SIZE_USD", "MAX_OPEN_POSITIONS", "MAX_DAILY_TRADES", "MIN_PRICE", "MAX_PRICE",
         "ALLOW_MULTIPLE_PER_GAME", "AGGRESSIVE_EXIT_PCT", "TRAILING_STOP_ACTIVATION_PCT",
         "TRAILING_STOP_PCT", "TAKE_PROFIT_THRESHOLD", "STOP_LOSS_THRESHOLD", "LET_IT_RIDE_THRESHOLD",
         "DAILY_LOSS_LIMIT_USD"]
DEFAULTS = {"MIN_EDGE_THRESHOLD": "0.05", "LEAGUE_MIN_EDGE_OVERRIDE": "league table", "MIN_NET_EDGE": "0.02",
            "KELLY_FRACTION": "0.25", "MAX_POSITION_SIZE_USD": "50", "MAX_OPEN_POSITIONS": "5",
            "MAX_DAILY_TRADES": "5", "MIN_PRICE": "0.15", "MAX_PRICE": "0.85", "ALLOW_MULTIPLE_PER_GAME": "false",
            "AGGRESSIVE_EXIT_PCT": "0.30", "TRAILING_STOP_ACTIVATION_PCT": "0.15", "TRAILING_STOP_PCT": "0.10",
            "TAKE_PROFIT_THRESHOLD": "0.05", "STOP_LOSS_THRESHOLD": "0.25", "LET_IT_RIDE_THRESHOLD": "0.70",
            "DAILY_LOSS_LIMIT_USD": "75"}


def load_env():
    env = {}
    try:
        for line in open(f"{HOME}/.polybot-board.env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip('"')
    except OSError:
        pass
    return env


def settings(name):
    label = "com.polymarket.bot" + ("" if name == "baseline" else f".{name}")
    try:
        pl = plistlib.load(open(f"{HOME}/Library/LaunchAgents/{label}.plist", "rb"))
        env = pl.get("EnvironmentVariables", {})
    except Exception:
        env = {}
    out = {}
    for k in KNOBS:
        v = env.get(f"POLYBOT_TRADING__{k}")
        out[k.lower()] = {"value": v if v is not None else DEFAULTS[k], "overridden": v is not None}
    return out


def profile(name, root):
    p = {"name": name, "alive": False, "hb_age_s": None, "cycle": None, "cash": None, "equity": None,
         "positions": [], "trades": 0, "wins": 0, "realized": 0.0, "avg_hold_min": 0.0, "exposure": 0.0,
         "decisions": 0, "executed": 0, "by_kind": {}, "closed": [], "rejections": [],
         "settings": settings(name)}
    try:
        h = json.load(open(f"{root}/data/heartbeat.json"))
        age = time.time() - h.get("timestamp", 0)
        p.update(hb_age_s=int(age), alive=age < 300, cycle=h.get("cycle"), cash=h.get("cash"),
                 equity=h.get("equity"), positions=h.get("positions", []))
    except Exception:
        pass
    db = f"{root}/data/trades.db"
    if os.path.exists(db):
        c = sqlite3.connect(db)
        try:
            r = c.execute("select count(*), coalesce(sum(realized_pnl>0),0), coalesce(sum(realized_pnl),0), "
                          "coalesce(avg((julianday(close_time)-julianday(entry_time))*1440),0) from trades").fetchone()
            p.update(trades=r[0], wins=r[1], realized=round(r[2], 2), avg_hold_min=round(r[3], 1))
            p["exposure"] = round(sum(x.get("size_usd", 0) for x in p["positions"]), 2)
            r = c.execute("select count(*), coalesce(sum(decision='executed'),0) from decision_log").fetchone()
            p.update(decisions=r[0], executed=r[1])
            p["by_kind"] = {k: round(v, 2) for k, v in c.execute(
                "select case when slug like 'asc-%' then 'spread' when slug like 'tsc-%' then 'total' else 'ml' end, "
                "sum(realized_pnl) from trades group by 1").fetchall()}
            p["closed"] = [dict(zip(("slug", "side", "entry_price", "close_price", "size_usd", "pnl", "reason",
                                     "entry_time", "close_time"), row))
                           for row in c.execute("select slug, side, entry_price, close_price, round(size_usd,2), "
                                                "round(realized_pnl,2), close_reason, entry_time, close_time "
                                                "from trades order by close_time desc limit 60")]
            p["rejections"] = [{"reason": a, "n": b} for a, b in c.execute(
                "select case when reason like 'price_%' then 'price outside band' "
                "when reason like 'already_have_position%' then 'already on this game' "
                "when reason like 'edge_%' then 'edge below league minimum' "
                "when reason like 'net_edge%' then 'net edge after costs' "
                "when reason like 'liquidity%' then 'book too thin' "
                "when reason like 'daily_limit%' then 'daily trade limit' "
                "when reason like 'last_5_minutes%' then 'last 5 minutes' "
                "when reason like 'spread_%' then 'spread too wide' "
                "when reason like 'only_%_books' then 'no book consensus' else reason end r, count(*) "
                "from decision_log where decision='rejected' group by r order by 2 desc limit 8")]
        finally:
            c.close()
    unreal = round(sum(x.get("unrealized_pnl", 0) for x in p["positions"]), 2)
    p["unrealized"] = unreal
    p["total"] = round(p["realized"] + unreal, 2)
    # Paper cash resets to the configured bankroll on every restart, so
    # equity is rebuilt from the ledger: bankroll + realized + unrealized.
    p["bankroll"] = 1000.0
    p["equity"] = round(1000.0 + p["realized"] + unreal, 2)
    p["cash"] = round(1000.0 + p["realized"] - p["exposure"], 2)
    return p


def games():
    out = []
    for url in ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
                "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=400"):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=10))
        except Exception:
            continue
        for e in d.get("events", []):
            # college: only live games and today's slate keep the strip readable
            if "college" in url and e["status"]["type"]["state"] not in ("in",) and \
                    e.get("date", "")[:10] != datetime.now(timezone.utc).date().isoformat():
                continue
            c = e["competitions"][0]
            t = {x["homeAway"]: x for x in c["competitors"]}
            out.append({"away": t["away"]["team"]["abbreviation"], "home": t["home"]["team"]["abbreviation"],
                        "away_score": t["away"].get("score"), "home_score": t["home"].get("score"),
                        "status": e["status"]["type"]["shortDetail"],
                        "state": e["status"]["type"]["state"],
                        "league": "cfb" if "college" in url else "nfl"})
    return out


def main():
    """Write data.json into the board site and redeploy it (static hosting,
    no external store). Vercel Hobby allows 100 deploys/day, so launchd runs
    this every 20 minutes. --print dumps the document instead."""
    import subprocess
    site = f"{HOME}/Projects/polybot-board/site"
    doc = {"updated_at": datetime.now(timezone.utc).isoformat(),
           "profiles": [profile(n, r) for n, r in PROFILES.items()],
           "games": games()}
    if "--print" in sys.argv:
        print(json.dumps(doc, indent=1)[:3000])
        return
    tmp = f"{site}/data.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, f"{site}/data.json")
    env = dict(os.environ, PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
    r = subprocess.run(["vercel", "deploy", "--prod", "--yes"], cwd=site, env=env,
                       capture_output=True, text=True, timeout=240)
    print(time.strftime("%H:%M:%S"), "deploy", "ok" if r.returncode == 0 else "FAILED",
          (r.stdout + r.stderr).strip().splitlines()[-1][:120] if (r.stdout + r.stderr).strip() else "")


if __name__ == "__main__":
    main()
