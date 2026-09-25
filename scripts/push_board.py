#!/usr/bin/env python3
"""Push the live strategy scoreboard for https://polybot-board.vercel.app.

launchd com.polymarket.board runs this every 30 s. It builds one JSON
document from each profile's heartbeat.json and trades.db plus the ESPN
NFL/college scoreboards and writes two Upstash Redis keys:

  polybot:board       the full document (~30-45 KB), rewritten only when its
                      content changes, or every 10 minutes
  polybot:board:live  timestamps, bot liveness and decision counters (~1 KB),
                      rewritten every run

The site's /api/board reads both with one MGET and overlays the live key, so
the page stays 30 s fresh while overnight runs send ~1 KB instead of ~40 KB.
The Redis database is shared with the FR triage board (upstash-kv-coffee-clock,
free plan: 10 GB bandwidth and 500K commands a month).

    python scripts/push_board.py            # push to Redis (30-second job)
    python scripts/push_board.py --deploy   # also redeploy the site (page edits)
    python scripts/push_board.py --print    # dump the document, push nothing

Config in ~/.polybot-board.env: REDIS_REST_URL, REDIS_REST_TOKEN.
Moved off Vercel Blob on 2026-09-25: Hobby Blob allows 2,000 put/list
operations a month and locks the store for 30 days when exceeded.
"""
import hashlib
import json
import os
import plistlib
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiles import PROFILES as _PROFILES, root as _root, plist_path as _plist_path  # noqa: E402

PROFILES = {name: _root(name) for name in _PROFILES}
# (config section, field) shown on the board; values from each profile's
# plist overrides, else the configs/config.yaml default below.
KNOBS = [("TRADING", k) for k in (
    "ENTRY_WINDOW", "HOLD_TO_SETTLEMENT", "PREGAME_MAX_HOURS", "MARKET_KINDS",
    "MAX_PORTFOLIO_EXPOSURE_USD", "MIN_EDGE_THRESHOLD", "MIN_NET_EDGE", "REQUIRE_ROUND_TRIP_EDGE", "KELLY_FRACTION",
    "MAX_POSITION_SIZE_USD", "MAX_OPEN_POSITIONS", "MAX_DAILY_TRADES", "MIN_PRICE", "MAX_PRICE",
    "REENTRY_AFTER_STOP", "AGGRESSIVE_EXIT_PCT", "TRAILING_STOP_ENABLED",
    "TRAILING_STOP_ACTIVATION_PCT", "TRAILING_STOP_PCT", "TAKE_PROFIT_THRESHOLD",
    "STOP_LOSS_THRESHOLD", "LET_IT_RIDE_THRESHOLD", "DAILY_LOSS_LIMIT_USD")] + [("SIGNALS", "LIVE_PRIMARY")]
DEFAULTS = {"ENTRY_WINDOW": "any", "HOLD_TO_SETTLEMENT": "false", "PREGAME_MAX_HOURS": "0",
            "MARKET_KINDS": "ml,spread,total", "MAX_PORTFOLIO_EXPOSURE_USD": "500",
            "MIN_EDGE_THRESHOLD": "0.05", "MIN_NET_EDGE": "0.02", "REQUIRE_ROUND_TRIP_EDGE": "true",
            "KELLY_FRACTION": "0.25", "MAX_POSITION_SIZE_USD": "50", "MAX_OPEN_POSITIONS": "8",
            "MAX_DAILY_TRADES": "15", "MIN_PRICE": "0.15", "MAX_PRICE": "0.85",
            "REENTRY_AFTER_STOP": "false", "AGGRESSIVE_EXIT_PCT": "0.30", "TRAILING_STOP_ENABLED": "true",
            "TRAILING_STOP_ACTIVATION_PCT": "0.15", "TRAILING_STOP_PCT": "0.10",
            "TAKE_PROFIT_THRESHOLD": "0.05", "STOP_LOSS_THRESHOLD": "0.25", "LET_IT_RIDE_THRESHOLD": "0.70",
            "DAILY_LOSS_LIMIT_USD": "75", "LIVE_PRIMARY": "books"}


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
    try:
        pl = plistlib.load(open(_plist_path(name), "rb"))
        env = pl.get("EnvironmentVariables", {})
    except Exception:
        env = {}
    out = {}
    for section, k in KNOBS:
        v = env.get(f"POLYBOT_{section}__{k}")
        out[k.lower()] = {"value": v if v is not None else DEFAULTS[k], "overridden": v is not None}
    return out


def profile(name, root):
    p = {"name": name, "description": _PROFILES.get(name, {}).get("description", ""), "alive": False, "hb_age_s": None, "cycle": None, "cash": None, "equity": None,
         "positions": [], "trades": 0, "wins": 0, "realized": 0.0, "avg_hold_min": 0.0, "exposure": 0.0,
         "decisions": 0, "executed": 0, "by_kind": {}, "closed": [], "rejections": [],
         "settings": settings(name)}
    hb_v2 = None
    try:
        h = json.load(open(f"{root}/data/heartbeat.json"))
        if h.get("accounting_version") == 2:
            hb_v2 = h
        age = time.time() - h.get("timestamp", 0)
        p.update(hb_age_s=int(age), alive=age < 300, cycle=h.get("cycle"), cash=h.get("cash"),
                 equity=h.get("equity"), positions=h.get("positions", []),
                 paused_until=h.get("entries_paused_until"))
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
    p["fees"] = 0.0
    if os.path.exists(db):
        c = sqlite3.connect(db)
        try:
            cols = {r[1] for r in c.execute("PRAGMA table_info(trades)")}
            if "entry_fees" in cols:
                p["fees"] = round(c.execute("select coalesce(sum(entry_fees + exit_fees),0) from trades").fetchone()[0], 2)
        finally:
            c.close()
    if hb_v2 is not None:
        # Accounting v2 (executable fills, both fees, persisted cash): the
        # bot's own heartbeat is the source of truth. Equity = cash + open
        # collateral + mark-to-exit value net of the exit fee.
        p["bankroll"] = float(hb_v2.get("initial_bankroll") or 1000.0)
        p["cash"] = round(float(hb_v2.get("cash", 0.0)), 2)
        p["equity"] = round(float(hb_v2.get("equity", 0.0)), 2)
        p["total"] = round(p["equity"] - p["bankroll"], 2)
        p["unrealized"] = round(p["total"] - p["realized"], 2)
        p["accounting"] = "v2"
        return p
    unreal = round(sum(x.get("unrealized_pnl", 0) for x in p["positions"]), 2)
    p["unrealized"] = unreal
    p["total"] = round(p["realized"] + unreal, 2)
    # Legacy ledgers reset paper cash on restart, so equity is rebuilt from
    # the ledger: bankroll + realized + unrealized.
    p["bankroll"] = 1000.0
    p["equity"] = round(1000.0 + p["realized"] + unreal, 2)
    p["cash"] = round(1000.0 + p["realized"] - p["exposure"], 2)
    p["accounting"] = "legacy"
    return p


GAMES_CACHE = f"{HOME}/Projects/polymarket-bot/data/shared/board_games.json"


def games():
    """ESPN NFL + college scoreboards, cached 90 s (the college payload is ~1 MB)."""
    try:
        if time.time() - os.path.getmtime(GAMES_CACHE) <= 90:
            with open(GAMES_CACHE) as fh:
                return json.load(fh)
    except (OSError, ValueError):
        pass
    out = _fetch_games()
    if out:
        try:
            with open(GAMES_CACHE, "w") as fh:
                json.dump(out, fh)
        except OSError:
            pass
    return out


def _fetch_games():
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


PUSH_STATE = f"{HOME}/Projects/polymarket-bot/data/shared/board_push_state.json"
# Fields that change every bot cycle without anything happening on the board.
LIVE_FIELDS = ("alive", "hb_age_s", "cycle", "decisions", "executed", "rejections")
FULL_EVERY_S = 600


def redis(env: dict, command: list):
    url, token = env.get("REDIS_REST_URL"), env.get("REDIS_REST_TOKEN")
    if not url or not token:
        raise RuntimeError("REDIS_REST_URL / REDIS_REST_TOKEN missing from ~/.polybot-board.env")
    req = urllib.request.Request(url, data=json.dumps(command).encode(), method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("result")


def push(doc: dict, env: dict) -> str:
    live = {"updated_at": doc["updated_at"],
            "p": {p["name"]: {k: p.get(k) for k in LIVE_FIELDS} for p in doc["profiles"]}}
    stable = {"games": doc["games"],
              "profiles": [{k: v for k, v in p.items() if k not in LIVE_FIELDS} for p in doc["profiles"]]}
    digest = hashlib.sha1(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    try:
        with open(PUSH_STATE) as fh:
            last = json.load(fh)
    except (OSError, ValueError):
        last = {}
    if last.get("digest") == digest and time.time() - last.get("full_at", 0) < FULL_EVERY_S:
        redis(env, ["SET", "polybot:board:live", json.dumps(live)])
        return ""
    body = json.dumps(doc)
    redis(env, ["MSET", "polybot:board", body, "polybot:board:live", json.dumps(live)])
    tmp = PUSH_STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"digest": digest, "full_at": time.time()}, fh)
    os.replace(tmp, PUSH_STATE)
    return f"full {len(body) // 1024} KB"


def main():
    import subprocess
    site = f"{HOME}/Projects/polybot-board/site"
    env = load_env()
    doc = {"updated_at": datetime.now(timezone.utc).isoformat(),
           "profiles": [profile(n, r) for n, r in PROFILES.items()],
           "games": games()}
    if "--print" in sys.argv:
        print(json.dumps(doc, indent=1)[:3000])
        return
    # static fallback, served only if the live store is unreachable
    tmp = f"{site}/data.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, f"{site}/data.json")
    try:
        line = push(doc, env)
    except Exception as e:  # one line per failed run, not a traceback every 30 s
        line = f"redis push FAILED: {type(e).__name__}: {str(e)[:120]}"
    if "--deploy" in sys.argv:
        penv = dict(os.environ, PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
        r = subprocess.run(["vercel", "deploy", "--prod", "--yes"], cwd=site, env=penv,
                           capture_output=True, text=True, timeout=240)
        tail = (r.stdout + r.stderr).strip().splitlines()
        line += f" | deploy {'ok' if r.returncode == 0 else 'FAILED'} {tail[-1][:80] if tail else ''}"
    if line:  # quiet when only the live key moved
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}", flush=True)


if __name__ == "__main__":
    main()
