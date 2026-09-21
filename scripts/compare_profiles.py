#!/usr/bin/env python3
"""Side-by-side scoreboard for the strategy profiles running in paper mode.

Each profile is a full copy of the repo with its own data/trades.db and
heartbeat; strategy differences live in its launchd plist as POLYBOT_*
environment overrides. Usage:

    python scripts/compare_profiles.py            # table
    python scripts/compare_profiles.py --trades   # plus every closed trade
"""
import json
import os
import sqlite3
import sys
import time

PROFILES = {
    "baseline":   os.path.expanduser("~/Projects/polymarket-bot"),
    "aggressive": os.path.expanduser("~/Projects/polybot-profiles/aggressive"),
    "ride":       os.path.expanduser("~/Projects/polybot-profiles/ride"),
    "balanced":       os.path.expanduser("~/Projects/polybot-profiles/balanced"),
}


def stats(root):
    db = os.path.join(root, "data", "trades.db")
    hb = os.path.join(root, "data", "heartbeat.json")
    out = {"trades": 0, "wins": 0, "pnl": 0.0, "open": 0, "exposure": 0.0,
           "avg_hold_min": 0.0, "decisions": 0, "executed": 0, "hb_age_s": None}
    if os.path.exists(hb):
        try:
            h = json.load(open(hb))
            out["hb_age_s"] = int(time.time() - h.get("timestamp", 0))
            out["cycle"] = h.get("cycle")
        except Exception:
            pass
    if not os.path.exists(db):
        return out
    c = sqlite3.connect(db)
    try:
        r = c.execute("select count(*), coalesce(sum(realized_pnl>0),0), coalesce(sum(realized_pnl),0), "
                      "coalesce(avg((julianday(close_time)-julianday(entry_time))*1440),0) from trades").fetchone()
        out.update(trades=r[0], wins=r[1], pnl=r[2], avg_hold_min=r[3])
        r = c.execute("select count(*), coalesce(sum(size_usd),0) from live_position_state").fetchone()
        out.update(open=r[0], exposure=r[1])
        r = c.execute("select count(*), coalesce(sum(decision='executed'),0) from decision_log").fetchone()
        out.update(decisions=r[0], executed=r[1])
        out["by_kind"] = dict(c.execute(
            "select case when slug like 'asc-%' then 'spread' when slug like 'tsc-%' then 'total' else 'ml' end k, "
            "round(sum(realized_pnl),2) from trades group by k").fetchall())
        out["trade_rows"] = c.execute(
            "select slug, side, entry_price, close_price, round(size_usd), round(realized_pnl,2), close_reason, "
            "substr(entry_time,12,5), substr(close_time,12,5) from trades order by close_time").fetchall()
    except sqlite3.Error as e:
        out["error"] = str(e)
    finally:
        c.close()
    return out


def main():
    show_trades = "--trades" in sys.argv
    rows = {name: stats(root) for name, root in PROFILES.items()}
    hdr = f"{'profile':11} {'alive':>6} {'trades':>6} {'wins':>5} {'win%':>5} {'pnl':>8} {'open':>4} {'expo$':>6} {'hold(m)':>7} {'exec/dec':>9}  by kind"
    print(hdr); print("-" * len(hdr))
    for name, s in rows.items():
        alive = "-" if s["hb_age_s"] is None else ("yes" if s["hb_age_s"] < 300 else f"{s['hb_age_s']//60}m")
        wr = f"{100*s['wins']/s['trades']:.0f}" if s["trades"] else "-"
        print(f"{name:11} {alive:>6} {s['trades']:>6} {s['wins']:>5} {wr:>5} {s['pnl']:>8.2f} {s['open']:>4} "
              f"{s['exposure']:>6.0f} {s['avg_hold_min']:>7.0f} {str(s['executed'])+'/'+str(s['decisions']):>9}  {s.get('by_kind', {})}")
    if show_trades:
        for name, s in rows.items():
            print(f"\n== {name} ==")
            for r in s.get("trade_rows", []):
                print("  ", r)


if __name__ == "__main__":
    main()
