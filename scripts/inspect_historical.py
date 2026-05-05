"""
Inspect the historical ingestion database.

Prints:
  1. Game count by date (moneyline_game markets only)
  2. Market count per game (grouped by espn_game_id)
  3. Snapshot count per market — sanity-checks density
  4. Settled outcome distribution
  5. Sample trade records for the two highest-density markets
"""
import datetime
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.historical_db import (
    get_all_markets,
    get_games_by_date,
    get_sample_snapshots,
    get_settled_outcome_distribution,
    get_snapshot_counts,
    get_conn,
)


def _ts_to_iso(ts: int) -> str:
    try:
        return datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def main() -> None:
    # ── 0. Quick existence check ──────────────────────────────────────────────
    conn = get_conn()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    if "historical_markets" not in tables or "historical_snapshots" not in tables:
        print("ERROR: historical tables not found. Run ingest_historical.py first.")
        sys.exit(1)

    markets   = get_all_markets()
    snap_cnts = get_snapshot_counts()   # [{slug, cnt}]
    snap_map  = {r["slug"]: r["cnt"] for r in snap_cnts}

    # ── 1. Market summary ─────────────────────────────────────────────────────
    total_markets   = len(markets)
    game_markets    = [m for m in markets if m["market_type"] == "moneyline_game"]
    outright_markets = [m for m in markets if m["market_type"] == "nba_outright"]
    total_snapshots = sum(snap_map.values())

    print("=" * 60)
    print("HISTORICAL DATABASE SUMMARY")
    print("=" * 60)
    print(f"Total markets ingested : {total_markets}")
    print(f"  moneyline_game       : {len(game_markets)}")
    print(f"  nba_outright         : {len(outright_markets)}")
    print(f"  other                : {total_markets - len(game_markets) - len(outright_markets)}")
    print(f"Total snapshots        : {total_snapshots}")
    print()

    # ── 2. Game count by date ─────────────────────────────────────────────────
    print("─" * 60)
    print("GAME MARKETS BY DATE")
    print("─" * 60)
    by_date = get_games_by_date()
    if by_date:
        for row in by_date:
            print(f"  {row['game_date']}  markets={row['market_count']}")
    else:
        print("  (no moneyline_game markets found)")
    print()

    # ── 3. Market count per ESPN game ─────────────────────────────────────────
    print("─" * 60)
    print("MARKET COUNT PER ESPN GAME")
    print("─" * 60)
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT espn_game_id,
               home_team, away_team, game_start_time,
               COUNT(*) as cnt
        FROM historical_markets
        WHERE market_type = 'moneyline_game'
          AND espn_game_id != ''
        GROUP BY espn_game_id
        ORDER BY game_start_time
        """
    ).fetchall()
    conn.close()

    if rows:
        for r in rows:
            date_part = str(r["game_start_time"])[:10]
            print(
                f"  ESPN {r['espn_game_id']}  "
                f"{r['away_team']} @ {r['home_team']}  "
                f"{date_part}  markets={r['cnt']}"
            )
    else:
        print("  (no game markets with ESPN IDs)")
    print()

    # ── 4. Snapshot count per market ──────────────────────────────────────────
    print("─" * 60)
    print("SNAPSHOT COUNT PER MARKET (all markets, sorted by density)")
    print("─" * 60)
    if snap_cnts:
        for row in snap_cnts:
            mtype = next(
                (m["market_type"] for m in markets if m["slug"] == row["slug"]), "?"
            )
            q = next(
                (m["question"] for m in markets if m["slug"] == row["slug"]), ""
            )
            print(
                f"  [{mtype:15s}] {row['slug'][:50]:<50}  pts={row['cnt']:>4}"
            )
            if q:
                print(f"    q: {q[:75]}")
    else:
        print("  (no snapshots found)")
    print()

    # ── 5. Settled outcome distribution ───────────────────────────────────────
    print("─" * 60)
    print("SETTLED OUTCOME DISTRIBUTION")
    print("─" * 60)
    outcomes = get_settled_outcome_distribution()
    if outcomes:
        for row in outcomes:
            label = row["settled_outcome"] or "(empty)"
            print(f"  {label:<10}  count={row['cnt']}")
    else:
        print("  (no markets)")
    print()

    # ── 6. Sample snapshot records for two highest-density markets ────────────
    print("─" * 60)
    print("SAMPLE SNAPSHOTS (top-2 markets by density)")
    print("─" * 60)
    top2 = snap_cnts[:2]
    for entry in top2:
        slug = entry["slug"]
        q    = next((m["question"] for m in markets if m["slug"] == slug), "")
        mtype = next((m["market_type"] for m in markets if m["slug"] == slug), "")
        snaps = get_sample_snapshots(slug, limit=5)

        print(f"\n  Market : {slug}")
        print(f"  Type   : {mtype}")
        print(f"  Q      : {q[:80]}")
        print(f"  Density: {entry['cnt']} snapshots")
        print(f"  {'Timestamp':<20}  {'Date (UTC)':<13}  {'Price':>8}")
        print(f"  {'-'*20}  {'-'*13}  {'-'*8}")
        for s in snaps:
            print(
                f"  {str(s['timestamp']):<20}  "
                f"{_ts_to_iso(s['timestamp']):<13}  "
                f"{s['polymarket_price']:>8.4f}"
            )

    print()
    print("=" * 60)
    print("Inspect complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
