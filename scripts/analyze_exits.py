#!/usr/bin/env python3
"""Canned exit telemetry queries against the exit_log table.

Usage:
    python scripts/analyze_exits.py [--days N]

Reads data/trades.db and prints analysis to stdout.
Requires at least one closed position with exit_log data to produce output.
"""

import argparse
import sqlite3
import os
import sys
import json
from datetime import datetime, timezone, timedelta

# Resolve DB path relative to this script's parent directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_REPO_ROOT, "data", "trades.db")


def _connect() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at {DB_PATH}")
        print("       Run the bot at least once to create it.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _divider(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Query 1: Distribution of close reasons
# ---------------------------------------------------------------------------

def q_close_reason_distribution(conn: sqlite3.Connection, since_iso: str):
    _divider("Close Reason Distribution")
    rows = conn.execute(
        """
        SELECT
            close_reason,
            COUNT(*)                                       AS n,
            ROUND(AVG(realized_pnl), 2)                   AS avg_pnl,
            ROUND(SUM(realized_pnl), 2)                   AS total_pnl,
            ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                           AS win_rate_pct
        FROM exit_log
        WHERE close_time >= ?
        GROUP BY close_reason
        ORDER BY n DESC
        """,
        (since_iso,),
    ).fetchall()

    if not rows:
        print("  No data in window.")
        return

    print(f"  {'Reason':<22} {'N':>5}  {'Win%':>6}  {'AvgP&L':>8}  {'TotalP&L':>10}")
    print(f"  {'─'*22} {'─'*5}  {'─'*6}  {'─'*8}  {'─'*10}")
    for r in rows:
        print(
            f"  {r['close_reason']:<22} {r['n']:>5}  "
            f"{r['win_rate_pct']:>5.1f}%  "
            f"${r['avg_pnl']:>7.2f}  "
            f"${r['total_pnl']:>9.2f}"
        )


# ---------------------------------------------------------------------------
# Query 2: Average peak unrealized P&L by close_reason
# ---------------------------------------------------------------------------

def q_peak_pnl_by_reason(conn: sqlite3.Connection, since_iso: str):
    _divider("Peak Unrealized P&L by Close Reason")
    rows = conn.execute(
        """
        SELECT
            close_reason,
            COUNT(*)                                        AS n,
            ROUND(AVG(max_favorable_pnl_usd), 2)           AS avg_peak_favorable,
            ROUND(AVG(max_adverse_pnl_usd), 2)             AS avg_peak_adverse,
            ROUND(AVG(peak_unrealized_pct) * 100, 1)       AS avg_peak_pct,
            ROUND(AVG(realized_pnl), 2)                    AS avg_realized
        FROM exit_log
        WHERE close_time >= ?
        GROUP BY close_reason
        ORDER BY avg_peak_favorable DESC
        """,
        (since_iso,),
    ).fetchall()

    if not rows:
        print("  No data in window.")
        return

    print(
        f"  {'Reason':<22} {'N':>5}  "
        f"{'PeakFav$':>9}  {'PeakAdv$':>9}  "
        f"{'Peak%':>7}  {'Realized$':>10}"
    )
    print(
        f"  {'─'*22} {'─'*5}  "
        f"{'─'*9}  {'─'*9}  {'─'*7}  {'─'*10}"
    )
    for r in rows:
        print(
            f"  {r['close_reason']:<22} {r['n']:>5}  "
            f"${r['avg_peak_favorable']:>8.2f}  "
            f"${r['avg_peak_adverse']:>8.2f}  "
            f"{r['avg_peak_pct']:>6.1f}%  "
            f"${r['avg_realized']:>9.2f}"
        )


# ---------------------------------------------------------------------------
# Query 3: Let-it-ride win rate (triggered vs not triggered)
# ---------------------------------------------------------------------------

def q_let_it_ride_win_rate(conn: sqlite3.Connection, since_iso: str):
    _divider("Win Rate: Let-It-Ride Triggered vs Not")
    rows = conn.execute(
        """
        SELECT
            let_it_ride_triggered,
            COUNT(*)                                                   AS n,
            ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                                       AS win_rate_pct,
            ROUND(AVG(realized_pnl), 2)                               AS avg_pnl,
            ROUND(AVG(num_let_it_ride_triggers), 1)                   AS avg_triggers
        FROM exit_log
        WHERE close_time >= ?
        GROUP BY let_it_ride_triggered
        ORDER BY let_it_ride_triggered DESC
        """,
        (since_iso,),
    ).fetchall()

    if not rows:
        print("  No data in window.")
        return

    print(f"  {'Let-it-Ride?':<15} {'N':>5}  {'Win%':>6}  {'AvgP&L':>8}  {'AvgTriggers':>12}")
    print(f"  {'─'*15} {'─'*5}  {'─'*6}  {'─'*8}  {'─'*12}")
    for r in rows:
        label = "YES" if r["let_it_ride_triggered"] else "NO"
        print(
            f"  {label:<15} {r['n']:>5}  "
            f"{r['win_rate_pct']:>5.1f}%  "
            f"${r['avg_pnl']:>7.2f}  "
            f"{r['avg_triggers']:>12.1f}"
        )

    # Follow-up: of LIR positions that eventually closed via stop_loss, how bad?
    sl_after_lir = conn.execute(
        """
        SELECT COUNT(*) AS n, ROUND(AVG(realized_pnl), 2) AS avg_pnl
        FROM exit_log
        WHERE close_time >= ?
          AND let_it_ride_triggered = 1
          AND close_reason = 'stop_loss'
        """,
        (since_iso,),
    ).fetchone()
    if sl_after_lir and sl_after_lir["n"] > 0:
        print(
            f"\n  ⚠  {sl_after_lir['n']} position(s) triggered let_it_ride "
            f"then closed via stop_loss  (avg P&L: ${sl_after_lir['avg_pnl']:.2f})"
        )


# ---------------------------------------------------------------------------
# Query 4: Stop-loss tightness — were we close to profitability?
# ---------------------------------------------------------------------------

def q_stop_loss_analysis(conn: sqlite3.Connection, since_iso: str):
    _divider("Stop-Loss Analysis")
    rows = conn.execute(
        """
        SELECT
            COUNT(*)                                                  AS n,
            ROUND(AVG(exit_threshold_distance) * 100, 2)             AS avg_dist_pct,
            ROUND(MIN(exit_threshold_distance) * 100, 2)             AS min_dist_pct,
            ROUND(AVG(max_favorable_pnl_usd), 2)                     AS avg_peak_before_sl,
            ROUND(AVG(realized_pnl), 2)                              AS avg_realized,
            SUM(CASE WHEN max_favorable_pnl_usd > 0 THEN 1 ELSE 0 END) AS was_winning_first
        FROM exit_log
        WHERE close_time >= ?
          AND close_reason = 'stop_loss'
        """,
        (since_iso,),
    ).fetchone()

    if not rows or rows["n"] == 0:
        print("  No stop_loss exits in window.")
        return

    r = rows
    print(f"  Total stop-loss exits:          {r['n']}")
    print(f"  Positions that were winning first: {r['was_winning_first']} "
          f"({100*r['was_winning_first']//r['n']:.0f}%)")
    print(f"  Avg peak favorable P&L before SL: ${r['avg_peak_before_sl']:.2f}")
    print(f"  Avg distance from SL boundary:    {r['avg_dist_pct']:.2f}% of entry")
    print(f"  Min distance from SL boundary:    {r['min_dist_pct']:.2f}% of entry")
    print(f"  Avg realized P&L at stop-loss:    ${r['avg_realized']:.2f}")

    if r["avg_dist_pct"] < 5.0:
        print("\n  ⚠  Avg distance < 5% suggests stop-loss may be catching normal noise.")
    if r["avg_peak_before_sl"] and r["avg_peak_before_sl"] > 2.0:
        print(f"\n  ⚠  Positions were profitable on average "
              f"(${r['avg_peak_before_sl']:.2f}) before being stopped out.")


# ---------------------------------------------------------------------------
# Query 5: Average hold time by close reason
# ---------------------------------------------------------------------------

def q_hold_time_by_reason(conn: sqlite3.Connection, since_iso: str):
    _divider("Average Hold Time by Close Reason")
    rows = conn.execute(
        """
        SELECT
            close_reason,
            COUNT(*)                            AS n,
            ROUND(AVG(hold_seconds) / 60, 1)   AS avg_minutes,
            ROUND(MIN(hold_seconds) / 60, 1)   AS min_minutes,
            ROUND(MAX(hold_seconds) / 60, 1)   AS max_minutes
        FROM exit_log
        WHERE close_time >= ?
        GROUP BY close_reason
        ORDER BY avg_minutes DESC
        """,
        (since_iso,),
    ).fetchall()

    if not rows:
        print("  No data in window.")
        return

    print(f"  {'Reason':<22} {'N':>5}  {'AvgMin':>8}  {'MinMin':>8}  {'MaxMin':>8}")
    print(f"  {'─'*22} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in rows:
        print(
            f"  {r['close_reason']:<22} {r['n']:>5}  "
            f"{r['avg_minutes']:>7.1f}m  "
            f"{r['min_minutes']:>7.1f}m  "
            f"{r['max_minutes']:>7.1f}m"
        )


# ---------------------------------------------------------------------------
# Query 6: P&L by entry_estimated_prob bucket
# ---------------------------------------------------------------------------

def q_pnl_by_prob_bucket(conn: sqlite3.Connection, since_iso: str):
    _divider("P&L by Entry Estimated Probability Bucket")
    rows = conn.execute(
        """
        SELECT
            CASE
                WHEN entry_estimated_prob < 0.40 THEN '<40%'
                WHEN entry_estimated_prob < 0.50 THEN '40-50%'
                WHEN entry_estimated_prob < 0.60 THEN '50-60%'
                WHEN entry_estimated_prob < 0.70 THEN '60-70%'
                ELSE '70%+'
            END                                            AS prob_bucket,
            COUNT(*)                                       AS n,
            ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                           AS win_rate_pct,
            ROUND(AVG(realized_pnl), 2)                   AS avg_pnl,
            ROUND(SUM(realized_pnl), 2)                   AS total_pnl
        FROM exit_log
        WHERE close_time >= ?
        GROUP BY prob_bucket
        ORDER BY entry_estimated_prob
        """,
        (since_iso,),
    ).fetchall()

    if not rows:
        print("  No data in window.")
        return

    print(f"  {'Prob Bucket':<12} {'N':>5}  {'Win%':>6}  {'AvgP&L':>8}  {'TotalP&L':>10}")
    print(f"  {'─'*12} {'─'*5}  {'─'*6}  {'─'*8}  {'─'*10}")
    for r in rows:
        print(
            f"  {r['prob_bucket']:<12} {r['n']:>5}  "
            f"{r['win_rate_pct']:>5.1f}%  "
            f"${r['avg_pnl']:>7.2f}  "
            f"${r['total_pnl']:>9.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze exit_log telemetry from trades.db")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Look-back window in days (default: 30)"
    )
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    since_iso = since.isoformat()

    conn = _connect()

    if not _table_exists(conn, "exit_log"):
        print("[ERROR] exit_log table does not exist yet.")
        print("       Run the bot once to create the table (it lazy-inits on startup).")
        conn.close()
        sys.exit(1)

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM exit_log WHERE close_time >= ?", (since_iso,)
    ).fetchone()[0]

    print(f"\n{'='*60}")
    print(f"  Exit Log Analysis  |  Last {args.days} day(s)")
    print(f"  DB: {DB_PATH}")
    print(f"  Rows in window: {total_rows}")
    print(f"{'='*60}")

    if total_rows == 0:
        print("\n  No exit_log rows in the requested window.")
        print("  Close a position first, then re-run this script.")
        conn.close()
        return

    q_close_reason_distribution(conn, since_iso)
    q_peak_pnl_by_reason(conn, since_iso)
    q_let_it_ride_win_rate(conn, since_iso)
    q_stop_loss_analysis(conn, since_iso)
    q_hold_time_by_reason(conn, since_iso)
    q_pnl_by_prob_bucket(conn, since_iso)

    print(f"\n{'='*60}\n")
    conn.close()


if __name__ == "__main__":
    main()
