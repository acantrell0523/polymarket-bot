"""Supervisor agent: daily report, morning briefing, kill switch (READ-ONLY).

The supervisor NEVER modifies config.yaml. It only:
  - Reports performance (daily review, edge validation)
  - Sends morning briefing with top edge opportunities
  - Activates kill switch to STOP the bot (does not change parameters)
  - Checks kill switch conditions every 15 minutes

All parameter changes are manual — only the operator decides.
"""

import os
import signal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.config import load_config
from utils.logger import TradingLogger
from bot.alerts import SlackAlerter, COLOR_GREEN, COLOR_RED, COLOR_GRAY
from bot import trade_db
from bot.edge_log import (
    generate_edge_validation_report,
    format_edge_report_slack,
    run_morning_scan,
    format_morning_briefing,
)


KILL_SWITCH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kill_switch")


class Supervisor:
    """Read-only supervisor. Reports performance, never changes config."""

    def __init__(self):
        self.config = load_config()
        self.logger = TradingLogger(
            name="supervisor",
            level="INFO",
            log_file="./reports/supervisor.log",
            console=True,
        )
        self.alerter = SlackAlerter(
            webhook_url=self.config.alerts.slack_webhook_url,
            enabled=self.config.alerts.enabled,
        )
        self._starting_value = self._get_account_value() or self.config.backtest.initial_bankroll_usd

        self.logger.info("supervisor_initialized", {
            "starting_value": self._starting_value,
            "mode": "READ-ONLY (no config changes)",
        })

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        try:
            from polymarket_us import PolymarketUS
            return PolymarketUS(
                key_id=self.config.wallet.key_id,
                secret_key=self.config.wallet.secret_key,
            )
        except Exception as e:
            self.logger.error("supervisor_client_failed", {"error": str(e)})
            return None

    def _get_account_value(self) -> Optional[float]:
        client = self._get_client()
        if not client:
            return None
        try:
            bal = client.account.balances()
            b = bal["balances"][0]
            buying_power = float(b.get("buyingPower", 0))
            positions = client.portfolio.positions()
            pos_value = sum(
                float(p.get("cashValue", {}).get("value", "0"))
                for p in positions.get("positions", {}).values()
                if int(p.get("netPosition", "0")) != 0
            )
            return buying_power + pos_value
        except Exception as e:
            self.logger.error("account_value_failed", {"error": str(e)})
            return None

    def _get_exchange_balance(self) -> float:
        client = self._get_client()
        if not client:
            return 0
        try:
            bal = client.account.balances()
            return float(bal["balances"][0].get("buyingPower", 0))
        except Exception:
            return 0

    def _get_open_positions(self) -> Dict[str, Any]:
        client = self._get_client()
        if not client:
            return {}
        try:
            result = client.portfolio.positions()
            return {
                slug: p for slug, p in result.get("positions", {}).items()
                if int(p.get("netPosition", "0")) != 0
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Market type classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_market_type(slug: str) -> str:
        if slug.startswith("asc-") or "spread" in slug or "pos-" in slug:
            return "spread"
        if slug.startswith("tsc-") or "pt5" in slug:
            return "totals"
        return "moneyline"

    # ------------------------------------------------------------------
    # Performance analysis (read-only)
    # ------------------------------------------------------------------

    def _compute_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl": 0, "avg_win": 0,
                "avg_loss": 0, "profit_factor": 0,
            }
        wins = [t for t in trades if t["realized_pnl"] > 0]
        losses = [t for t in trades if t["realized_pnl"] <= 0]
        total_win = sum(t["realized_pnl"] for t in wins)
        total_loss = abs(sum(t["realized_pnl"] for t in losses))

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": sum(t["realized_pnl"] for t in trades),
            "avg_win": total_win / len(wins) if wins else 0,
            "avg_loss": -total_loss / len(losses) if losses else 0,
            "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
        }

    def _compute_by_type(self, trades: List[Dict]) -> Dict[str, Dict]:
        by_type = {}
        for mtype in ("moneyline", "spread", "totals"):
            typed = [t for t in trades if self._classify_market_type(t["slug"]) == mtype]
            by_type[mtype] = self._compute_metrics(typed)
        return by_type

    # ------------------------------------------------------------------
    # Exit telemetry helpers (read-only queries on exit_log)
    # ------------------------------------------------------------------

    def _get_exit_insights(self, since_iso: str) -> Optional[Dict[str, Any]]:
        """Query exit_log for the given window.

        Runs four diagnostic queries:
          1. Close-reason distribution + avg peak unrealized P&L per reason
          2. Let-it-ride triggered count and P&L summary
          3. Let-it-ride positions that subsequently stopped out
          4. Stop-loss tightness: count of stop_loss exits where take_profit
             edge was within 2% at close time (reads exit_proximity_json)

        Returns a dict on success, or None when the window is empty or any
        DB error occurs.  Never raises — all errors are logged at WARNING.
        """
        try:
            conn = trade_db._get_conn()

            total = conn.execute(
                "SELECT COUNT(*) FROM exit_log WHERE close_time >= ?",
                (since_iso,),
            ).fetchone()[0]

            if total == 0:
                conn.close()
                return None

            # 1. Close-reason distribution + avg peak unrealized P&L
            reasons = conn.execute(
                "SELECT close_reason, COUNT(*) AS n, "
                "ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) "
                "/ COUNT(*), 1) AS win_rate_pct, "
                "ROUND(AVG(realized_pnl), 2) AS avg_pnl, "
                "ROUND(AVG(max_favorable_pnl_usd), 2) AS avg_peak_favorable_usd "
                "FROM exit_log WHERE close_time >= ? "
                "GROUP BY close_reason ORDER BY n DESC",
                (since_iso,),
            ).fetchall()

            # 2. Let-it-ride summary
            lir = conn.execute(
                "SELECT COUNT(*) AS n, ROUND(AVG(realized_pnl), 2) AS avg_pnl "
                "FROM exit_log WHERE close_time >= ? AND let_it_ride_triggered = 1",
                (since_iso,),
            ).fetchone()

            # 3. Let-it-ride positions that subsequently stopped out
            lir_sl = conn.execute(
                "SELECT COUNT(*) AS n, ROUND(AVG(realized_pnl), 2) AS avg_pnl "
                "FROM exit_log WHERE close_time >= ? "
                "AND let_it_ride_triggered = 1 AND close_reason = 'stop_loss'",
                (since_iso,),
            ).fetchone()

            # 4. Stop-loss tightness: stop_loss exits where take_profit edge
            #    was within 2% at close (positive value > -0.02 means nearly converged)
            sl_tightness = conn.execute(
                "SELECT COUNT(*) AS total_sl, "
                "SUM(CASE WHEN exit_proximity_json IS NOT NULL "
                "AND exit_proximity_json != '{}' "
                "AND CAST(json_extract(exit_proximity_json, "
                "'$.take_profit_edge_distance') AS REAL) > -0.02 "
                "THEN 1 ELSE 0 END) AS tight_count "
                "FROM exit_log WHERE close_time >= ? AND close_reason = 'stop_loss'",
                (since_iso,),
            ).fetchone()

            conn.close()

            return {
                "total": total,
                "close_reasons": [dict(r) for r in reasons],
                "let_it_ride_count": lir["n"] if lir else 0,
                "let_it_ride_avg_pnl": lir["avg_pnl"] if lir else 0.0,
                "let_it_ride_then_stop_loss": lir_sl["n"] if lir_sl else 0,
                "let_it_ride_sl_avg_pnl": lir_sl["avg_pnl"] if lir_sl else 0.0,
                "total_stop_loss": sl_tightness["total_sl"] if sl_tightness else 0,
                "tight_stop_count": int(sl_tightness["tight_count"] or 0) if sl_tightness else 0,
            }

        except Exception as e:
            self.logger.warning("exit_insights_failed", {"error": str(e)})
            return None

    def _format_exit_section(self, insights: Dict[str, Any]) -> str:
        """Render exit telemetry insights as a Slack text block.

        Four blocks:
          1. Close-reason table (count, win rate, avg realized P&L, avg peak P&L)
          2. Let-it-ride summary (only when triggered ≥1 time)
          3. Let-it-ride → subsequent stop-loss warning (only when present)
          4. Stop-loss tightness diagnostic (only when stop_loss exits exist)
        """
        lines = ["\n*Exit Telemetry (last 24h)*"]

        # Block 1: Close-reason distribution + avg peak unrealized P&L
        if insights.get("close_reasons"):
            lines.append("")
            for r in insights["close_reasons"]:
                peak = r.get("avg_peak_favorable_usd") or 0.0
                lines.append(
                    f"  `{r['close_reason']:<20}` "
                    f"`{r['n']:>2}` exit(s) | "
                    f"Win `{r['win_rate_pct']:.0f}%` | "
                    f"Avg P&L `{r['avg_pnl']:+.2f}` | "
                    f"Avg Peak `+${peak:.2f}`"
                )

        # Block 2 + 3: Let-it-ride summary (only when triggered)
        lir_count = insights.get("let_it_ride_count", 0)
        if lir_count:
            lines.append("")
            lir_avg = insights.get("let_it_ride_avg_pnl") or 0.0
            lir_sl = insights.get("let_it_ride_then_stop_loss", 0)
            lir_sl_avg = insights.get("let_it_ride_sl_avg_pnl") or 0.0
            lines.append(
                f"  :rocket: Let-it-ride fired: `{lir_count}` position(s) "
                f"(avg `{lir_avg:+.2f}`)"
            )
            if lir_sl:
                lines.append(
                    f"  :warning: `{lir_sl}` let-it-ride position(s) later stopped out "
                    f"(avg `{lir_sl_avg:+.2f}`)"
                )

        # Block 4: Stop-loss tightness diagnostic
        total_sl = insights.get("total_stop_loss", 0)
        tight = insights.get("tight_stop_count", 0)
        if total_sl:
            lines.append("")
            if tight:
                lines.append(
                    f"  :warning: Stop-loss tightness: `{tight}` of `{total_sl}` "
                    f"stop_loss exit(s) had take_profit edge within 2% — "
                    f"stop may be firing prematurely"
                )
            else:
                lines.append(
                    f"  :white_check_mark: Stop-loss tightness: `{total_sl}` "
                    f"stop_loss exit(s), none within 2% of take_profit"
                )

        return "\n".join(lines)

    def _format_exit_oneliner(self, insights: Optional[Dict[str, Any]]) -> Optional[str]:
        """One-line stop-loss tightness observation for the morning briefing.

        Derived from the tightness diagnostic (item 4) since that is the most
        actionable forward-looking signal — it tells the operator whether
        yesterday's stop-losses fired too early.

        Returns None when there are no stop_loss exits so the morning briefing
        is not padded with an empty or irrelevant line.
        """
        if not insights:
            return None
        total_sl = insights.get("total_stop_loss", 0)
        if not total_sl:
            return None
        tight = insights.get("tight_stop_count", 0)
        if tight:
            return (
                f":warning: *Yesterday's stop-loss tightness:* "
                f"`{tight}` of `{total_sl}` stop_loss exit(s) had take_profit "
                f"edge within 2% — consider reviewing stop-loss width"
            )
        return (
            f":white_check_mark: *Yesterday's stop-loss tightness:* "
            f"`{total_sl}` stop_loss exit(s), none within 2% of take_profit — "
            f"stops look reasonable"
        )

    # ------------------------------------------------------------------
    # Kill switch (STOP only — never changes parameters)
    # ------------------------------------------------------------------

    def _activate_kill_switch(self, reason: str):
        with open(KILL_SWITCH_PATH, "w") as f:
            f.write(reason)
        self.alerter._post(COLOR_RED,
            f":rotating_light: *KILL SWITCH ACTIVATED*\n"
            f"{reason}\n"
            f"Trading has been halted. Remove `data/kill_switch` to resume."
        )
        self.logger.error("kill_switch_activated", {"reason": reason})

    def check_kill_switch(self):
        """Check if account value dropped below 50% of starting. STOPS bot, never changes config."""
        account_value = self._get_account_value()
        if account_value is not None:
            threshold = self._starting_value * 0.50
            if account_value < threshold:
                self._activate_kill_switch(
                    f"Account value `${account_value:.2f}` dropped below 50% "
                    f"of starting value `${self._starting_value:.2f}`"
                )

    # ------------------------------------------------------------------
    # Daily review (report only)
    # ------------------------------------------------------------------

    def daily_review(self):
        self.logger.info("daily_review_start", {})

        try:
            yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
            trades_24h = trade_db.get_trades_since(yesterday)
            metrics = self._compute_metrics(trades_24h)
            by_type = self._compute_by_type(trades_24h)

            # Exit telemetry for the same 24h window
            yesterday_iso = yesterday.isoformat()
            exit_insights = self._get_exit_insights(yesterday_iso)

            # Save daily summary
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            summary = {
                "date": today_str,
                **metrics,
                "bankroll": self._get_exchange_balance(),
                "moneyline_trades": by_type["moneyline"]["total_trades"],
                "moneyline_win_rate": by_type["moneyline"]["win_rate"],
                "spread_trades": by_type["spread"]["total_trades"],
                "spread_win_rate": by_type["spread"]["win_rate"],
                "totals_trades": by_type["totals"]["total_trades"],
                "totals_win_rate": by_type["totals"]["win_rate"],
            }
            trade_db.insert_daily_summary(summary)

            # Check kill switch
            self.check_kill_switch()

            # Send report (NO auto-tuning)
            self._send_daily_report(metrics, by_type, exit_insights)

            # Edge validation
            self.run_edge_validation()

            self.logger.info("daily_review_complete", {
                "trades": metrics["total_trades"],
                "pnl": metrics["total_pnl"],
            })

        except Exception as e:
            self.logger.error("daily_review_failed", {"error": str(e)})
            self.alerter.error("Daily review failed", str(e))

    def _send_daily_report(self, metrics: Dict, by_type: Dict, exit_insights: Optional[Dict] = None):
        pnl = metrics["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        color = COLOR_GREEN if pnl >= 0 else COLOR_RED
        wr = metrics["win_rate"] * 100

        lines = [
            f":bar_chart: *Daily Report*",
            f"",
            f"Trades: `{metrics['total_trades']}` ({metrics['wins']}W / {metrics['losses']}L)",
            f"Win Rate: `{wr:.0f}%`",
            f"P&L: `{pnl_sign}${pnl:.2f}`",
            f"Avg Win: `${metrics['avg_win']:.2f}` | Avg Loss: `${metrics['avg_loss']:.2f}`",
            f"Profit Factor: `{metrics['profit_factor']:.2f}`",
        ]

        # By market type
        lines.append(f"\n*By Market Type*")
        for mtype in ("moneyline", "spread", "totals"):
            m = by_type[mtype]
            if m["total_trades"] > 0:
                lines.append(
                    f"  {mtype.title()}: `{m['total_trades']}` trades, "
                    f"`{m['win_rate']*100:.0f}%` WR, `${m['total_pnl']:+.2f}`"
                )

        # Account state from exchange
        account_value = self._get_account_value()
        balance = self._get_exchange_balance()
        open_pos = self._get_open_positions()
        lines.append(f"\n*Account*")
        lines.append(f"Cash: `${balance:.2f}` | Total: `${account_value:.2f}`" if account_value else f"Cash: `${balance:.2f}`")
        lines.append(f"Open Positions: `{len(open_pos)}`")

        if open_pos:
            lines.append("")
            for slug, p in list(open_pos.items())[:10]:
                net = int(p.get("netPosition", "0"))
                cost = float(p.get("cost", {}).get("value", "0"))
                value = float(p.get("cashValue", {}).get("value", "0"))
                upnl = value - cost
                side = "LONG" if net > 0 else "SHORT"
                emoji = ":small_green_triangle:" if upnl >= 0 else ":small_red_triangle_down:"
                lines.append(f"  {emoji} `{slug}` {side} `${upnl:+.2f}`")

        # Exit telemetry section — appended to the same Slack message
        if exit_insights:
            lines.append(self._format_exit_section(exit_insights))
        else:
            lines.append(
                "\n*Exit Telemetry (last 24h)*\n"
                "  _No exit_log data yet for this window._"
            )

        self.alerter._post(color, "\n".join(lines))

    # ------------------------------------------------------------------
    # Edge validation
    # ------------------------------------------------------------------

    def run_edge_validation(self):
        try:
            report = generate_edge_validation_report(days=1)
            if report.get("completed_entries", 0) == 0:
                return
            text = format_edge_report_slack(report)
            self.alerter.edge_report(text)
        except Exception as e:
            self.logger.error("edge_validation_failed", {"error": str(e)})

    # ------------------------------------------------------------------
    # Morning briefing
    # ------------------------------------------------------------------

    def morning_briefing(self):
        self.logger.info("morning_briefing_start", {})
        try:
            from bot.market_data import MarketDataClient
            from bot.signals.estimator import ProbabilityEstimator
            from bot.signals.odds_api import OddsCache
            from bot.signals.sports_data import ESPNCache, GameContextAnalyzer

            odds_cache = OddsCache(api_key=self.config.odds_api_key, cache_ttl=300)
            espn_cache = ESPNCache(cache_ttl=300)

            market_data = MarketDataClient(
                self.config.api, self.logger, self.config.filters
            )
            estimator = ProbabilityEstimator(
                self.config.signals, odds_cache,
                espn_cache=espn_cache,
                game_context_analyzer=GameContextAnalyzer(espn_cache),
            )

            opportunities = run_morning_scan(
                market_data, estimator, odds_cache, self.config
            )
            text = format_morning_briefing(opportunities)

            # Append yesterday's stop-loss tightness one-liner (item 4 of exit
            # telemetry) — the most actionable forward-looking signal.
            yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
            exit_insights = self._get_exit_insights(yesterday.isoformat())
            oneliner = self._format_exit_oneliner(exit_insights)
            if oneliner:
                text = text + "\n\n" + oneliner

            self.alerter.morning_briefing(text)

            self.logger.info("morning_briefing_sent", {
                "opportunities": len(opportunities),
            })
        except Exception as e:
            self.logger.error("morning_briefing_failed", {"error": str(e)})

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def run(self):
        self.logger.info("supervisor_starting", {})
        self.alerter._post(COLOR_GRAY,
            ":robot_face: *Supervisor Started (READ-ONLY)*\n"
            "Morning briefing 8 AM ET | Daily report 6 AM ET | Kill switch every 15 min\n"
            "Supervisor does NOT modify config. All parameter changes are manual."
        )

        scheduler = BlockingScheduler(timezone="US/Eastern")

        scheduler.add_job(
            self.daily_review,
            CronTrigger(hour=6, minute=0),
            id="daily_review",
            name="Daily Report",
            misfire_grace_time=3600,
        )

        scheduler.add_job(
            self.morning_briefing,
            CronTrigger(hour=8, minute=0),
            id="morning_briefing",
            name="Morning Edge Briefing",
            misfire_grace_time=3600,
        )

        scheduler.add_job(
            self.check_kill_switch,
            "interval",
            minutes=15,
            id="kill_switch_check",
            name="Kill Switch Check",
        )

        def shutdown(sig, frame):
            self.logger.info("supervisor_stopping", {})
            scheduler.shutdown(wait=False)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass


def main():
    supervisor = Supervisor()
    supervisor.run()


if __name__ == "__main__":
    main()
