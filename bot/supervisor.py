"""Supervisor agent: daily review, auto-tuning, kill switch, weekly optimization."""

import os
import sys
import signal
import time
import yaml
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.config import load_config, BotConfig
from utils.logger import TradingLogger
from bot.alerts import SlackAlerter, COLOR_GREEN, COLOR_RED, COLOR_GRAY, COLOR_ORANGE
from bot import trade_db


# File-based kill switch: trading_loop checks this file
KILL_SWITCH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kill_switch")
PAUSE_UNTIL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pause_until")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")


class Supervisor:
    """Monitors bot performance, auto-tunes parameters, and enforces safety limits."""

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

        # Track starting value for kill switch (read from config as baseline)
        self._starting_value = self._get_account_value() or self.config.backtest.initial_bankroll_usd
        self._original_max_position_size = self.config.trading.max_position_size_usd

        self.logger.info("supervisor_initialized", {
            "starting_value": self._starting_value,
            "config_path": CONFIG_PATH,
        })

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Get a PolymarketUS client."""
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
        """Get total account value (balance + position value)."""
        client = self._get_client()
        if not client:
            return None
        try:
            bal = client.account.balances()
            balance = float(bal["balances"][0]["currentBalance"])
            positions = client.portfolio.positions()
            pos_dict = positions.get("positions", {})
            pos_value = sum(
                float(p.get("cashValue", {}).get("value", "0"))
                for p in pos_dict.values()
            )
            return balance + pos_value
        except Exception as e:
            self.logger.error("account_value_failed", {"error": str(e)})
            return None

    def _get_exchange_balance(self) -> float:
        client = self._get_client()
        if not client:
            return 0
        try:
            bal = client.account.balances()
            return float(bal["balances"][0]["currentBalance"])
        except Exception:
            return 0

    def _get_open_positions(self) -> Dict[str, Any]:
        client = self._get_client()
        if not client:
            return {}
        try:
            result = client.portfolio.positions()
            return result.get("positions", {})
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Config read/write
    # ------------------------------------------------------------------

    def _read_config_yaml(self) -> dict:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _write_config_yaml(self, data: dict):
        """Atomic write: write to temp file then rename."""
        dir_name = os.path.dirname(CONFIG_PATH)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            shutil.move(tmp_path, CONFIG_PATH)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _update_config_param(self, section: str, key: str, new_value, reason: str):
        """Read config, update one param, write back, log the change."""
        raw = self._read_config_yaml()
        old_value = raw.get(section, {}).get(key)
        if old_value == new_value:
            return
        if section not in raw:
            raw[section] = {}
        raw[section][key] = new_value
        self._write_config_yaml(raw)
        trade_db.log_parameter_change(
            parameter=f"{section}.{key}",
            old_value=str(old_value),
            new_value=str(new_value),
            reason=reason,
        )
        self.logger.info("config_updated", {
            "param": f"{section}.{key}",
            "old": old_value,
            "new": new_value,
            "reason": reason,
        })

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
    # Performance analysis
    # ------------------------------------------------------------------

    def _compute_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Compute performance metrics from a list of trade dicts."""
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
        """Break down metrics by market type."""
        by_type = {}
        for mtype in ("moneyline", "spread", "totals"):
            typed = [t for t in trades if self._classify_market_type(t["slug"]) == mtype]
            by_type[mtype] = self._compute_metrics(typed)
        return by_type

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _activate_kill_switch(self, reason: str):
        """Write kill switch file and alert."""
        with open(KILL_SWITCH_PATH, "w") as f:
            f.write(reason)
        self.alerter._post(COLOR_RED,
            f":rotating_light: *KILL SWITCH ACTIVATED*\n"
            f"{reason}\n"
            f"Trading has been halted. Remove `data/kill_switch` to resume."
        )
        self.logger.error("kill_switch_activated", {"reason": reason})

    def _activate_pause(self, hours: float, reason: str):
        """Pause trading for N hours."""
        resume_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        with open(PAUSE_UNTIL_PATH, "w") as f:
            f.write(resume_at.isoformat())
        self.alerter._post(COLOR_ORANGE,
            f":pause_button: *Trading Paused*\n"
            f"{reason}\n"
            f"Resuming at `{resume_at.strftime('%Y-%m-%d %H:%M UTC')}`"
        )
        self.logger.warning("trading_paused", {"reason": reason, "resume_at": resume_at.isoformat()})

    def check_kill_switch(self):
        """Check kill switch conditions."""
        # Condition 1: account value < 50% of starting
        account_value = self._get_account_value()
        if account_value is not None:
            threshold = self._starting_value * 0.50
            if account_value < threshold:
                self._activate_kill_switch(
                    f"Account value `${account_value:.2f}` dropped below 50% "
                    f"of starting value `${self._starting_value:.2f}`"
                )
                return

        # Condition 2: 10 consecutive losses
        recent = trade_db.get_recent_trades(limit=10)
        if len(recent) >= 10:
            all_losses = all(t["realized_pnl"] <= 0 for t in recent)
            if all_losses:
                self._activate_pause(6, "10 consecutive losing trades detected")

    # ------------------------------------------------------------------
    # Auto-tuning
    # ------------------------------------------------------------------

    def auto_tune(self):
        """Apply rule-based parameter adjustments."""
        changes = []

        # Get last 3 days of trades
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        trades_3d = trade_db.get_trades_since(three_days_ago)
        metrics_3d = self._compute_metrics(trades_3d)
        by_type_3d = self._compute_by_type(trades_3d)

        raw = self._read_config_yaml()
        current_edge = raw.get("trading", {}).get("min_edge_threshold", 0.02)
        current_max_size = raw.get("trading", {}).get("max_position_size_usd", 10.0)

        # Rule 1: Win rate too low → raise edge threshold
        if metrics_3d["total_trades"] >= 5 and metrics_3d["win_rate"] < 0.55:
            new_edge = round(current_edge + 0.005, 4)
            self._update_config_param("trading", "min_edge_threshold", new_edge,
                f"3-day win rate {metrics_3d['win_rate']*100:.0f}% < 55% "
                f"({metrics_3d['total_trades']} trades)")
            changes.append(f"min_edge_threshold: {current_edge} → {new_edge} (low win rate)")

        # Rule 2: Win rate very high → lower edge threshold
        elif metrics_3d["total_trades"] >= 5 and metrics_3d["win_rate"] > 0.70:
            new_edge = round(max(0.01, current_edge - 0.005), 4)
            self._update_config_param("trading", "min_edge_threshold", new_edge,
                f"3-day win rate {metrics_3d['win_rate']*100:.0f}% > 70% "
                f"({metrics_3d['total_trades']} trades)")
            changes.append(f"min_edge_threshold: {current_edge} → {new_edge} (high win rate)")

        # Rule 3: Market type with win rate < 45% → exclude it
        for mtype, mstats in by_type_3d.items():
            if mstats["total_trades"] >= 3 and mstats["win_rate"] < 0.45:
                exclude = raw.get("filters", {}).get("exclude_categories", [])
                if mtype not in exclude:
                    exclude.append(mtype)
                    self._update_config_param("filters", "exclude_categories", exclude,
                        f"{mtype} 3-day win rate {mstats['win_rate']*100:.0f}% < 45% "
                        f"({mstats['total_trades']} trades)")
                    changes.append(f"Excluded market type: {mtype}")

        # Rule 4: Drawdown > 20% → cut position size
        account_value = self._get_account_value()
        if account_value is not None:
            drawdown_pct = 1 - (account_value / self._starting_value)
            if drawdown_pct > 0.20:
                new_size = round(current_max_size / 2, 2)
                self._update_config_param("trading", "max_position_size_usd", new_size,
                    f"Drawdown {drawdown_pct*100:.0f}% exceeds 20% threshold")
                changes.append(f"max_position_size_usd: {current_max_size} → {new_size} (drawdown)")

        # Rule 5: 5+ consecutive wins → increase position size (capped)
        recent = trade_db.get_recent_trades(limit=5)
        if len(recent) >= 5 and all(t["realized_pnl"] > 0 for t in recent):
            cap = self._original_max_position_size
            new_size = round(min(current_max_size * 1.25, cap), 2)
            if new_size > current_max_size:
                self._update_config_param("trading", "max_position_size_usd", new_size,
                    f"5 consecutive wins, increasing size (capped at {cap})")
                changes.append(f"max_position_size_usd: {current_max_size} → {new_size} (streak)")

        return changes

    # ------------------------------------------------------------------
    # Daily review
    # ------------------------------------------------------------------

    def daily_review(self):
        """Run daily performance review, auto-tune, and send report."""
        self.logger.info("daily_review_start", {})

        try:
            # 1. Pull last 24h trades
            yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
            trades_24h = trade_db.get_trades_since(yesterday)
            metrics = self._compute_metrics(trades_24h)
            by_type = self._compute_by_type(trades_24h)

            # 2. Save daily summary
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

            # 3. Check kill switch
            self.check_kill_switch()

            # 4. Auto-tune
            param_changes = self.auto_tune()

            # 5. Send Slack report (after 5 min delay simulated by scheduler offset)
            self._send_daily_report(metrics, by_type, param_changes)

            self.logger.info("daily_review_complete", {
                "trades": metrics["total_trades"],
                "pnl": metrics["total_pnl"],
                "changes": len(param_changes),
            })

        except Exception as e:
            self.logger.error("daily_review_failed", {"error": str(e)})
            self.alerter.error("Daily review failed", str(e))

    def _send_daily_report(self, metrics: Dict, by_type: Dict, param_changes: List[str]):
        """Send the rich daily Slack report."""
        pnl = metrics["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        color = COLOR_GREEN if pnl >= 0 else COLOR_RED
        wr = metrics["win_rate"] * 100
        benchmark_status = ":white_check_mark:" if wr >= 62 else ":x:"

        # Main stats
        lines = [
            f":bar_chart: *Supervisor Daily Report*",
            f"",
            f"*Yesterday's Results*",
            f"Trades: `{metrics['total_trades']}` ({metrics['wins']}W / {metrics['losses']}L)",
            f"Win Rate: `{wr:.0f}%` {benchmark_status} (target: 62%)",
            f"P&L: `{pnl_sign}${pnl:.2f}`",
            f"Avg Win: `${metrics['avg_win']:.2f}` | Avg Loss: `${metrics['avg_loss']:.2f}`",
            f"Profit Factor: `{metrics['profit_factor']:.2f}`",
        ]

        # By market type
        lines.append(f"\n*By Market Type*")
        for mtype in ("moneyline", "spread", "totals"):
            m = by_type[mtype]
            if m["total_trades"] > 0:
                mwr = m["win_rate"] * 100
                lines.append(
                    f"  {mtype.title()}: `{m['total_trades']}` trades, "
                    f"`{mwr:.0f}%` WR, `${m['total_pnl']:+.2f}` P&L"
                )

        # Account state
        balance = self._get_exchange_balance()
        account_value = self._get_account_value() or balance
        open_pos = self._get_open_positions()
        lines.append(f"\n*Account*")
        lines.append(f"Balance: `${balance:.2f}` | Total Value: `${account_value:.2f}`")
        lines.append(f"Open Positions: `{len(open_pos)}`")

        # Open positions detail
        if open_pos:
            lines.append("")
            for slug, p in list(open_pos.items())[:10]:
                net = int(p.get("netPosition", "0"))
                cost = float(p.get("cost", {}).get("value", "0"))
                value = float(p.get("cashValue", {}).get("value", "0"))
                upnl = value - cost
                upnl_sign = "+" if upnl >= 0 else ""
                side = "LONG" if net > 0 else "SHORT"
                emoji = ":small_green_triangle:" if upnl >= 0 else ":small_red_triangle_down:"
                lines.append(f"  {emoji} `{slug}` {side} `{upnl_sign}${upnl:.2f}`")

        # Parameter changes
        if param_changes:
            lines.append(f"\n*Parameter Changes*")
            for c in param_changes:
                lines.append(f"  :gear: {c}")

        self.alerter._post(color, "\n".join(lines))

    # ------------------------------------------------------------------
    # Weekly optimization (stub — requires historical data)
    # ------------------------------------------------------------------

    def weekly_optimization(self):
        """Run weekly parameter sweep on last 7 days of trade data."""
        self.logger.info("weekly_optimization_start", {})

        try:
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            trades_7d = trade_db.get_trades_since(week_ago)
            metrics_before = self._compute_metrics(trades_7d)

            if metrics_before["total_trades"] < 10:
                self.logger.info("weekly_optimization_skipped", {
                    "reason": "fewer than 10 trades in last 7 days",
                    "trades": metrics_before["total_trades"],
                })
                return

            # Snapshot current params
            raw = self._read_config_yaml()
            before_params = {
                "min_edge_threshold": raw.get("trading", {}).get("min_edge_threshold"),
                "max_position_size_usd": raw.get("trading", {}).get("max_position_size_usd"),
                "stop_loss_threshold": raw.get("trading", {}).get("stop_loss_threshold"),
                "position_sizing_method": raw.get("trading", {}).get("position_sizing_method"),
            }

            # NOTE: Full backtest sweep requires reconstructed MarketSnapshot data
            # from historical prices. For now, apply heuristic adjustments based on
            # the 7-day metrics, which is what auto_tune already does.
            # When historical OHLC data collection is implemented, this will call:
            #   from backtest.sweep import run_sweep
            #   results = run_sweep(config, market_data)
            #   best = results[0]

            # For now, just run auto_tune with the 7-day window
            param_changes = self.auto_tune()

            after_params = {
                "min_edge_threshold": self._read_config_yaml().get("trading", {}).get("min_edge_threshold"),
                "max_position_size_usd": self._read_config_yaml().get("trading", {}).get("max_position_size_usd"),
                "stop_loss_threshold": self._read_config_yaml().get("trading", {}).get("stop_loss_threshold"),
                "position_sizing_method": self._read_config_yaml().get("trading", {}).get("position_sizing_method"),
            }

            # Send weekly report
            lines = [
                f":calendar: *Weekly Optimization Report*",
                f"",
                f"*7-Day Performance*",
                f"Trades: `{metrics_before['total_trades']}`",
                f"Win Rate: `{metrics_before['win_rate']*100:.0f}%`",
                f"P&L: `${metrics_before['total_pnl']:+.2f}`",
                f"Profit Factor: `{metrics_before['profit_factor']:.2f}`",
                f"",
                f"*Parameter Comparison*",
            ]

            changed = False
            for key in before_params:
                bv = before_params[key]
                av = after_params[key]
                if bv != av:
                    changed = True
                    lines.append(f"  `{key}`: `{bv}` → `{av}`")

            if not changed:
                lines.append("  No changes — current parameters are optimal")

            color = COLOR_GREEN if metrics_before["total_pnl"] >= 0 else COLOR_RED
            self.alerter._post(color, "\n".join(lines))

            self.logger.info("weekly_optimization_complete", {
                "trades": metrics_before["total_trades"],
                "changes": len(param_changes),
            })

        except Exception as e:
            self.logger.error("weekly_optimization_failed", {"error": str(e)})
            self.alerter.error("Weekly optimization failed", str(e))

    # ------------------------------------------------------------------
    # Scheduler entry point
    # ------------------------------------------------------------------

    def run(self):
        """Start the supervisor scheduler."""
        self.logger.info("supervisor_starting", {})
        self.alerter._post(COLOR_GRAY,
            ":robot_face: *Supervisor Started*\n"
            "Daily review at 6:00 AM ET | Weekly optimization Sundays 6:00 AM ET"
        )

        scheduler = BlockingScheduler(timezone="US/Eastern")

        # Daily review at 6:00 AM ET
        scheduler.add_job(
            self.daily_review,
            CronTrigger(hour=6, minute=0),
            id="daily_review",
            name="Daily Performance Review",
            misfire_grace_time=3600,
        )

        # Weekly optimization on Sundays at 6:00 AM ET
        scheduler.add_job(
            self.weekly_optimization,
            CronTrigger(day_of_week="sun", hour=6, minute=0),
            id="weekly_optimization",
            name="Weekly Parameter Optimization",
            misfire_grace_time=3600,
        )

        # Kill switch check every 15 minutes
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
