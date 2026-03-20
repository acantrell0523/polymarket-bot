"""Slack alerting for trade events and daily summaries."""

import requests
from typing import Optional, Dict


class SlackAlerter:
    """Sends formatted alerts to a Slack webhook."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        self.webhook_url = webhook_url
        self.enabled = enabled and bool(webhook_url)

    def _post(self, text: str):
        """Post a message to Slack. Fails silently."""
        if not self.enabled:
            return
        try:
            requests.post(self.webhook_url, json={"text": text}, timeout=5)
        except Exception:
            pass

    def send_test(self):
        """Send a test message to verify the webhook works."""
        self._post(":white_check_mark: *Polymarket Bot* — Slack integration test successful.")

    def bot_started(self, mode: str, bankroll: float, markets_count: int):
        self._post(
            f":rocket: *Bot Started — {mode} MODE*\n"
            f">Bankroll: `${bankroll:.2f}`\n"
            f">Markets in scope: `{markets_count}`"
        )

    def trade_opened(
        self,
        question: str,
        slug: str,
        side: str,
        price: float,
        size_usd: float,
        edge: float,
        is_live: bool = False,
    ):
        live_tag = " :red_circle: LIVE" if is_live else ""
        side_emoji = ":chart_with_upwards_trend:" if side == "buy" else ":chart_with_downwards_trend:"
        self._post(
            f"{side_emoji} *Trade Opened*{live_tag}\n"
            f">*{question}*\n"
            f">`{slug}`\n"
            f">Side: `{side.upper()}`  |  Price: `{price:.3f}`  |  Size: `${size_usd:.2f}`\n"
            f">Edge: `{edge*100:+.1f}%`"
        )

    def trade_closed(
        self,
        question: str,
        slug: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        close_reason: str,
    ):
        emoji = ":large_green_circle:" if pnl > 0 else ":red_circle:"
        pnl_sign = "+" if pnl > 0 else ""
        self._post(
            f"{emoji} *Trade Closed — {close_reason}*\n"
            f">*{question}*\n"
            f">`{slug}`\n"
            f">Side: `{side.upper()}`  |  Entry: `{entry_price:.3f}`  |  Exit: `{exit_price:.3f}`\n"
            f">P&L: `{pnl_sign}${pnl:.2f}`"
        )

    def daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        daily_pnl: float,
        open_positions: int,
        bankroll: float,
    ):
        pnl_emoji = ":large_green_circle:" if daily_pnl >= 0 else ":red_circle:"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        self._post(
            f":bar_chart: *Daily Summary*\n"
            f">Trades: `{total_trades}` ({wins}W / {losses}L — `{win_rate:.0f}%` win rate)\n"
            f">Daily P&L: {pnl_emoji} `{pnl_sign}${daily_pnl:.2f}`\n"
            f">Open positions: `{open_positions}`\n"
            f">Bankroll: `${bankroll:.2f}`"
        )

    def error(self, message: str, details: str = ""):
        text = f":warning: *Bot Error*\n>{message}"
        if details:
            text += f"\n>```{details}```"
        self._post(text)
