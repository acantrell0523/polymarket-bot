"""Slack alerting for trade events and daily summaries."""

import requests
from typing import Optional

# Color palette
COLOR_GREEN = "#36a64f"
COLOR_RED = "#dc3545"
COLOR_BLUE = "#2196F3"
COLOR_ORANGE = "#ff9800"
COLOR_GRAY = "#9e9e9e"


class SlackAlerter:
    """Sends color-coded Slack attachment alerts."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        self.webhook_url = webhook_url
        self.enabled = enabled and bool(webhook_url)

    def _post(self, color: str, text: str):
        """Post a color-coded attachment to Slack."""
        if not self.enabled:
            return
        try:
            resp = requests.post(self.webhook_url, json={
                "attachments": [{
                    "color": color,
                    "text": text,
                    "footer": "Polymarket Bot",
                }]
            }, timeout=5)
            if resp.status_code != 200:
                import sys
                print(f"[SLACK ERROR] {resp.status_code}: {resp.text}", file=sys.stderr)
        except Exception as e:
            import sys
            print(f"[SLACK EXCEPTION] {e}", file=sys.stderr)

    def send_test(self):
        """Send test messages for each color."""
        self._post(COLOR_GREEN, ":white_check_mark: *Test* — Green (trade closed, profit)")
        self._post(COLOR_RED, ":red_circle: *Test* — Red (trade closed, loss)")
        self._post(COLOR_BLUE, ":chart_with_upwards_trend: *Test* — Blue (trade opened, pre-game)")
        self._post(COLOR_ORANGE, ":zap: *Test* — Orange (trade opened, live game)")
        self._post(COLOR_GRAY, ":bar_chart: *Test* — Gray (summary / startup)")

    def bot_started(self, mode: str, bankroll: float, markets_count: int):
        self._post(COLOR_GRAY,
            f":rocket: *Bot Started — {mode} MODE*\n"
            f"Bankroll: `${bankroll:.2f}`\n"
            f"Markets in scope: `{markets_count}`"
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
        color = COLOR_ORANGE if is_live else COLOR_BLUE
        live_tag = " :zap: LIVE" if is_live else ""
        side_emoji = ":chart_with_upwards_trend:" if side == "buy" else ":chart_with_downwards_trend:"
        self._post(color,
            f"{side_emoji} *Trade Opened*{live_tag}\n"
            f"*{question}*\n"
            f"`{slug}`\n"
            f"Side: `{side.upper()}`  |  Price: `{price:.3f}`  |  Size: `${size_usd:.2f}`\n"
            f"Edge: `{edge*100:+.1f}%`"
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
        color = COLOR_GREEN if pnl > 0 else COLOR_RED
        pnl_sign = "+" if pnl > 0 else ""
        emoji = ":large_green_circle:" if pnl > 0 else ":red_circle:"
        self._post(color,
            f"{emoji} *Trade Closed — {close_reason}*\n"
            f"*{question}*\n"
            f"`{slug}`\n"
            f"Side: `{side.upper()}`  |  Entry: `{entry_price:.3f}`  |  Exit: `{exit_price:.3f}`\n"
            f"P&L: `{pnl_sign}${pnl:.2f}`"
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
        pnl_sign = "+" if daily_pnl >= 0 else ""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        self._post(COLOR_GRAY,
            f":bar_chart: *Daily Summary*\n"
            f"Trades: `{total_trades}` ({wins}W / {losses}L — `{win_rate:.0f}%` win rate)\n"
            f"Daily P&L: `{pnl_sign}${daily_pnl:.2f}`\n"
            f"Open positions: `{open_positions}`\n"
            f"Bankroll: `${bankroll:.2f}`"
        )

    def morning_briefing(self, text: str):
        """Post the morning edge briefing."""
        self._post(COLOR_BLUE, text)

    def edge_report(self, text: str):
        """Post the edge validation report."""
        self._post(COLOR_GRAY, text)

    def error(self, message: str, details: str = ""):
        text = f":warning: *Bot Error*\n{message}"
        if details:
            text += f"\n```{details}```"
        self._post(COLOR_RED, text)
