"""
ScalForex — Telegram Alert Bot
Real-time trade alerts, killswitch warnings, daily summaries.
Anti-spam: rate limiting + message batching.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import requests

from utils.logger import setup_logger

log = setup_logger("utils.telegram_bot")


class TelegramBot:
    """Telegram notification bot for ScalForex.

    Features:
        - Trade open/close alerts with full details
        - Killswitch emergency alerts
        - Rate limiting (max 1 msg/sec to avoid API throttle)
        - Formatted messages with emoji for quick scanning

    Args:
        token: Bot API token (from @BotFather).
        chat_id: Target chat/group ID.
        rate_limit: Minimum seconds between messages.
    """

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        rate_limit: float = 1.0,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.rate_limit = rate_limit
        self._last_send_time = 0.0
        self._base_url = f"https://api.telegram.org/bot{self.token}"

        if not self.token or not self.chat_id:
            log.warning(
                "Telegram bot not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _rate_limit_wait(self) -> None:
        """Wait if sending too fast (anti-spam)."""
        elapsed = time.time() - self._last_send_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat.

        Args:
            text: Message text (supports HTML formatting).
            parse_mode: Telegram parse mode (HTML or Markdown).

        Returns:
            True if sent successfully.
        """
        if not self.is_configured:
            log.debug(f"Telegram not configured, would send: {text[:100]}...")
            return False

        self._rate_limit_wait()

        try:
            response = requests.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            self._last_send_time = time.time()

            if response.status_code == 200:
                return True

            log.warning(f"Telegram send failed: {response.status_code} {response.text}")
            return False

        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False

    # ── Formatted Alert Methods ──

    def alert_trade_open(
        self,
        symbol: str,
        side: str,
        lot: float,
        entry: float,
        sl: float,
        session: str = "",
        confidence: float = 0.0,
    ) -> bool:
        """Alert: New trade opened.

        Args:
            symbol: Trading symbol.
            side: BUY or SELL.
            lot: Position size.
            entry: Entry price.
            sl: Stop loss price.
            session: Trading session (Asian/EU/US).
            confidence: Model confidence score.

        Returns:
            True if sent.
        """
        emoji = "🟢" if side == "BUY" else "🔴"
        text = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {side}\n"
            f"📏 Lot: {lot}\n"
            f"💰 Entry: {entry:.5f}\n"
            f"🛡️ SL: {sl:.5f}\n"
            f"📈 Confidence: {confidence:.1%}\n"
        )
        if session:
            text += f"🕐 Session: {session}\n"
        text += f"⏰ {datetime.now().strftime('%H:%M:%S')}"

        return self.send_message(text)

    def alert_trade_close(
        self,
        symbol: str,
        side: str,
        pnl: float,
        reason: str = "Signal",
        bars_held: int = 0,
    ) -> bool:
        """Alert: Trade closed.

        Args:
            symbol: Trading symbol.
            side: Original direction.
            pnl: Realized PnL.
            reason: Close reason (Signal/SL/TP/Killswitch).
            bars_held: Duration in M5 bars.

        Returns:
            True if sent.
        """
        emoji = "💰" if pnl >= 0 else "💸"
        pnl_sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {side}\n"
            f"💵 PnL: {pnl_sign}${pnl:.2f}\n"
            f"📋 Reason: {reason}\n"
            f"⏱️ Duration: {bars_held} bars ({bars_held * 5}min)\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        return self.send_message(text)

    def alert_killswitch(
        self,
        drawdown_pct: float,
        equity: float,
        positions_closed: int,
    ) -> bool:
        """Alert: KILLSWITCH ACTIVATED — emergency trade closure.

        Args:
            drawdown_pct: Current drawdown percentage.
            equity: Current account equity.
            positions_closed: Number of positions force-closed.

        Returns:
            True if sent.
        """
        text = (
            f"🚨🚨🚨 <b>KILLSWITCH ACTIVATED</b> 🚨🚨🚨\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Drawdown: {drawdown_pct:.1%}\n"
            f"💰 Equity: ${equity:.2f}\n"
            f"🔒 Positions closed: {positions_closed}\n"
            f"⛔ AI Trading DISABLED\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
            f"\n"
            f"Manual intervention required."
        )
        return self.send_message(text)

    def alert_daily_summary(
        self,
        total_trades: int,
        total_pnl: float,
        win_rate: float,
        equity: float,
        drawdown_pct: float,
    ) -> bool:
        """Alert: Daily performance summary.

        Args:
            total_trades: Trades today.
            total_pnl: Net PnL today.
            win_rate: Win rate fraction.
            equity: Current equity.
            drawdown_pct: Current drawdown.

        Returns:
            True if sent.
        """
        emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""
        text = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Trades: {total_trades}\n"
            f"💵 PnL: {pnl_sign}${total_pnl:.2f}\n"
            f"✅ Win Rate: {win_rate:.0%}\n"
            f"💰 Equity: ${equity:.2f}\n"
            f"📉 Drawdown: {drawdown_pct:.1%}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_watchdog(self, event: str, detail: str = "") -> bool:
        """Alert: Watchdog event (process crash, restart, etc).

        Args:
            event: Event type (CRASH/RESTART/TIMEOUT).
            detail: Additional details.

        Returns:
            True if sent.
        """
        text = (
            f"🐕 <b>WATCHDOG: {event}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{detail}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        return self.send_message(text)

    def send_test(self) -> bool:
        """Send test message to verify bot setup."""
        return self.send_message(
            "🤖 <b>ScalForex AI Bot</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "✅ Telegram integration active\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
