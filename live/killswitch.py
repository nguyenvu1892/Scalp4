"""
ScalForex — Killswitch
Continuous DD monitoring. If DD > threshold → CLOSE ALL + disable AI.
Last line of defense against catastrophic losses.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from utils.logger import setup_logger

log = setup_logger("live.killswitch")


class Killswitch:
    """Drawdown killswitch — emergency position closure.

    Continuously monitors account drawdown.
    If DD exceeds threshold → closes ALL positions and disables trading.

    This is the LAST LINE OF DEFENSE. Hard SL on broker is first.
    Killswitch catches scenarios where multiple positions accumulate
    losses beyond the total account threshold.

    Args:
        mt5_bridge: MT5 bridge instance.
        telegram: Telegram bot for alerts.
        max_dd_pct: Drawdown threshold to trigger (0.45 = 45%).
        check_interval: Seconds between DD checks.
        initial_balance: Account starting balance for DD calculation.
    """

    def __init__(
        self,
        mt5_bridge: object,
        telegram: Optional[object] = None,
        max_dd_pct: float = 0.45,
        check_interval: float = 5.0,
        initial_balance: float = 200.0,
    ) -> None:
        self.mt5 = mt5_bridge
        self.telegram = telegram
        self.max_dd_pct = max_dd_pct
        self.check_interval = check_interval
        self.initial_balance = initial_balance
        self.peak_equity = initial_balance
        self._armed = True
        self._triggered = False
        self._trigger_count = 0

    @property
    def is_armed(self) -> bool:
        """Whether killswitch is active and monitoring."""
        return self._armed and not self._triggered

    def arm(self) -> None:
        """Re-arm the killswitch after manual review."""
        self._armed = True
        self._triggered = False
        log.info("Killswitch RE-ARMED")

    def disarm(self) -> None:
        """Temporarily disarm (e.g. during maintenance)."""
        self._armed = False
        log.warning("⚠️ Killswitch DISARMED — trading unprotected!")

    def check_drawdown(self, current_equity: float) -> dict:
        """Check if drawdown exceeds threshold.

        Args:
            current_equity: Current account equity.

        Returns:
            Dict with dd_pct, peak_equity, triggered status.
        """
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown from peak
        dd_pct = (self.peak_equity - current_equity) / max(self.peak_equity, 1e-10)

        result = {
            "equity": current_equity,
            "peak_equity": self.peak_equity,
            "dd_pct": dd_pct,
            "threshold": self.max_dd_pct,
            "triggered": False,
        }

        if dd_pct >= self.max_dd_pct and self.is_armed:
            result["triggered"] = True
            self._trigger(current_equity, dd_pct)

        return result

    def _trigger(self, equity: float, dd_pct: float) -> None:
        """Execute killswitch — close all positions and disable trading.

        Args:
            equity: Current equity at trigger time.
            dd_pct: Drawdown percentage at trigger time.
        """
        self._triggered = True
        self._trigger_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log.critical(
            f"🚨🚨🚨 KILLSWITCH TRIGGERED #{self._trigger_count}! "
            f"DD={dd_pct:.1%} >= {self.max_dd_pct:.1%} "
            f"Equity=${equity:.2f} at {timestamp}"
        )

        # Close all positions via MT5
        positions_closed = 0
        if hasattr(self.mt5, "close_all_positions"):
            results = self.mt5.close_all_positions()
            positions_closed = sum(1 for r in results if r.success)
            log.critical(f"Closed {positions_closed}/{len(results)} positions")

        # Alert via Telegram
        if self.telegram and hasattr(self.telegram, "alert_killswitch"):
            self.telegram.alert_killswitch(
                drawdown_pct=dd_pct,
                equity=equity,
                positions_closed=positions_closed,
            )

        log.critical("⛔ AI TRADING DISABLED. Manual intervention required.")

    def simulate_trigger(self) -> dict:
        """Simulate killswitch trigger for testing.

        Does NOT actually close positions. Just tests the detection logic.

        Returns:
            Dict with simulated trigger result.
        """
        fake_equity = self.initial_balance * (1 - self.max_dd_pct - 0.01)
        log.warning(
            f"🧪 SIMULATING killswitch: fake equity=${fake_equity:.2f} "
            f"(DD would be {self.max_dd_pct + 0.01:.1%})"
        )

        # Check without actually triggering
        dd_pct = (self.peak_equity - fake_equity) / max(self.peak_equity, 1e-10)
        would_trigger = dd_pct >= self.max_dd_pct

        return {
            "simulated_equity": fake_equity,
            "simulated_dd": dd_pct,
            "would_trigger": would_trigger,
            "threshold": self.max_dd_pct,
        }
