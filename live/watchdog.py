"""
ScalForex — Process Watchdog
Monitors AI trading process health. Auto-restart on crash/hang.
Runs as a separate process checking heartbeat every 60s.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

log = setup_logger("live.watchdog")


class Watchdog:
    """Process health monitor for the AI trading system.

    Checks heartbeat file updated by the main trading process.
    If heartbeat stale > timeout → kills process, cleans up, restarts.

    Args:
        heartbeat_path: Path to heartbeat file (main process writes timestamp).
        timeout: Seconds before declaring process dead.
        check_interval: Seconds between health checks.
        main_script: Path to main trading script to restart.
        telegram: Optional Telegram bot for alerts.
        mt5_bridge: Optional MT5 bridge for position cleanup.
    """

    def __init__(
        self,
        heartbeat_path: Path = Path("logs/heartbeat.txt"),
        timeout: float = 120.0,
        check_interval: float = 60.0,
        main_script: str = "scripts/live_trade.py",
        telegram: Optional[object] = None,
        mt5_bridge: Optional[object] = None,
    ) -> None:
        self.heartbeat_path = heartbeat_path
        self.timeout = timeout
        self.check_interval = check_interval
        self.main_script = main_script
        self.telegram = telegram
        self.mt5 = mt5_bridge
        self._running = False
        self._restart_count = 0
        self._child_process: Optional[subprocess.Popen] = None

    def write_heartbeat(self) -> None:
        """Write current timestamp to heartbeat file.

        Called by the MAIN trading process periodically.
        """
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(
            str(time.time()), encoding="utf-8"
        )

    def read_heartbeat(self) -> Optional[float]:
        """Read last heartbeat timestamp.

        Returns:
            Unix timestamp of last heartbeat, or None if file missing.
        """
        if not self.heartbeat_path.exists():
            return None
        try:
            return float(self.heartbeat_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

    def is_process_alive(self) -> bool:
        """Check if main trading process is alive.

        Returns:
            True if heartbeat is recent enough.
        """
        last_beat = self.read_heartbeat()
        if last_beat is None:
            return False
        return (time.time() - last_beat) < self.timeout

    def _cleanup_positions(self) -> None:
        """Close all positions on process death (safety net)."""
        if self.mt5 and hasattr(self.mt5, "close_all_positions"):
            log.warning("Watchdog: cleaning up positions after process death...")
            self.mt5.close_all_positions()

    def _send_alert(self, event: str, detail: str) -> None:
        """Send watchdog alert via Telegram."""
        if self.telegram and hasattr(self.telegram, "alert_watchdog"):
            self.telegram.alert_watchdog(event=event, detail=detail)

    def _restart_process(self) -> None:
        """Kill dead process and start fresh."""
        self._restart_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        log.warning(
            f"🐕 Watchdog: Restarting main process "
            f"(restart #{self._restart_count}) at {timestamp}"
        )

        # Kill existing child if any
        if self._child_process and self._child_process.poll() is None:
            log.warning("Terminating stale child process...")
            self._child_process.terminate()
            try:
                self._child_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._child_process.kill()

        # Cleanup positions
        self._cleanup_positions()

        # Alert
        self._send_alert(
            "RESTART",
            f"Process restarted (#{self._restart_count}). "
            f"Positions cleaned up at {timestamp}."
        )

        # Restart
        try:
            self._child_process = subprocess.Popen(
                [sys.executable, self.main_script],
                cwd=str(Path.cwd()),
            )
            log.info(f"Main process restarted: PID={self._child_process.pid}")
        except Exception as e:
            log.error(f"Failed to restart: {e}")
            self._send_alert("RESTART_FAILED", str(e))

    def run(self) -> None:
        """Start watchdog monitoring loop.

        Blocks indefinitely. Run in a separate process/container.
        """
        self._running = True
        log.info(
            f"🐕 Watchdog started: checking every {self.check_interval}s, "
            f"timeout={self.timeout}s"
        )

        while self._running:
            if not self.is_process_alive():
                log.warning("🐕 Watchdog: Heartbeat STALE or MISSING!")
                self._send_alert(
                    "TIMEOUT",
                    f"Main process heartbeat stale > {self.timeout}s"
                )
                self._restart_process()
            else:
                last = self.read_heartbeat()
                age = time.time() - last if last else 0
                log.debug(f"Watchdog: heartbeat OK ({age:.0f}s ago)")

            time.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop watchdog monitoring."""
        self._running = False
        log.info("🐕 Watchdog stopped")
