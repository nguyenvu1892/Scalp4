"""Tests for Killswitch, Watchdog, MT5Bridge (unit tests without real MT5)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Killswitch Tests ──

class TestKillswitch:
    """Tests for the DD killswitch."""

    def _make_killswitch(self, max_dd=0.45, initial=200.0):
        from live.killswitch import Killswitch
        mt5_mock = MagicMock()
        mt5_mock.close_all_positions.return_value = []
        telegram_mock = MagicMock()
        return Killswitch(
            mt5_bridge=mt5_mock,
            telegram=telegram_mock,
            max_dd_pct=max_dd,
            initial_balance=initial,
        )

    def test_no_trigger_below_threshold(self) -> None:
        """DD below threshold should NOT trigger."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        result = ks.check_drawdown(180.0)  # 10% DD
        assert result["triggered"] is False

    def test_trigger_at_threshold(self) -> None:
        """DD at threshold SHOULD trigger."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        result = ks.check_drawdown(110.0)  # 45% DD exactly
        assert result["triggered"] is True

    def test_trigger_above_threshold(self) -> None:
        """DD above threshold SHOULD trigger."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        result = ks.check_drawdown(100.0)  # 50% DD
        assert result["triggered"] is True
        assert ks._triggered is True

    def test_close_all_called_on_trigger(self) -> None:
        """MT5 close_all_positions should be called when triggered."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        ks.check_drawdown(100.0)
        ks.mt5.close_all_positions.assert_called_once()

    def test_telegram_alert_on_trigger(self) -> None:
        """Telegram alert should fire on killswitch trigger."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        ks.check_drawdown(100.0)
        ks.telegram.alert_killswitch.assert_called_once()

    def test_arm_disarm(self) -> None:
        """Killswitch can be armed/disarmed."""
        ks = self._make_killswitch()
        assert ks.is_armed is True
        ks.disarm()
        assert ks.is_armed is False
        # Should NOT trigger when disarmed
        result = ks.check_drawdown(50.0)
        assert result["triggered"] is False

    def test_re_arm_after_trigger(self) -> None:
        """Can re-arm after trigger."""
        ks = self._make_killswitch(max_dd=0.45, initial=200.0)
        ks.check_drawdown(100.0)  # Triggers
        assert ks.is_armed is False
        ks.arm()
        assert ks.is_armed is True

    def test_peak_tracking(self) -> None:
        """Peak equity should track highest value."""
        ks = self._make_killswitch(initial=200.0)
        ks.check_drawdown(250.0)  # New peak
        assert ks.peak_equity == 250.0
        ks.check_drawdown(240.0)  # Dropped but still high
        assert ks.peak_equity == 250.0  # Peak unchanged

    def test_simulate_no_actual_close(self) -> None:
        """simulate_trigger should NOT call close_all_positions."""
        ks = self._make_killswitch()
        result = ks.simulate_trigger()
        assert result["would_trigger"] is True
        ks.mt5.close_all_positions.assert_not_called()


# ── Watchdog Tests ──

class TestWatchdog:
    """Tests for the process watchdog."""

    def test_write_and_read_heartbeat(self) -> None:
        """Heartbeat write/read round-trip."""
        from live.watchdog import Watchdog
        with tempfile.TemporaryDirectory() as td:
            wd = Watchdog(heartbeat_path=Path(td) / "hb.txt")
            wd.write_heartbeat()
            ts = wd.read_heartbeat()
            assert ts is not None
            assert abs(ts - time.time()) < 2.0

    def test_process_alive_fresh_heartbeat(self) -> None:
        """Process should be alive with fresh heartbeat."""
        from live.watchdog import Watchdog
        with tempfile.TemporaryDirectory() as td:
            wd = Watchdog(heartbeat_path=Path(td) / "hb.txt", timeout=5.0)
            wd.write_heartbeat()
            assert wd.is_process_alive() is True

    def test_process_dead_no_heartbeat(self) -> None:
        """Process should be dead with no heartbeat file."""
        from live.watchdog import Watchdog
        wd = Watchdog(heartbeat_path=Path("/nonexistent/hb.txt"))
        assert wd.is_process_alive() is False

    def test_process_dead_stale_heartbeat(self) -> None:
        """Process should be dead with stale heartbeat."""
        from live.watchdog import Watchdog
        with tempfile.TemporaryDirectory() as td:
            hb_path = Path(td) / "hb.txt"
            hb_path.write_text(str(time.time() - 300), encoding="utf-8")
            wd = Watchdog(heartbeat_path=hb_path, timeout=5.0)
            assert wd.is_process_alive() is False


# ── Telegram Bot Tests ──

class TestTelegramBot:
    """Tests for Telegram bot formatting (no actual API calls)."""

    def test_not_configured(self) -> None:
        """Bot without token should report not configured."""
        from utils.telegram_bot import TelegramBot
        bot = TelegramBot(token="", chat_id="")
        assert bot.is_configured is False

    def test_send_returns_false_when_not_configured(self) -> None:
        """send_message should return False when not configured."""
        from utils.telegram_bot import TelegramBot
        bot = TelegramBot(token="", chat_id="")
        assert bot.send_message("test") is False


# ── MT5 Bridge Tests ──

class TestMT5Bridge:
    """Tests for MT5 bridge (without real MT5 connection)."""

    def test_not_connected_by_default(self) -> None:
        """Bridge should not be connected initially."""
        from live.mt5_bridge import MT5Bridge
        bridge = MT5Bridge()
        assert bridge.is_connected is False

    def test_get_account_info_not_connected(self) -> None:
        """Should return error dict when not connected."""
        from live.mt5_bridge import MT5Bridge
        bridge = MT5Bridge()
        info = bridge.get_account_info()
        assert "error" in info

    def test_close_all_not_connected(self) -> None:
        """close_all_positions when not connected returns error."""
        from live.mt5_bridge import MT5Bridge
        bridge = MT5Bridge()
        results = bridge.close_all_positions()
        assert len(results) == 1
        assert results[0].success is False
