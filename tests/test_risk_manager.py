"""TDD Tests for RiskManager — live/risk_manager.py
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

import pytest


class TestRiskManager:
    """Tests for position sizing and risk management."""

    def _make_manager(self, balance: float = 200.0) -> "RiskManager":
        from live.risk_manager import RiskManager
        from configs.validator import load_trading_rules
        from pathlib import Path

        config = load_trading_rules(Path(__file__).parent.parent / "configs")
        return RiskManager(config=config, balance=balance)

    # ── Lot calculation ──

    def test_xauusd_fixed_lot_below_threshold(self) -> None:
        """XAUUSD must use fixed 0.01 lot when balance < $600."""
        mgr = self._make_manager(balance=200.0)
        lot = mgr.calculate_lot(
            symbol="XAUUSD",
            entry_price=2000.0,
            sl_price=1990.0,
            risk_fraction=1.0,
        )
        assert lot == 0.01

    def test_xauusd_dynamic_lot_above_threshold(self) -> None:
        """XAUUSD should use 3% rule when balance >= $600."""
        mgr = self._make_manager(balance=1000.0)
        lot = mgr.calculate_lot(
            symbol="XAUUSD",
            entry_price=2000.0,
            sl_price=1990.0,
            risk_fraction=1.0,
        )
        assert lot > 0.01  # Should be dynamic now

    def test_other_symbol_dynamic_lot(self) -> None:
        """Non-XAUUSD symbols always use 3% risk rule."""
        mgr = self._make_manager(balance=200.0)
        lot = mgr.calculate_lot(
            symbol="BTCUSD",
            entry_price=60000.0,
            sl_price=59500.0,
            risk_fraction=0.5,
        )
        assert lot > 0

    def test_risk_fraction_scales_lot(self) -> None:
        """Higher risk_fraction → bigger lot (for non-XAUUSD)."""
        mgr = self._make_manager(balance=500.0)
        lot_half = mgr.calculate_lot("BTCUSD", 60000.0, 59500.0, risk_fraction=0.5)
        lot_full = mgr.calculate_lot("BTCUSD", 60000.0, 59500.0, risk_fraction=1.0)
        assert lot_full > lot_half

    def test_lot_respects_min_max(self) -> None:
        """Lot should be clamped to [min_lot, max_lot]."""
        mgr = self._make_manager(balance=200.0)
        lot = mgr.calculate_lot("US30", 40000.0, 39999.0, risk_fraction=1.0)
        assert lot >= 0.01  # min_lot

    # ── Drawdown ──

    def test_drawdown_calculation(self) -> None:
        """DD should be (peak - current) / peak."""
        mgr = self._make_manager(balance=200.0)
        mgr.update_balance(200.0)  # peak = 200
        mgr.update_balance(180.0)  # current = 180
        assert mgr.current_dd == pytest.approx(0.10, abs=0.01)  # 10% DD

    def test_killswitch_at_45_pct(self) -> None:
        """Killswitch should trigger at 45% DD."""
        mgr = self._make_manager(balance=200.0)
        mgr.update_balance(200.0)
        mgr.update_balance(110.0)  # 45% DD
        assert mgr.should_killswitch() is True

    def test_force_terminate_at_50_pct(self) -> None:
        """Force terminate at 50% DD."""
        mgr = self._make_manager(balance=200.0)
        mgr.update_balance(200.0)
        mgr.update_balance(100.0)  # 50% DD
        assert mgr.should_force_terminate() is True

    def test_no_killswitch_at_low_dd(self) -> None:
        """No killswitch at normal DD levels."""
        mgr = self._make_manager(balance=200.0)
        mgr.update_balance(200.0)
        mgr.update_balance(190.0)  # 5% DD
        assert mgr.should_killswitch() is False

    # ── Max positions ──

    def test_max_positions_check(self) -> None:
        """Should reject new positions when at max."""
        mgr = self._make_manager(balance=200.0)
        assert mgr.can_open_position(current_open=1) is True
        assert mgr.can_open_position(current_open=2) is False  # max=2

    # ── Confidence threshold ──

    def test_confidence_threshold(self) -> None:
        """Should reject trades below confidence threshold."""
        mgr = self._make_manager(balance=200.0)
        assert mgr.passes_confidence(confidence=0.5) is True
        assert mgr.passes_confidence(confidence=0.2) is False  # < 0.3
