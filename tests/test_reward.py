"""TDD Tests for RewardEngine — environments/reward_engine.py
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

import pytest

from configs.validator import RewardConfig


class TestRewardEngine:
    """Tests for 8-component reward system."""

    def _make_engine(self) -> "RewardEngine":
        from environments.reward_engine import RewardEngine
        config = RewardConfig()
        return RewardEngine(config)

    # ── R_pnl ──

    def test_realized_pnl_positive(self) -> None:
        """Positive PnL trade should give positive reward."""
        engine = self._make_engine()
        r = engine.compute_pnl_reward(realized_pnl=10.0, balance=200.0)
        assert r > 0

    def test_realized_pnl_negative(self) -> None:
        """Negative PnL should give negative reward."""
        engine = self._make_engine()
        r = engine.compute_pnl_reward(realized_pnl=-5.0, balance=200.0)
        assert r < 0

    # ── R_shaping ──

    def test_shaping_unrealized_positive(self) -> None:
        """Positive unrealized PnL = small positive shaping."""
        engine = self._make_engine()
        r = engine.compute_shaping_reward(unrealized_pnl=3.0, balance=200.0)
        assert r > 0
        assert r < 0.1  # Should be small (w_shaping=0.05)

    # ── R_scalp_bonus ──

    def test_scalp_bonus_fast_close(self) -> None:
        """Close < 6 bars with profit → fast bonus."""
        engine = self._make_engine()
        r = engine.compute_scalp_bonus(bars_held=3, realized_pnl=5.0)
        assert r > 0
        assert r >= 0.5  # scalp_bonus_fast = 0.5

    def test_scalp_bonus_medium_close(self) -> None:
        """Close 6-30 bars with profit → medium bonus."""
        engine = self._make_engine()
        r = engine.compute_scalp_bonus(bars_held=15, realized_pnl=5.0)
        assert r > 0
        assert r == pytest.approx(0.2, abs=0.01)

    def test_scalp_bonus_slow_close(self) -> None:
        """Close > 30 bars → no bonus."""
        engine = self._make_engine()
        r = engine.compute_scalp_bonus(bars_held=50, realized_pnl=5.0)
        assert r == 0.0

    def test_scalp_bonus_loss_no_bonus(self) -> None:
        """Close with loss → no bonus even if fast."""
        engine = self._make_engine()
        r = engine.compute_scalp_bonus(bars_held=3, realized_pnl=-2.0)
        assert r == 0.0

    # ── R_rr_bonus ──

    def test_rr_bonus_good_ratio(self) -> None:
        """Good risk/reward ratio should give bonus."""
        engine = self._make_engine()
        r = engine.compute_rr_bonus(realized_pnl=6.0, risk_amount=3.0)
        assert r > 0  # R:R = 2.0

    def test_rr_bonus_bad_ratio(self) -> None:
        """Loss trade → no R:R bonus."""
        engine = self._make_engine()
        r = engine.compute_rr_bonus(realized_pnl=-3.0, risk_amount=3.0)
        assert r == 0.0

    # ── R_cost ──

    def test_cost_penalty_always_negative(self) -> None:
        """Cost penalty should always be negative."""
        engine = self._make_engine()
        r = engine.compute_cost_penalty(
            spread_cost=0.20, commission=0.0, balance=200.0,
        )
        assert r < 0

    def test_cost_penalty_scaled_by_w_cost(self) -> None:
        """Cost should be amplified by w_cost=1.5."""
        engine = self._make_engine()
        r = engine.compute_cost_penalty(
            spread_cost=1.0, commission=0.0, balance=200.0,
        )
        # w_cost=1.5 means penalty is 1.5× actual cost
        assert abs(r) > 1.0 / 200.0  # More than raw normalized cost

    # ── R_hold_penalty ──

    def test_hold_penalty_increases_with_bars(self) -> None:
        """Longer hold → bigger penalty."""
        engine = self._make_engine()
        r1 = engine.compute_hold_penalty(bars_held=5)
        r2 = engine.compute_hold_penalty(bars_held=20)
        assert r2 < r1  # Both negative, r2 more negative

    # ── R_overtrade ──

    def test_overtrade_penalty_within_limit(self) -> None:
        """No penalty if trades_today < max."""
        engine = self._make_engine()
        r = engine.compute_overtrade_penalty(trades_today=5)
        assert r == 0.0

    def test_overtrade_penalty_exceeds_limit(self) -> None:
        """Heavy penalty if trades_today > max_trades_per_day."""
        engine = self._make_engine()
        r = engine.compute_overtrade_penalty(trades_today=15)
        assert r < 0

    # ── R_drawdown ──

    def test_dd_penalty_low_dd(self) -> None:
        """Low DD (10%) → small penalty."""
        engine = self._make_engine()
        r = engine.compute_dd_penalty(current_dd=0.10)
        assert r <= 0
        assert r > -2.0  # Should be moderate at 10% DD

    def test_dd_penalty_high_dd(self) -> None:
        """High DD (45%) → massive penalty (exponential)."""
        engine = self._make_engine()
        r_low = engine.compute_dd_penalty(current_dd=0.10)
        r_high = engine.compute_dd_penalty(current_dd=0.45)
        assert r_high < r_low  # Much more negative

    # ── Total ──

    def test_total_reward_combines_all(self) -> None:
        """Total reward should combine all 8 components."""
        engine = self._make_engine()
        total = engine.compute_total_reward(
            realized_pnl=5.0,
            unrealized_pnl=1.0,
            bars_held=3,
            risk_amount=3.0,
            spread_cost=0.2,
            commission=0.0,
            trades_today=2,
            current_dd=0.05,
            balance=200.0,
        )
        assert isinstance(total, float)
